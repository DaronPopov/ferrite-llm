//! Pluggable scheduling policies for the multi-tenant scheduler.
//!
//! Policies control two aspects of scheduling:
//! 1. **Ordering**: Which jobs should be dispatched first.
//! 2. **Admission**: Whether a job should be admitted given current tenant quotas.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use super::job::Job;
use super::tenant::Tenant;

/// The decision returned by an admission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// The job is admitted for scheduling.
    Admit,
    /// The job is denied with a specific reason.
    Deny(DenialReason),
}

impl AdmissionDecision {
    /// Returns true if the decision is to admit.
    pub fn is_admitted(&self) -> bool {
        matches!(self, AdmissionDecision::Admit)
    }
}

/// Reason a job was denied admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenialReason {
    /// Tenant has exceeded its VRAM quota.
    VramQuotaExceeded,
    /// Tenant has exceeded its stream quota.
    StreamQuotaExceeded,
    /// Tenant has reached its concurrent job limit.
    ConcurrentJobLimit,
    /// Tenant has exhausted its cumulative runtime budget.
    RuntimeBudgetExhausted,
    /// Tenant has been administratively suspended.
    TenantSuspended,
}

impl std::fmt::Display for DenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenialReason::VramQuotaExceeded => write!(f, "VRAM quota exceeded"),
            DenialReason::StreamQuotaExceeded => write!(f, "stream quota exceeded"),
            DenialReason::ConcurrentJobLimit => write!(f, "concurrent job limit reached"),
            DenialReason::RuntimeBudgetExhausted => write!(f, "runtime budget exhausted"),
            DenialReason::TenantSuspended => write!(f, "tenant suspended"),
        }
    }
}

/// Trait for pluggable scheduling policies.
///
/// Implementations control how jobs are ordered and whether they pass admission.
pub trait SchedulerPolicy: Send + Sync {
    /// Return a permutation of job indices defining the dispatch order.
    ///
    /// The returned vec contains indices into the `jobs` slice, ordered from
    /// highest priority (first to dispatch) to lowest.
    fn order(&self, jobs: &[&Job]) -> Vec<usize>;

    /// Decide whether a job should be admitted given the current tenant state.
    fn admit(&self, job: &Job, tenant: &Tenant) -> AdmissionDecision;

    /// Human-readable name of this policy (for diagnostics).
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// FifoPolicy
// ---------------------------------------------------------------------------

/// Simple first-in-first-out scheduling policy.
///
/// Jobs are ordered by submission time. Admission checks enforce tenant quotas.
#[derive(Debug)]
pub struct FifoPolicy;

impl SchedulerPolicy for FifoPolicy {
    fn order(&self, jobs: &[&Job]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..jobs.len()).collect();
        indices.sort_by_key(|&i| jobs[i].submitted_at);
        indices
    }

    fn admit(&self, job: &Job, tenant: &Tenant) -> AdmissionDecision {
        check_quotas(job, tenant)
    }

    fn name(&self) -> &'static str {
        "fifo"
    }
}

// ---------------------------------------------------------------------------
// FairSharePolicy
// ---------------------------------------------------------------------------

/// Fair-share scheduling policy with tenant-aware round-robin.
///
/// Jobs are interleaved across tenants so that no single tenant can starve others.
/// Within each tenant's slice, jobs are ordered by priority then submission time.
/// Admission checks enforce all tenant quotas.
pub struct FairSharePolicy {
    /// Tracks which tenant index to serve next for round-robin fairness.
    next_tenant_slot: AtomicUsize,
}

impl FairSharePolicy {
    /// Create a new fair-share policy.
    pub fn new() -> Self {
        Self {
            next_tenant_slot: AtomicUsize::new(0),
        }
    }
}

impl Default for FairSharePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FairSharePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FairSharePolicy")
            .field("next_tenant_slot", &self.next_tenant_slot.load(AtomicOrdering::Relaxed))
            .finish()
    }
}

impl SchedulerPolicy for FairSharePolicy {
    fn order(&self, jobs: &[&Job]) -> Vec<usize> {
        if jobs.is_empty() {
            return Vec::new();
        }

        // Group job indices by tenant
        let mut tenant_jobs: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
        for (i, job) in jobs.iter().enumerate() {
            tenant_jobs.entry(job.tenant_id.0).or_default().push(i);
        }

        // Sort each tenant's jobs by priority (lower enum value = higher priority),
        // then by submission time (earlier = first)
        for indices in tenant_jobs.values_mut() {
            indices.sort_by(|&a, &b| {
                let pa = jobs[a].priority as u32;
                let pb = jobs[b].priority as u32;
                pa.cmp(&pb).then_with(|| jobs[a].submitted_at.cmp(&jobs[b].submitted_at))
            });
        }

        // Collect tenant IDs in a stable order
        let mut tenant_ids: Vec<u64> = tenant_jobs.keys().copied().collect();
        tenant_ids.sort();

        // Round-robin interleave: rotate starting tenant based on our counter
        let start = self.next_tenant_slot.fetch_add(1, AtomicOrdering::Relaxed) % tenant_ids.len();

        let mut result = Vec::with_capacity(jobs.len());
        let mut cursors: Vec<usize> = vec![0; tenant_ids.len()];
        let max_len = tenant_jobs.values().map(|v| v.len()).max().unwrap_or(0);

        for round in 0..max_len {
            for offset in 0..tenant_ids.len() {
                let ti = (start + offset + round) % tenant_ids.len();
                let tid = tenant_ids[ti];
                let tenant_list = &tenant_jobs[&tid];
                // Each tenant contributes one job per round
                let cursor = &mut cursors[ti];
                if *cursor < tenant_list.len() {
                    result.push(tenant_list[*cursor]);
                    *cursor += 1;
                }
            }
        }

        // Deduplicate (in case round-robin logic produces duplicates at edges)
        let mut seen = vec![false; jobs.len()];
        result.retain(|&idx| {
            if seen[idx] {
                false
            } else {
                seen[idx] = true;
                true
            }
        });

        result
    }

    fn admit(&self, job: &Job, tenant: &Tenant) -> AdmissionDecision {
        check_quotas(job, tenant)
    }

    fn name(&self) -> &'static str {
        "fair-share"
    }
}

/// Shared quota checking logic used by all built-in policies.
///
/// Checks all four quota dimensions. For VRAM, the job's `estimated_vram_bytes`
/// is added to the tenant's current usage to determine whether the allocation
/// would exceed the quota.
fn check_quotas(job: &Job, tenant: &Tenant) -> AdmissionDecision {
    if !tenant.runtime_within_budget() {
        return AdmissionDecision::Deny(DenialReason::RuntimeBudgetExhausted);
    }
    if !tenant.jobs_within_quota() {
        return AdmissionDecision::Deny(DenialReason::ConcurrentJobLimit);
    }
    // VRAM: current usage + this job's estimate must not exceed quota
    if tenant.quotas.max_vram_bytes != u64::MAX {
        let current = tenant.usage.current_vram_bytes.load(std::sync::atomic::Ordering::Relaxed);
        if current.saturating_add(job.estimated_vram_bytes) > tenant.quotas.max_vram_bytes {
            return AdmissionDecision::Deny(DenialReason::VramQuotaExceeded);
        }
    } else if !tenant.vram_within_quota() {
        return AdmissionDecision::Deny(DenialReason::VramQuotaExceeded);
    }
    if !tenant.streams_within_quota() {
        return AdmissionDecision::Deny(DenialReason::StreamQuotaExceeded);
    }
    AdmissionDecision::Admit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::tenant::{TenantId, TenantQuotas};
    use crate::stream::StreamPriority;
    use std::sync::atomic::Ordering as AO;

    fn make_job(tenant_id: u64, priority: StreamPriority) -> Job {
        Job::new(TenantId(tenant_id), priority)
    }

    #[test]
    fn test_fifo_ordering() {
        let j1 = make_job(1, StreamPriority::Normal);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let j2 = make_job(1, StreamPriority::High);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let j3 = make_job(2, StreamPriority::Low);

        let jobs: Vec<&Job> = vec![&j3, &j1, &j2];
        let policy = FifoPolicy;
        let order = policy.order(&jobs);

        // j1 submitted first, j2 second, j3 third
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn test_fifo_admit_within_quota() {
        let tenant = Tenant::default_tenant();
        let job = make_job(0, StreamPriority::Normal);
        let policy = FifoPolicy;

        assert_eq!(policy.admit(&job, &tenant), AdmissionDecision::Admit);
    }

    #[test]
    fn test_admit_deny_vram() {
        let quotas = TenantQuotas {
            max_vram_bytes: 100,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        };
        let tenant = Tenant::new(TenantId(1), "test", quotas);
        tenant.usage.current_vram_bytes.store(200, AO::Relaxed);

        let job = make_job(1, StreamPriority::Normal);
        let policy = FifoPolicy;
        assert_eq!(
            policy.admit(&job, &tenant),
            AdmissionDecision::Deny(DenialReason::VramQuotaExceeded)
        );
    }

    #[test]
    fn test_admit_deny_streams() {
        let quotas = TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: 2,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        };
        let tenant = Tenant::new(TenantId(1), "test", quotas);
        tenant.usage.active_streams.store(3, AO::Relaxed);

        let job = make_job(1, StreamPriority::Normal);
        let policy = FifoPolicy;
        assert_eq!(
            policy.admit(&job, &tenant),
            AdmissionDecision::Deny(DenialReason::StreamQuotaExceeded)
        );
    }

    #[test]
    fn test_admit_deny_concurrent_jobs() {
        let quotas = TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: 1,
            max_runtime_budget_ms: u64::MAX,
        };
        let tenant = Tenant::new(TenantId(1), "test", quotas);
        tenant.usage.active_jobs.store(1, AO::Relaxed);

        let job = make_job(1, StreamPriority::Normal);
        let policy = FifoPolicy;
        assert_eq!(
            policy.admit(&job, &tenant),
            AdmissionDecision::Deny(DenialReason::ConcurrentJobLimit)
        );
    }

    #[test]
    fn test_admit_deny_runtime_budget() {
        let quotas = TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: 1000,
        };
        let tenant = Tenant::new(TenantId(1), "test", quotas);
        tenant.usage.consumed_runtime_ms.store(2000, AO::Relaxed);

        let job = make_job(1, StreamPriority::Normal);
        let policy = FifoPolicy;
        assert_eq!(
            policy.admit(&job, &tenant),
            AdmissionDecision::Deny(DenialReason::RuntimeBudgetExhausted)
        );
    }

    #[test]
    fn test_fair_share_interleaving() {
        // Create jobs from two tenants
        let j1_t1 = make_job(1, StreamPriority::Normal);
        let j2_t1 = make_job(1, StreamPriority::Normal);
        let j1_t2 = make_job(2, StreamPriority::Normal);
        let j2_t2 = make_job(2, StreamPriority::Normal);

        let jobs: Vec<&Job> = vec![&j1_t1, &j2_t1, &j1_t2, &j2_t2];
        let policy = FairSharePolicy::new();
        let order = policy.order(&jobs);

        // Should interleave between tenants, all 4 jobs present
        assert_eq!(order.len(), 4);

        // Verify no duplicates
        let mut seen = std::collections::HashSet::new();
        for &idx in &order {
            assert!(seen.insert(idx), "duplicate index {} in order", idx);
        }
    }

    #[test]
    fn test_fair_share_empty() {
        let jobs: Vec<&Job> = vec![];
        let policy = FairSharePolicy::new();
        let order = policy.order(&jobs);
        assert!(order.is_empty());
    }

    #[test]
    fn test_fair_share_single_tenant() {
        let j1 = make_job(1, StreamPriority::High);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let j2 = make_job(1, StreamPriority::Normal);

        let jobs: Vec<&Job> = vec![&j2, &j1];
        let policy = FairSharePolicy::new();
        let order = policy.order(&jobs);

        // With single tenant, should order by priority then time
        // j1 is High (1), j2 is Normal (2), so j1 first
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn test_denial_reason_display() {
        assert_eq!(DenialReason::VramQuotaExceeded.to_string(), "VRAM quota exceeded");
        assert_eq!(DenialReason::TenantSuspended.to_string(), "tenant suspended");
    }

    #[test]
    fn test_policy_names() {
        assert_eq!(FifoPolicy.name(), "fifo");
        assert_eq!(FairSharePolicy::new().name(), "fair-share");
    }
}
