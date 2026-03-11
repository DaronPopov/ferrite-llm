//! Multi-tenant GPU scheduler for Ferrite-OS.
//!
//! This module provides a complete scheduling subsystem for managing GPU workloads
//! across multiple tenants with resource isolation, fair dispatch, and quota enforcement.
//!
//! # Architecture
//!
//! The scheduler is composed of several collaborating components:
//!
//! - **Tenants** ([`tenant`]): Identity, quotas, and live usage tracking per workload owner.
//! - **Jobs** ([`job`]): Discrete GPU work units with a well-defined lifecycle state machine.
//! - **Policies** ([`policy`]): Pluggable ordering and admission strategies (FIFO, fair-share).
//! - **Registry** ([`registry`]): Central tenant registry with quota management.
//! - **Dispatcher** ([`dispatcher`]): Stream assignment, backpressure, and active job tracking.
//! - **Diagnostics** ([`diagnostics`]): Structured event logging and health monitoring.
//!
//! # Backward Compatibility
//!
//! The default tenant (id=0, unlimited quotas) ensures that existing single-user
//! workloads continue to function without any scheduler configuration. New tenant-aware
//! APIs are purely additive.
//!
//! # Example
//!
//! ```ignore
//! use ptx_runtime::scheduler::*;
//!
//! // Create a scheduler with fair-share policy
//! let mut scheduler = Scheduler::new(SchedulerConfig::default());
//!
//! // Register tenants
//! scheduler.register_tenant(TenantId(1), "inference", TenantQuotas {
//!     max_vram_bytes: 4 * 1024 * 1024 * 1024,
//!     max_streams: 8,
//!     max_concurrent_jobs: 16,
//!     max_runtime_budget_ms: u64::MAX,
//! });
//!
//! // Submit and dispatch jobs
//! let job = Job::new(TenantId(1), StreamPriority::Normal);
//! scheduler.submit(job).unwrap();
//! let dispatched = scheduler.run_cycle(&pool);
//! ```

pub mod tenant;
pub mod job;
pub mod policy;
pub mod registry;
pub mod dispatcher;
pub mod diagnostics;

// Re-export primary types for convenience.
pub use tenant::{Tenant, TenantId, TenantQuotas, TenantUsage, TenantUsageSnapshot};
pub use job::{Job, JobId, JobState, InvalidTransition};
pub use policy::{
    AdmissionDecision, DenialReason, FairSharePolicy, FifoPolicy, SchedulerPolicy,
};
pub use registry::{TenantRegistry, RegistryError};
pub use dispatcher::{DispatchRecord, Dispatcher};
pub use diagnostics::{SchedulerDiagnostics, SchedulerEvent, TenantQuotaSummary};

use serde::Serialize;

use crate::error::Result;
use crate::stream::StreamPool;

/// Point-in-time snapshot of scheduler state for control-plane reporting.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerStateSnapshot {
    pub queue_depth: usize,
    pub active_jobs: usize,
    pub total_completed: u64,
    pub total_failed: u64,
    pub total_denied: u64,
    pub total_dispatched: u64,
    pub total_cancelled: u64,
    pub policy_name: String,
}

/// High-level scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Scheduling policy name: "fair-share" or "fifo".
    pub policy: String,
    /// Default quotas applied to newly registered tenants.
    pub default_tenant_quotas: TenantQuotas,
    /// Maximum number of jobs that can be queued in the dispatcher.
    pub max_queued_jobs: usize,
    /// Maximum number of queued jobs per tenant (0 = no per-tenant limit).
    pub max_queued_jobs_per_tenant: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: "fair-share".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 4096,
            max_queued_jobs_per_tenant: 0,
        }
    }
}

/// The top-level scheduler that composes the registry, policy, and dispatcher.
///
/// This is the primary entry point for multi-tenant scheduling. It owns the
/// tenant registry, the chosen policy, and the job dispatcher.
pub struct Scheduler {
    registry: TenantRegistry,
    policy: Box<dyn SchedulerPolicy>,
    dispatcher: Dispatcher,
    config: SchedulerConfig,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        let policy: Box<dyn SchedulerPolicy> = match config.policy.as_str() {
            "fifo" => Box::new(FifoPolicy),
            _ => Box::new(FairSharePolicy::new()),
        };

        let dispatcher = Dispatcher::with_per_tenant_limit(
            config.max_queued_jobs,
            config.max_queued_jobs_per_tenant,
        );

        tracing::info!(
            policy = policy.name(),
            max_queued_jobs = config.max_queued_jobs,
            "scheduler initialized"
        );

        Self {
            registry: TenantRegistry::new(),
            policy,
            dispatcher,
            config,
        }
    }

    /// Register a new tenant with specific quotas.
    pub fn register_tenant(
        &mut self,
        id: TenantId,
        label: impl Into<String>,
        quotas: TenantQuotas,
    ) -> std::result::Result<(), RegistryError> {
        self.registry.register(id, label, quotas)?;
        Ok(())
    }

    /// Register a new tenant with default quotas from the scheduler config.
    pub fn register_tenant_default(
        &mut self,
        id: TenantId,
        label: impl Into<String>,
    ) -> std::result::Result<(), RegistryError> {
        self.registry
            .register(id, label, self.config.default_tenant_quotas.clone())?;
        Ok(())
    }

    /// Submit a job to the scheduler queue.
    pub fn submit(&mut self, job: Job) -> Result<JobId> {
        self.dispatcher.submit(job)
    }

    /// Run one scheduling cycle: order, admit, dispatch queued jobs.
    pub fn run_cycle(&mut self, pool: &StreamPool) -> Vec<DispatchRecord> {
        self.dispatcher
            .dispatch_cycle(pool, self.policy.as_ref(), &self.registry)
    }

    /// Mark a job as completed.
    pub fn complete_job(&mut self, job_id: JobId) -> Option<DispatchRecord> {
        self.dispatcher.complete_job(job_id, &self.registry)
    }

    /// Mark a job as failed.
    pub fn fail_job(&mut self, job_id: JobId, reason: &str) -> Option<DispatchRecord> {
        self.dispatcher.fail_job(job_id, reason, &self.registry)
    }

    /// Cancel an active (dispatched) job and release its resources.
    pub fn cancel_job(&mut self, job_id: JobId, reason: &str) -> Option<DispatchRecord> {
        self.dispatcher.cancel_job(job_id, reason, &self.registry)
    }

    /// Cancel a queued (not yet dispatched) job.
    pub fn cancel_queued_job(&mut self, job_id: JobId) -> bool {
        self.dispatcher.cancel_queued_job(job_id)
    }

    /// Get a reference to the tenant registry.
    pub fn registry(&self) -> &TenantRegistry {
        &self.registry
    }

    /// Get a mutable reference to the tenant registry.
    pub fn registry_mut(&mut self) -> &mut TenantRegistry {
        &mut self.registry
    }

    /// Get a reference to the dispatcher.
    pub fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }

    /// Get a mutable reference to the dispatcher.
    pub fn dispatcher_mut(&mut self) -> &mut Dispatcher {
        &mut self.dispatcher
    }

    /// Get the active scheduling policy name.
    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    /// Get the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Update tenant VRAM usage (used by memory allocation hooks).
    pub fn update_vram_usage(&self, tenant_id: TenantId, delta: i64) -> Option<u64> {
        self.registry.update_vram_usage(tenant_id, delta)
    }

    /// Take a point-in-time snapshot of the scheduler state for reporting.
    pub fn state_snapshot(&self) -> SchedulerStateSnapshot {
        let diag = self.dispatcher.diagnostics();
        SchedulerStateSnapshot {
            queue_depth: self.dispatcher.queue_len(),
            active_jobs: self.dispatcher.active_count(),
            total_completed: diag.total_completed,
            total_failed: diag.total_failed,
            total_denied: diag.total_denied,
            total_dispatched: diag.total_dispatched,
            total_cancelled: diag.total_cancelled,
            policy_name: self.policy.name().to_string(),
        }
    }

    /// Check if the stream pool is saturated.
    pub fn is_saturated(&self, pool: &StreamPool) -> bool {
        self.dispatcher.is_saturated(pool)
    }

    /// Get the scheduler health diagnostic.
    pub fn health_diagnostic(&self) -> crate::telemetry::DiagnosticEvent {
        self.dispatcher.diagnostics().health_diagnostic()
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("policy", &self.policy.name())
            .field("registry", &self.registry)
            .field("dispatcher", &self.dispatcher)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{Stream, StreamPriority};

    fn mock_pool(n: usize) -> StreamPool {
        let streams: Vec<Stream> = (0..n)
            .map(|i| Stream::new(std::ptr::null_mut(), i as i32))
            .collect();
        StreamPool::new(streams)
    }

    #[test]
    fn test_scheduler_default_config() {
        let config = SchedulerConfig::default();
        assert_eq!(config.policy, "fair-share");
        assert_eq!(config.max_queued_jobs, 4096);
    }

    #[test]
    fn test_scheduler_lifecycle() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        // Default tenant exists
        assert!(sched.registry().get(TenantId::DEFAULT).is_some());

        // Submit a job for the default tenant
        let job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        let job_id = sched.submit(job).unwrap();

        // Run dispatch cycle
        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 1);
        assert_eq!(dispatched[0].job_id, job_id);

        // Complete the job
        let record = sched.complete_job(job_id).unwrap();
        assert_eq!(record.job_id, job_id);
    }

    #[test]
    fn test_scheduler_multi_tenant() {
        let config = SchedulerConfig {
            policy: "fair-share".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 100,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);
        let pool = mock_pool(8);

        sched
            .register_tenant(
                TenantId(1),
                "training",
                TenantQuotas {
                    max_vram_bytes: u64::MAX,
                    max_streams: 4,
                    max_concurrent_jobs: 10,
                    max_runtime_budget_ms: u64::MAX,
                },
            )
            .unwrap();
        sched
            .register_tenant(
                TenantId(2),
                "inference",
                TenantQuotas {
                    max_vram_bytes: u64::MAX,
                    max_streams: 4,
                    max_concurrent_jobs: 10,
                    max_runtime_budget_ms: u64::MAX,
                },
            )
            .unwrap();

        // Submit jobs from both tenants
        for _ in 0..3 {
            sched
                .submit(Job::new(TenantId(1), StreamPriority::Normal))
                .unwrap();
            sched
                .submit(Job::new(TenantId(2), StreamPriority::Normal))
                .unwrap();
        }

        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 6);

        let t1_count = dispatched.iter().filter(|r| r.tenant_id == TenantId(1)).count();
        let t2_count = dispatched.iter().filter(|r| r.tenant_id == TenantId(2)).count();
        assert_eq!(t1_count, 3);
        assert_eq!(t2_count, 3);
    }

    #[test]
    fn test_scheduler_fifo_policy() {
        let config = SchedulerConfig {
            policy: "fifo".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 100,
            ..Default::default()
        };
        let sched = Scheduler::new(config);
        assert_eq!(sched.policy_name(), "fifo");
    }

    #[test]
    fn test_scheduler_vram_tracking() {
        let sched = Scheduler::new(SchedulerConfig::default());

        let new_usage = sched.update_vram_usage(TenantId::DEFAULT, 4096);
        assert_eq!(new_usage, Some(4096));

        let new_usage = sched.update_vram_usage(TenantId::DEFAULT, -1024);
        assert_eq!(new_usage, Some(3072));
    }

    #[test]
    fn test_scheduler_health() {
        let sched = Scheduler::new(SchedulerConfig::default());
        let health = sched.health_diagnostic();
        assert_eq!(health.status, crate::telemetry::DiagnosticStatus::PASS);
    }

    #[test]
    fn test_register_tenant_default_quotas() {
        let config = SchedulerConfig {
            policy: "fifo".to_string(),
            default_tenant_quotas: TenantQuotas {
                max_vram_bytes: 1_000_000,
                max_streams: 2,
                max_concurrent_jobs: 5,
                max_runtime_budget_ms: 30_000,
            },
            max_queued_jobs: 100,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);
        sched.register_tenant_default(TenantId(10), "auto-tenant").unwrap();

        let tenant = sched.registry().get(TenantId(10)).unwrap();
        assert_eq!(tenant.quotas.max_vram_bytes, 1_000_000);
        assert_eq!(tenant.quotas.max_streams, 2);
    }

    // ====================================================================
    // Multi-tenant isolation and quota enforcement integration tests
    // ====================================================================

    /// Helper: verify all usage counters are zero for a tenant.
    fn assert_counters_zero(sched: &Scheduler, tenant_id: TenantId) {
        let t = sched.registry().get(tenant_id).unwrap();
        let snap = t.usage.snapshot();
        assert_eq!(snap.active_streams, 0, "active_streams not zero for {}", tenant_id);
        assert_eq!(snap.active_jobs, 0, "active_jobs not zero for {}", tenant_id);
        assert_eq!(snap.current_vram_bytes, 0, "current_vram_bytes not zero for {}", tenant_id);
    }

    #[test]
    fn test_counter_symmetry_complete_path() {
        // Submit, dispatch, complete → all counters return to zero.
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "t1", TenantQuotas {
            max_vram_bytes: 10_000,
            max_streams: 4,
            max_concurrent_jobs: 4,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let job = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 2048);
        let job_id = sched.submit(job).unwrap();
        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 1);

        // While running: counters should be non-zero
        let t = sched.registry().get(TenantId(1)).unwrap();
        assert_eq!(t.usage.active_jobs.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(t.usage.active_streams.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(t.usage.current_vram_bytes.load(std::sync::atomic::Ordering::Relaxed), 2048);

        // Complete the job
        sched.complete_job(job_id);

        // All counters back to zero (except runtime, which is cumulative)
        assert_counters_zero(&sched, TenantId(1));
    }

    #[test]
    fn test_counter_symmetry_fail_path() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "t1", TenantQuotas {
            max_vram_bytes: 10_000,
            max_streams: 4,
            max_concurrent_jobs: 4,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let job = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 1024);
        let job_id = sched.submit(job).unwrap();
        sched.run_cycle(&pool);

        // Fail the job
        sched.fail_job(job_id, "GPU error");

        assert_counters_zero(&sched, TenantId(1));
    }

    #[test]
    fn test_counter_symmetry_cancel_active_path() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "t1", TenantQuotas {
            max_vram_bytes: 10_000,
            max_streams: 4,
            max_concurrent_jobs: 4,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let job = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 512);
        let job_id = sched.submit(job).unwrap();
        sched.run_cycle(&pool);

        // Cancel active job
        let record = sched.cancel_job(job_id, "user cancelled");
        assert!(record.is_some());

        assert_counters_zero(&sched, TenantId(1));
    }

    #[test]
    fn test_cancel_queued_job() {
        let mut sched = Scheduler::new(SchedulerConfig::default());

        let job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        let job_id = sched.submit(job).unwrap();
        assert_eq!(sched.dispatcher().queue_len(), 1);

        // Cancel before dispatch
        let removed = sched.cancel_queued_job(job_id);
        assert!(removed);
        assert_eq!(sched.dispatcher().queue_len(), 0);
    }

    #[test]
    fn test_vram_admission_uses_estimated_bytes() {
        // Tenant with 4KB VRAM quota. Job estimates 3KB → admitted.
        // Second 3KB job → denied (3+3 > 4).
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "limited-vram", TenantQuotas {
            max_vram_bytes: 4096,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let j1 = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 3072);
        let j2 = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 3072);
        sched.submit(j1).unwrap();
        sched.submit(j2).unwrap();

        let dispatched = sched.run_cycle(&pool);
        // First job passes (3072 <= 4096), second denied (3072+3072 > 4096)
        assert_eq!(dispatched.len(), 1);
        assert_eq!(sched.dispatcher().queue_len(), 1); // second job still queued
    }

    #[test]
    fn test_runtime_budget_blocks_new_jobs() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "limited-rt", TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: 100, // very small budget
        }).unwrap();

        // Artificially exhaust the runtime budget
        sched.registry().get(TenantId(1)).unwrap()
            .usage.consumed_runtime_ms
            .store(200, std::sync::atomic::Ordering::Relaxed);

        let job = Job::new(TenantId(1), StreamPriority::Normal);
        sched.submit(job).unwrap();

        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 0, "job should be denied when runtime budget exhausted");
    }

    #[test]
    fn test_tenant_cannot_starve_another() {
        // Tenant 1 submits many jobs, tenant 2 submits one.
        // With fair-share, tenant 2 should get dispatched.
        let config = SchedulerConfig {
            policy: "fair-share".to_string(),
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "greedy", TenantQuotas::unlimited()).unwrap();
        sched.register_tenant(TenantId(2), "modest", TenantQuotas::unlimited()).unwrap();

        // Tenant 1 submits 10 jobs
        for _ in 0..10 {
            sched.submit(Job::new(TenantId(1), StreamPriority::Normal)).unwrap();
        }
        // Tenant 2 submits 1 job
        sched.submit(Job::new(TenantId(2), StreamPriority::Normal)).unwrap();

        let dispatched = sched.run_cycle(&pool);
        // 4 streams available — both tenants should get at least 1
        let t2_dispatched = dispatched.iter().filter(|r| r.tenant_id == TenantId(2)).count();
        assert!(t2_dispatched >= 1, "tenant 2 should not be starved; got {} dispatches", t2_dispatched);
    }

    #[test]
    fn test_concurrent_job_limit_per_tenant() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(8);

        sched.register_tenant(TenantId(1), "limited-jobs", TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: 2,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        // Submit 5 jobs
        for _ in 0..5 {
            sched.submit(Job::new(TenantId(1), StreamPriority::Normal)).unwrap();
        }

        let dispatched = sched.run_cycle(&pool);
        // Only 2 should dispatch (concurrent limit)
        assert_eq!(dispatched.len(), 2);
        assert_eq!(sched.dispatcher().queue_len(), 3);
    }

    #[test]
    fn test_per_tenant_queue_limit() {
        let config = SchedulerConfig {
            max_queued_jobs: 100,
            max_queued_jobs_per_tenant: 3,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);

        sched.register_tenant(TenantId(1), "t1", TenantQuotas::unlimited()).unwrap();
        sched.register_tenant(TenantId(2), "t2", TenantQuotas::unlimited()).unwrap();

        // Tenant 1 can queue up to 3
        for _ in 0..3 {
            sched.submit(Job::new(TenantId(1), StreamPriority::Normal)).unwrap();
        }
        // 4th should fail
        let err = sched.submit(Job::new(TenantId(1), StreamPriority::Normal));
        assert!(err.is_err(), "4th job for tenant 1 should be rejected");

        // Tenant 2 can still queue (independent limit)
        sched.submit(Job::new(TenantId(2), StreamPriority::Normal)).unwrap();
    }

    #[test]
    fn test_multi_tenant_isolation_vram() {
        // Two tenants each with 4KB VRAM. One exhausting quota shouldn't
        // affect the other.
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(8);

        for i in 1..=2 {
            sched.register_tenant(TenantId(i), &format!("t{}", i), TenantQuotas {
                max_vram_bytes: 4096,
                max_streams: u64::MAX,
                max_concurrent_jobs: u64::MAX,
                max_runtime_budget_ms: u64::MAX,
            }).unwrap();
        }

        // T1 submits 3KB job, T2 submits 3KB job — both should pass
        sched.submit(Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 3072)).unwrap();
        sched.submit(Job::with_vram_estimate(TenantId(2), StreamPriority::Normal, 3072)).unwrap();

        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 2);

        // T1 submits another 3KB job — should be denied (3072+3072 > 4096)
        sched.submit(Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 3072)).unwrap();
        let dispatched2 = sched.run_cycle(&pool);
        assert_eq!(dispatched2.len(), 0, "t1 second job should be denied");

        // T2 submits another 1KB job — should pass (3072+1024 = 4096)
        sched.submit(Job::with_vram_estimate(TenantId(2), StreamPriority::Normal, 1024)).unwrap();
        let dispatched3 = sched.run_cycle(&pool);
        assert_eq!(dispatched3.len(), 1, "t2 should still have VRAM headroom");
        assert_eq!(dispatched3[0].tenant_id, TenantId(2));
    }

    #[test]
    fn test_runtime_accounting_on_completion() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "rt-test", TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let job = Job::new(TenantId(1), StreamPriority::Normal);
        let job_id = sched.submit(job).unwrap();
        sched.run_cycle(&pool);

        // Sleep a bit to accumulate measurable runtime
        std::thread::sleep(std::time::Duration::from_millis(10));

        sched.complete_job(job_id);

        let t = sched.registry().get(TenantId(1)).unwrap();
        let consumed = t.usage.consumed_runtime_ms.load(std::sync::atomic::Ordering::Relaxed);
        assert!(consumed > 0, "consumed runtime should be > 0 after completion, got {}", consumed);
    }

    #[test]
    fn test_vram_released_on_failure() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "t1", TenantQuotas {
            max_vram_bytes: 8192,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let j1 = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 4096);
        let j1_id = sched.submit(j1).unwrap();
        sched.run_cycle(&pool);

        // VRAM is reserved
        let t = sched.registry().get(TenantId(1)).unwrap();
        assert_eq!(t.usage.current_vram_bytes.load(std::sync::atomic::Ordering::Relaxed), 4096);

        // Fail the job → VRAM should be released
        sched.fail_job(j1_id, "kernel panic");

        let t = sched.registry().get(TenantId(1)).unwrap();
        assert_eq!(t.usage.current_vram_bytes.load(std::sync::atomic::Ordering::Relaxed), 0);

        // Now a new 5KB job should be admitted
        sched.submit(Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 5120)).unwrap();
        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 1);
    }

    #[test]
    fn test_quota_denial_diagnostics() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        sched.register_tenant(TenantId(1), "denied", TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: 1,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        for _ in 0..3 {
            sched.submit(Job::new(TenantId(1), StreamPriority::Normal)).unwrap();
        }

        sched.run_cycle(&pool);

        let diag = sched.dispatcher().diagnostics();
        assert!(diag.total_denied > 0, "should have recorded denial events");
        assert_eq!(diag.total_dispatched, 1);
    }

    #[test]
    fn test_dispatch_single_full_quota_checks() {
        let pool = mock_pool(4);
        let mut registry = TenantRegistry::new();
        registry.register(TenantId(1), "limited", TenantQuotas {
            max_vram_bytes: 1024,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        let mut disp = Dispatcher::new(100);

        // dispatch_single with VRAM estimate exceeding quota
        let mut job = Job::with_vram_estimate(TenantId(1), StreamPriority::Normal, 2048);
        let result = disp.dispatch_single(&mut job, &pool, &registry);
        assert!(result.is_err(), "dispatch_single should enforce VRAM quota");
    }
}
