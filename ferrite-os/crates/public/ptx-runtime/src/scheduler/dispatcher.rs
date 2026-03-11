//! Execution dispatcher for the multi-tenant scheduler.
//!
//! The dispatcher is responsible for assigning admitted jobs to streams,
//! tracking active jobs, and enforcing backpressure when the stream pool
//! is saturated.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::error::{Error, Result};
use crate::stream::{Stream, StreamPool};

use super::job::{Job, JobId};
use super::policy::{AdmissionDecision, DenialReason, SchedulerPolicy};
use super::registry::TenantRegistry;
use super::diagnostics::SchedulerDiagnostics;
use super::tenant::TenantId;

/// Maps a dispatched job to the stream it was assigned to.
#[derive(Debug, Clone)]
pub struct DispatchRecord {
    pub job_id: JobId,
    pub tenant_id: TenantId,
    pub stream_id: i32,
    /// VRAM bytes reserved for this job (used for accounting on completion/failure).
    pub estimated_vram_bytes: u64,
    /// When the job started running (for runtime budget accounting).
    pub started_at: Instant,
}

/// The execution dispatcher.
///
/// Bridges the scheduling policy and the stream pool. It maintains the job queue,
/// runs admission checks, assigns streams, and tracks active jobs.
pub struct Dispatcher {
    /// Jobs waiting to be admitted and dispatched.
    queue: Vec<Job>,
    /// Currently running jobs, keyed by JobId.
    active_jobs: HashMap<u64, DispatchRecord>,
    /// Maximum number of jobs that can be queued.
    max_queued_jobs: usize,
    /// Maximum number of queued jobs per tenant (0 = no limit).
    max_queued_per_tenant: usize,
    /// Per-tenant round-robin cursor into the stream pool.
    tenant_stream_cursors: HashMap<u64, usize>,
    /// Diagnostics emitter.
    diagnostics: SchedulerDiagnostics,
}

impl Dispatcher {
    /// Create a new dispatcher.
    pub fn new(max_queued_jobs: usize) -> Self {
        Self::with_per_tenant_limit(max_queued_jobs, 0)
    }

    /// Create a new dispatcher with a per-tenant queue limit.
    ///
    /// A `max_queued_per_tenant` of 0 means no per-tenant limit.
    pub fn with_per_tenant_limit(max_queued_jobs: usize, max_queued_per_tenant: usize) -> Self {
        Self {
            queue: Vec::new(),
            active_jobs: HashMap::new(),
            max_queued_jobs,
            max_queued_per_tenant,
            tenant_stream_cursors: HashMap::new(),
            diagnostics: SchedulerDiagnostics::new(),
        }
    }

    /// Submit a job to the dispatcher queue.
    ///
    /// Returns an error if the global queue is full or if the per-tenant queue
    /// limit would be exceeded (backpressure).
    pub fn submit(&mut self, job: Job) -> Result<JobId> {
        if self.queue.len() >= self.max_queued_jobs {
            tracing::warn!(
                queue_len = self.queue.len(),
                max = self.max_queued_jobs,
                "job queue full, applying backpressure"
            );
            return Err(Error::SchedulerError {
                detail: format!(
                    "job queue full ({}/{}), backpressure applied",
                    self.queue.len(),
                    self.max_queued_jobs
                ),
            });
        }

        // Per-tenant queue limit
        if self.max_queued_per_tenant > 0 {
            let tenant_queued = self
                .queue
                .iter()
                .filter(|j| j.tenant_id == job.tenant_id)
                .count();
            if tenant_queued >= self.max_queued_per_tenant {
                tracing::warn!(
                    tenant_id = job.tenant_id.0,
                    tenant_queued,
                    limit = self.max_queued_per_tenant,
                    "per-tenant queue limit reached"
                );
                return Err(Error::QuotaExceeded {
                    tenant_id: job.tenant_id.0,
                    resource: "queued_jobs".to_string(),
                    limit: self.max_queued_per_tenant as u64,
                    current: tenant_queued as u64,
                });
            }
        }

        let job_id = job.id;
        let tenant_id = job.tenant_id;
        self.diagnostics.record_job_queued(job_id, tenant_id);
        self.queue.push(job);

        tracing::debug!(job_id = job_id.0, tenant_id = tenant_id.0, "job queued");
        Ok(job_id)
    }

    /// Run one scheduling cycle: order, admit, and dispatch as many queued jobs
    /// as possible.
    ///
    /// Returns the list of newly dispatched jobs.
    pub fn dispatch_cycle(
        &mut self,
        pool: &StreamPool,
        policy: &dyn SchedulerPolicy,
        registry: &TenantRegistry,
    ) -> Vec<DispatchRecord> {
        if self.queue.is_empty() {
            return Vec::new();
        }

        // Collect references for ordering
        let job_refs: Vec<&Job> = self.queue.iter().collect();
        let order = policy.order(&job_refs);

        let mut dispatched = Vec::new();
        let mut to_remove: Vec<usize> = Vec::new();

        // Track how many streams are currently occupied
        let occupied_streams = self.active_jobs.len();
        let total_streams = pool.len();
        let mut available_streams = total_streams.saturating_sub(occupied_streams);

        for &idx in &order {
            if available_streams == 0 {
                tracing::debug!("all streams occupied, stopping dispatch cycle");
                break;
            }

            let job = &self.queue[idx];
            let tenant_id = job.tenant_id;

            // Look up tenant
            let tenant = match registry.get(tenant_id) {
                Some(t) => t,
                None => {
                    tracing::warn!(
                        tenant_id = tenant_id.0,
                        job_id = job.id.0,
                        "tenant not found in registry, skipping job"
                    );
                    continue;
                }
            };

            // Admission check
            let decision = policy.admit(job, tenant);
            match decision {
                AdmissionDecision::Admit => {
                    self.diagnostics.record_job_admitted(job.id, tenant_id);
                }
                AdmissionDecision::Deny(reason) => {
                    self.diagnostics.record_job_denied(job.id, tenant_id, &reason);
                    tracing::info!(
                        job_id = job.id.0,
                        tenant_id = tenant_id.0,
                        reason = %reason,
                        "job admission denied"
                    );
                    continue;
                }
            }

            // Check per-tenant stream quota
            if !tenant.streams_within_quota() {
                self.diagnostics.record_job_denied(
                    job.id,
                    tenant_id,
                    &DenialReason::StreamQuotaExceeded,
                );
                continue;
            }

            // Assign a stream (tenant-aware round-robin within the pool)
            let cursor = self
                .tenant_stream_cursors
                .entry(tenant_id.0)
                .or_insert(0);
            let stream_idx = *cursor % total_streams;
            *cursor = cursor.wrapping_add(1);

            let stream = match pool.get(stream_idx) {
                Some(s) => s,
                None => {
                    tracing::error!(
                        stream_idx,
                        pool_size = total_streams,
                        "stream index out of bounds"
                    );
                    continue;
                }
            };

            to_remove.push(idx);

            let now = Instant::now();
            let record = DispatchRecord {
                job_id: job.id,
                tenant_id,
                stream_id: stream.id(),
                estimated_vram_bytes: job.estimated_vram_bytes,
                started_at: now,
            };

            // Update tenant usage atomics
            tenant.usage.active_streams.fetch_add(1, Ordering::Relaxed);
            tenant.usage.active_jobs.fetch_add(1, Ordering::Relaxed);
            if job.estimated_vram_bytes > 0 {
                tenant.usage.current_vram_bytes.fetch_add(job.estimated_vram_bytes, Ordering::Relaxed);
            }

            self.diagnostics
                .record_job_dispatched(job.id, tenant_id, stream.id());

            tracing::info!(
                job_id = job.id.0,
                tenant_id = tenant_id.0,
                stream_id = stream.id(),
                "job dispatched to stream"
            );

            dispatched.push(record.clone());
            self.active_jobs.insert(job.id.0, record);
            available_streams -= 1;
        }

        // Remove dispatched jobs from the queue (in reverse index order to preserve positions)
        to_remove.sort_unstable();
        to_remove.dedup();
        for &idx in to_remove.iter().rev() {
            let mut job = self.queue.remove(idx);
            // Transition: Queued -> Admitted -> Running
            let _ = job.admit();
            let _ = job.start();
        }

        // Check for starvation: tenants with queued jobs that never got dispatched
        self.detect_starvation(registry);

        // Emit quota warnings
        self.emit_quota_warnings(registry);

        dispatched
    }

    /// Mark a job as completed and release its resources.
    ///
    /// Decrements all tenant usage counters (streams, jobs, VRAM) and accounts
    /// the elapsed wall-clock runtime against the tenant's runtime budget.
    pub fn complete_job(
        &mut self,
        job_id: JobId,
        registry: &TenantRegistry,
    ) -> Option<DispatchRecord> {
        let record = self.active_jobs.remove(&job_id.0)?;

        Self::release_tenant_resources(&record, registry);

        self.diagnostics
            .record_job_completed(job_id, record.tenant_id);

        tracing::info!(
            job_id = job_id.0,
            tenant_id = record.tenant_id.0,
            stream_id = record.stream_id,
            "job completed"
        );

        Some(record)
    }

    /// Mark a job as failed and release its resources.
    ///
    /// Decrements all tenant usage counters symmetrically with dispatch.
    pub fn fail_job(
        &mut self,
        job_id: JobId,
        reason: &str,
        registry: &TenantRegistry,
    ) -> Option<DispatchRecord> {
        let record = self.active_jobs.remove(&job_id.0)?;

        Self::release_tenant_resources(&record, registry);

        self.diagnostics.record_job_failed(job_id, record.tenant_id);

        tracing::warn!(
            job_id = job_id.0,
            tenant_id = record.tenant_id.0,
            reason = reason,
            "job failed"
        );

        Some(record)
    }

    /// Cancel an active (dispatched) job and release its resources.
    ///
    /// Returns `None` if the job is not currently active. For queued jobs that
    /// have not yet been dispatched, use [`cancel_queued_job`] instead.
    pub fn cancel_job(
        &mut self,
        job_id: JobId,
        reason: &str,
        registry: &TenantRegistry,
    ) -> Option<DispatchRecord> {
        let record = self.active_jobs.remove(&job_id.0)?;

        Self::release_tenant_resources(&record, registry);

        self.diagnostics
            .record_job_cancelled(job_id, record.tenant_id);

        tracing::info!(
            job_id = job_id.0,
            tenant_id = record.tenant_id.0,
            reason = reason,
            "active job cancelled"
        );

        Some(record)
    }

    /// Remove a queued (not yet dispatched) job from the queue.
    ///
    /// Returns true if the job was found and removed.
    pub fn cancel_queued_job(&mut self, job_id: JobId) -> bool {
        if let Some(pos) = self.queue.iter().position(|j| j.id == job_id) {
            let mut job = self.queue.remove(pos);
            let _ = job.cancel("cancelled while queued");
            self.diagnostics
                .record_job_cancelled(job_id, job.tenant_id);
            tracing::info!(
                job_id = job_id.0,
                tenant_id = job.tenant_id.0,
                "queued job cancelled"
            );
            true
        } else {
            false
        }
    }

    /// Dispatch a single pre-created job directly to a stream from the pool.
    ///
    /// This is a convenience method for simpler use cases where the caller manages
    /// the job lifecycle externally. All tenant quotas are enforced.
    pub fn dispatch_single(
        &mut self,
        job: &mut Job,
        pool: &StreamPool,
        registry: &TenantRegistry,
    ) -> Result<Stream> {
        let tenant = registry.get(job.tenant_id).ok_or_else(|| Error::SchedulerError {
            detail: format!("tenant {} not found", job.tenant_id),
        })?;

        // Full quota check (all four dimensions)
        if !tenant.runtime_within_budget() {
            return Err(Error::QuotaExceeded {
                tenant_id: job.tenant_id.0,
                resource: "runtime_budget_ms".to_string(),
                limit: tenant.quotas.max_runtime_budget_ms,
                current: tenant.usage.consumed_runtime_ms.load(Ordering::Relaxed),
            });
        }
        if !tenant.jobs_within_quota() {
            return Err(Error::QuotaExceeded {
                tenant_id: job.tenant_id.0,
                resource: "concurrent_jobs".to_string(),
                limit: tenant.quotas.max_concurrent_jobs,
                current: tenant.usage.active_jobs.load(Ordering::Relaxed),
            });
        }
        if tenant.quotas.max_vram_bytes != u64::MAX {
            let current_vram = tenant.usage.current_vram_bytes.load(Ordering::Relaxed);
            if current_vram.saturating_add(job.estimated_vram_bytes) > tenant.quotas.max_vram_bytes {
                return Err(Error::QuotaExceeded {
                    tenant_id: job.tenant_id.0,
                    resource: "vram_bytes".to_string(),
                    limit: tenant.quotas.max_vram_bytes,
                    current: current_vram,
                });
            }
        }
        if !tenant.streams_within_quota() {
            return Err(Error::QuotaExceeded {
                tenant_id: job.tenant_id.0,
                resource: "streams".to_string(),
                limit: tenant.quotas.max_streams,
                current: tenant.usage.active_streams.load(Ordering::Relaxed),
            });
        }

        if pool.is_empty() {
            return Err(Error::StreamError {
                message: "stream pool is empty".to_string(),
            });
        }

        // All streams occupied check
        let occupied = self.active_jobs.len();
        if occupied >= pool.len() {
            return Err(Error::SchedulerError {
                detail: format!(
                    "all {} streams occupied ({} active jobs), backpressure",
                    pool.len(),
                    occupied
                ),
            });
        }

        let stream = pool.next();
        job.admit().map_err(|e| Error::SchedulerError {
            detail: e.to_string(),
        })?;
        job.start().map_err(|e| Error::SchedulerError {
            detail: e.to_string(),
        })?;

        tenant.usage.active_streams.fetch_add(1, Ordering::Relaxed);
        tenant.usage.active_jobs.fetch_add(1, Ordering::Relaxed);
        if job.estimated_vram_bytes > 0 {
            tenant.usage.current_vram_bytes.fetch_add(job.estimated_vram_bytes, Ordering::Relaxed);
        }

        let now = Instant::now();
        let record = DispatchRecord {
            job_id: job.id,
            tenant_id: job.tenant_id,
            stream_id: stream.id(),
            estimated_vram_bytes: job.estimated_vram_bytes,
            started_at: now,
        };
        self.active_jobs.insert(job.id.0, record);

        self.diagnostics
            .record_job_dispatched(job.id, job.tenant_id, stream.id());

        Ok(stream)
    }

    /// Release tenant resources for a completed/failed/cancelled job.
    ///
    /// This is the symmetric counterpart to the resource acquisition in
    /// `dispatch_cycle` / `dispatch_single`. It decrements streams, jobs,
    /// VRAM, and accounts the elapsed runtime.
    fn release_tenant_resources(record: &DispatchRecord, registry: &TenantRegistry) {
        if let Some(tenant) = registry.get(record.tenant_id) {
            tenant
                .usage
                .active_streams
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(1))
                })
                .ok();
            tenant
                .usage
                .active_jobs
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(1))
                })
                .ok();
            // Release VRAM reservation
            if record.estimated_vram_bytes > 0 {
                tenant
                    .usage
                    .current_vram_bytes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(record.estimated_vram_bytes))
                    })
                    .ok();
            }
            // Account elapsed runtime against the tenant's budget
            let elapsed_ms = record.started_at.elapsed().as_millis() as u64;
            if elapsed_ms > 0 {
                tenant
                    .usage
                    .consumed_runtime_ms
                    .fetch_add(elapsed_ms, Ordering::Relaxed);
            }
        }
    }

    /// Get the number of queued jobs.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get the number of active (dispatched) jobs.
    pub fn active_count(&self) -> usize {
        self.active_jobs.len()
    }

    /// Get an immutable reference to the queued jobs for inspection.
    pub fn queued_jobs(&self) -> &[Job] {
        &self.queue
    }

    /// Get a snapshot of all active dispatch records.
    pub fn active_records(&self) -> Vec<&DispatchRecord> {
        self.active_jobs.values().collect()
    }

    /// Get a reference to the diagnostics subsystem.
    pub fn diagnostics(&self) -> &SchedulerDiagnostics {
        &self.diagnostics
    }

    /// Check if all streams would be occupied (backpressure condition).
    pub fn is_saturated(&self, pool: &StreamPool) -> bool {
        self.active_jobs.len() >= pool.len()
    }

    /// Detect tenants whose jobs are starving (sitting in queue while others dispatch).
    fn detect_starvation(&mut self, _registry: &TenantRegistry) {
        // Build a set of tenant IDs that have queued jobs
        let mut queued_tenants: HashMap<u64, usize> = HashMap::new();
        for job in &self.queue {
            *queued_tenants.entry(job.tenant_id.0).or_insert(0) += 1;
        }

        // Build a set of tenant IDs that have active jobs
        let mut active_tenants: HashMap<u64, usize> = HashMap::new();
        for record in self.active_jobs.values() {
            *active_tenants.entry(record.tenant_id.0).or_insert(0) += 1;
        }

        // A tenant is "starving" if it has queued jobs but no active jobs,
        // while other tenants DO have active jobs.
        let any_active = !active_tenants.is_empty();
        for (&tenant_id, &queued_count) in &queued_tenants {
            if any_active && !active_tenants.contains_key(&tenant_id) && queued_count > 0 {
                let tid = TenantId(tenant_id);
                self.diagnostics.record_starvation(tid);
                tracing::warn!(
                    tenant_id,
                    queued_jobs = queued_count,
                    "starvation detected: tenant has queued jobs but no active dispatches"
                );
            }
        }
    }

    /// Emit quota warnings for tenants approaching their limits.
    fn emit_quota_warnings(&mut self, registry: &TenantRegistry) {
        for tenant_id in registry.list() {
            if let Some(tenant) = registry.get(tenant_id) {
                // VRAM warning at 80%
                let vram_pct = tenant.vram_utilization_pct();
                if vram_pct > 80.0 {
                    self.diagnostics
                        .record_quota_warning(tenant_id, "vram", vram_pct);
                }

                // Concurrent jobs warning at 80%
                if tenant.quotas.max_concurrent_jobs != u64::MAX {
                    let jobs_pct = (tenant.usage.active_jobs.load(Ordering::Relaxed) as f64
                        / tenant.quotas.max_concurrent_jobs as f64)
                        * 100.0;
                    if jobs_pct > 80.0 {
                        self.diagnostics
                            .record_quota_warning(tenant_id, "concurrent_jobs", jobs_pct);
                    }
                }

                // Stream quota warning at 80%
                if tenant.quotas.max_streams != u64::MAX {
                    let streams_pct = (tenant.usage.active_streams.load(Ordering::Relaxed) as f64
                        / tenant.quotas.max_streams as f64)
                        * 100.0;
                    if streams_pct > 80.0 {
                        self.diagnostics
                            .record_quota_warning(tenant_id, "streams", streams_pct);
                    }
                }

                // Runtime budget warning at 80%
                if tenant.quotas.max_runtime_budget_ms != u64::MAX {
                    let rt_pct = (tenant.usage.consumed_runtime_ms.load(Ordering::Relaxed) as f64
                        / tenant.quotas.max_runtime_budget_ms as f64)
                        * 100.0;
                    if rt_pct > 80.0 {
                        self.diagnostics
                            .record_quota_warning(tenant_id, "runtime_budget", rt_pct);
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for Dispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dispatcher")
            .field("queued_jobs", &self.queue.len())
            .field("active_jobs", &self.active_jobs.len())
            .field("max_queued_jobs", &self.max_queued_jobs)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::job::{Job, JobState};
    use crate::scheduler::policy::{FairSharePolicy, FifoPolicy};
    use crate::scheduler::tenant::{TenantId, TenantQuotas};
    use crate::stream::{Stream, StreamPriority};

    /// Create a mock stream pool for testing (no real CUDA streams).
    fn mock_pool(n: usize) -> StreamPool {
        let streams: Vec<Stream> = (0..n)
            .map(|i| Stream::new(std::ptr::null_mut(), i as i32))
            .collect();
        StreamPool::new(streams)
    }

    #[test]
    fn test_submit_job() {
        let mut disp = Dispatcher::new(100);
        let job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        let id = disp.submit(job).unwrap();
        assert_eq!(disp.queue_len(), 1);
        assert!(id.0 > 0);
    }

    #[test]
    fn test_submit_backpressure() {
        let mut disp = Dispatcher::new(2);
        disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
            .unwrap();
        disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
            .unwrap();

        let result = disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal));
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_cycle_fifo() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(4);
        let registry = TenantRegistry::new();
        let policy = FifoPolicy;

        // Submit 3 jobs
        for _ in 0..3 {
            disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
                .unwrap();
        }

        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);
        assert_eq!(dispatched.len(), 3);
        assert_eq!(disp.queue_len(), 0);
        assert_eq!(disp.active_count(), 3);
    }

    #[test]
    fn test_dispatch_cycle_saturated() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(2);
        let registry = TenantRegistry::new();
        let policy = FifoPolicy;

        // Submit 5 jobs but only 2 streams
        for _ in 0..5 {
            disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
                .unwrap();
        }

        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);
        assert_eq!(dispatched.len(), 2);
        assert_eq!(disp.queue_len(), 3);
        assert_eq!(disp.active_count(), 2);
        assert!(disp.is_saturated(&pool));
    }

    #[test]
    fn test_complete_job() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(4);
        let registry = TenantRegistry::new();
        let policy = FifoPolicy;

        disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
            .unwrap();
        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);
        assert_eq!(dispatched.len(), 1);

        let record = disp.complete_job(dispatched[0].job_id, &registry).unwrap();
        assert_eq!(record.job_id, dispatched[0].job_id);
        assert_eq!(disp.active_count(), 0);
    }

    #[test]
    fn test_fail_job() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(4);
        let registry = TenantRegistry::new();
        let policy = FifoPolicy;

        disp.submit(Job::new(TenantId::DEFAULT, StreamPriority::Normal))
            .unwrap();
        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);

        let record = disp
            .fail_job(dispatched[0].job_id, "test failure", &registry)
            .unwrap();
        assert_eq!(record.tenant_id, TenantId::DEFAULT);
        assert_eq!(disp.active_count(), 0);
    }

    #[test]
    fn test_multi_tenant_dispatch() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(4);
        let mut registry = TenantRegistry::new();
        registry
            .register(TenantId(1), "tenant-1", TenantQuotas::unlimited())
            .unwrap();
        registry
            .register(TenantId(2), "tenant-2", TenantQuotas::unlimited())
            .unwrap();
        let policy = FairSharePolicy::new();

        // 2 jobs from each tenant
        for _ in 0..2 {
            disp.submit(Job::new(TenantId(1), StreamPriority::Normal))
                .unwrap();
            disp.submit(Job::new(TenantId(2), StreamPriority::Normal))
                .unwrap();
        }

        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);
        assert_eq!(dispatched.len(), 4);

        // Both tenants should have dispatched jobs
        let t1_dispatched = dispatched.iter().filter(|r| r.tenant_id == TenantId(1)).count();
        let t2_dispatched = dispatched.iter().filter(|r| r.tenant_id == TenantId(2)).count();
        assert_eq!(t1_dispatched, 2);
        assert_eq!(t2_dispatched, 2);
    }

    #[test]
    fn test_quota_enforcement_in_dispatch() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(4);
        let mut registry = TenantRegistry::new();
        let quotas = TenantQuotas {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: 1, // Only 1 concurrent job allowed
            max_runtime_budget_ms: u64::MAX,
        };
        registry.register(TenantId(1), "limited", quotas).unwrap();
        let policy = FifoPolicy;

        // Submit 3 jobs for the limited tenant
        for _ in 0..3 {
            disp.submit(Job::new(TenantId(1), StreamPriority::Normal))
                .unwrap();
        }

        // First cycle: should dispatch 1 (then concurrent limit kicks in for rest)
        let dispatched = disp.dispatch_cycle(&pool, &policy, &registry);
        assert_eq!(dispatched.len(), 1);
        // Remaining jobs stay in queue because concurrent job limit prevents admission
        assert_eq!(disp.queue_len(), 2);
    }

    #[test]
    fn test_dispatch_single() {
        let mut disp = Dispatcher::new(100);
        let pool = mock_pool(2);
        let registry = TenantRegistry::new();

        let mut job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        let stream = disp.dispatch_single(&mut job, &pool, &registry).unwrap();
        assert!(stream.id() >= 0);
        assert_eq!(job.state, JobState::Running);
        assert_eq!(disp.active_count(), 1);
    }
}
