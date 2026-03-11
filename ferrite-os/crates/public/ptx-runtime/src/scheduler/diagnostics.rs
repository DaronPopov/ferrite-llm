//! Scheduler diagnostics and structured event tracking.
//!
//! Provides structured observability for scheduling decisions, quota warnings,
//! and anomaly detection (starvation). Integrates with the existing
//! `DiagnosticEvent` / `emit_diag` telemetry infrastructure.

use std::collections::VecDeque;
use std::time::Instant;

use crate::telemetry::{emit_diag, DiagnosticEvent, DiagnosticStatus};

use super::job::JobId;
use super::policy::DenialReason;
use super::tenant::TenantId;

/// Maximum number of events retained in the ring buffer.
const MAX_EVENT_BUFFER: usize = 1024;

/// A structured scheduler event for observability.
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    /// A job was added to the queue.
    JobQueued {
        job_id: JobId,
        tenant_id: TenantId,
        timestamp: Instant,
    },
    /// A job passed admission checks.
    JobAdmitted {
        job_id: JobId,
        tenant_id: TenantId,
        timestamp: Instant,
    },
    /// A job was denied admission.
    JobDenied {
        job_id: JobId,
        tenant_id: TenantId,
        reason: String,
        timestamp: Instant,
    },
    /// A job was dispatched to a stream.
    JobDispatched {
        job_id: JobId,
        tenant_id: TenantId,
        stream_id: i32,
        timestamp: Instant,
    },
    /// A job completed successfully.
    JobCompleted {
        job_id: JobId,
        tenant_id: TenantId,
        timestamp: Instant,
    },
    /// A job failed during execution.
    JobFailed {
        job_id: JobId,
        tenant_id: TenantId,
        timestamp: Instant,
    },
    /// A job was cancelled.
    JobCancelled {
        job_id: JobId,
        tenant_id: TenantId,
        timestamp: Instant,
    },
    /// A tenant is approaching a resource quota limit.
    QuotaWarning {
        tenant_id: TenantId,
        resource: String,
        usage_pct: f64,
        timestamp: Instant,
    },
    /// A tenant has queued jobs but no dispatches while others are running.
    StarvationDetected {
        tenant_id: TenantId,
        timestamp: Instant,
    },
}

impl SchedulerEvent {
    /// Get the timestamp of this event.
    pub fn timestamp(&self) -> Instant {
        match self {
            SchedulerEvent::JobQueued { timestamp, .. }
            | SchedulerEvent::JobAdmitted { timestamp, .. }
            | SchedulerEvent::JobDenied { timestamp, .. }
            | SchedulerEvent::JobDispatched { timestamp, .. }
            | SchedulerEvent::JobCompleted { timestamp, .. }
            | SchedulerEvent::JobFailed { timestamp, .. }
            | SchedulerEvent::JobCancelled { timestamp, .. }
            | SchedulerEvent::QuotaWarning { timestamp, .. }
            | SchedulerEvent::StarvationDetected { timestamp, .. } => *timestamp,
        }
    }

    /// Get the diagnostic code for this event type.
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            SchedulerEvent::JobQueued { .. } => "SCHED-EVT-0001",
            SchedulerEvent::JobAdmitted { .. } => "SCHED-EVT-0002",
            SchedulerEvent::JobDenied { .. } => "SCHED-EVT-0003",
            SchedulerEvent::JobDispatched { .. } => "SCHED-EVT-0004",
            SchedulerEvent::JobCompleted { .. } => "SCHED-EVT-0005",
            SchedulerEvent::JobFailed { .. } => "SCHED-EVT-0006",
            SchedulerEvent::JobCancelled { .. } => "SCHED-EVT-0009",
            SchedulerEvent::QuotaWarning { .. } => "SCHED-EVT-0007",
            SchedulerEvent::StarvationDetected { .. } => "SCHED-EVT-0008",
        }
    }
}

/// Per-tenant quota utilization summary for diagnostics.
#[derive(Debug, Clone)]
pub struct TenantQuotaSummary {
    pub tenant_id: TenantId,
    pub label: String,
    /// VRAM utilization percentage (None if unlimited).
    pub vram_usage_pct: Option<f64>,
    /// Stream utilization percentage (None if unlimited).
    pub stream_usage_pct: Option<f64>,
    /// Concurrent jobs utilization percentage (None if unlimited).
    pub job_usage_pct: Option<f64>,
    /// Runtime budget utilization percentage (None if unlimited).
    pub runtime_usage_pct: Option<f64>,
    /// Raw usage snapshot.
    pub usage: super::tenant::TenantUsageSnapshot,
}

/// Scheduler diagnostics collector.
///
/// Maintains a bounded ring buffer of recent scheduler events and emits
/// structured diagnostics via the runtime telemetry infrastructure.
pub struct SchedulerDiagnostics {
    events: VecDeque<SchedulerEvent>,
    /// Total jobs queued since creation.
    pub total_queued: u64,
    /// Total jobs admitted since creation.
    pub total_admitted: u64,
    /// Total jobs denied since creation.
    pub total_denied: u64,
    /// Total jobs dispatched since creation.
    pub total_dispatched: u64,
    /// Total jobs completed since creation.
    pub total_completed: u64,
    /// Total jobs failed since creation.
    pub total_failed: u64,
    /// Total jobs cancelled since creation.
    pub total_cancelled: u64,
    /// Total starvation events detected since creation.
    pub total_starvation_events: u64,
}

impl SchedulerDiagnostics {
    /// Create a new diagnostics collector.
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(MAX_EVENT_BUFFER),
            total_queued: 0,
            total_admitted: 0,
            total_denied: 0,
            total_dispatched: 0,
            total_completed: 0,
            total_failed: 0,
            total_cancelled: 0,
            total_starvation_events: 0,
        }
    }

    /// Record a job-queued event.
    pub fn record_job_queued(&mut self, job_id: JobId, tenant_id: TenantId) {
        self.total_queued += 1;
        self.push_event(SchedulerEvent::JobQueued {
            job_id,
            tenant_id,
            timestamp: Instant::now(),
        });
    }

    /// Record a job-admitted event.
    pub fn record_job_admitted(&mut self, job_id: JobId, tenant_id: TenantId) {
        self.total_admitted += 1;
        self.push_event(SchedulerEvent::JobAdmitted {
            job_id,
            tenant_id,
            timestamp: Instant::now(),
        });

        emit_diag(&DiagnosticEvent::new(
            "scheduler.admission",
            DiagnosticStatus::PASS,
            "SCHED-EVT-0002",
            format!("job {} admitted for tenant {}", job_id, tenant_id),
            "none",
        ));
    }

    /// Record a job-denied event.
    pub fn record_job_denied(
        &mut self,
        job_id: JobId,
        tenant_id: TenantId,
        reason: &DenialReason,
    ) {
        self.total_denied += 1;
        let reason_str = reason.to_string();

        self.push_event(SchedulerEvent::JobDenied {
            job_id,
            tenant_id,
            reason: reason_str.clone(),
            timestamp: Instant::now(),
        });

        emit_diag(&DiagnosticEvent::new(
            "scheduler.admission",
            DiagnosticStatus::WARN,
            "SCHED-EVT-0003",
            format!(
                "job {} denied for tenant {}: {}",
                job_id, tenant_id, reason_str
            ),
            "review tenant quotas or reduce workload",
        ));
    }

    /// Record a job-dispatched event.
    pub fn record_job_dispatched(
        &mut self,
        job_id: JobId,
        tenant_id: TenantId,
        stream_id: i32,
    ) {
        self.total_dispatched += 1;
        self.push_event(SchedulerEvent::JobDispatched {
            job_id,
            tenant_id,
            stream_id,
            timestamp: Instant::now(),
        });
    }

    /// Record a job-completed event.
    pub fn record_job_completed(&mut self, job_id: JobId, tenant_id: TenantId) {
        self.total_completed += 1;
        self.push_event(SchedulerEvent::JobCompleted {
            job_id,
            tenant_id,
            timestamp: Instant::now(),
        });
    }

    /// Record a job-failed event.
    pub fn record_job_failed(&mut self, job_id: JobId, tenant_id: TenantId) {
        self.total_failed += 1;
        self.push_event(SchedulerEvent::JobFailed {
            job_id,
            tenant_id,
            timestamp: Instant::now(),
        });

        emit_diag(&DiagnosticEvent::new(
            "scheduler.execution",
            DiagnosticStatus::FAIL,
            "SCHED-EVT-0006",
            format!("job {} failed for tenant {}", job_id, tenant_id),
            "inspect job logs and GPU state",
        ));
    }

    /// Record a job-cancelled event.
    pub fn record_job_cancelled(&mut self, job_id: JobId, tenant_id: TenantId) {
        self.total_cancelled += 1;
        self.push_event(SchedulerEvent::JobCancelled {
            job_id,
            tenant_id,
            timestamp: Instant::now(),
        });
    }

    /// Record a quota warning.
    pub fn record_quota_warning(
        &mut self,
        tenant_id: TenantId,
        resource: &str,
        usage_pct: f64,
    ) {
        self.push_event(SchedulerEvent::QuotaWarning {
            tenant_id,
            resource: resource.to_string(),
            usage_pct,
            timestamp: Instant::now(),
        });

        emit_diag(&DiagnosticEvent::new(
            "scheduler.quotas",
            DiagnosticStatus::WARN,
            "SCHED-EVT-0007",
            format!(
                "tenant {} {} usage at {:.1}% of quota",
                tenant_id, resource, usage_pct
            ),
            "consider increasing tenant quota or shedding load",
        ));
    }

    /// Record a starvation detection event.
    pub fn record_starvation(&mut self, tenant_id: TenantId) {
        self.total_starvation_events += 1;

        self.push_event(SchedulerEvent::StarvationDetected {
            tenant_id,
            timestamp: Instant::now(),
        });

        emit_diag(&DiagnosticEvent::new(
            "scheduler.fairness",
            DiagnosticStatus::WARN,
            "SCHED-EVT-0008",
            format!(
                "starvation detected for tenant {}: queued jobs but no active dispatches",
                tenant_id
            ),
            "verify fair-share policy is active and tenant quotas are not blocking admission",
        ));
    }

    /// Get the most recent events (up to `count`).
    pub fn recent_events(&self, count: usize) -> Vec<&SchedulerEvent> {
        self.events.iter().rev().take(count).collect()
    }

    /// Get all buffered events.
    pub fn all_events(&self) -> &VecDeque<SchedulerEvent> {
        &self.events
    }

    /// Produce a summary diagnostic for the scheduler's overall health.
    pub fn health_diagnostic(&self) -> DiagnosticEvent {
        let (status, summary) = if self.total_failed > 0 || self.total_starvation_events > 0 {
            (
                DiagnosticStatus::WARN,
                format!(
                    "queued={} admitted={} denied={} dispatched={} completed={} failed={} starvation_events={}",
                    self.total_queued,
                    self.total_admitted,
                    self.total_denied,
                    self.total_dispatched,
                    self.total_completed,
                    self.total_failed,
                    self.total_starvation_events
                ),
            )
        } else {
            (
                DiagnosticStatus::PASS,
                format!(
                    "queued={} dispatched={} completed={} denied={}",
                    self.total_queued, self.total_dispatched, self.total_completed, self.total_denied
                ),
            )
        };

        DiagnosticEvent::new(
            "scheduler.health",
            status,
            "SCHED-HEALTH-0001",
            summary,
            if status == DiagnosticStatus::WARN {
                "investigate failed jobs and starvation events"
            } else {
                "none"
            },
        )
    }

    /// Produce per-tenant quota utilization summaries.
    ///
    /// Returns a list of `(TenantId, resource, usage_pct)` tuples for all
    /// tenants that have non-unlimited quotas.
    pub fn tenant_quota_summaries(
        &self,
        registry: &super::registry::TenantRegistry,
    ) -> Vec<TenantQuotaSummary> {
        let mut summaries = Vec::new();
        for tenant_id in registry.list() {
            if let Some(tenant) = registry.get(tenant_id) {
                let snap = tenant.usage.snapshot();
                let q = &tenant.quotas;

                let vram_pct = if q.max_vram_bytes != u64::MAX {
                    Some((snap.current_vram_bytes as f64 / q.max_vram_bytes as f64) * 100.0)
                } else {
                    None
                };
                let streams_pct = if q.max_streams != u64::MAX {
                    Some((snap.active_streams as f64 / q.max_streams as f64) * 100.0)
                } else {
                    None
                };
                let jobs_pct = if q.max_concurrent_jobs != u64::MAX {
                    Some((snap.active_jobs as f64 / q.max_concurrent_jobs as f64) * 100.0)
                } else {
                    None
                };
                let runtime_pct = if q.max_runtime_budget_ms != u64::MAX {
                    Some((snap.consumed_runtime_ms as f64 / q.max_runtime_budget_ms as f64) * 100.0)
                } else {
                    None
                };

                summaries.push(TenantQuotaSummary {
                    tenant_id,
                    label: tenant.label.clone(),
                    vram_usage_pct: vram_pct,
                    stream_usage_pct: streams_pct,
                    job_usage_pct: jobs_pct,
                    runtime_usage_pct: runtime_pct,
                    usage: snap,
                });
            }
        }
        summaries
    }

    /// Push an event into the ring buffer, evicting the oldest if full.
    fn push_event(&mut self, event: SchedulerEvent) {
        if self.events.len() >= MAX_EVENT_BUFFER {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}

impl Default for SchedulerDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SchedulerDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerDiagnostics")
            .field("buffered_events", &self.events.len())
            .field("total_queued", &self.total_queued)
            .field("total_dispatched", &self.total_dispatched)
            .field("total_completed", &self.total_completed)
            .field("total_denied", &self.total_denied)
            .field("total_failed", &self.total_failed)
            .field("total_cancelled", &self.total_cancelled)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::policy::DenialReason;

    #[test]
    fn test_event_recording() {
        let mut diag = SchedulerDiagnostics::new();
        let job_id = JobId(1);
        let tenant_id = TenantId(0);

        diag.record_job_queued(job_id, tenant_id);
        assert_eq!(diag.total_queued, 1);
        assert_eq!(diag.all_events().len(), 1);

        diag.record_job_admitted(job_id, tenant_id);
        assert_eq!(diag.total_admitted, 1);

        diag.record_job_dispatched(job_id, tenant_id, 0);
        assert_eq!(diag.total_dispatched, 1);

        diag.record_job_completed(job_id, tenant_id);
        assert_eq!(diag.total_completed, 1);

        assert_eq!(diag.all_events().len(), 4);
    }

    #[test]
    fn test_denial_recording() {
        let mut diag = SchedulerDiagnostics::new();
        diag.record_job_denied(
            JobId(1),
            TenantId(1),
            &DenialReason::VramQuotaExceeded,
        );
        assert_eq!(diag.total_denied, 1);

        if let SchedulerEvent::JobDenied { reason, .. } = &diag.all_events()[0] {
            assert_eq!(reason, "VRAM quota exceeded");
        } else {
            panic!("expected JobDenied event");
        }
    }

    #[test]
    fn test_starvation_recording() {
        let mut diag = SchedulerDiagnostics::new();
        diag.record_starvation(TenantId(5));
        assert_eq!(diag.total_starvation_events, 1);

        match &diag.all_events()[0] {
            SchedulerEvent::StarvationDetected { tenant_id, .. } => {
                assert_eq!(*tenant_id, TenantId(5));
            }
            _ => panic!("expected StarvationDetected event"),
        }
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut diag = SchedulerDiagnostics::new();
        for i in 0..MAX_EVENT_BUFFER + 100 {
            diag.record_job_queued(JobId(i as u64), TenantId(0));
        }
        assert_eq!(diag.all_events().len(), MAX_EVENT_BUFFER);
    }

    #[test]
    fn test_recent_events() {
        let mut diag = SchedulerDiagnostics::new();
        for i in 0..10 {
            diag.record_job_queued(JobId(i), TenantId(0));
        }

        let recent = diag.recent_events(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_health_diagnostic_pass() {
        let diag = SchedulerDiagnostics::new();
        let health = diag.health_diagnostic();
        assert_eq!(health.status, DiagnosticStatus::PASS);
    }

    #[test]
    fn test_health_diagnostic_warn() {
        let mut diag = SchedulerDiagnostics::new();
        diag.record_job_failed(JobId(1), TenantId(0));
        let health = diag.health_diagnostic();
        assert_eq!(health.status, DiagnosticStatus::WARN);
    }

    #[test]
    fn test_diagnostic_codes() {
        let event = SchedulerEvent::JobQueued {
            job_id: JobId(1),
            tenant_id: TenantId(0),
            timestamp: Instant::now(),
        };
        assert_eq!(event.diagnostic_code(), "SCHED-EVT-0001");
    }
}
