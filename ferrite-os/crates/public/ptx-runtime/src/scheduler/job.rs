//! Job model for the multi-tenant scheduler.
//!
//! A job represents a discrete unit of GPU work submitted by a tenant.
//! Jobs progress through a well-defined state machine:
//!
//! ```text
//! Queued -> Admitted -> Running -> Completed
//!                   \           \-> Failed
//!                    \-> Cancelled
//! ```

use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::stream::StreamPriority;
use super::tenant::TenantId;

/// Monotonically increasing job ID counter.
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl JobId {
    /// Allocate the next unique job ID.
    pub fn next() -> Self {
        JobId(NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "job-{}", self.0)
    }
}

/// The lifecycle state of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// Job is in the queue waiting for admission.
    Queued,
    /// Job has passed admission checks and is ready for dispatch.
    Admitted,
    /// Job has been dispatched to a stream and is executing on the GPU.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed during execution.
    Failed,
    /// Job was cancelled before completion.
    Cancelled,
}

impl JobState {
    /// Returns true if this state is a terminal state (Completed, Failed, or Cancelled).
    pub fn is_terminal(self) -> bool {
        matches!(self, JobState::Completed | JobState::Failed | JobState::Cancelled)
    }
}

impl std::fmt::Display for JobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobState::Queued => write!(f, "Queued"),
            JobState::Admitted => write!(f, "Admitted"),
            JobState::Running => write!(f, "Running"),
            JobState::Completed => write!(f, "Completed"),
            JobState::Failed => write!(f, "Failed"),
            JobState::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// A scheduled GPU job with lifecycle tracking.
#[derive(Debug)]
pub struct Job {
    /// Unique job identifier.
    pub id: JobId,
    /// The tenant that submitted this job.
    pub tenant_id: TenantId,
    /// Scheduling priority.
    pub priority: StreamPriority,
    /// Current lifecycle state.
    pub state: JobState,
    /// When the job was submitted to the scheduler.
    pub submitted_at: Instant,
    /// When the job transitioned to Running (set by dispatcher).
    pub started_at: Option<Instant>,
    /// When the job reached a terminal state.
    pub completed_at: Option<Instant>,
    /// Optional reason code for failure or cancellation.
    pub reason_code: Option<String>,
    /// Estimated VRAM requirement in bytes (used for admission).
    pub estimated_vram_bytes: u64,
}

impl Job {
    /// Create a new job in the Queued state.
    pub fn new(tenant_id: TenantId, priority: StreamPriority) -> Self {
        Self {
            id: JobId::next(),
            tenant_id,
            priority,
            state: JobState::Queued,
            submitted_at: Instant::now(),
            started_at: None,
            completed_at: None,
            reason_code: None,
            estimated_vram_bytes: 0,
        }
    }

    /// Create a new job with an estimated VRAM requirement.
    pub fn with_vram_estimate(tenant_id: TenantId, priority: StreamPriority, vram_bytes: u64) -> Self {
        let mut job = Self::new(tenant_id, priority);
        job.estimated_vram_bytes = vram_bytes;
        job
    }

    /// Transition the job to the Admitted state.
    ///
    /// Valid only from: Queued
    pub fn admit(&mut self) -> Result<(), InvalidTransition> {
        self.transition(JobState::Admitted)
    }

    /// Transition the job to the Running state.
    ///
    /// Valid only from: Admitted
    pub fn start(&mut self) -> Result<(), InvalidTransition> {
        self.transition(JobState::Running)?;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    /// Transition the job to the Completed state.
    ///
    /// Valid only from: Running
    pub fn complete(&mut self) -> Result<(), InvalidTransition> {
        self.transition(JobState::Completed)?;
        self.completed_at = Some(Instant::now());
        Ok(())
    }

    /// Transition the job to the Failed state with a reason.
    ///
    /// Valid only from: Running
    pub fn fail(&mut self, reason: impl Into<String>) -> Result<(), InvalidTransition> {
        self.transition(JobState::Failed)?;
        self.completed_at = Some(Instant::now());
        self.reason_code = Some(reason.into());
        Ok(())
    }

    /// Cancel the job.
    ///
    /// Valid from: Queued, Admitted
    pub fn cancel(&mut self, reason: impl Into<String>) -> Result<(), InvalidTransition> {
        self.transition(JobState::Cancelled)?;
        self.completed_at = Some(Instant::now());
        self.reason_code = Some(reason.into());
        Ok(())
    }

    /// Compute the elapsed wall-clock runtime for a running or completed job.
    pub fn elapsed_runtime_ms(&self) -> Option<u64> {
        let started = self.started_at?;
        let end = self.completed_at.unwrap_or_else(Instant::now);
        Some(end.duration_since(started).as_millis() as u64)
    }

    /// Compute the time spent in the queue before being admitted.
    pub fn queue_latency_ms(&self) -> Option<u64> {
        let started = self.started_at?;
        Some(started.duration_since(self.submitted_at).as_millis() as u64)
    }

    /// Validate and apply a state transition.
    fn transition(&mut self, target: JobState) -> Result<(), InvalidTransition> {
        if self.is_valid_transition(target) {
            self.state = target;
            Ok(())
        } else {
            Err(InvalidTransition {
                job_id: self.id,
                from: self.state,
                to: target,
            })
        }
    }

    /// Check if a state transition is valid.
    fn is_valid_transition(&self, target: JobState) -> bool {
        matches!(
            (self.state, target),
            (JobState::Queued, JobState::Admitted)
                | (JobState::Queued, JobState::Cancelled)
                | (JobState::Admitted, JobState::Running)
                | (JobState::Admitted, JobState::Cancelled)
                | (JobState::Running, JobState::Completed)
                | (JobState::Running, JobState::Failed)
        )
    }
}

/// Error returned when an invalid job state transition is attempted.
#[derive(Debug, Clone)]
pub struct InvalidTransition {
    pub job_id: JobId,
    pub from: JobState,
    pub to: JobState,
}

impl std::fmt::Display for InvalidTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid job state transition for {}: {} -> {}",
            self.job_id, self.from, self.to
        )
    }
}

impl std::error::Error for InvalidTransition {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id_monotonic() {
        let a = JobId::next();
        let b = JobId::next();
        assert!(b.0 > a.0);
    }

    #[test]
    fn test_job_state_display() {
        assert_eq!(JobState::Queued.to_string(), "Queued");
        assert_eq!(JobState::Running.to_string(), "Running");
        assert_eq!(JobState::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_terminal_states() {
        assert!(!JobState::Queued.is_terminal());
        assert!(!JobState::Admitted.is_terminal());
        assert!(!JobState::Running.is_terminal());
        assert!(JobState::Completed.is_terminal());
        assert!(JobState::Failed.is_terminal());
        assert!(JobState::Cancelled.is_terminal());
    }

    #[test]
    fn test_happy_path_transitions() {
        let mut job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        assert_eq!(job.state, JobState::Queued);

        job.admit().unwrap();
        assert_eq!(job.state, JobState::Admitted);

        job.start().unwrap();
        assert_eq!(job.state, JobState::Running);
        assert!(job.started_at.is_some());

        job.complete().unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_failure_transition() {
        let mut job = Job::new(TenantId(1), StreamPriority::High);
        job.admit().unwrap();
        job.start().unwrap();
        job.fail("OOM during kernel launch").unwrap();

        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.reason_code.as_deref(), Some("OOM during kernel launch"));
    }

    #[test]
    fn test_cancel_from_queued() {
        let mut job = Job::new(TenantId(2), StreamPriority::Low);
        job.cancel("user requested").unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[test]
    fn test_cancel_from_admitted() {
        let mut job = Job::new(TenantId(2), StreamPriority::Low);
        job.admit().unwrap();
        job.cancel("quota exceeded").unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[test]
    fn test_invalid_transition() {
        let mut job = Job::new(TenantId(3), StreamPriority::Normal);

        // Cannot go directly from Queued to Running
        let err = job.start().unwrap_err();
        assert_eq!(err.from, JobState::Queued);
        assert_eq!(err.to, JobState::Running);

        // Cannot go from Queued to Completed
        let err = job.complete().unwrap_err();
        assert_eq!(err.from, JobState::Queued);
        assert_eq!(err.to, JobState::Completed);
    }

    #[test]
    fn test_no_transition_from_terminal() {
        let mut job = Job::new(TenantId(4), StreamPriority::Normal);
        job.admit().unwrap();
        job.start().unwrap();
        job.complete().unwrap();

        // Cannot transition from Completed
        assert!(job.admit().is_err());
        assert!(job.start().is_err());
        assert!(job.fail("late failure").is_err());
    }

    #[test]
    fn test_with_vram_estimate() {
        let job = Job::with_vram_estimate(TenantId(5), StreamPriority::Realtime, 1024 * 1024);
        assert_eq!(job.estimated_vram_bytes, 1024 * 1024);
        assert_eq!(job.state, JobState::Queued);
    }
}
