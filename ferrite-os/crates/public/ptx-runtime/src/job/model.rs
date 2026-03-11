//! Durable job model.
//!
//! `DurableJob` is the primary entity tracked by the job supervisor. It
//! bundles identity, command specification, restart policy, lifecycle state,
//! failure accounting, and (optionally) a running process handle.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::job::policy::{RestartDecision, RestartPolicy};
use crate::job::state::{JobLifecycleState, JobStateMachine};

// ---------------------------------------------------------------------------
// DurableJobId
// ---------------------------------------------------------------------------

/// Monotonically increasing job identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurableJobId(pub u64);

impl std::fmt::Display for DurableJobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "job-{}", self.0)
    }
}

/// Global counter used to mint new `DurableJobId` values.
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

impl DurableJobId {
    /// Generate the next unique job ID.
    pub fn next() -> Self {
        Self(NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Ensure the global counter is at least `min` so that IDs loaded from
    /// disk do not collide with newly generated ones.
    pub fn advance_counter_to(min: u64) {
        NEXT_JOB_ID.fetch_max(min + 1, Ordering::Relaxed);
    }

    /// The raw numeric identifier.
    pub fn raw(self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// DurableJob
// ---------------------------------------------------------------------------

/// A durable, supervised job.
///
/// The `process_pid` field is transient -- it is not persisted across daemon
/// restarts. On recovery the reconciler will re-discover running processes
/// by PID and update this field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableJob {
    pub id: DurableJobId,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub restart_policy: RestartPolicy,
    pub state_machine: JobStateMachine,
    pub failure_count: u32,
    pub last_failure: Option<String>,
    pub created_at: SystemTime,

    /// PID of the child process (transient, only valid while running).
    #[serde(skip)]
    pub process_pid: Option<u32>,
}

impl DurableJob {
    /// Create a new job in the `Created` state.
    pub fn new(
        name: String,
        command: String,
        args: Vec<String>,
        restart_policy: RestartPolicy,
    ) -> Self {
        Self {
            id: DurableJobId::next(),
            name,
            command,
            args,
            restart_policy,
            state_machine: JobStateMachine::new(),
            failure_count: 0,
            last_failure: None,
            created_at: SystemTime::now(),
            process_pid: None,
        }
    }

    // -- convenience accessors ------------------------------------------

    /// Current lifecycle state.
    pub fn state(&self) -> JobLifecycleState {
        self.state_machine.current_state()
    }

    // -- lifecycle operations -------------------------------------------

    /// Transition the job from Created -> Queued.
    pub fn enqueue(&mut self) -> Result<()> {
        self.state_machine
            .transition(JobLifecycleState::Queued, "enqueued for execution".into())
    }

    /// Transition from Queued/Retrying -> Starting.
    pub fn mark_starting(&mut self) -> Result<()> {
        self.state_machine
            .transition(JobLifecycleState::Starting, "starting process".into())
    }

    /// Transition from Starting -> Running and record the process PID.
    pub fn mark_running(&mut self, pid: u32) -> Result<()> {
        self.process_pid = Some(pid);
        self.state_machine
            .transition(JobLifecycleState::Running, format!("running (pid {})", pid))
    }

    /// Record a successful completion.
    pub fn record_success(&mut self) -> Result<()> {
        self.process_pid = None;
        self.state_machine
            .transition(JobLifecycleState::Succeeded, "process exited successfully".into())
    }

    /// Record a failure and update failure accounting.
    pub fn record_failure(&mut self, error_message: String) -> Result<()> {
        self.process_pid = None;
        self.failure_count += 1;
        self.last_failure = Some(error_message.clone());
        self.state_machine
            .transition(JobLifecycleState::Failed, error_message)
    }

    /// Evaluate the restart policy and, if appropriate, transition to Retrying.
    /// Returns the `RestartDecision` so the caller knows the delay.
    pub fn should_restart(&mut self, error: &Error) -> Result<RestartDecision> {
        let decision = self.restart_policy.evaluate(self.failure_count, error);
        if let RestartDecision::Restart { .. } = &decision {
            self.state_machine.transition(
                JobLifecycleState::Retrying,
                format!("retry #{}", self.failure_count),
            )?;
        }
        Ok(decision)
    }

    /// Cancel the job.
    pub fn cancel(&mut self, reason: String) -> Result<()> {
        self.process_pid = None;
        self.state_machine
            .transition(JobLifecycleState::Cancelled, reason)
    }

    /// Terminate the job (system-initiated).
    pub fn terminate(&mut self, reason: String) -> Result<()> {
        self.process_pid = None;
        self.state_machine
            .transition(JobLifecycleState::Terminated, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::policy::BackoffConfig;

    fn make_job(policy: RestartPolicy) -> DurableJob {
        DurableJob::new(
            "test-job".into(),
            "/bin/true".into(),
            vec![],
            policy,
        )
    }

    #[test]
    fn id_generation_unique() {
        let a = DurableJobId::next();
        let b = DurableJobId::next();
        assert_ne!(a, b);
        assert!(b.0 > a.0);
    }

    #[test]
    fn full_lifecycle() {
        let mut job = make_job(RestartPolicy::Never);
        assert_eq!(job.state(), JobLifecycleState::Created);

        job.enqueue().unwrap();
        assert_eq!(job.state(), JobLifecycleState::Queued);

        job.mark_starting().unwrap();
        assert_eq!(job.state(), JobLifecycleState::Starting);

        job.mark_running(12345).unwrap();
        assert_eq!(job.state(), JobLifecycleState::Running);
        assert_eq!(job.process_pid, Some(12345));

        job.record_success().unwrap();
        assert_eq!(job.state(), JobLifecycleState::Succeeded);
        assert_eq!(job.process_pid, None);
    }

    #[test]
    fn failure_and_retry() {
        let policy = RestartPolicy::OnFailure {
            max_retries: 3,
            backoff: BackoffConfig::default(),
        };
        let mut job = make_job(policy);
        job.enqueue().unwrap();
        job.mark_starting().unwrap();
        job.mark_running(111).unwrap();
        job.record_failure("exit code 1".into()).unwrap();
        assert_eq!(job.failure_count, 1);

        let err = Error::CudaError {
            code: 2,
            message: "oom".into(),
        };
        let decision = job.should_restart(&err).unwrap();
        assert!(matches!(decision, RestartDecision::Restart { .. }));
        assert_eq!(job.state(), JobLifecycleState::Retrying);
    }

    #[test]
    fn cancel_clears_pid() {
        let mut job = make_job(RestartPolicy::Never);
        job.enqueue().unwrap();
        job.mark_starting().unwrap();
        job.mark_running(999).unwrap();
        job.cancel("operator".into()).unwrap();
        assert_eq!(job.process_pid, None);
        assert_eq!(job.state(), JobLifecycleState::Cancelled);
    }
}
