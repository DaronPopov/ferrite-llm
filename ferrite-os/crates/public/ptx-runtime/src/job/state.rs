//! Job lifecycle state machine.
//!
//! Defines the finite set of states a durable job can occupy and the
//! validated transitions between them. Every transition is recorded so
//! the full lifecycle history can be inspected after the fact.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Every state a durable job can pass through during its lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobLifecycleState {
    /// Job has been created but not yet queued for execution.
    Created,
    /// Job is waiting in the run queue.
    Queued,
    /// Job is in the process of being started (spawning a process).
    Starting,
    /// Job process is running.
    Running,
    /// Job completed successfully (exit code 0).
    Succeeded,
    /// Job process exited with an error or was lost.
    Failed,
    /// Job is waiting for a restart attempt after a failure.
    Retrying,
    /// Job was explicitly cancelled by an operator.
    Cancelled,
    /// Job was terminated by the system (e.g. shutdown).
    Terminated,
}

impl std::fmt::Display for JobLifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Created => "Created",
            Self::Queued => "Queued",
            Self::Starting => "Starting",
            Self::Running => "Running",
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::Retrying => "Retrying",
            Self::Cancelled => "Cancelled",
            Self::Terminated => "Terminated",
        };
        write!(f, "{}", label)
    }
}

impl JobLifecycleState {
    /// Returns `true` if the state is a terminal state (no further transitions
    /// except to Cancelled / Terminated).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Terminated
        )
    }
}

/// A single recorded state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStateTransition {
    pub from_state: JobLifecycleState,
    pub to_state: JobLifecycleState,
    pub timestamp: SystemTime,
    pub reason: String,
}

/// Tracks the current lifecycle state and the full transition history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStateMachine {
    current_state: JobLifecycleState,
    history: Vec<JobStateTransition>,
}

impl JobStateMachine {
    /// Create a new state machine starting in the `Created` state.
    pub fn new() -> Self {
        Self {
            current_state: JobLifecycleState::Created,
            history: Vec::new(),
        }
    }

    /// The current lifecycle state.
    pub fn current_state(&self) -> JobLifecycleState {
        self.current_state
    }

    /// The full ordered list of transitions that have occurred.
    pub fn history(&self) -> &[JobStateTransition] {
        &self.history
    }

    /// Attempt a validated transition. Returns an error if the transition
    /// is not allowed by the lifecycle matrix.
    pub fn transition(&mut self, to: JobLifecycleState, reason: String) -> Result<()> {
        if !Self::is_valid_transition(self.current_state, to) {
            return Err(Error::JobStateInvalid {
                from: self.current_state.to_string(),
                to: to.to_string(),
            });
        }

        let transition = JobStateTransition {
            from_state: self.current_state,
            to_state: to,
            timestamp: SystemTime::now(),
            reason,
        };

        tracing::debug!(
            from = %transition.from_state,
            to = %transition.to_state,
            reason = %transition.reason,
            "job state transition"
        );

        self.history.push(transition);
        self.current_state = to;
        Ok(())
    }

    /// The valid transitions matrix.
    ///
    /// | From       | Allowed targets                              |
    /// |------------|----------------------------------------------|
    /// | Created    | Queued, Cancelled, Terminated                |
    /// | Queued     | Starting, Cancelled, Terminated              |
    /// | Starting   | Running, Failed, Cancelled, Terminated       |
    /// | Running    | Succeeded, Failed, Cancelled, Terminated     |
    /// | Succeeded  | Cancelled, Terminated                        |
    /// | Failed     | Retrying, Cancelled, Terminated              |
    /// | Retrying   | Starting, Cancelled, Terminated              |
    /// | Cancelled  | (none -- terminal)                           |
    /// | Terminated | (none -- terminal)                           |
    fn is_valid_transition(from: JobLifecycleState, to: JobLifecycleState) -> bool {
        use JobLifecycleState::*;

        // Cancelled and Terminated are absorbing states -- nothing leaves them.
        if from == Cancelled || from == Terminated {
            return false;
        }

        // Any non-terminal state can move to Cancelled or Terminated.
        if to == Cancelled || to == Terminated {
            return true;
        }

        matches!(
            (from, to),
            (Created, Queued)
                | (Queued, Starting)
                | (Starting, Running)
                | (Starting, Failed)
                | (Running, Succeeded)
                | (Running, Failed)
                | (Failed, Retrying)
                | (Retrying, Starting)
        )
    }
}

impl Default for JobStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path() {
        let mut sm = JobStateMachine::new();
        assert_eq!(sm.current_state(), JobLifecycleState::Created);

        sm.transition(JobLifecycleState::Queued, "submitted".into())
            .unwrap();
        sm.transition(JobLifecycleState::Starting, "supervisor tick".into())
            .unwrap();
        sm.transition(JobLifecycleState::Running, "process alive".into())
            .unwrap();
        sm.transition(JobLifecycleState::Succeeded, "exit 0".into())
            .unwrap();

        assert_eq!(sm.current_state(), JobLifecycleState::Succeeded);
        assert_eq!(sm.history().len(), 4);
    }

    #[test]
    fn retry_path() {
        let mut sm = JobStateMachine::new();
        sm.transition(JobLifecycleState::Queued, "submitted".into())
            .unwrap();
        sm.transition(JobLifecycleState::Starting, "tick".into())
            .unwrap();
        sm.transition(JobLifecycleState::Running, "alive".into())
            .unwrap();
        sm.transition(JobLifecycleState::Failed, "exit 1".into())
            .unwrap();
        sm.transition(JobLifecycleState::Retrying, "policy says retry".into())
            .unwrap();
        sm.transition(JobLifecycleState::Starting, "retry #1".into())
            .unwrap();
        sm.transition(JobLifecycleState::Running, "alive again".into())
            .unwrap();
        sm.transition(JobLifecycleState::Succeeded, "exit 0".into())
            .unwrap();

        assert_eq!(sm.current_state(), JobLifecycleState::Succeeded);
    }

    #[test]
    fn invalid_transition_rejected() {
        let mut sm = JobStateMachine::new();
        let err = sm
            .transition(JobLifecycleState::Running, "skip ahead".into())
            .unwrap_err();
        assert!(matches!(err, Error::JobStateInvalid { .. }));
    }

    #[test]
    fn cancel_from_any_non_terminal() {
        for start in &[
            JobLifecycleState::Created,
            JobLifecycleState::Queued,
            JobLifecycleState::Failed,
        ] {
            let mut sm = JobStateMachine::new();
            if *start == JobLifecycleState::Queued {
                sm.transition(JobLifecycleState::Queued, "q".into())
                    .unwrap();
            } else if *start == JobLifecycleState::Failed {
                sm.transition(JobLifecycleState::Queued, "q".into())
                    .unwrap();
                sm.transition(JobLifecycleState::Starting, "s".into())
                    .unwrap();
                sm.transition(JobLifecycleState::Running, "r".into())
                    .unwrap();
                sm.transition(JobLifecycleState::Failed, "f".into())
                    .unwrap();
            }
            sm.transition(JobLifecycleState::Cancelled, "operator".into())
                .unwrap();
            assert_eq!(sm.current_state(), JobLifecycleState::Cancelled);
        }
    }

    #[test]
    fn terminal_states_absorbing() {
        let mut sm = JobStateMachine::new();
        sm.transition(JobLifecycleState::Queued, "q".into())
            .unwrap();
        sm.transition(JobLifecycleState::Cancelled, "cancel".into())
            .unwrap();
        let err = sm
            .transition(JobLifecycleState::Queued, "re-queue".into())
            .unwrap_err();
        assert!(matches!(err, Error::JobStateInvalid { .. }));
    }
}
