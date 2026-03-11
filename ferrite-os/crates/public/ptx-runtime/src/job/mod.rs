//! Durable job runtime and recovery.
//!
//! This module provides the core types for the Plan-B durable job system:
//!
//! - **`state`** -- Job lifecycle state machine with validated transitions
//! - **`policy`** -- Restart policies (Never, OnFailure, Always) with backoff
//! - **`failure`** -- Failure classification (Transient / Permanent / Unknown)
//! - **`model`** -- The `DurableJob` entity and `DurableJobId` newtype
//!
//! For backward compatibility with the legacy `job` re-export, this module
//! also re-exports the scheduler's job types.

pub mod failure;
pub mod model;
pub mod policy;
pub mod state;

pub use failure::{FailureClass, FailureClassifier};
pub use model::{DurableJob, DurableJobId};
pub use policy::{BackoffConfig, RestartDecision, RestartPolicy};
pub use state::{JobLifecycleState, JobStateMachine, JobStateTransition};

// Backward compatibility: re-export scheduler job types that were previously
// exposed via the stub `job.rs` module.
pub use crate::scheduler::job::*;
