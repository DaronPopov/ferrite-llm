//! Failure classification for durable jobs.
//!
//! Maps runtime errors into a small set of failure classes so the restart
//! policy can decide whether retrying makes sense.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Coarse classification of a runtime error with respect to retryability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureClass {
    /// The error is likely transient (GPU memory pressure, timeout, etc.)
    /// and retrying may succeed.
    Transient,
    /// The error is permanent (bad configuration, unsupported operation) and
    /// retrying will not help.
    Permanent,
    /// The error could not be classified. Treat as transient by default.
    Unknown,
}

impl std::fmt::Display for FailureClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Permanent => write!(f, "Permanent"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Stateless classifier that maps `Error` variants to `FailureClass`.
pub struct FailureClassifier;

impl FailureClassifier {
    /// Classify an error into a failure class.
    ///
    /// Classification rules:
    ///
    /// | Error variant       | Class      | Rationale                         |
    /// |---------------------|------------|-----------------------------------|
    /// | CudaError           | Transient  | GPU can recover on retry          |
    /// | AllocationFailed    | Transient  | Memory pressure may clear         |
    /// | StreamError         | Transient  | Stream state may reset            |
    /// | GraphError          | Transient  | Graph replay can be retried       |
    /// | CublasError         | Transient  | cuBLAS handles can be recreated   |
    /// | StableApi           | Transient  | Transient API glitch              |
    /// | InitFailed          | Permanent  | Fundamental init problem          |
    /// | InvalidPointer      | Permanent  | Programming error                 |
    /// | InvalidStreamId     | Permanent  | Invalid stream index              |
    /// | InvalidPtrOwnership | Permanent  | Pointer not owned by runtime      |
    /// | NotInitialized      | Permanent  | Must initialize first             |
    /// | NotSupported        | Permanent  | Feature not available             |
    /// | InvalidOperation    | Permanent  | (mapped via Internal)             |
    /// | ShapeMismatch       | Permanent  | Data error                        |
    /// | DTypeMismatch       | Permanent  | Data error                        |
    /// | Internal            | Unknown    | Unclassifiable                    |
    /// | JobStateInvalid     | Permanent  | Logic error                       |
    /// | JobNotFound         | Permanent  | Logic error                       |
    /// | JobPersistenceError | Transient  | Filesystem may recover            |
    pub fn classify(error: &Error) -> FailureClass {
        match error {
            // Transient: GPU / resource pressure can resolve itself.
            Error::CudaError { .. } => FailureClass::Transient,
            Error::AllocationFailed { .. } => FailureClass::Transient,
            Error::StreamError { .. } => FailureClass::Transient,
            Error::GraphError { .. } => FailureClass::Transient,
            Error::CublasError { .. } => FailureClass::Transient,
            Error::StableApi { .. } => FailureClass::Transient,
            Error::JobPersistenceError { .. } => FailureClass::Transient,

            // Permanent: no point retrying.
            Error::InitFailed { .. } => FailureClass::Permanent,
            Error::InvalidPointer => FailureClass::Permanent,
            Error::InvalidStreamId { .. } => FailureClass::Permanent,
            Error::InvalidPointerOwnership { .. } => FailureClass::Permanent,
            Error::NotInitialized => FailureClass::Permanent,
            Error::NotSupported { .. } => FailureClass::Permanent,
            Error::ShapeMismatch { .. } => FailureClass::Permanent,
            Error::DTypeMismatch { .. } => FailureClass::Permanent,
            Error::JobStateInvalid { .. } => FailureClass::Permanent,
            Error::JobNotFound { .. } => FailureClass::Permanent,
            Error::InvalidLaunchContext { .. } => FailureClass::Permanent,
            Error::EmptyStreamPool => FailureClass::Permanent,

            // Scheduler / control-plane errors -- generally transient.
            Error::QuotaExceeded { .. } => FailureClass::Transient,
            Error::AdmissionDenied { .. } => FailureClass::Transient,
            Error::SchedulerError { .. } => FailureClass::Transient,
            Error::PolicyDenied { .. } => FailureClass::Permanent,
            Error::AuditError { .. } => FailureClass::Transient,
            Error::ControlPlaneError { .. } => FailureClass::Transient,

            // Unknown: cannot determine.
            Error::Internal { .. } => FailureClass::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_error_is_transient() {
        let err = Error::CudaError {
            code: 2,
            message: "out of memory".into(),
        };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Transient);
    }

    #[test]
    fn init_failed_is_permanent() {
        let err = Error::InitFailed { device_id: 0 };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Permanent);
    }

    #[test]
    fn allocation_failed_is_transient() {
        let err = Error::AllocationFailed { size: 1024 };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Transient);
    }

    #[test]
    fn internal_is_unknown() {
        let err = Error::Internal {
            message: "something".into(),
        };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Unknown);
    }

    #[test]
    fn not_supported_is_permanent() {
        let err = Error::NotSupported {
            message: "fp16 not available".into(),
        };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Permanent);
    }

    #[test]
    fn job_persistence_is_transient() {
        let err = Error::JobPersistenceError {
            detail: "disk full".into(),
        };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Transient);
    }

    #[test]
    fn invalid_stream_id_is_permanent() {
        let err = Error::InvalidStreamId { id: 99, pool_size: 4 };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Permanent);
    }

    #[test]
    fn invalid_pointer_ownership_is_permanent() {
        let err = Error::InvalidPointerOwnership {
            ptr_debug: "0xdeadbeef".into(),
        };
        assert_eq!(FailureClassifier::classify(&err), FailureClass::Permanent);
    }
}
