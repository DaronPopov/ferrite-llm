//! Error types for PTX-OS runtime operations.

use std::ffi::CStr;
use thiserror::Error;
use crate::telemetry::{DiagnosticEvent, DiagnosticStatus};

/// Result type for PTX-OS operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during PTX-OS runtime operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Runtime initialization failed
    #[error("Failed to initialize PTX-OS runtime on device {device_id}")]
    InitFailed { device_id: i32 },

    /// Memory allocation failed
    #[error("Failed to allocate {size} bytes on GPU")]
    AllocationFailed { size: usize },

    /// Invalid pointer passed to operation
    #[error("Invalid GPU pointer")]
    InvalidPointer,

    /// Stream ID out of range
    #[error("Invalid stream ID {id}: pool has {pool_size} streams")]
    InvalidStreamId { id: i32, pool_size: usize },

    /// Pointer not owned by this runtime (rejected before async free or kernel launch)
    #[error("Pointer {ptr_debug} not owned by this runtime")]
    InvalidPointerOwnership { ptr_debug: String },

    /// Stream operation failed
    #[error("Stream operation failed: {message}")]
    StreamError { message: String },

    /// CUDA graph operation failed
    #[error("CUDA graph operation failed: {message}")]
    GraphError { message: String },

    /// cuBLAS operation failed
    #[error("cuBLAS error: {status}")]
    CublasError { status: i32 },

    /// CUDA error
    #[error("CUDA error {code}: {message}")]
    CudaError { code: i32, message: String },

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type mismatch
    #[error("Data type mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        expected: ptx_sys::PTXDType,
        actual: ptx_sys::PTXDType,
    },

    /// Runtime not initialized
    #[error("PTX-OS runtime not initialized")]
    NotInitialized,

    /// Stable runtime API failure
    #[error("PTX stable API error {status}: {message}")]
    StableApi { status: i32, message: String },

    /// Operation not supported
    #[error("Operation not supported: {message}")]
    NotSupported { message: String },

    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// Invalid job state transition
    #[error("Invalid job state transition: {from} -> {to}")]
    JobStateInvalid { from: String, to: String },

    /// Job not found
    #[error("Job not found: id={id}")]
    JobNotFound { id: u64 },

    /// Job persistence error
    #[error("Job persistence error: {detail}")]
    JobPersistenceError { detail: String },

    /// Tenant resource quota exceeded
    #[error("Quota exceeded for tenant {tenant_id}: {resource} limit={limit}, current={current}")]
    QuotaExceeded {
        tenant_id: u64,
        resource: String,
        limit: u64,
        current: u64,
    },

    /// Scheduler admission denied
    #[error("Admission denied: {reason}")]
    AdmissionDenied { reason: String },

    /// General scheduler error
    #[error("Scheduler error: {detail}")]
    SchedulerError { detail: String },

    /// Policy enforcement denied the requested action
    #[error("Policy denied action '{action}': {reason} — {remediation}")]
    PolicyDenied {
        action: String,
        reason: String,
        remediation: String,
    },

    /// Audit subsystem error
    #[error("Audit error: {detail}")]
    AuditError { detail: String },

    /// Control plane error
    #[error("Control plane error: {detail}")]
    ControlPlaneError { detail: String },

    /// Invalid kernel launch context (null stream or runtime pointer)
    #[error("Invalid launch context: {detail}")]
    InvalidLaunchContext { detail: String },

    /// Stream pool is empty (no streams available for scheduling)
    #[error("Stream pool is empty: no streams available for scheduling")]
    EmptyStreamPool,
}

impl Error {
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            Error::InitFailed { .. } => "RT-ERR-0001",
            Error::AllocationFailed { .. } => "RT-ERR-0002",
            Error::InvalidPointer => "RT-ERR-0003",
            Error::InvalidStreamId { .. } => "RT-ERR-0014",
            Error::InvalidPointerOwnership { .. } => "RT-ERR-0015",
            Error::StreamError { .. } => "RT-ERR-0004",
            Error::GraphError { .. } => "RT-ERR-0005",
            Error::CublasError { .. } => "RT-ERR-0006",
            Error::CudaError { .. } => "RT-ERR-0007",
            Error::ShapeMismatch { .. } => "RT-ERR-0008",
            Error::DTypeMismatch { .. } => "RT-ERR-0009",
            Error::NotInitialized => "RT-ERR-0010",
            Error::StableApi { .. } => "RT-ERR-0011",
            Error::NotSupported { .. } => "RT-ERR-0012",
            Error::Internal { .. } => "RT-ERR-0013",
            Error::JobStateInvalid { .. } => "JOB-ERR-0001",
            Error::JobNotFound { .. } => "JOB-ERR-0002",
            Error::JobPersistenceError { .. } => "JOB-ERR-0003",
            Error::QuotaExceeded { .. } => "SCHED-ERR-0001",
            Error::AdmissionDenied { .. } => "SCHED-ERR-0002",
            Error::SchedulerError { .. } => "SCHED-ERR-0003",
            Error::PolicyDenied { .. } => "POLICY-ERR-0001",
            Error::AuditError { .. } => "AUDIT-ERR-0001",
            Error::ControlPlaneError { .. } => "CP-ERR-0001",
            Error::InvalidLaunchContext { .. } => "RT-ERR-0016",
            Error::EmptyStreamPool => "RT-ERR-0017",
        }
    }

    pub fn remediation(&self) -> &'static str {
        match self {
            Error::InitFailed { .. } => "verify CUDA driver/runtime compatibility and device availability",
            Error::AllocationFailed { .. } => "reduce allocation size or increase available GPU memory",
            Error::InvalidPointer => "validate pointer ownership and lifetime before calling runtime APIs",
            Error::InvalidStreamId { .. } => "use a stream ID in range [0, num_streams); call num_streams() to query pool size",
            Error::InvalidPointerOwnership { .. } => "verify pointer was allocated by this runtime before free_async or kernel launch",
            Error::StreamError { .. } => "check stream state and synchronize before dependent operations",
            Error::GraphError { .. } => "rebuild CUDA graph capture sequence and verify kernel launch params",
            Error::CublasError { .. } => "check matrix dimensions/types and cuBLAS handle initialization",
            Error::CudaError { .. } => "inspect CUDA error code and verify runtime/kernel preconditions",
            Error::ShapeMismatch { .. } => "align tensor shapes before invoking operation",
            Error::DTypeMismatch { .. } => "align tensor data types before invoking operation",
            Error::NotInitialized => "initialize runtime before using allocation/stream APIs",
            Error::StableApi { .. } => "check stable API status and ABI version compatibility",
            Error::NotSupported { .. } => "use a supported operation for this runtime build/device",
            Error::Internal { .. } => "capture logs and report issue with reproduction details",
            Error::JobStateInvalid { .. } => "verify job lifecycle; only valid transitions are allowed (see state machine docs)",
            Error::JobNotFound { .. } => "check job ID against job-list output; the job may have been purged",
            Error::JobPersistenceError { .. } => "verify state directory permissions and available disk space",
            Error::QuotaExceeded { .. } => "increase tenant quota or reduce resource usage before retrying",
            Error::AdmissionDenied { .. } => "check tenant quotas, active job counts, and runtime budget",
            Error::SchedulerError { .. } => "inspect scheduler state and verify configuration",
            Error::PolicyDenied { .. } => "review the denied action's policy rules and tenant authorization status",
            Error::AuditError { .. } => "check audit log configuration and available memory for audit entries",
            Error::ControlPlaneError { .. } => "inspect control plane configuration and scheduler state",
            Error::InvalidLaunchContext { .. } => "ensure runtime is initialized and stream is valid before launching kernels",
            Error::EmptyStreamPool => "initialize the runtime with at least one stream (max_streams >= 1)",
        }
    }

    pub fn as_diagnostic(&self, component: &'static str) -> DiagnosticEvent {
        DiagnosticEvent::new(
            component,
            DiagnosticStatus::FAIL,
            self.diagnostic_code(),
            self.to_string(),
            self.remediation(),
        )
    }

    /// Create a CUDA error from error code
    pub fn cuda(code: i32) -> Self {
        let message = unsafe {
            let ptr = ptx_sys::cudaGetErrorString(code);
            if ptr.is_null() {
                "Unknown CUDA error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error::CudaError { code, message }
    }

    /// Create a cuBLAS error from status code
    pub fn cublas(status: i32) -> Self {
        Error::CublasError { status }
    }

    /// Check CUDA result and return error if not success
    pub fn check_cuda(code: i32) -> Result<()> {
        if code == ptx_sys::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::cuda(code))
        }
    }

    /// Check cuBLAS result and return error if not success
    pub fn check_cublas(status: i32) -> Result<()> {
        if status == ptx_sys::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::cublas(status))
        }
    }

    /// Create a stable runtime API error from status code.
    pub fn stable(status: ptx_sys::PTXStableStatus) -> Self {
        let message = unsafe {
            let ptr = ptx_sys::ptx_stable_strerror(status);
            if ptr.is_null() {
                "Unknown stable API error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error::StableApi {
            status: status as i32,
            message,
        }
    }

    /// Check stable API status and return error if not OK.
    pub fn check_stable(status: ptx_sys::PTXStableStatus) -> Result<()> {
        if status == ptx_sys::PTXStableStatus::Ok {
            Ok(())
        } else {
            Err(Error::stable(status))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_stream_id_display() {
        let err = Error::InvalidStreamId { id: -1, pool_size: 4 };
        let msg = err.to_string();
        assert!(msg.contains("-1"), "should contain stream id");
        assert!(msg.contains("4"), "should contain pool size");
    }

    #[test]
    fn invalid_stream_id_diagnostic() {
        let err = Error::InvalidStreamId { id: 10, pool_size: 4 };
        assert_eq!(err.diagnostic_code(), "RT-ERR-0014");
        assert!(err.remediation().contains("num_streams"));
    }

    #[test]
    fn invalid_pointer_ownership_display() {
        let err = Error::InvalidPointerOwnership {
            ptr_debug: "0xdeadbeef".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("0xdeadbeef"));
        assert!(msg.contains("not owned"));
    }

    #[test]
    fn invalid_pointer_ownership_diagnostic() {
        let err = Error::InvalidPointerOwnership {
            ptr_debug: "0x1".into(),
        };
        assert_eq!(err.diagnostic_code(), "RT-ERR-0015");
        assert!(err.remediation().contains("allocated by this runtime"));
    }

    #[test]
    fn invalid_launch_context_display() {
        let err = Error::InvalidLaunchContext {
            detail: "null stream pointer".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("null stream pointer"));
        assert!(msg.contains("Invalid launch context"));
    }

    #[test]
    fn invalid_launch_context_diagnostic() {
        let err = Error::InvalidLaunchContext {
            detail: "null runtime".into(),
        };
        assert_eq!(err.diagnostic_code(), "RT-ERR-0016");
        assert!(err.remediation().contains("runtime is initialized"));
    }

    #[test]
    fn empty_stream_pool_display() {
        let err = Error::EmptyStreamPool;
        let msg = err.to_string();
        assert!(msg.contains("empty"));
        assert!(msg.contains("no streams"));
    }

    #[test]
    fn empty_stream_pool_diagnostic() {
        let err = Error::EmptyStreamPool;
        assert_eq!(err.diagnostic_code(), "RT-ERR-0017");
        assert!(err.remediation().contains("max_streams"));
    }

    #[test]
    fn all_remediation_hints_are_actionable() {
        let errors: Vec<Error> = vec![
            Error::InitFailed { device_id: 0 },
            Error::AllocationFailed { size: 0 },
            Error::InvalidPointer,
            Error::InvalidStreamId { id: 0, pool_size: 0 },
            Error::InvalidPointerOwnership { ptr_debug: "0x0".into() },
            Error::StreamError { message: "test".into() },
            Error::GraphError { message: "test".into() },
            Error::CublasError { status: 0 },
            Error::CudaError { code: 0, message: "test".into() },
            Error::NotInitialized,
            Error::NotSupported { message: "test".into() },
            Error::Internal { message: "test".into() },
            Error::InvalidLaunchContext { detail: "test".into() },
            Error::EmptyStreamPool,
        ];
        for err in &errors {
            let hint = err.remediation();
            assert!(
                !hint.is_empty(),
                "{} remediation should not be empty",
                err.diagnostic_code()
            );
            // Actionable hints should contain a verb
            assert!(
                hint.contains("check") || hint.contains("verify") || hint.contains("ensure")
                    || hint.contains("reduce") || hint.contains("increase") || hint.contains("use")
                    || hint.contains("align") || hint.contains("initialize") || hint.contains("inspect")
                    || hint.contains("validate") || hint.contains("capture") || hint.contains("rebuild")
                    || hint.contains("review"),
                "{} remediation '{}' should contain an actionable verb",
                err.diagnostic_code(),
                hint
            );
        }
    }
}
