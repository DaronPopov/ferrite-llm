//! PTX-OS runtime wrapper with RAII semantics.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::cublas::CublasHandle;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::graph::{CudaGraph, GraphCapture};
use crate::memory::{GpuPtr, PtxRuntimeInner};
use crate::stream::{Stream, StreamPool, StreamPriority};

mod lifecycle;
mod scheduling;
mod resilience;

/// The main PTX-OS runtime handle.
///
/// This is the primary interface for interacting with the GPU through PTX-OS.
/// It manages memory allocation, streams, and CUDA graphs.
///
/// # Thread Safety
///
/// The runtime is thread-safe and can be shared between threads using `Arc<PtxRuntime>`.
///
/// # Example
///
/// ```no_run
/// use ptx_runtime::PtxRuntime;
///
/// let runtime = PtxRuntime::new(0).expect("Failed to init");
/// let ptr = runtime.alloc(1024).expect("Failed to alloc");
/// runtime.sync_all().expect("Failed to sync");
/// ```
pub struct PtxRuntime {
    inner: Arc<PtxRuntimeInner>,
    device: Device,
    streams: StreamPool,
    cublas: Mutex<Option<CublasHandle>>,
}

// The runtime is thread-safe
unsafe impl Send for PtxRuntime {}
unsafe impl Sync for PtxRuntime {}

/// Global runtime instance for convenience.
static GLOBAL_RUNTIME: once_cell::sync::OnceCell<Arc<PtxRuntime>> = once_cell::sync::OnceCell::new();

/// Get or initialize the global runtime on device 0.
pub fn global_runtime() -> Result<Arc<PtxRuntime>> {
    GLOBAL_RUNTIME
        .get_or_try_init(|| PtxRuntime::new(0).map(Arc::new))
        .cloned()
}

/// Initialize the global runtime with a specific device.
pub fn init_global_runtime(device_id: i32) -> Result<Arc<PtxRuntime>> {
    GLOBAL_RUNTIME
        .get_or_try_init(|| PtxRuntime::new(device_id).map(Arc::new))
        .cloned()
}

/// Install an existing runtime as the process-global runtime.
///
/// If a global runtime was already installed, the existing instance is retained
/// and returned.
pub fn install_global_runtime(runtime: Arc<PtxRuntime>) -> Arc<PtxRuntime> {
    if GLOBAL_RUNTIME.set(Arc::clone(&runtime)).is_ok() {
        runtime
    } else {
        GLOBAL_RUNTIME
            .get()
            .expect("global runtime should exist after set failure")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a CUDA-capable GPU and the PTX-OS library
    #[test]
    #[ignore]
    fn test_runtime_init() {
        let runtime = PtxRuntime::new(0).expect("Failed to init");
        assert!(runtime.device().is_gpu());
    }

    #[test]
    #[ignore]
    fn test_alloc_free() {
        let runtime = PtxRuntime::new(0).expect("Failed to init");
        let ptr = runtime.alloc(1024).expect("Failed to alloc");
        assert!(ptr.is_valid());
        // ptr is automatically freed when dropped
    }
}
