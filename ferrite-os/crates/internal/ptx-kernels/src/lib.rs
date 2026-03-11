//! PTX Kernels - Candle CUDA kernel bindings for PTX-OS runtime
//!
//! This crate provides FFI bindings to optimized CUDA kernels from Candle,
//! designed to work seamlessly with the PTX-OS TLSF allocator.
//!
//! # Architecture
//!
//! - Raw CUDA kernels compiled from .cu files
//! - C launcher wrappers for kernel invocation
//! - Rust FFI bindings (this crate)
//! - Safe high-level wrappers (future: ptx-compute)
//!
//! # Memory Model
//!
//! All kernels accept raw GPU pointers allocated via PTX-OS TLSF allocator:
//! - `gpu_hot_alloc()` or `gpu_hot_alloc_async()`
//! - No framework-specific tensor types
//! - Direct pointer passing for zero overhead

#![allow(non_camel_case_types)]

pub mod test_kernels;
pub mod sha256;
pub mod guards;
pub mod candle;  // Now enabled with architecture fixes
pub mod safe_api;

// Re-export commonly used types
pub use guards::{GuardedBuffer, KernelContext, GuardError, GuardResult};
pub use safe_api::{unary, binary, gather, scan, topk, indexing, sort, ternary};

/// Re-export cudaStream_t from ptx-sys
pub use ptx_sys::cudaStream_t;

/// Check CUDA errors and return a structured error on failure.
#[inline]
pub fn check_cuda(err: ptx_sys::cudaError_t, msg: &str) -> GuardResult<()> {
    if err != ptx_sys::CUDA_SUCCESS {
        let err_str = unsafe {
            let ptr = ptx_sys::cudaGetErrorString(err);
            std::ffi::CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("unknown error")
        };
        return Err(GuardError::CudaError {
            operation: msg.to_string(),
            error_code: err,
            message: err_str.to_string(),
        });
    }
    Ok(())
}

/// Synchronize a CUDA stream and check for errors.
#[inline]
pub fn sync_stream(stream: cudaStream_t) -> GuardResult<()> {
    let err = unsafe { ptx_sys::cudaStreamSynchronize(stream) };
    check_cuda(err, "Stream synchronization failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_loads() {
        // Just verify the module compiles and links
        assert!(true);
    }

    #[test]
    fn check_cuda_success_returns_ok() {
        let result = check_cuda(ptx_sys::CUDA_SUCCESS, "test op");
        assert!(result.is_ok());
    }

    #[test]
    fn check_cuda_failure_returns_err() {
        // Use a non-zero error code (1 = cudaErrorInvalidValue on most CUDA versions)
        let result = check_cuda(1, "test op");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GuardError::CudaError { operation, error_code, .. } => {
                assert_eq!(operation, "test op");
                assert_eq!(error_code, 1);
            }
            _ => panic!("expected CudaError variant"),
        }
    }
}
