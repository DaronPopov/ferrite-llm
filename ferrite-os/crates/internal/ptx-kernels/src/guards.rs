//! Memory guards and validation layer for PTX-OS TLSF allocator
//!
//! This module provides a thin abstraction that validates memory operations
//! before executing Candle kernels, ensuring that:
//! - All pointers are owned by the TLSF allocator
//! - Memory bounds are valid
//! - Stream ordering is preserved
//! - Operations are type-safe

use ptx_sys::cudaStream_t;
use libc::c_void;

/// Result type for guarded operations
pub type GuardResult<T> = Result<T, GuardError>;

/// Errors that can occur during guarded kernel execution.
///
/// Each variant carries a diagnostic code (retrievable via [`GuardError::diagnostic_code`])
/// and a remediation hint (via [`GuardError::remediation`]).
#[derive(Debug, Clone)]
pub enum GuardError {
    /// Pointer is not owned by the TLSF allocator.
    InvalidPointer { ptr: usize },

    /// Buffer size is too small for the operation.
    BufferTooSmall {
        required: usize,
        available: usize,
    },

    /// Null pointer provided where valid pointer expected.
    NullPointer { operation: &'static str },

    /// Zero-element operation (would produce a zero-size kernel launch).
    ZeroElements { kernel: &'static str },

    /// CUDA kernel launch failed.
    KernelLaunchFailed {
        kernel: &'static str,
        error_code: i32,
    },

    /// Stream synchronization failed.
    StreamSyncFailed { error_code: i32 },

    /// General CUDA runtime error.
    CudaError {
        operation: String,
        error_code: i32,
        message: String,
    },

    /// Invalid kernel launch context (null runtime or stream pointer).
    InvalidLaunchContext { detail: &'static str },
}

impl GuardError {
    /// Return a structured diagnostic code for this error category.
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            GuardError::InvalidPointer { .. } => "KERN-GUARD-0001",
            GuardError::BufferTooSmall { .. } => "KERN-GUARD-0002",
            GuardError::NullPointer { .. } => "KERN-GUARD-0003",
            GuardError::ZeroElements { .. } => "KERN-GUARD-0004",
            GuardError::KernelLaunchFailed { .. } => "KERN-GUARD-0005",
            GuardError::StreamSyncFailed { .. } => "KERN-GUARD-0006",
            GuardError::CudaError { .. } => "KERN-GUARD-0007",
            GuardError::InvalidLaunchContext { .. } => "KERN-GUARD-0008",
        }
    }

    /// Return a human-readable remediation hint.
    pub fn remediation(&self) -> &'static str {
        match self {
            GuardError::InvalidPointer { .. } => {
                "ensure the pointer was allocated by the TLSF allocator and has not been freed"
            }
            GuardError::BufferTooSmall { .. } => {
                "allocate a larger buffer or reduce the number of elements"
            }
            GuardError::NullPointer { .. } => {
                "provide a valid non-null pointer from the TLSF allocator"
            }
            GuardError::ZeroElements { .. } => {
                "provide a non-zero numel; zero-element kernels are not supported"
            }
            GuardError::KernelLaunchFailed { .. } => {
                "check CUDA device state, driver version, and kernel parameters"
            }
            GuardError::StreamSyncFailed { .. } => {
                "check for prior asynchronous errors on this stream"
            }
            GuardError::CudaError { .. } => {
                "inspect the CUDA error code and ensure the device is in a healthy state"
            }
            GuardError::InvalidLaunchContext { .. } => {
                "ensure runtime is initialized and stream is valid before creating a KernelContext"
            }
        }
    }
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardError::InvalidPointer { ptr } => {
                write!(f, "[{}] pointer 0x{:x} not owned by TLSF allocator", self.diagnostic_code(), ptr)
            }
            GuardError::BufferTooSmall { required, available } => {
                write!(f, "[{}] buffer too small: need {} bytes, have {}", self.diagnostic_code(), required, available)
            }
            GuardError::NullPointer { operation } => {
                write!(f, "[{}] null pointer in operation: {}", self.diagnostic_code(), operation)
            }
            GuardError::ZeroElements { kernel } => {
                write!(f, "[{}] zero-element launch rejected for kernel: {}", self.diagnostic_code(), kernel)
            }
            GuardError::KernelLaunchFailed { kernel, error_code } => {
                write!(f, "[{}] kernel '{}' launch failed (cuda error {})", self.diagnostic_code(), kernel, error_code)
            }
            GuardError::StreamSyncFailed { error_code } => {
                write!(f, "[{}] stream sync failed (cuda error {})", self.diagnostic_code(), error_code)
            }
            GuardError::CudaError { operation, error_code, message } => {
                write!(f, "[{}] CUDA error in {}: {} (code {})", self.diagnostic_code(), operation, message, error_code)
            }
            GuardError::InvalidLaunchContext { detail } => {
                write!(f, "[{}] invalid launch context: {}", self.diagnostic_code(), detail)
            }
        }
    }
}

impl std::error::Error for GuardError {}

/// A guarded GPU buffer that validates TLSF ownership
#[derive(Debug)]
pub struct GuardedBuffer {
    pub(crate) ptr: *mut c_void,
    pub(crate) size_bytes: usize,
    pub(crate) runtime: *mut ptx_sys::GPUHotRuntime,
}

impl GuardedBuffer {
    /// Create a new guarded buffer from a raw pointer
    ///
    /// # Safety
    ///
    /// - `ptr` must be allocated by the given runtime's TLSF allocator
    /// - `size_bytes` must be the actual allocation size
    pub unsafe fn new(
        ptr: *mut c_void,
        size_bytes: usize,
        runtime: *mut ptx_sys::GPUHotRuntime,
    ) -> GuardResult<Self> {
        // Validate pointer is not null
        if ptr.is_null() {
            return Err(GuardError::NullPointer {
                operation: "GuardedBuffer::new"
            });
        }

        // Validate TLSF ownership
        if !ptx_sys::gpu_hot_owns_ptr(runtime, ptr) {
            return Err(GuardError::InvalidPointer { ptr: ptr as usize });
        }

        Ok(Self {
            ptr,
            size_bytes,
            runtime,
        })
    }

    /// Get the raw pointer (for kernel calls)
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the raw pointer as a typed pointer
    #[inline]
    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get the size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the number of elements of type T
    #[inline]
    pub fn len<T>(&self) -> usize {
        self.size_bytes / std::mem::size_of::<T>()
    }

    /// Check if this buffer can hold `count` elements of type T
    pub fn can_hold<T>(&self, count: usize) -> bool {
        count * std::mem::size_of::<T>() <= self.size_bytes
    }

    /// Validate that this buffer can hold `count` elements of type T
    pub fn validate_capacity<T>(&self, count: usize) -> GuardResult<()> {
        let required = count * std::mem::size_of::<T>();
        if required > self.size_bytes {
            return Err(GuardError::BufferTooSmall {
                required,
                available: self.size_bytes,
            });
        }
        Ok(())
    }

    /// Re-validate TLSF ownership (useful after long operations)
    pub fn revalidate(&self) -> GuardResult<()> {
        if !unsafe { ptx_sys::gpu_hot_owns_ptr(self.runtime, self.ptr) } {
            return Err(GuardError::InvalidPointer { ptr: self.ptr as usize });
        }
        Ok(())
    }
}

// GuardedBuffer is safe to send between threads (GPU pointer)
unsafe impl Send for GuardedBuffer {}
unsafe impl Sync for GuardedBuffer {}

/// A validated kernel launch context
///
/// This ensures all parameters are valid before launching kernels
#[derive(Debug)]
pub struct KernelContext {
    #[allow(dead_code)]
    pub(crate) runtime: *mut ptx_sys::GPUHotRuntime,
    pub(crate) stream: cudaStream_t,
}

impl KernelContext {
    /// Create a new kernel context with validated pointers.
    ///
    /// # Errors
    ///
    /// Returns `GuardError::InvalidLaunchContext` if `runtime` is null.
    pub fn new(runtime: *mut ptx_sys::GPUHotRuntime, stream: cudaStream_t) -> GuardResult<Self> {
        if runtime.is_null() {
            return Err(GuardError::InvalidLaunchContext {
                detail: "null runtime pointer",
            });
        }
        Ok(Self { runtime, stream })
    }

    /// Get the CUDA stream
    #[inline]
    pub fn stream(&self) -> cudaStream_t {
        self.stream
    }

    /// Synchronize the stream and check for errors
    pub fn sync(&self) -> GuardResult<()> {
        let err = unsafe { ptx_sys::cudaStreamSynchronize(self.stream) };
        if err != ptx_sys::CUDA_SUCCESS {
            return Err(GuardError::StreamSyncFailed { error_code: err });
        }
        Ok(())
    }

    /// Check last CUDA error without blocking.
    pub fn check_last_error(&self) -> GuardResult<()> {
        self.check_last_error_for("unknown")
    }

    /// Check last CUDA error, attributing any failure to the named kernel.
    pub fn check_last_error_for(&self, kernel: &'static str) -> GuardResult<()> {
        let err = unsafe { ptx_sys::cudaGetLastError() };
        if err != ptx_sys::CUDA_SUCCESS {
            return Err(GuardError::KernelLaunchFailed {
                kernel,
                error_code: err,
            });
        }
        Ok(())
    }
}

/// Validated parameters for unary operations.
#[derive(Debug)]
pub struct UnaryOpGuard<'a> {
    pub input: &'a GuardedBuffer,
    pub output: &'a GuardedBuffer,
    pub numel: usize,
    pub context: &'a KernelContext,
}

impl<'a> UnaryOpGuard<'a> {
    /// Create and validate a unary operation.
    ///
    /// Rejects zero-element operations, validates buffer capacities, and
    /// re-checks TLSF ownership.
    pub fn new(
        input: &'a GuardedBuffer,
        output: &'a GuardedBuffer,
        numel: usize,
        context: &'a KernelContext,
    ) -> GuardResult<Self> {
        // Reject zero-element launches
        if numel == 0 {
            return Err(GuardError::ZeroElements { kernel: "unary" });
        }

        // Validate input buffer
        input.validate_capacity::<f32>(numel)?;
        input.revalidate()?;

        // Validate output buffer
        output.validate_capacity::<f32>(numel)?;
        output.revalidate()?;

        Ok(Self {
            input,
            output,
            numel,
            context,
        })
    }

    /// Get parameters for kernel launch (contiguous tensors)
    #[inline]
    pub fn kernel_params(&self) -> (usize, usize, *const usize, *const f32, *mut f32, cudaStream_t) {
        (
            self.numel,
            0, // num_dims = 0 for contiguous
            std::ptr::null(), // info = null for contiguous
            self.input.as_ptr_typed::<f32>(),
            self.output.as_ptr_typed::<f32>(),
            self.context.stream(),
        )
    }
}

/// Validated parameters for binary operations
#[derive(Debug)]
pub struct BinaryOpGuard<'a> {
    pub left: &'a GuardedBuffer,
    pub right: &'a GuardedBuffer,
    pub output: &'a GuardedBuffer,
    pub numel: usize,
    pub context: &'a KernelContext,
}

impl<'a> BinaryOpGuard<'a> {
    /// Create and validate a binary operation.
    ///
    /// Rejects zero-element operations, validates all three buffer capacities,
    /// and re-checks TLSF ownership.
    pub fn new(
        left: &'a GuardedBuffer,
        right: &'a GuardedBuffer,
        output: &'a GuardedBuffer,
        numel: usize,
        context: &'a KernelContext,
    ) -> GuardResult<Self> {
        // Reject zero-element launches
        if numel == 0 {
            return Err(GuardError::ZeroElements { kernel: "binary" });
        }

        // Validate all buffers
        left.validate_capacity::<f32>(numel)?;
        left.revalidate()?;

        right.validate_capacity::<f32>(numel)?;
        right.revalidate()?;

        output.validate_capacity::<f32>(numel)?;
        output.revalidate()?;

        Ok(Self {
            left,
            right,
            output,
            numel,
            context,
        })
    }

    /// Get parameters for kernel launch (contiguous tensors)
    #[inline]
    pub fn kernel_params(&self) -> (
        usize,
        usize,
        *const usize,
        *const usize,
        *const f32,
        *const usize,
        *const f32,
        *mut f32,
        cudaStream_t
    ) {
        (
            self.numel,
            0, // num_dims = 0 for contiguous
            std::ptr::null(), // dims = null for contiguous
            std::ptr::null(), // left_strides = null for contiguous
            self.left.as_ptr_typed::<f32>(),
            std::ptr::null(), // right_strides = null for contiguous
            self.right.as_ptr_typed::<f32>(),
            self.output.as_ptr_typed::<f32>(),
            self.context.stream(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_error_display() {
        let err = GuardError::BufferTooSmall {
            required: 1024,
            available: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_buffer_capacity() {
        // This test just verifies the logic, doesn't need actual GPU
        let size = 1024;
        let count_ok = 256; // 256 * 4 = 1024 bytes
        let count_fail = 257; // 257 * 4 = 1028 bytes

        assert_eq!(count_ok * std::mem::size_of::<f32>(), size);
        assert!(count_fail * std::mem::size_of::<f32>() > size);
    }

    #[test]
    fn test_cuda_error_display() {
        let err = GuardError::CudaError {
            operation: "cudaMalloc".into(),
            error_code: 2,
            message: "out of memory".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("cudaMalloc"));
        assert!(msg.contains("out of memory"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_null_pointer_display() {
        let err = GuardError::NullPointer { operation: "free_async" };
        assert!(err.to_string().contains("free_async"));
    }

    // ---- Diagnostic code tests ----

    #[test]
    fn test_diagnostic_codes_are_unique() {
        let errors: Vec<GuardError> = vec![
            GuardError::InvalidPointer { ptr: 0 },
            GuardError::BufferTooSmall { required: 1, available: 0 },
            GuardError::NullPointer { operation: "test" },
            GuardError::ZeroElements { kernel: "test" },
            GuardError::KernelLaunchFailed { kernel: "test", error_code: 1 },
            GuardError::StreamSyncFailed { error_code: 1 },
            GuardError::CudaError { operation: "t".into(), error_code: 1, message: "t".into() },
            GuardError::InvalidLaunchContext { detail: "test" },
        ];
        let codes: Vec<&str> = errors.iter().map(|e| e.diagnostic_code()).collect();
        // All codes should be unique
        let mut deduped = codes.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(codes.len(), deduped.len(), "diagnostic codes must be unique");
    }

    #[test]
    fn test_diagnostic_codes_in_display() {
        let err = GuardError::InvalidPointer { ptr: 0 };
        assert!(err.to_string().contains("KERN-GUARD-0001"));

        let err = GuardError::BufferTooSmall { required: 100, available: 50 };
        assert!(err.to_string().contains("KERN-GUARD-0002"));

        let err = GuardError::ZeroElements { kernel: "usqrt_f32" };
        let msg = err.to_string();
        assert!(msg.contains("KERN-GUARD-0004"));
        assert!(msg.contains("usqrt_f32"));
    }

    #[test]
    fn test_remediation_hints_non_empty() {
        let errors: Vec<GuardError> = vec![
            GuardError::InvalidPointer { ptr: 0 },
            GuardError::BufferTooSmall { required: 1, available: 0 },
            GuardError::NullPointer { operation: "test" },
            GuardError::ZeroElements { kernel: "test" },
            GuardError::KernelLaunchFailed { kernel: "test", error_code: 1 },
            GuardError::StreamSyncFailed { error_code: 1 },
            GuardError::CudaError { operation: "t".into(), error_code: 1, message: "t".into() },
            GuardError::InvalidLaunchContext { detail: "test" },
        ];
        for err in &errors {
            assert!(
                !err.remediation().is_empty(),
                "remediation for {:?} should not be empty",
                err.diagnostic_code()
            );
        }
    }

    // ---- ZeroElements tests ----

    #[test]
    fn test_zero_elements_display() {
        let err = GuardError::ZeroElements { kernel: "ugelu_f32" };
        let msg = err.to_string();
        assert!(msg.contains("zero-element"));
        assert!(msg.contains("ugelu_f32"));
    }

    // ---- KernelLaunchFailed kernel name propagation ----

    #[test]
    fn test_kernel_launch_failed_carries_name() {
        let err = GuardError::KernelLaunchFailed {
            kernel: "badd_f32",
            error_code: 700,
        };
        let msg = err.to_string();
        assert!(msg.contains("badd_f32"), "error should carry kernel name");
        assert!(msg.contains("700"), "error should carry CUDA error code");
        assert!(msg.contains("KERN-GUARD-0005"));
    }

    #[test]
    fn test_stream_sync_failed_display() {
        let err = GuardError::StreamSyncFailed { error_code: 6 };
        let msg = err.to_string();
        assert!(msg.contains("KERN-GUARD-0006"));
        assert!(msg.contains("6"));
    }

    // ---- GuardError is std::error::Error ----

    #[test]
    fn test_guard_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(GuardError::ZeroElements { kernel: "test" });
        assert!(err.to_string().contains("zero-element"));
    }

    // ---- InvalidLaunchContext tests ----

    #[test]
    fn test_invalid_launch_context_display() {
        let err = GuardError::InvalidLaunchContext { detail: "null runtime pointer" };
        let msg = err.to_string();
        assert!(msg.contains("KERN-GUARD-0008"));
        assert!(msg.contains("null runtime pointer"));
    }

    #[test]
    fn test_invalid_launch_context_remediation() {
        let err = GuardError::InvalidLaunchContext { detail: "test" };
        assert!(!err.remediation().is_empty());
        assert!(err.remediation().contains("runtime is initialized"));
    }

    #[test]
    fn test_kernel_context_rejects_null_runtime() {
        let result = KernelContext::new(std::ptr::null_mut(), std::ptr::null_mut());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GuardError::InvalidLaunchContext { .. }));
        assert_eq!(err.diagnostic_code(), "KERN-GUARD-0008");
    }

    #[test]
    fn test_kernel_context_accepts_valid_runtime() {
        // Use a non-null (but fake) runtime pointer — only tests the null check,
        // not actual CUDA operations.
        let fake_runtime = 0x1234usize as *mut ptx_sys::GPUHotRuntime;
        let result = KernelContext::new(fake_runtime, std::ptr::null_mut());
        assert!(result.is_ok());
    }

    // ---- GuardedBuffer null-pointer rejection ----

    #[test]
    fn test_guarded_buffer_rejects_null_ptr() {
        let fake_runtime = 0x1234usize as *mut ptx_sys::GPUHotRuntime;
        let result = unsafe { GuardedBuffer::new(std::ptr::null_mut(), 1024, fake_runtime) };
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GuardError::NullPointer { .. }));
        assert_eq!(err.diagnostic_code(), "KERN-GUARD-0003");
    }

    // ---- Buffer capacity validation ----

    #[test]
    fn test_validate_capacity_ok() {
        // Fake buffer with known size
        let buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 4096,
            runtime: std::ptr::null_mut(),
        };
        assert!(buf.validate_capacity::<f32>(1024).is_ok()); // 1024 * 4 = 4096
        assert!(buf.validate_capacity::<f32>(512).is_ok());  // 512 * 4 = 2048
    }

    #[test]
    fn test_validate_capacity_too_small() {
        let buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 4096,
            runtime: std::ptr::null_mut(),
        };
        let result = buf.validate_capacity::<f32>(1025); // 1025 * 4 = 4100 > 4096
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GuardError::BufferTooSmall { required: 4100, available: 4096 }));
    }

    #[test]
    fn test_can_hold_boundary() {
        let buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 100,
            runtime: std::ptr::null_mut(),
        };
        assert!(buf.can_hold::<u8>(100));
        assert!(!buf.can_hold::<u8>(101));
        assert!(buf.can_hold::<f32>(25));  // 25 * 4 = 100
        assert!(!buf.can_hold::<f32>(26)); // 26 * 4 = 104
    }

    #[test]
    fn test_buffer_len_calculation() {
        let buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 4096,
            runtime: std::ptr::null_mut(),
        };
        assert_eq!(buf.len::<f32>(), 1024);
        assert_eq!(buf.len::<u8>(), 4096);
        assert_eq!(buf.len::<f64>(), 512);
    }

    // ---- UnaryOpGuard zero-element rejection ----

    #[test]
    fn test_unary_guard_rejects_zero_elements() {
        let fake_buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 1024,
            runtime: std::ptr::null_mut(),
        };
        let fake_runtime = 0x1234usize as *mut ptx_sys::GPUHotRuntime;
        let ctx = KernelContext::new(fake_runtime, std::ptr::null_mut()).unwrap();

        let result = UnaryOpGuard::new(&fake_buf, &fake_buf, 0, &ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GuardError::ZeroElements { .. }));
    }

    // ---- BinaryOpGuard zero-element rejection ----

    #[test]
    fn test_binary_guard_rejects_zero_elements() {
        let fake_buf = GuardedBuffer {
            ptr: 0xDEAD_BEEFusize as *mut c_void,
            size_bytes: 1024,
            runtime: std::ptr::null_mut(),
        };
        let fake_runtime = 0x1234usize as *mut ptx_sys::GPUHotRuntime;
        let ctx = KernelContext::new(fake_runtime, std::ptr::null_mut()).unwrap();

        let result = BinaryOpGuard::new(&fake_buf, &fake_buf, &fake_buf, 0, &ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GuardError::ZeroElements { .. }));
    }
}
