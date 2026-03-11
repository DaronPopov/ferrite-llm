//! Matrix multiplication operations.
//!
//! This module provides high-level APIs for matrix multiplication (GEMM)
//! operations using cuBLAS, with support for single/double precision,
//! batched operations, and multi-stream parallelism.

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, CublasHandle, GemmOp, Stream, Result};

/// High-level matrix multiplication helper.
///
/// Manages cuBLAS handles and provides ergonomic APIs for matrix
/// multiplication operations.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use ptx_runtime::PtxRuntime;
/// use ptx_compute::gemm::Matmul;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = Arc::new(PtxRuntime::new(0)?);
/// let matmul = Matmul::new(&runtime)?;
///
/// // C = A @ B where A is (m, k), B is (k, n), C is (m, n)
/// let (m, n, k) = (1024, 1024, 1024);
/// # let (a_ptr, b_ptr, c_ptr) = (std::ptr::null(), std::ptr::null(), std::ptr::null_mut());
/// unsafe {
///     matmul.multiply_f32(a_ptr, b_ptr, c_ptr, m, n, k)?;
/// }
/// # Ok(())
/// # }
/// ```
pub struct Matmul {
    handle: CublasHandle,
    #[allow(dead_code)]
    runtime: Arc<PtxRuntime>,
}

impl Matmul {
    /// Create a new Matmul helper.
    pub fn new(runtime: &Arc<PtxRuntime>) -> Result<Self> {
        Ok(Self {
            handle: CublasHandle::new()?,
            runtime: Arc::clone(runtime),
        })
    }

    /// Set the stream for this matmul operation.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Matrix multiply (FP32): C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to matrix A (m × k)
    /// * `b` - Pointer to matrix B (k × n)
    /// * `c` - Pointer to output matrix C (m × n)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A and rows in B
    ///
    /// # Safety
    ///
    /// Pointers must be valid GPU memory addresses with correct sizes.
    pub unsafe fn multiply_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // cuBLAS uses column-major, so we compute: C = A @ B
        // In column-major this is: C^T = B^T @ A^T
        self.handle.sgemm(
            GemmOp::None,
            GemmOp::None,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b, n as i32,
            a, k as i32,
            0.0,
            c, n as i32,
        )
    }

    /// Matrix multiply (FP64): C = A @ B
    pub unsafe fn multiply_f64(
        &self,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.handle.dgemm(
            GemmOp::None,
            GemmOp::None,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b, n as i32,
            a, k as i32,
            0.0,
            c, n as i32,
        )
    }

    /// Batched matrix multiply (FP32): C[i] = A[i] @ B[i]
    ///
    /// Performs multiple independent matrix multiplications in a single call.
    pub unsafe fn batched_multiply_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        self.handle.sgemm_strided_batched(
            GemmOp::None,
            GemmOp::None,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b, n as i32, stride_b,
            a, k as i32, stride_a,
            0.0,
            c, n as i32, stride_c,
            batch_size as i32,
        )
    }

    /// Calculate FLOPS for a matrix multiplication.
    pub fn flops(m: usize, n: usize, k: usize) -> f64 {
        2.0 * m as f64 * n as f64 * k as f64
    }
}

/// Multi-stream parallel matrix multiplication helper.
///
/// Distributes matrix multiplications across multiple GPU streams
/// for maximum parallelism and throughput.
pub struct ParallelMatmul {
    handles: Vec<CublasHandle>,
    runtime: Arc<PtxRuntime>,
}

impl ParallelMatmul {
    /// Create a parallel matmul helper with the specified number of streams.
    pub fn new(runtime: &Arc<PtxRuntime>, num_streams: usize) -> Result<Self> {
        let handles: Result<Vec<_>> = (0..num_streams)
            .map(|_| CublasHandle::new())
            .collect();

        Ok(Self {
            handles: handles?,
            runtime: Arc::clone(runtime),
        })
    }

    /// Get the number of streams.
    pub fn num_streams(&self) -> usize {
        self.handles.len()
    }

    /// Perform matrix multiplications in parallel across all streams.
    ///
    /// Distributes `operations` matrix multiplications across the available
    /// streams, executing them in parallel.
    ///
    /// # Arguments
    ///
    /// * `operations` - Number of matmul operations to perform
    /// * `m, n, k` - Matrix dimensions (all operations use same dimensions)
    /// * `a_ptrs` - Array of input A matrix pointers
    /// * `b_ptrs` - Array of input B matrix pointers
    /// * `c_ptrs` - Array of output C matrix pointers
    pub unsafe fn parallel_multiply_f32(
        &self,
        operations: usize,
        m: usize,
        n: usize,
        k: usize,
        a_ptrs: &[*const f32],
        b_ptrs: &[*const f32],
        c_ptrs: &[*mut f32],
    ) -> Result<()> {
        assert_eq!(a_ptrs.len(), operations);
        assert_eq!(b_ptrs.len(), operations);
        assert_eq!(c_ptrs.len(), operations);

        // Launch operations across streams
        for (op_id, ((a, b), c)) in a_ptrs.iter().zip(b_ptrs).zip(c_ptrs).enumerate() {
            let stream_id = op_id % self.handles.len();
            let handle = &self.handles[stream_id];
            let stream = self.runtime.stream(stream_id as i32)?;

            handle.set_stream(&stream)?;
            handle.sgemm(
                GemmOp::None,
                GemmOp::None,
                n as i32,
                m as i32,
                k as i32,
                1.0,
                *b, n as i32,
                *a, k as i32,
                0.0,
                *c, n as i32,
            )?;
        }

        // Synchronize all streams
        self.runtime.sync_all()?;
        Ok(())
    }
}
