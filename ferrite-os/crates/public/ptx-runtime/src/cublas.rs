//! cuBLAS handle management and GEMM operations.

use crate::stream::Stream;
use crate::error::{Error, Result};

/// A cuBLAS handle for matrix operations.
pub struct CublasHandle {
    handle: ptx_sys::cublasHandle_t,
}

impl CublasHandle {
    /// Create a new cuBLAS handle.
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ptx_sys::cublasCreate_v2(&mut handle) };
        Error::check_cublas(status)?;
        Ok(Self { handle })
    }

    /// Get the raw handle.
    pub fn raw(&self) -> ptx_sys::cublasHandle_t {
        self.handle
    }

    /// Set the stream for this handle.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let status = unsafe { ptx_sys::cublasSetStream_v2(self.handle, stream.raw()) };
        Error::check_cublas(status)
    }

    /// Get the current stream.
    pub fn get_stream(&self) -> Result<ptx_sys::cudaStream_t> {
        let mut stream = std::ptr::null_mut();
        let status = unsafe { ptx_sys::cublasGetStream_v2(self.handle, &mut stream) };
        Error::check_cublas(status)?;
        Ok(stream)
    }

    /// Single-precision GEMM: C = alpha * op(A) * op(B) + beta * C
    ///
    /// # Arguments
    ///
    /// * `transa` - Operation on A (None, Transpose, ConjTranspose)
    /// * `transb` - Operation on B
    /// * `m` - Number of rows in op(A) and C
    /// * `n` - Number of columns in op(B) and C
    /// * `k` - Number of columns in op(A) and rows in op(B)
    /// * `alpha` - Scalar multiplier for A*B
    /// * `a` - Pointer to matrix A
    /// * `lda` - Leading dimension of A
    /// * `b` - Pointer to matrix B
    /// * `ldb` - Leading dimension of B
    /// * `beta` - Scalar multiplier for C
    /// * `c` - Pointer to matrix C (output)
    /// * `ldc` - Leading dimension of C
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<()> {
        let status = ptx_sys::cublasSgemm_v2(
            self.handle,
            transa.to_cublas(),
            transb.to_cublas(),
            m, n, k,
            &alpha,
            a, lda,
            b, ldb,
            &beta,
            c, ldc,
        );
        Error::check_cublas(status)
    }

    /// Double-precision GEMM: C = alpha * op(A) * op(B) + beta * C
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dgemm(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    ) -> Result<()> {
        let status = ptx_sys::cublasDgemm_v2(
            self.handle,
            transa.to_cublas(),
            transb.to_cublas(),
            m, n, k,
            &alpha,
            a, lda,
            b, ldb,
            &beta,
            c, ldc,
        );
        Error::check_cublas(status)
    }

    /// Strided batched GEMM for batch matrix multiplication.
    ///
    /// Computes C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i] for i in 0..batch_count
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_strided_batched(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = ptx_sys::cublasSgemmStridedBatched(
            self.handle,
            transa.to_cublas(),
            transb.to_cublas(),
            m, n, k,
            &alpha,
            a, lda, stride_a,
            b, ldb, stride_b,
            &beta,
            c, ldc, stride_c,
            batch_count,
        );
        Error::check_cublas(status)
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            ptx_sys::cublasDestroy_v2(self.handle);
        }
    }
}

// Safety: cuBLAS handles are thread-safe when used with proper synchronization
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// GEMM operation type (transpose, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GemmOp {
    /// No transpose
    #[default]
    None,
    /// Transpose
    Transpose,
    /// Conjugate transpose (same as transpose for real matrices)
    ConjTranspose,
}

impl GemmOp {
    /// Convert to cuBLAS operation type.
    pub fn to_cublas(self) -> ptx_sys::cublasOperation_t {
        match self {
            GemmOp::None => ptx_sys::cublasOperation_t::CUBLAS_OP_N,
            GemmOp::Transpose => ptx_sys::cublasOperation_t::CUBLAS_OP_T,
            GemmOp::ConjTranspose => ptx_sys::cublasOperation_t::CUBLAS_OP_C,
        }
    }
}

/// High-level matrix multiplication helper.
///
/// Computes C = A @ B (standard matrix multiply).
pub struct Gemm {
    handle: CublasHandle,
}

impl Gemm {
    /// Create a new GEMM helper with its own cuBLAS handle.
    pub fn new() -> Result<Self> {
        Ok(Self {
            handle: CublasHandle::new()?,
        })
    }

    /// Get a reference to the underlying cuBLAS handle.
    pub fn handle(&self) -> &CublasHandle {
        &self.handle
    }

    /// Set the stream for GEMM operations.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Matrix multiply: C = A @ B
    ///
    /// A: (m, k), B: (k, n), C: (m, n)
    ///
    /// Note: cuBLAS uses column-major order, so we compute B^T @ A^T = (A @ B)^T
    /// which gives us the correct result in row-major order.
    pub unsafe fn matmul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // For row-major: C = A @ B is equivalent to C^T = B^T @ A^T in column-major
        // So we swap A and B, and their dimensions
        self.handle.sgemm(
            GemmOp::None,      // B (no transpose)
            GemmOp::None,      // A (no transpose)
            n as i32,          // rows of B^T = cols of B
            m as i32,          // cols of A^T = rows of A
            k as i32,          // shared dimension
            1.0,               // alpha
            b,                 // B
            n as i32,          // ldb
            a,                 // A
            k as i32,          // lda
            0.0,               // beta
            c,                 // C
            n as i32,          // ldc
        )
    }

    /// Batched matrix multiply: C[i] = A[i] @ B[i]
    pub unsafe fn bmm_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        batch: usize,
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
            batch as i32,
        )
    }
}
