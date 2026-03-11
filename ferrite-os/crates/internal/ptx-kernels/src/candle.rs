//! Candle CUDA kernel FFI bindings
//!
//! These are direct bindings to the C launcher functions that invoke
//! the actual CUDA kernels from Candle.
//!
//! # Kernel Categories
//!
//! - **Unary**: Element-wise operations on single tensor
//! - **Binary**: Element-wise operations on two tensors
//! - **Ternary**: Three-input operations (where, clamp, etc.)
//! - **Reductions**: Sum, mean, max, min, etc.
//! - **Affine**: Scale and bias transformations
//!
//! # Memory Layout
//!
//! Kernels support both contiguous and strided tensors:
//! - Set `info` to `null()` for contiguous tensors
//! - Provide dims + strides for strided tensors

use libc::size_t;
use crate::cudaStream_t;

// ============================================================================
// Unary Operations (F32)
// ============================================================================

extern "C" {
    /// GELU activation (Gaussian Error Linear Unit) - Tanh approximation
    ///
    /// Formula: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    pub fn candle_launch_ugelu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// ReLU activation (Rectified Linear Unit)
    ///
    /// Formula: `max(0, x)`
    pub fn candle_launch_urelu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// SiLU activation (Sigmoid Linear Unit) aka Swish
    ///
    /// Formula: `x / (1 + exp(-x))`
    pub fn candle_launch_usilu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Sigmoid activation
    ///
    /// Formula: `1 / (1 + exp(-x))`
    pub fn candle_launch_usigmoid_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Absolute value
    pub fn candle_launch_uabs_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Square root
    pub fn candle_launch_usqrt_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Exponential (e^x)
    pub fn candle_launch_uexp_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Natural logarithm
    pub fn candle_launch_ulog_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Hyperbolic tangent
    pub fn candle_launch_utanh_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Gather/Scatter Operations (F32)
// ============================================================================

extern "C" {
    /// Gather elements from input along a dimension using integer indices.
    ///
    /// Tensor viewed as [outer, input_dim_size, inner]; indices have shape
    /// [outer, idx_dim_size, inner].  Output has same shape as indices.
    pub fn candle_launch_gather_f32(
        input: *const f32,
        indices: *const i32,
        output: *mut f32,
        outer: size_t,
        input_dim_size: size_t,
        idx_dim_size: size_t,
        inner: size_t,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Scan/Prefix Operations (F32)
// ============================================================================

extern "C" {
    /// Inclusive prefix sum (cumulative sum) along a dimension.
    ///
    /// Tensor is viewed as [outer, dim_size, inner] and the scan runs along
    /// the middle axis.  Output has the same shape as input.
    pub fn candle_launch_cumsum_f32(
        input: *const f32,
        output: *mut f32,
        outer: size_t,
        dim_size: size_t,
        inner: size_t,
        stream: cudaStream_t,
    );
}

// ============================================================================
// TopK / Selection Operations (F32)
// ============================================================================

extern "C" {
    /// Select top-k values and indices along a dimension.
    ///
    /// Tensor is viewed as [outer, dim_size, inner].  Selects k largest
    /// (or smallest) elements along the middle axis.
    /// values_out  shape: [outer, k, inner]
    /// indices_out shape: [outer, k, inner]  (int32)
    pub fn candle_launch_topk_f32(
        input: *const f32,
        values_out: *mut f32,
        indices_out: *mut i32,
        outer: size_t,
        dim_size: size_t,
        inner: size_t,
        k: size_t,
        largest: libc::c_int,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Binary Operations (F32)
// ============================================================================

extern "C" {
    /// Element-wise addition
    ///
    /// Formula: `out[i] = left[i] + right[i]`
    pub fn candle_launch_badd_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise multiplication
    ///
    /// Formula: `out[i] = left[i] * right[i]`
    pub fn candle_launch_bmul_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise subtraction
    ///
    /// Formula: `out[i] = left[i] - right[i]`
    pub fn candle_launch_bsub_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise division
    ///
    /// Formula: `out[i] = left[i] / right[i]`
    pub fn candle_launch_bdiv_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Safe Wrappers (Optional - for convenience)
// ============================================================================

#[deprecated(note = "use ptx_kernels::safe_api::unary instead — these bypass the guard layer")]
/// Safe wrapper for unary operations on contiguous F32 tensors
pub mod unary_f32 {
    use super::*;

    /// Apply GELU activation in-place or to output buffer
    #[inline]
    pub unsafe fn gelu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_ugelu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply ReLU activation
    #[inline]
    pub unsafe fn relu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_urelu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply SiLU/Swish activation
    #[inline]
    pub unsafe fn silu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usilu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply Sigmoid activation
    #[inline]
    pub unsafe fn sigmoid(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usigmoid_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute absolute value
    #[inline]
    pub unsafe fn abs(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_uabs_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute square root
    #[inline]
    pub unsafe fn sqrt(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usqrt_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute exponential
    #[inline]
    pub unsafe fn exp(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_uexp_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute natural logarithm
    #[inline]
    pub unsafe fn log(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_ulog_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute hyperbolic tangent
    #[inline]
    pub unsafe fn tanh(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_utanh_f32(numel, 0, std::ptr::null(), input, output, stream);
    }
}

#[deprecated(note = "use ptx_kernels::safe_api::gather instead — these bypass the guard layer")]
/// Safe wrapper for gather/scatter operations on contiguous F32 tensors
pub mod gather_f32 {
    use super::*;

    /// Gather elements from input along a dimension using i32 indices.
    #[inline]
    pub unsafe fn gather(
        input: *const f32,
        indices: *const i32,
        output: *mut f32,
        outer: usize,
        input_dim_size: usize,
        idx_dim_size: usize,
        inner: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_gather_f32(
            input, indices, output, outer, input_dim_size, idx_dim_size, inner, stream,
        );
    }
}

#[deprecated(note = "use ptx_kernels::safe_api::scan instead — these bypass the guard layer")]
/// Safe wrapper for scan/prefix operations on contiguous F32 tensors
pub mod scan_f32 {
    use super::*;

    /// Inclusive cumulative sum along a dimension.
    ///
    /// Tensor viewed as [outer, dim_size, inner]; scan runs along middle axis.
    #[inline]
    pub unsafe fn cumsum(
        input: *const f32,
        output: *mut f32,
        outer: usize,
        dim_size: usize,
        inner: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_cumsum_f32(input, output, outer, dim_size, inner, stream);
    }
}

#[deprecated(note = "use ptx_kernels::safe_api::topk instead — these bypass the guard layer")]
/// Safe wrapper for top-k selection on contiguous F32 tensors
pub mod topk_f32 {
    use super::*;

    /// Select top-k values and their indices along a dimension.
    ///
    /// Tensor viewed as [outer, dim_size, inner]; selects k extreme values
    /// along the middle axis.  `largest` controls whether max (true) or
    /// min (false) elements are selected.
    #[inline]
    pub unsafe fn topk(
        input: *const f32,
        values_out: *mut f32,
        indices_out: *mut i32,
        outer: usize,
        dim_size: usize,
        inner: usize,
        k: usize,
        largest: bool,
        stream: cudaStream_t,
    ) {
        candle_launch_topk_f32(
            input, values_out, indices_out, outer, dim_size, inner, k,
            if largest { 1 } else { 0 },
            stream,
        );
    }
}

// ============================================================================
// Indexing Operations (F32) — candle indexing.cu via indexing_launchers.cu
// ============================================================================

extern "C" {
    /// Index-select: gather slices along a dimension by index.
    /// Tensor viewed as [left, src_dim, right]; indices select from src_dim.
    pub fn candle_launch_index_select_f32(
        input: *const f32,
        ids: *const u32,
        output: *mut f32,
        left_size: size_t,
        src_dim_size: size_t,
        ids_dim_size: size_t,
        right_size: size_t,
        stream: cudaStream_t,
    );

    /// Scatter-add: accumulate src values into output at positions given by ids.
    /// Output must be pre-initialized (e.g. zeros).
    pub fn candle_launch_scatter_add_f32(
        ids: *const u32,
        src: *const f32,
        output: *mut f32,
        left_size: size_t,
        src_dim_size: size_t,
        dst_dim_size: size_t,
        right_size: size_t,
        stream: cudaStream_t,
    );

    /// Index-add: accumulate src into output using a 1D index vector.
    pub fn candle_launch_index_add_f32(
        ids: *const u32,
        ids_dim_size: size_t,
        src: *const f32,
        output: *mut f32,
        left_size: size_t,
        src_dim_size: size_t,
        dst_dim_size: size_t,
        right_size: size_t,
        stream: cudaStream_t,
    );
}

#[deprecated(note = "use ptx_kernels::safe_api::indexing instead — these bypass the guard layer")]
/// Safe wrapper for indexing operations on contiguous F32 tensors
pub mod indexing_f32 {
    use super::*;

    #[inline]
    pub unsafe fn index_select(
        input: *const f32,
        ids: *const u32,
        output: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_index_select_f32(
            input, ids, output, left_size, src_dim_size, ids_dim_size, right_size, stream,
        );
    }

    #[inline]
    pub unsafe fn scatter_add(
        ids: *const u32,
        src: *const f32,
        output: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_scatter_add_f32(
            ids, src, output, left_size, src_dim_size, dst_dim_size, right_size, stream,
        );
    }
}

// ============================================================================
// Sort Operations (F32) — candle sort.cu via sort_launchers.cu
// ============================================================================

extern "C" {
    /// Argsort: sort each row of a [nrows, ncols] matrix.
    /// Output is uint32 indices. ascending=1 for ascending, 0 for descending.
    pub fn candle_launch_argsort_f32(
        input: *const f32,
        output: *mut u32,
        nrows: size_t,
        ncols: size_t,
        ascending: libc::c_int,
        stream: cudaStream_t,
    );
}

#[deprecated(note = "use ptx_kernels::safe_api::sort instead — these bypass the guard layer")]
/// Safe wrapper for sort operations on contiguous F32 tensors
pub mod sort_f32 {
    use super::*;

    #[inline]
    pub unsafe fn argsort(
        input: *const f32,
        output: *mut u32,
        nrows: usize,
        ncols: usize,
        ascending: bool,
        stream: cudaStream_t,
    ) {
        candle_launch_argsort_f32(
            input, output, nrows, ncols, if ascending { 1 } else { 0 }, stream,
        );
    }
}

// ============================================================================
// Where/Ternary Operations (F32) — candle ternary.cu via where_launchers.cu
// ============================================================================

extern "C" {
    /// Where: element-wise conditional: out[i] = cond[i] ? true_val[i] : false_val[i]
    pub fn candle_launch_where_f32(
        cond: *const u8,
        true_val: *const f32,
        false_val: *const f32,
        output: *mut f32,
        numel: size_t,
        stream: cudaStream_t,
    );
}

#[deprecated(note = "use ptx_kernels::safe_api::ternary instead — these bypass the guard layer")]
/// Safe wrapper for where/ternary operations on contiguous F32 tensors
pub mod where_f32 {
    use super::*;

    #[inline]
    pub unsafe fn where_cond(
        cond: *const u8,
        true_val: *const f32,
        false_val: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_where_f32(cond, true_val, false_val, output, numel, stream);
    }
}

#[deprecated(note = "use ptx_kernels::safe_api::binary instead — these bypass the guard layer")]
/// Safe wrapper for binary operations on contiguous F32 tensors
pub mod binary_f32 {
    use super::*;

    /// Element-wise addition (contiguous tensors)
    #[inline]
    pub unsafe fn add(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_badd_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise multiplication (contiguous tensors)
    #[inline]
    pub unsafe fn mul(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bmul_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise subtraction (contiguous tensors)
    #[inline]
    pub unsafe fn sub(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bsub_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise division (contiguous tensors)
    #[inline]
    pub unsafe fn div(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bdiv_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }
}

// ============================================================================
// Linker-force references
// ============================================================================
//
// The ptx_tensor_* aliases live in launcher .cu files that are compiled into
