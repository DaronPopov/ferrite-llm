//! Parallel reduction operations.
//!
//! This module provides high-level APIs for parallel reduction operations
//! like sum, mean, max, and min across large arrays.

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, Stream, Result};

/// Parallel reduction helper.
///
/// Provides methods for common reduction operations (sum, mean, max, min)
/// with support for multi-dimensional arrays and multi-stream parallelism.
pub struct Reducer {
    #[allow(dead_code)]
    runtime: Arc<PtxRuntime>,
}

impl Reducer {
    /// Create a new Reducer.
    pub fn new(runtime: &Arc<PtxRuntime>) -> Self {
        Self {
            runtime: Arc::clone(runtime),
        }
    }

    /// Sum reduction: compute sum of all elements.
    ///
    /// # Arguments
    ///
    /// * `input` - Input array pointer
    /// * `output` - Output scalar pointer (will contain the sum)
    /// * `size` - Number of elements
    /// * `stream` - CUDA stream to use
    ///
    /// # Safety
    ///
    /// Pointers must be valid GPU memory addresses.
    pub unsafe fn sum_f32(
        &self,
        input: *mut f32,
        output: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_reduce_sum_f32(
            input,
            output,
            1,
            size,
            1,
            stream.raw(),
        );
        Ok(())
    }

    /// Mean reduction: compute mean of all elements.
    pub unsafe fn mean_f32(
        &self,
        input: *mut f32,
        output: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_reduce_mean_f32(
            input,
            output,
            1,
            size,
            1,
            stream.raw(),
        );
        Ok(())
    }

    /// Max reduction: find maximum element.
    pub unsafe fn max_f32(
        &self,
        input: *mut f32,
        output: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_reduce_max_f32(
            input,
            output,
            1,
            size,
            1,
            stream.raw(),
        );
        Ok(())
    }

    /// Min reduction: find minimum element.
    pub unsafe fn min_f32(
        &self,
        input: *mut f32,
        output: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_reduce_min_f32(
            input,
            output,
            1,
            size,
            1,
            stream.raw(),
        );
        Ok(())
    }

    /// Perform all reduction operations in sequence.
    ///
    /// Computes sum, mean, max, and min in a single call.
    pub unsafe fn all_reductions_f32(
        &self,
        input: *mut f32,
        sum_out: *mut f32,
        mean_out: *mut f32,
        max_out: *mut f32,
        min_out: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.sum_f32(input, sum_out, size, stream)?;
        self.mean_f32(input, mean_out, size, stream)?;
        self.max_f32(input, max_out, size, stream)?;
        self.min_f32(input, min_out, size, stream)?;
        Ok(())
    }
}

/// Multi-stream parallel reduction.
///
/// Distributes reduction operations across multiple streams for
/// maximum throughput when processing multiple arrays.
pub struct ParallelReducer {
    runtime: Arc<PtxRuntime>,
    num_streams: usize,
}

impl ParallelReducer {
    /// Create a parallel reducer with specified number of streams.
    pub fn new(runtime: &Arc<PtxRuntime>, num_streams: usize) -> Self {
        Self {
            runtime: Arc::clone(runtime),
            num_streams,
        }
    }

    /// Perform sum reductions in parallel across multiple arrays.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Array of input pointers
    /// * `outputs` - Array of output pointers
    /// * `sizes` - Array of sizes for each input
    pub unsafe fn parallel_sum_f32(
        &self,
        inputs: &[*mut f32],
        outputs: &[*mut f32],
        sizes: &[usize],
    ) -> Result<()> {
        assert_eq!(inputs.len(), outputs.len());
        assert_eq!(inputs.len(), sizes.len());

        for (i, ((input, output), size)) in inputs.iter()
            .zip(outputs.iter())
            .zip(sizes.iter())
            .enumerate()
        {
            let stream_id = i % self.num_streams;
            let stream = self.runtime.stream(stream_id as i32)?;

            ptx_sys::ptx_tensor_reduce_sum_f32(
                *input,
                *output,
                1,
                *size,
                1,
                stream.raw(),
            );
        }

        self.runtime.sync_all()?;
        Ok(())
    }

    /// Perform mean reductions in parallel.
    pub unsafe fn parallel_mean_f32(
        &self,
        inputs: &[*mut f32],
        outputs: &[*mut f32],
        sizes: &[usize],
    ) -> Result<()> {
        assert_eq!(inputs.len(), outputs.len());
        assert_eq!(inputs.len(), sizes.len());

        for (i, ((input, output), size)) in inputs.iter()
            .zip(outputs.iter())
            .zip(sizes.iter())
            .enumerate()
        {
            let stream_id = i % self.num_streams;
            let stream = self.runtime.stream(stream_id as i32)?;

            ptx_sys::ptx_tensor_reduce_mean_f32(
                *input,
                *output,
                1,
                *size,
                1,
                stream.raw(),
            );
        }

        self.runtime.sync_all()?;
        Ok(())
    }

    /// Perform all reduction types in parallel across multiple arrays.
    pub unsafe fn parallel_all_reductions_f32(
        &self,
        inputs: &[*mut f32],
        sum_outputs: &[*mut f32],
        mean_outputs: &[*mut f32],
        max_outputs: &[*mut f32],
        min_outputs: &[*mut f32],
        sizes: &[usize],
    ) -> Result<()> {
        let n = inputs.len();
        assert_eq!(sum_outputs.len(), n);
        assert_eq!(mean_outputs.len(), n);
        assert_eq!(max_outputs.len(), n);
        assert_eq!(min_outputs.len(), n);
        assert_eq!(sizes.len(), n);

        for (i, input) in inputs.iter().enumerate() {
            let stream_id = i % self.num_streams;
            let stream = self.runtime.stream(stream_id as i32)?;
            let size = sizes[i];

            // Queue all 4 reduction operations on the same stream
            ptx_sys::ptx_tensor_reduce_sum_f32(
                *input,
                sum_outputs[i],
                1,
                size,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_mean_f32(
                *input,
                mean_outputs[i],
                1,
                size,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_max_f32(
                *input,
                max_outputs[i],
                1,
                size,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_min_f32(
                *input,
                min_outputs[i],
                1,
                size,
                1,
                stream.raw(),
            );
        }

        self.runtime.sync_all()?;
        Ok(())
    }
}
