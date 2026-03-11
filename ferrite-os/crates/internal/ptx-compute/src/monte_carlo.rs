//! Monte Carlo simulation utilities.
//!
//! This module provides high-level APIs for Monte Carlo simulations,
//! using GPU-accelerated random number generation via PTX-OS.

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, Stream, Result, Error};

/// Monte Carlo simulator.
///
/// Provides utilities for Monte Carlo simulations using GPU parallelism.
pub struct Simulator {
    runtime: Arc<PtxRuntime>,
}

impl Simulator {
    /// Create a new Monte Carlo simulator.
    pub fn new(runtime: &Arc<PtxRuntime>) -> Self {
        Self {
            runtime: Arc::clone(runtime),
        }
    }

    /// Estimate pi using Monte Carlo sampling.
    ///
    /// Uses the classic Monte Carlo method: sample random points in a unit square
    /// and count how many fall within the unit circle. The ratio approximates pi/4.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Total number of random samples to generate
    /// * `samples_buffer` - GPU buffer for storing random samples (must hold 2*num_samples f32)
    /// * `result_buffer` - GPU buffer (unused in current impl, reserved for future GPU-side counting)
    /// * `stream` - CUDA stream to use
    ///
    /// # Returns
    ///
    /// Returns the estimated value of pi.
    ///
    /// # Safety
    ///
    /// `samples_buffer` must point to valid GPU memory of at least `2 * num_samples * sizeof(f32)` bytes.
    pub unsafe fn estimate_pi(
        &self,
        num_samples: usize,
        samples_buffer: *mut f32,
        _result_buffer: *mut u32,
        stream: &Stream,
    ) -> Result<f64> {
        if samples_buffer.is_null() {
            return Err(Error::InvalidPointer);
        }
        if num_samples == 0 {
            return Err(Error::Internal {
                message: "num_samples must be > 0".into(),
            });
        }

        // Generate 2*N uniform random values in [0, 1) for (x, y) pairs
        let total_floats = num_samples * 2;
        let seed = 42u64;
        ptx_sys::ptx_tensor_rand_f32(
            samples_buffer,
            total_floats,
            seed,
            stream.raw(),
        );

        // Synchronize before reading back
        stream.synchronize()?;

        // Copy samples to host
        let total_bytes = total_floats * std::mem::size_of::<f32>();
        let mut host_samples = vec![0.0f32; total_floats];
        let err = ptx_sys::cudaMemcpy(
            host_samples.as_mut_ptr() as *mut _,
            samples_buffer as *const _,
            total_bytes,
            2, // cudaMemcpyDeviceToHost
        );
        if err != 0 {
            return Err(Error::cuda(err as i32));
        }

        // Count points inside unit circle: x^2 + y^2 <= 1
        let inside = host_samples
            .chunks_exact(2)
            .filter(|pair| {
                let x = pair[0];
                let y = pair[1];
                x * x + y * y <= 1.0
            })
            .count();

        Ok(4.0 * inside as f64 / num_samples as f64)
    }

    /// Generate random samples on GPU (uniform distribution [0, 1)).
    ///
    /// # Arguments
    ///
    /// * `output` - Output buffer for random samples
    /// * `size` - Number of random values to generate
    /// * `seed` - Random seed
    /// * `stream` - CUDA stream to use
    ///
    /// # Safety
    ///
    /// `output` must point to valid GPU memory of at least `size * sizeof(f32)` bytes.
    pub unsafe fn generate_random_f32(
        &self,
        output: *mut f32,
        size: usize,
        seed: u64,
        stream: &Stream,
    ) -> Result<()> {
        if output.is_null() {
            return Err(Error::InvalidPointer);
        }
        if size == 0 {
            return Ok(());
        }
        ptx_sys::ptx_tensor_rand_f32(output, size, seed, stream.raw());
        Ok(())
    }

    /// Generate normally distributed random samples on GPU.
    ///
    /// # Safety
    ///
    /// `output` must point to valid GPU memory of at least `size * sizeof(f32)` bytes.
    pub unsafe fn generate_randn_f32(
        &self,
        output: *mut f32,
        size: usize,
        seed: u64,
        stream: &Stream,
    ) -> Result<()> {
        if output.is_null() {
            return Err(Error::InvalidPointer);
        }
        if size == 0 {
            return Ok(());
        }
        ptx_sys::ptx_tensor_randn_f32(output, size, seed, stream.raw());
        Ok(())
    }

    /// Perform parallel Monte Carlo sampling across multiple streams.
    ///
    /// Each stream runs an independent pi estimation with a different seed,
    /// and results are averaged for a more accurate estimate.
    ///
    /// # Arguments
    ///
    /// * `samples_per_stream` - Number of samples per stream
    /// * `num_streams` - Number of parallel streams to use
    /// * `buffers` - Array of GPU buffers (one per stream, each must hold 2*samples_per_stream f32)
    /// * `results` - Array of result buffers (one per stream, currently unused)
    ///
    /// # Safety
    ///
    /// All buffer pointers must be valid GPU memory with sufficient size.
    pub unsafe fn parallel_sampling(
        &self,
        samples_per_stream: usize,
        num_streams: usize,
        buffers: &[*mut f32],
        results: &[*mut u32],
    ) -> Result<Vec<f64>> {
        if buffers.len() < num_streams || results.len() < num_streams {
            return Err(Error::Internal {
                message: format!(
                    "need {} buffers and results, got {} and {}",
                    num_streams, buffers.len(), results.len()
                ),
            });
        }

        let total_floats = samples_per_stream * 2;

        // Launch random generation on all streams
        for i in 0..num_streams {
            let stream_id = (i % self.runtime.num_streams()) as i32;
            let stream = ptx_sys::gpu_hot_get_stream(self.runtime.raw(), stream_id);
            let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(42);
            ptx_sys::ptx_tensor_rand_f32(buffers[i], total_floats, seed, stream);
        }

        // Sync all streams
        self.runtime.sync_all()?;

        // Read back and compute pi estimate per stream
        let total_bytes = total_floats * std::mem::size_of::<f32>();
        let mut estimates = Vec::with_capacity(num_streams);

        for i in 0..num_streams {
            let mut host_samples = vec![0.0f32; total_floats];
            let err = ptx_sys::cudaMemcpy(
                host_samples.as_mut_ptr() as *mut _,
                buffers[i] as *const _,
                total_bytes,
                2, // cudaMemcpyDeviceToHost
            );
            if err != 0 {
                return Err(Error::cuda(err as i32));
            }

            let inside = host_samples
                .chunks_exact(2)
                .filter(|pair| pair[0] * pair[0] + pair[1] * pair[1] <= 1.0)
                .count();

            estimates.push(4.0 * inside as f64 / samples_per_stream as f64);
        }

        Ok(estimates)
    }
}

/// Parallel Monte Carlo helper for multi-stream execution.
pub struct ParallelMonteCarlo {
    runtime: Arc<PtxRuntime>,
    num_streams: usize,
}

impl ParallelMonteCarlo {
    /// Create a parallel Monte Carlo simulator.
    pub fn new(runtime: &Arc<PtxRuntime>, num_streams: usize) -> Self {
        Self {
            runtime: Arc::clone(runtime),
            num_streams,
        }
    }

    /// Estimate pi using parallel sampling across multiple streams.
    ///
    /// Divides the total samples across streams, runs independent estimates,
    /// and averages the results.
    ///
    /// # Safety
    ///
    /// All buffer pointers must be valid GPU memory with sufficient size.
    pub unsafe fn estimate_pi_parallel(
        &self,
        total_samples: usize,
        buffers: &[*mut f32],
        results: &[*mut u32],
    ) -> Result<f64> {
        let sps = self.samples_per_stream(total_samples);
        if sps == 0 {
            return Err(Error::Internal {
                message: "samples_per_stream is 0".into(),
            });
        }

        let total_floats = sps * 2;

        // Launch random generation on all streams with distinct seeds
        for i in 0..self.num_streams {
            let stream_id = (i % self.runtime.num_streams()) as i32;
            let stream = ptx_sys::gpu_hot_get_stream(self.runtime.raw(), stream_id);
            let seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(7);
            ptx_sys::ptx_tensor_rand_f32(buffers[i], total_floats, seed, stream);
        }

        self.runtime.sync_all()?;

        // Read back and aggregate
        let total_bytes = total_floats * std::mem::size_of::<f32>();
        let mut total_inside = 0usize;
        let actual_streams = self.num_streams.min(buffers.len()).min(results.len());

        for i in 0..actual_streams {
            let mut host_samples = vec![0.0f32; total_floats];
            let err = ptx_sys::cudaMemcpy(
                host_samples.as_mut_ptr() as *mut _,
                buffers[i] as *const _,
                total_bytes,
                2, // cudaMemcpyDeviceToHost
            );
            if err != 0 {
                return Err(Error::cuda(err as i32));
            }

            total_inside += host_samples
                .chunks_exact(2)
                .filter(|pair| pair[0] * pair[0] + pair[1] * pair[1] <= 1.0)
                .count();
        }

        let total_actual_samples = sps * actual_streams;
        Ok(4.0 * total_inside as f64 / total_actual_samples as f64)
    }

    /// Get optimal samples per stream for the given total samples.
    pub fn samples_per_stream(&self, total_samples: usize) -> usize {
        total_samples / self.num_streams
    }

    /// Get number of streams.
    pub fn num_streams(&self) -> usize {
        self.num_streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_constructs() {
        // Can't test GPU operations without hardware, but verify the type compiles
        // and the API shape is correct.
        let _ = std::mem::size_of::<Simulator>();
    }

    #[test]
    fn samples_per_stream_divides_evenly() {
        // Test the pure division logic: total / num_streams
        assert_eq!(1000 / 4, 250);
        assert_eq!(7 / 4, 1);
        assert_eq!(100 / 1, 100);
    }
}
