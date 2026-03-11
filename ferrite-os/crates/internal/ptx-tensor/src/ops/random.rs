//! Random tensor generation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{PtxRuntime, Result, Error};

/// Global seed counter for unique seeds across calls.
static SEED_COUNTER: AtomicU64 = AtomicU64::new(42);

fn next_seed() -> u64 {
    SEED_COUNTER.fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

impl Tensor {
    /// Create a tensor filled with uniform random values in [0, 1).
    pub fn rand(shape: &[usize], runtime: &Arc<PtxRuntime>) -> Result<Tensor> {
        let out_shape = Shape::from_slice(shape);
        let n: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(n, DType::F32, runtime)?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let stream = runtime.next_stream();
        let seed = next_seed();

        unsafe {
            ptx_sys::ptx_tensor_rand_f32(
                output.data_ptr_typed::<f32>(),
                n, seed, stream.raw(),
            );
        }

        stream.synchronize()?;
        Ok(output)
    }

    /// Create a tensor filled with standard normal random values (mean=0, std=1).
    pub fn randn(shape: &[usize], runtime: &Arc<PtxRuntime>) -> Result<Tensor> {
        let out_shape = Shape::from_slice(shape);
        let n: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(n, DType::F32, runtime)?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let stream = runtime.next_stream();
        let seed = next_seed();

        unsafe {
            ptx_sys::ptx_tensor_randn_f32(
                output.data_ptr_typed::<f32>(),
                n, seed, stream.raw(),
            );
        }

        stream.synchronize()?;
        Ok(output)
    }

    /// Create a tensor like this one filled with uniform random values in [0, 1).
    pub fn rand_like(&self) -> Result<Tensor> {
        Tensor::rand(self.shape(), self.runtime())
    }

    /// Create a tensor like this one filled with standard normal random values.
    pub fn randn_like(&self) -> Result<Tensor> {
        Tensor::randn(self.shape(), self.runtime())
    }
}
