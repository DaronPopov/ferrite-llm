//! Tensor builder — enables `ctx.tensor(&[1024, 1024], DType::F32)?.randn()?`.

use std::sync::Arc;

use ptx_runtime::PtxRuntime;
use ptx_tensor::{DType, Tensor};

use crate::error::AppError;

/// A staged tensor builder that holds shape, dtype, and runtime reference.
///
/// Created via [`Ctx::tensor`](crate::ctx::Ctx::tensor), then finalized
/// with one of the fill methods.
pub struct TensorBuilder {
    shape: Vec<usize>,
    dtype: DType,
    runtime: Arc<PtxRuntime>,
}

impl TensorBuilder {
    pub(crate) fn new(shape: Vec<usize>, dtype: DType, runtime: Arc<PtxRuntime>) -> Self {
        Self {
            shape,
            dtype,
            runtime,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(self) -> Result<Tensor, AppError> {
        Tensor::zeros(&self.shape, self.dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor filled with ones.
    pub fn ones(self) -> Result<Tensor, AppError> {
        Tensor::ones(&self.shape, self.dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor filled with standard-normal random values (mean=0, std=1).
    pub fn randn(self) -> Result<Tensor, AppError> {
        Tensor::randn(&self.shape, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor filled with a constant value.
    pub fn fill(self, val: f32) -> Result<Tensor, AppError> {
        Tensor::full(&self.shape, val, self.dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor from host data.
    pub fn from_slice<T: bytemuck::Pod>(self, data: &[T]) -> Result<Tensor, AppError> {
        Tensor::from_slice(data, &self.shape, self.dtype, &self.runtime).map_err(AppError::from)
    }
}
