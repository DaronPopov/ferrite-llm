//! Core tensor type.

use std::fmt;
use std::sync::Arc;

use ptx_runtime::{Error, PtxRuntime, Result};

use crate::dtype::DType;
use crate::shape::{
    checked_elem_count, contiguous_strides, elem_count, is_contiguous, size_bytes, Shape, Strides,
};
use crate::storage::Storage;

mod construct;
mod metadata;
mod views_exec;

/// A multi-dimensional array stored on the GPU.
///
/// Tensors support:
/// - Element-wise operations (add, mul, etc.)
/// - Unary operations (exp, log, relu, etc.)
/// - Reductions (sum, mean, max, etc.)
/// - Matrix operations (matmul via cuBLAS)
/// - Automatic memory management
///
/// # Example
///
/// ```no_run
/// use ptx_tensor::{Tensor, DType};
/// use ptx_runtime::PtxRuntime;
/// use std::sync::Arc;
///
/// let runtime = Arc::new(PtxRuntime::new(0).unwrap());
///
/// // Create tensors
/// let a = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
/// let b = Tensor::ones(&[2, 3], DType::F32, &runtime).unwrap();
///
/// // Operations
/// let c = a.add(&b).unwrap();
/// let d = c.relu().unwrap();
/// let e = d.sum(1).unwrap();  // Sum along dimension 1
/// ```
#[derive(Clone)]
pub struct Tensor {
    /// Underlying storage (may be shared with other tensors).
    storage: Storage,
    /// Shape of the tensor.
    shape: Shape,
    /// Strides for each dimension (in elements).
    strides: Strides,
    /// Offset into storage (in elements).
    offset: usize,
    /// Data type.
    dtype: DType,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, contiguous={})",
            self.shape.as_slice(),
            self.dtype,
            self.is_contiguous()
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a CUDA-capable GPU
    #[test]
    #[ignore]
    fn test_tensor_creation() {
        let runtime = Arc::new(PtxRuntime::new(0).unwrap());
        let tensor = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.elem_count(), 6);
        assert!(tensor.is_contiguous());
    }

    #[test]
    #[ignore]
    fn test_tensor_reshape() {
        let runtime = Arc::new(PtxRuntime::new(0).unwrap());
        let tensor = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
        let reshaped = tensor.reshape(&[6]).unwrap();
        assert_eq!(reshaped.shape(), &[6]);
        assert_eq!(reshaped.elem_count(), 6);
    }
}
