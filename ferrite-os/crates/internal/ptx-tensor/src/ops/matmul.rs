//! Matrix multiplication operations via cuBLAS.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{Shape, contiguous_strides};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops, cublas::Gemm};

impl Tensor {
    /// Matrix multiplication: C = A @ B
    ///
    /// For 2D tensors:
    /// - A: (M, K)
    /// - B: (K, N)
    /// - C: (M, N)
    ///
    /// For batched 3D tensors:
    /// - A: (batch, M, K)
    /// - B: (batch, K, N)
    /// - C: (batch, M, N)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Check dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype().to_ptx(),
                actual: other.dtype().to_ptx(),
            });
        }

        // Check dimensions
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d(other),
            (3, 3) => self.bmm(other),
            _ => Err(Error::NotSupported {
                message: format!(
                    "matmul not supported for shapes {:?} @ {:?}",
                    self.shape(),
                    other.shape()
                ),
            }),
        }
    }

    /// 2D matrix multiplication.
    fn matmul_2d(&self, other: &Tensor) -> Result<Tensor> {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = other.shape()[0];
        let n = other.shape()[1];

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        // Create output tensor
        let out_shape = Shape::from_slice(&[m, n]);
        let out_storage = Storage::new(m * n, input.dtype(), input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // Get or create cuBLAS handle
        let gemm = Gemm::new()?;
        let stream = input.runtime().next_stream();
        gemm.set_stream(&stream)?;

        match input.dtype() {
            DType::F32 => unsafe {
                gemm.matmul_f32(
                    input.data_ptr_typed::<f32>(),
                    rhs.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    m,
                    n,
                    k,
                )?;
            },
            dtype => return Err(Error::NotSupported {
                message: format!("matmul not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Batched matrix multiplication: C[i] = A[i] @ B[i]
    ///
    /// - A: (batch, M, K)
    /// - B: (batch, K, N)
    /// - C: (batch, M, N)
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(Error::Internal {
                message: "bmm requires 3D tensors".to_string(),
            });
        }

        let batch = self.shape()[0];
        let m = self.shape()[1];
        let k = self.shape()[2];
        let batch2 = other.shape()[0];
        let k2 = other.shape()[1];
        let n = other.shape()[2];

        if batch != batch2 {
            return Err(Error::ShapeMismatch {
                expected: vec![batch],
                actual: vec![batch2],
            });
        }

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        // Create output tensor
        let out_shape = Shape::from_slice(&[batch, m, n]);
        let out_storage = Storage::new(batch * m * n, input.dtype(), input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // Get or create cuBLAS handle
        let gemm = Gemm::new()?;
        let stream = input.runtime().next_stream();
        gemm.set_stream(&stream)?;

        match input.dtype() {
            DType::F32 => unsafe {
                gemm.bmm_f32(
                    input.data_ptr_typed::<f32>(),
                    rhs.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    batch,
                    m,
                    n,
                    k,
                )?;
            },
            dtype => return Err(Error::NotSupported {
                message: format!("bmm not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    // transpose() is now in tensor.rs with proper (dim0, dim1) signature.
    // Use .t() for 2D matrix transpose, or .transpose(dim0, dim1) for general case.

    /// Linear layer: out = x @ weight.T + bias
    ///
    /// - x: (*, in_features)
    /// - weight: (out_features, in_features)
    /// - bias: (out_features,) or None
    /// - Returns: (*, out_features)
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let wt = weight.t()?.contiguous()?;
        let out = self.matmul(&wt)?;
        match bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

/// Convenience function for matrix multiply.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.matmul(b)
}

/// Convenience function for batched matrix multiply.
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.bmm(b)
}
