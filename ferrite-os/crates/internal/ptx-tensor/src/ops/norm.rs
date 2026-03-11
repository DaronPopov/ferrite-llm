//! Norm operations (composed from existing ops).

use crate::tensor::Tensor;
use ptx_runtime::Result;

impl Tensor {
    /// L1 norm along a dimension: sum(|x|).
    pub fn norm_l1(&self, dim: i32) -> Result<Tensor> {
        self.abs()?.sum(dim)
    }

    /// L2 norm along a dimension: sqrt(sum(x^2)).
    pub fn norm_l2(&self, dim: i32) -> Result<Tensor> {
        self.sqr()?.sum(dim)?.sqrt()
    }

    /// L1 norm of all elements.
    pub fn norm_l1_all(&self) -> Result<Tensor> {
        self.abs()?.sum_all()
    }

    /// L2 norm of all elements (Frobenius norm for matrices).
    pub fn norm_l2_all(&self) -> Result<Tensor> {
        self.sqr()?.sum_all()?.sqrt()
    }

    /// Normalize along a dimension (L2 normalization: x / ||x||_2).
    pub fn normalize(&self, dim: i32) -> Result<Tensor> {
        let norm = self.sqr()?.sum_keepdim(dim)?.sqrt()?.add_scalar(1e-8)?;
        self.broadcast_div(&norm)
    }
}
