//! Comparison operations (return U8 boolean tensors).

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

/// Comparison operation type.
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Tensor {
    /// Generic comparison operation (returns U8 tensor: 1 = true, 0 = false).
    fn comparison_op(&self, other: &Tensor, op: ComparisonOp) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype().to_ptx(),
                actual: other.dtype().to_ptx(),
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        let out_shape = Shape::from_slice(input.shape());
        let out_storage = Storage::new(
            input.elem_count(),
            DType::U8,
            input.runtime(),
        )?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let a = input.data_ptr_typed::<f32>();
                let b = rhs.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<u8>();

                match op {
                    ComparisonOp::Eq => ptx_sys::ptx_tensor_cmp_eq_f32(a, b, out, n, stream.raw()),
                    ComparisonOp::Ne => ptx_sys::ptx_tensor_cmp_ne_f32(a, b, out, n, stream.raw()),
                    ComparisonOp::Lt => ptx_sys::ptx_tensor_cmp_lt_f32(a, b, out, n, stream.raw()),
                    ComparisonOp::Le => ptx_sys::ptx_tensor_cmp_le_f32(a, b, out, n, stream.raw()),
                    ComparisonOp::Gt => ptx_sys::ptx_tensor_cmp_gt_f32(a, b, out, n, stream.raw()),
                    ComparisonOp::Ge => ptx_sys::ptx_tensor_cmp_ge_f32(a, b, out, n, stream.raw()),
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Comparison not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Element-wise equality (returns U8 tensor).
    pub fn eq(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Eq)
    }

    /// Element-wise not-equal (returns U8 tensor).
    pub fn ne(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Ne)
    }

    /// Element-wise less-than (returns U8 tensor).
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Lt)
    }

    /// Element-wise less-than-or-equal (returns U8 tensor).
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Le)
    }

    /// Element-wise greater-than (returns U8 tensor).
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Gt)
    }

    /// Element-wise greater-than-or-equal (returns U8 tensor).
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        self.comparison_op(other, ComparisonOp::Ge)
    }
}
