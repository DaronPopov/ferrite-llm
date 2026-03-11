//! Where (conditional select) operation.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::contiguous_strides;
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Where — element-wise conditional: out[i] = cond[i] ? self[i] : other[i]
    ///
    /// `cond` must be a U8 tensor (non-zero = true). `self` and `other` must
    /// have the same shape and dtype. Output has the same shape and dtype.
    pub fn where_cond(&self, cond: &Tensor, other: &Tensor) -> Result<Tensor> {
        if cond.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: format!("Where condition must be U8, got {:?}", cond.dtype()),
            });
        }
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        if cond.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: cond.shape().to_vec(),
            });
        }

        let input = self.require_contiguous()?;
        let cond_c = cond.require_contiguous()?;
        let other_c = other.require_contiguous()?;

        let out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        let numel = input.elem_count();

        let out_storage = Storage::new(numel, input.dtype(), input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let c = cond_c.data_ptr_typed::<u8>();
                let t = input.data_ptr_typed::<f32>();
                let f = other_c.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();
                ptx_sys::ptx_tensor_where_f32(c, t, f, out, numel, stream.raw());
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Where not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }
}
