//! Gather/scatter operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Gather elements along a dimension using an index tensor.
    ///
    /// `indices` must be an I32 tensor. The output has the same shape as `indices`.
    /// For dim=d:
    ///   output[i0..id..in] = self[i0..indices[i0..id..in]..in]
    pub fn gather(&self, dim: i32, indices: &Tensor) -> Result<Tensor> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        if indices.dtype() != DType::I32 {
            return Err(Error::NotSupported {
                message: format!("Gather indices must be I32, got {:?}", indices.dtype()),
            });
        }

        let input = self.require_contiguous()?;
        let idx_c = indices.require_contiguous()?;

        // Output shape = indices shape
        let out_shape: smallvec::SmallVec<[usize; 8]> = idx_c.shape().iter().copied().collect();

        // Compute outer / input_dim_size / idx_dim_size / inner
        let outer: usize = input.shape()[..dim].iter().product();
        let input_dim_size = input.shape()[dim];
        let idx_dim_size = idx_c.shape()[dim];
        let inner: usize = input.shape()[dim + 1..].iter().product();

        let out_elems = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(
            out_elems,
            input.dtype(),
            input.runtime(),
        )?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let inp = input.data_ptr_typed::<f32>();
                let idx = idx_c.data_ptr_typed::<i32>();
                let out = output.data_ptr_typed::<f32>();
                ptx_sys::ptx_tensor_gather_f32(
                    inp, idx, out, outer, input_dim_size, idx_dim_size, inner, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Gather not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }
}
