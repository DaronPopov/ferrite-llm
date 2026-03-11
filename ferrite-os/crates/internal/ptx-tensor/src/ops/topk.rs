//! Top-K selection operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Select the top-k values (and their indices) along a dimension.
    ///
    /// Returns `(values, indices)` where both have the same shape as `self`
    /// except the `dim`-th dimension is replaced by `k`.
    /// `indices` is an I32 tensor with the original positions along `dim`.
    /// When `largest` is true, returns the k largest elements; otherwise the k smallest.
    pub fn topk(&self, k: usize, dim: i32, largest: bool) -> Result<(Tensor, Tensor)> {
        let input = self.require_contiguous()?;
        let ndim = input.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        let dim_size = input.shape()[dim];
        if k == 0 || k > dim_size {
            return Err(Error::Internal {
                message: format!("k={} out of range for dimension size {}", k, dim_size),
            });
        }

        // Compute outer / inner
        let outer: usize = input.shape()[..dim].iter().product();
        let inner: usize = input.shape()[dim + 1..].iter().product();

        // Output shape: same as input but dim-th dimension is k
        let mut out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        out_shape[dim] = k;

        let out_elems: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;

        let val_storage = Storage::new(out_elems, input.dtype(), input.runtime())?;
        let values = Tensor::from_storage(
            val_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let idx_storage = Storage::new(out_elems, DType::I32, input.runtime())?;
        let indices = Tensor::from_storage(
            idx_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let inp = input.data_ptr_typed::<f32>();
                let vals = values.data_ptr_typed::<f32>();
                let idxs = indices.data_ptr_typed::<i32>();
                ptx_sys::ptx_tensor_topk_f32(
                    inp, vals, idxs,
                    outer, dim_size, inner, k,
                    if largest { 1 } else { 0 },
                    stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("TopK not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok((values, indices))
    }
}
