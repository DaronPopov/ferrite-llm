//! Argsort operation.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Argsort — returns a U32 tensor of indices that sort each row.
    ///
    /// The input is viewed as [nrows, ncols] where the sort runs along
    /// the last dimension. For higher-rank tensors, dimensions before `dim`
    /// are flattened into nrows.
    pub fn argsort(&self, dim: i32, ascending: bool) -> Result<Tensor> {
        let input = self.require_contiguous()?;
        let ndim = input.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        // Argsort operates along one dimension: reshape to [nrows, ncols]
        let nrows: usize = input.shape()[..dim].iter().product();
        let ncols = input.shape()[dim];
        let trailing: usize = input.shape()[dim + 1..].iter().product();
        if trailing != 1 {
            return Err(Error::NotSupported {
                message: "Argsort currently only supports the last dimension".to_string(),
            });
        }

        // Output has same shape but dtype U32
        let out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        let out_elems = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(
            out_elems,
            DType::U32,
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
                let out = output.data_ptr_typed::<u32>();
                ptx_sys::ptx_tensor_argsort_f32(
                    inp, out, nrows, ncols,
                    if ascending { 1 } else { 0 },
                    stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Argsort not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }
}
