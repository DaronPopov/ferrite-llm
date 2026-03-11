//! Index-select and scatter-add operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Index-select: select slices along a dimension using an index tensor.
    ///
    /// `indices` must be an I32 tensor of shape [..., ids_dim, ...].
    /// Output shape is input shape with dim replaced by ids_dim_size.
    ///
    /// For dim=d:
    ///   output[i0..id..in] = self[i0..indices[id]..in]
    pub fn index_select(&self, dim: i32, indices: &Tensor) -> Result<Tensor> {
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
                message: format!("IndexSelect indices must be I32, got {:?}", indices.dtype()),
            });
        }

        let input = self.require_contiguous()?;
        let idx_c = indices.require_contiguous()?;

        // indices is 1D: ids_dim_size = indices.elem_count()
        let ids_dim_size = idx_c.elem_count();

        // Compute (left, src_dim, right)
        let left: usize = input.shape()[..dim].iter().product();
        let src_dim_size = input.shape()[dim];
        let right: usize = input.shape()[dim + 1..].iter().product();

        // Output shape: input shape with dim replaced by ids_dim_size
        let mut out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        out_shape[dim] = ids_dim_size;

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
                let ids = idx_c.data_ptr_typed::<i32>();
                let out = output.data_ptr_typed::<f32>();
                ptx_sys::ptx_tensor_index_select_f32(
                    inp, ids, out, left, src_dim_size, ids_dim_size, right, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("IndexSelect not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }

    /// Scatter-add: accumulate src values into a new output tensor at positions given by ids.
    ///
    /// `self` is the source tensor, `indices` selects where to accumulate.
    /// `dst_dim_size` is the size of the output along the scatter dimension.
    /// Output is zero-initialized, then src values are atomically added at index positions.
    pub fn scatter_add(&self, dim: i32, indices: &Tensor, dst_dim_size: usize) -> Result<Tensor> {
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
                message: format!("ScatterAdd indices must be I32, got {:?}", indices.dtype()),
            });
        }

        let input = self.require_contiguous()?;
        let idx_c = indices.require_contiguous()?;

        let left: usize = input.shape()[..dim].iter().product();
        let src_dim_size = input.shape()[dim];
        let right: usize = input.shape()[dim + 1..].iter().product();

        // Output shape: input shape with dim replaced by dst_dim_size
        let mut out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        out_shape[dim] = dst_dim_size;

        let out_numel: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(out_numel, input.dtype(), input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // Zero-initialize output
        let stream = input.runtime().next_stream();
        unsafe {
            ptx_sys::cudaMemsetAsync(
                output.data_ptr() as *mut libc::c_void,
                0,
                out_numel * input.dtype().size_bytes(),
                stream.raw(),
            );
        }

        match input.dtype() {
            DType::F32 => unsafe {
                let ids = idx_c.data_ptr_typed::<i32>();
                let src = input.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();
                ptx_sys::ptx_tensor_scatter_add_f32(
                    ids, src, out, left, src_dim_size, dst_dim_size, right, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("ScatterAdd not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }
}
