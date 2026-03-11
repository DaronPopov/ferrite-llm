//! Tensor transformation operations: cat, pad, repeat, masked_fill.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Concatenate tensors along a dimension.
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(Error::Internal {
                message: "cat requires at least one tensor".to_string(),
            });
        }

        let first = tensors[0];
        let ndim = first.ndim();
        if dim >= ndim {
            return Err(Error::Internal {
                message: format!("cat dim {} out of range for {}D tensor", dim, ndim),
            });
        }

        // All tensors must have same shape except along cat dim
        for t in &tensors[1..] {
            if t.ndim() != ndim {
                return Err(Error::ShapeMismatch {
                    expected: first.shape().to_vec(),
                    actual: t.shape().to_vec(),
                });
            }
            for d in 0..ndim {
                if d != dim && t.shape()[d] != first.shape()[d] {
                    return Err(Error::ShapeMismatch {
                        expected: first.shape().to_vec(),
                        actual: t.shape().to_vec(),
                    });
                }
            }
            if t.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    expected: first.dtype().to_ptx(),
                    actual: t.dtype().to_ptx(),
                });
            }
        }

        // Compute output shape
        let cat_size: usize = tensors.iter().map(|t| t.shape()[dim]).sum();
        let mut out_shape = Shape::from_slice(first.shape());
        out_shape[dim] = cat_size;

        let out_elems = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(
            out_elems,
            first.dtype(),
            first.runtime(),
        )?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = first.runtime().next_stream();

        // Copy each tensor's data into the output using cudaMemcpyAsync
        // For dim=0 on contiguous tensors, this is a simple sequence of copies
        let outer: usize = first.shape()[..dim].iter().product();
        let inner: usize = first.shape()[dim + 1..].iter().product();
        let elem_size = first.dtype().size_bytes();

        let mut offset = 0usize;
        for t in tensors {
            let t_dim_size = t.shape()[dim];
            let chunk = t_dim_size * inner * elem_size;

            if outer == 1 {
                // Simple contiguous copy
                unsafe {
                    ptx_sys::cudaMemcpyAsync(
                        (output.data_ptr() as *mut u8).add(offset * inner * elem_size) as *mut libc::c_void,
                        t.data_ptr() as *const libc::c_void,
                        t.elem_count() * elem_size,
                        3, // cudaMemcpyDeviceToDevice
                        stream.raw(),
                    );
                }
            } else {
                // Need to interleave: for each outer slice, copy the chunk
                let out_dim_stride = cat_size * inner * elem_size;
                let in_dim_stride = t_dim_size * inner * elem_size;
                for o in 0..outer {
                    unsafe {
                        ptx_sys::cudaMemcpyAsync(
                            (output.data_ptr() as *mut u8).add(o * out_dim_stride + offset * inner * elem_size) as *mut libc::c_void,
                            (t.data_ptr() as *mut u8).add(o * in_dim_stride) as *const libc::c_void,
                            chunk,
                            3,
                            stream.raw(),
                        );
                    }
                }
            }
            offset += t_dim_size;
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }

    /// Stack tensors along a new dimension.
    pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(Error::Internal {
                message: "stack requires at least one tensor".to_string(),
            });
        }

        // Unsqueeze each tensor at dim, then cat
        let unsqueezed: Vec<Tensor> = tensors.iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        Tensor::cat(&refs, dim)
    }

    /// Repeat tensor along dimensions.
    /// `repeats` specifies how many times to repeat along each dim.
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "repeat only supported for F32".to_string(),
            });
        }
        if repeats.len() != self.ndim() {
            return Err(Error::Internal {
                message: format!("repeat requires {} dims, got {}", self.ndim(), repeats.len()),
            });
        }

        let mut out_shape = Shape::with_capacity(self.ndim());
        for i in 0..self.ndim() {
            out_shape.push(self.shape()[i] * repeats[i]);
        }

        let out_n: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(out_n, DType::F32, self.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = self.runtime().next_stream();
        let ndim = self.ndim() as i32;

        // Copy shape arrays to GPU
        let meta_size = self.ndim() * std::mem::size_of::<usize>();
        let alloc = self.runtime().alloc(meta_size * 2)?;
        let meta_ptr = alloc.as_ptr() as *mut usize;

        unsafe {
            let in_shape_ptr = meta_ptr;
            let out_shape_ptr = meta_ptr.add(self.ndim());
            let mut in_shape_vec: Vec<usize> = self.shape().to_vec();
            let mut out_shape_vec: Vec<usize> = out_shape.to_vec();

            ptx_sys::cudaMemcpyAsync(
                in_shape_ptr as *mut libc::c_void,
                in_shape_vec.as_mut_ptr() as *const libc::c_void,
                meta_size, 1, stream.raw(),
            );
            ptx_sys::cudaMemcpyAsync(
                out_shape_ptr as *mut libc::c_void,
                out_shape_vec.as_mut_ptr() as *const libc::c_void,
                meta_size, 1, stream.raw(),
            );

            ptx_sys::ptx_tensor_repeat_f32(
                self.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                out_n,
                in_shape_ptr as *mut usize,
                out_shape_ptr as *mut usize,
                ndim,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Masked fill: set elements to `value` where `mask` is true (non-zero).
    /// Modifies the tensor in-place and returns a clone.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "masked_fill only supported for F32".to_string(),
            });
        }
        if mask.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: "masked_fill mask must be U8".to_string(),
            });
        }
        if self.shape() != mask.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: mask.shape().to_vec(),
            });
        }

        // Clone tensor first, then fill in-place
        let output = self.clone_tensor()?;
        let n = self.elem_count();
        let stream = self.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_masked_fill_f32(
                output.data_ptr_typed::<f32>(),
                mask.data_ptr_typed::<u8>(),
                value,
                n,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }
}
