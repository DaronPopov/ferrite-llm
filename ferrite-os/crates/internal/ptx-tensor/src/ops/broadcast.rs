//! Broadcasting support for binary operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{broadcast_shapes, contiguous_strides, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

/// Compute broadcast strides: for dims where original size is 1, stride becomes 0.
fn broadcast_strides(original_shape: &[usize], broadcast_shape: &[usize]) -> Vec<usize> {
    let ndim = broadcast_shape.len();
    let offset = ndim - original_shape.len();
    let mut strides = vec![0usize; ndim];

    // Compute contiguous strides for original shape
    let orig_strides = contiguous_strides(original_shape);

    for i in 0..original_shape.len() {
        if original_shape[i] == broadcast_shape[i + offset] {
            strides[i + offset] = orig_strides[i];
        }
        // else stride stays 0 (broadcast)
    }

    strides
}

/// Binary op codes matching the CUDA kernel.
const OP_ADD: i32 = 0;
const OP_SUB: i32 = 1;
const OP_MUL: i32 = 2;
const OP_DIV: i32 = 3;
#[allow(dead_code)]
const OP_MAX: i32 = 4;
#[allow(dead_code)]
const OP_MIN: i32 = 5;
#[allow(dead_code)]
const OP_MOD: i32 = 6;

impl Tensor {
    /// Broadcast binary operation. Shapes must be broadcastable.
    fn broadcast_binary_op(&self, other: &Tensor, op: i32) -> Result<Tensor> {
        if self.dtype() != DType::F32 || other.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "broadcast binary ops only supported for F32".to_string(),
            });
        }

        let out_shape = broadcast_shapes(self.shape(), other.shape())
            .ok_or_else(|| Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            })?;

        let n: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let ndim = out_shape.len();

        let a_strides = broadcast_strides(self.shape(), &out_shape);
        let b_strides = broadcast_strides(other.shape(), &out_shape);

        let out_storage = Storage::new(n, DType::F32, self.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = self.runtime().next_stream();

        // Copy shape/strides to GPU
        let mut out_shape_vec: Vec<usize> = out_shape.to_vec();
        let mut a_strides_vec: Vec<usize> = a_strides;
        let mut b_strides_vec: Vec<usize> = b_strides;

        // Allocate GPU buffers for shape/strides metadata
        let meta_size = ndim * std::mem::size_of::<usize>();
        let alloc = self.runtime().alloc(meta_size * 3)?;
        let meta_ptr = alloc.as_ptr() as *mut usize;

        unsafe {
            let shape_ptr = meta_ptr;
            let a_str_ptr = meta_ptr.add(ndim);
            let b_str_ptr = meta_ptr.add(ndim * 2);

            ptx_sys::cudaMemcpyAsync(
                shape_ptr as *mut libc::c_void,
                out_shape_vec.as_mut_ptr() as *const libc::c_void,
                meta_size,
                1, // cudaMemcpyHostToDevice
                stream.raw(),
            );
            ptx_sys::cudaMemcpyAsync(
                a_str_ptr as *mut libc::c_void,
                a_strides_vec.as_mut_ptr() as *const libc::c_void,
                meta_size,
                1,
                stream.raw(),
            );
            ptx_sys::cudaMemcpyAsync(
                b_str_ptr as *mut libc::c_void,
                b_strides_vec.as_mut_ptr() as *const libc::c_void,
                meta_size,
                1,
                stream.raw(),
            );

            ptx_sys::ptx_tensor_broadcast_binary_f32(
                self.data_ptr_typed::<f32>(),
                other.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                shape_ptr as *mut usize,
                a_str_ptr as *mut usize,
                b_str_ptr as *mut usize,
                ndim as i32,
                op,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Broadcast addition: shapes must be broadcastable.
    pub fn broadcast_add(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_binary_op(other, OP_ADD)
    }

    /// Broadcast subtraction.
    pub fn broadcast_sub(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_binary_op(other, OP_SUB)
    }

    /// Broadcast multiplication.
    pub fn broadcast_mul(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_binary_op(other, OP_MUL)
    }

    /// Broadcast division.
    pub fn broadcast_div(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_binary_op(other, OP_DIV)
    }
}
