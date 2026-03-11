//! Type casting operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Cast tensor to F32.
    pub fn to_f32(&self) -> Result<Tensor> {
        if self.dtype() == DType::F32 {
            return Ok(self.clone());
        }

        let input = self.require_contiguous()?;
        let out_shape = Shape::from_slice(input.shape());
        let n = input.elem_count();
        let out_storage = Storage::new(n, DType::F32, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F16 => unsafe {
                ptx_sys::ptx_tensor_cast_f16_to_f32(
                    input.data_ptr_typed::<ptx_sys::__half>(),
                    output.data_ptr_typed::<f32>(),
                    n, stream.raw(),
                );
            },
            DType::I32 => unsafe {
                ptx_sys::ptx_tensor_cast_i32_to_f32(
                    input.data_ptr_typed::<i32>(),
                    output.data_ptr_typed::<f32>(),
                    n, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Cast from {:?} to F32 not supported", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Cast tensor to F16.
    pub fn to_f16(&self) -> Result<Tensor> {
        if self.dtype() == DType::F16 {
            return Ok(self.clone());
        }

        let input = self.require_contiguous()?;
        let out_shape = Shape::from_slice(input.shape());
        let n = input.elem_count();
        let out_storage = Storage::new(n, DType::F16, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_cast_f32_to_f16(
                    input.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<ptx_sys::__half>(),
                    n, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Cast from {:?} to F16 not supported", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Cast tensor to I32.
    pub fn to_i32(&self) -> Result<Tensor> {
        if self.dtype() == DType::I32 {
            return Ok(self.clone());
        }

        let input = self.require_contiguous()?;
        let out_shape = Shape::from_slice(input.shape());
        let n = input.elem_count();
        let out_storage = Storage::new(n, DType::I32, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_cast_f32_to_i32(
                    input.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<i32>(),
                    n, stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Cast from {:?} to I32 not supported", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }
}
