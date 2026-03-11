//! Logical operations on U8 boolean tensors.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Logical AND of two boolean (U8) tensors.
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::U8 || other.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: "logical_and requires U8 tensors".to_string(),
            });
        }
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        let out_shape = Shape::from_slice(input.shape());
        let out_storage = Storage::new(input.elem_count(), DType::U8, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_logical_and_u8(
                input.data_ptr_typed::<u8>(),
                rhs.data_ptr_typed::<u8>(),
                output.data_ptr_typed::<u8>(),
                n, stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Logical OR of two boolean (U8) tensors.
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::U8 || other.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: "logical_or requires U8 tensors".to_string(),
            });
        }
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        let out_shape = Shape::from_slice(input.shape());
        let out_storage = Storage::new(input.elem_count(), DType::U8, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_logical_or_u8(
                input.data_ptr_typed::<u8>(),
                rhs.data_ptr_typed::<u8>(),
                output.data_ptr_typed::<u8>(),
                n, stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Logical NOT of a boolean (U8) tensor.
    pub fn logical_not(&self) -> Result<Tensor> {
        if self.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: "logical_not requires U8 tensor".to_string(),
            });
        }

        let input = self.require_contiguous()?;

        let out_shape = Shape::from_slice(input.shape());
        let out_storage = Storage::new(input.elem_count(), DType::U8, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_logical_not_u8(
                input.data_ptr_typed::<u8>(),
                output.data_ptr_typed::<u8>(),
                n, stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Logical XOR of two boolean (U8) tensors.
    pub fn logical_xor(&self, other: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::U8 || other.dtype() != DType::U8 {
            return Err(Error::NotSupported {
                message: "logical_xor requires U8 tensors".to_string(),
            });
        }
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        let input = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        let out_shape = Shape::from_slice(input.shape());
        let out_storage = Storage::new(input.elem_count(), DType::U8, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_logical_xor_u8(
                input.data_ptr_typed::<u8>(),
                rhs.data_ptr_typed::<u8>(),
                output.data_ptr_typed::<u8>(),
                n, stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }
}
