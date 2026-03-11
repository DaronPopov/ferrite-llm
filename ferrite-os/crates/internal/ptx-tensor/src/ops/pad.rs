//! Padding operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, Shape, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Pad a 4D tensor [N, C, H, W] with constant value.
    ///
    /// `padding` is [top, bottom, left, right].
    pub fn pad2d(&self, padding: [usize; 4], value: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "pad2d only supported for F32".to_string(),
            });
        }
        if self.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("pad2d requires 4D tensor [N,C,H,W], got {}D", self.ndim()),
            });
        }

        let input = self.require_contiguous()?;

        let n = input.shape()[0] as i32;
        let c = input.shape()[1] as i32;
        let h = input.shape()[2] as i32;
        let w = input.shape()[3] as i32;
        let [pad_top, pad_bottom, pad_left, pad_right] = [
            padding[0] as i32, padding[1] as i32,
            padding[2] as i32, padding[3] as i32,
        ];

        let oh = (h + pad_top + pad_bottom) as usize;
        let ow = (w + pad_left + pad_right) as usize;
        let out_shape = Shape::from_slice(&[n as usize, c as usize, oh, ow]);
        let out_n: usize = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;

        let out_storage = Storage::new(out_n, DType::F32, input.runtime())?;
        let output = Tensor::from_storage(out_storage, out_shape.clone(), contiguous_strides(&out_shape), 0);

        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_pad2d_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n, c, h, w,
                pad_top, pad_bottom, pad_left, pad_right,
                value,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// Symmetric padding shorthand: pad all sides equally.
    pub fn pad2d_uniform(&self, pad: usize, value: f32) -> Result<Tensor> {
        self.pad2d([pad, pad, pad, pad], value)
    }
}
