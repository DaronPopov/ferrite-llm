//! Pooling operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{Shape, contiguous_strides};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Max pooling 2D.
    ///
    /// - self (input): (N, C, H, W)
    /// - kernel_size: [kH, kW]
    /// - stride: [strideH, strideW]
    /// - padding: [padH, padW]
    /// - Returns: (N, C, H_out, W_out)
    pub fn max_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor> {
        self.pool2d_impl(kernel_size, stride, padding, true)
    }

    /// Average pooling 2D.
    ///
    /// - self (input): (N, C, H, W)
    /// - kernel_size: [kH, kW]
    /// - stride: [strideH, strideW]
    /// - padding: [padH, padW]
    /// - Returns: (N, C, H_out, W_out)
    pub fn avg_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor> {
        self.pool2d_impl(kernel_size, stride, padding, false)
    }

    /// Adaptive average pooling 2D.
    ///
    /// Automatically computes kernel_size, stride, and padding to produce
    /// the desired output size.
    ///
    /// - self (input): (N, C, H, W)
    /// - output_size: [H_out, W_out]
    /// - Returns: (N, C, H_out, W_out)
    pub fn adaptive_avg_pool2d(&self, output_size: [usize; 2]) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("adaptive_avg_pool2d expects (N, C, H, W) input, got {:?}", self.shape()),
            });
        }

        let h = self.shape()[2];
        let w = self.shape()[3];
        let ho = output_size[0];
        let wo = output_size[1];

        // Compute stride and kernel_size to cover input evenly
        let stride_h = h / ho;
        let stride_w = w / wo;
        let kernel_h = h - (ho - 1) * stride_h;
        let kernel_w = w - (wo - 1) * stride_w;

        self.avg_pool2d([kernel_h, kernel_w], [stride_h, stride_w], [0, 0])
    }

    fn pool2d_impl(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        is_max: bool,
    ) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "pool2d only supported for F32".to_string(),
            });
        }
        if self.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("pool2d expects (N, C, H, W) input, got {:?}", self.shape()),
            });
        }

        let input = self.require_contiguous()?;

        let n = input.shape()[0] as i32;
        let c = input.shape()[1] as i32;
        let h = input.shape()[2] as i32;
        let w = input.shape()[3] as i32;
        let kh = kernel_size[0] as i32;
        let kw = kernel_size[1] as i32;
        let sh = stride[0] as i32;
        let sw = stride[1] as i32;
        let ph = padding[0] as i32;
        let pw = padding[1] as i32;

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        if h_out <= 0 || w_out <= 0 {
            return Err(Error::Internal {
                message: format!("pool2d output size non-positive: h_out={}, w_out={}", h_out, w_out),
            });
        }

        let out_shape = Shape::from_slice(&[n as usize, c as usize, h_out as usize, w_out as usize]);
        let out_elems = (n as usize) * (c as usize) * (h_out as usize) * (w_out as usize);
        let out_storage = Storage::new(out_elems, DType::F32, input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = input.runtime().next_stream();

        unsafe {
            if is_max {
                ptx_sys::ptx_tensor_max_pool2d_f32(
                    input.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    n, c, h, w,
                    kh, kw, sh, sw, ph, pw,
                    h_out, w_out,
                    stream.raw(),
                );
            } else {
                ptx_sys::ptx_tensor_avg_pool2d_f32(
                    input.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    n, c, h, w,
                    kh, kw, sh, sw, ph, pw,
                    h_out, w_out,
                    stream.raw(),
                );
            }
        }

        increment_ops();
        Ok(output)
    }
}
