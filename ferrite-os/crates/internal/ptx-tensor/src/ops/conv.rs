//! Convolution operations using im2col + cuBLAS GEMM.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{Shape, contiguous_strides};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops, cublas::Gemm};

impl Tensor {
    /// 2D convolution.
    ///
    /// - self (input): (N, C_in, H, W)
    /// - weight: (C_out, C_in, kH, kW)
    /// - bias: (C_out,) optional
    /// - stride: [strideH, strideW]
    /// - padding: [padH, padW]
    /// - dilation: [dilationH, dilationW]
    /// - Returns: (N, C_out, H_out, W_out)
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "conv2d only supported for F32".to_string(),
            });
        }
        if self.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("conv2d expects (N, C_in, H, W) input, got {:?}", self.shape()),
            });
        }
        if weight.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("conv2d expects (C_out, C_in, kH, kW) weight, got {:?}", weight.shape()),
            });
        }

        let input = self.require_contiguous()?;
        let weight = weight.require_contiguous()?;

        let n = input.shape()[0] as i32;
        let c_in = input.shape()[1] as i32;
        let h = input.shape()[2] as i32;
        let w = input.shape()[3] as i32;

        let c_out = weight.shape()[0];
        let wc_in = weight.shape()[1];
        let kh = weight.shape()[2] as i32;
        let kw = weight.shape()[3] as i32;

        if c_in != wc_in as i32 {
            return Err(Error::ShapeMismatch {
                expected: vec![c_in as usize],
                actual: vec![wc_in],
            });
        }

        let pad_h = padding[0] as i32;
        let pad_w = padding[1] as i32;
        let stride_h = stride[0] as i32;
        let stride_w = stride[1] as i32;
        let dilation_h = dilation[0] as i32;
        let dilation_w = dilation[1] as i32;

        let h_out = (h + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
        let w_out = (w + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;

        if h_out <= 0 || w_out <= 0 {
            return Err(Error::Internal {
                message: format!(
                    "conv2d output size is non-positive: h_out={}, w_out={}",
                    h_out, w_out
                ),
            });
        }

        // Allocate im2col buffer: (N, C_in*kH*kW, H_out*W_out)
        let col_size = (n as usize) * (c_in as usize) * (kh as usize) * (kw as usize) * (h_out as usize) * (w_out as usize);
        let col_storage = Storage::new(col_size, DType::F32, input.runtime())?;
        let col_ptr = col_storage.as_ptr() as *mut f32;

        // Run im2col
        let stream = input.runtime().next_stream();
        unsafe {
            ptx_sys::ptx_tensor_im2col_f32(
                input.data_ptr_typed::<f32>(),
                col_ptr,
                n, c_in, h, w,
                kh, kw,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w,
                h_out, w_out,
                stream.raw(),
            );
        }

        // Output: (N, C_out, H_out, W_out)
        let out_h = h_out as usize;
        let out_w = w_out as usize;
        let out_elems = (n as usize) * c_out * out_h * out_w;
        let out_shape = Shape::from_slice(&[n as usize, c_out, out_h, out_w]);
        let out_storage = Storage::new(out_elems, DType::F32, input.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // GEMM: for each batch element:
        //   weight_2d (C_out, C_in*kH*kW) @ col_b (C_in*kH*kW, H_out*W_out) = out_b (C_out, H_out*W_out)
        let gemm = Gemm::new()?;
        gemm.set_stream(&stream)?;

        let ckk = (c_in as usize) * (kh as usize) * (kw as usize);
        let hw_out = out_h * out_w;
        let weight_ptr = weight.data_ptr_typed::<f32>();
        let out_ptr = output.data_ptr_typed::<f32>();

        for b in 0..n as usize {
            let col_b = unsafe { col_ptr.add(b * ckk * hw_out) };
            let out_b = unsafe { out_ptr.add(b * c_out * hw_out) };
            unsafe {
                gemm.matmul_f32(
                    weight_ptr,
                    col_b,
                    out_b,
                    c_out,     // M
                    hw_out,    // N
                    ckk,       // K
                )?;
            }
        }

        // Add bias if provided
        let result = if let Some(b) = bias {
            // Reshape bias to (1, C_out, 1, 1) for broadcasting
            let b_reshaped = b.reshape(&[1, c_out, 1, 1])?;
            output.broadcast_add(&b_reshaped)?
        } else {
            output
        };

        increment_ops();
        Ok(result)
    }
}
