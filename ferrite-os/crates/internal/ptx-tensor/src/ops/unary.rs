//! Unary element-wise operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error, increment_ops};

/// Unary operation type.
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Log2,
    Log10,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sinh,
    Cosh,
    Ceil,
    Floor,
    Round,
    Sign,
    Sqr,
    Recip,
    Erf,
}

impl Tensor {
    /// Generic unary operation.
    fn unary_op(&self, op: UnaryOp) -> Result<Tensor> {
        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        if n == 0 {
            return Ok(output);
        }
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let inp = input.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();

                match op {
                    UnaryOp::Neg => ptx_sys::ptx_tensor_neg_f32(inp, out, n, stream.raw()),
                    UnaryOp::Abs => ptx_sys::ptx_tensor_abs_f32(inp, out, n, stream.raw()),
                    UnaryOp::Exp => ptx_sys::ptx_tensor_exp_f32(inp, out, n, stream.raw()),
                    UnaryOp::Log => ptx_sys::ptx_tensor_log_f32(inp, out, n, stream.raw()),
                    UnaryOp::Log2 => ptx_sys::ptx_tensor_log2_f32(inp, out, n, stream.raw()),
                    UnaryOp::Log10 => ptx_sys::ptx_tensor_log10_f32(inp, out, n, stream.raw()),
                    UnaryOp::Sqrt => ptx_sys::ptx_tensor_sqrt_f32(inp, out, n, stream.raw()),
                    UnaryOp::Rsqrt => ptx_sys::ptx_tensor_rsqrt_f32(inp, out, n, stream.raw()),
                    UnaryOp::Sin => ptx_sys::ptx_tensor_sin_f32(inp, out, n, stream.raw()),
                    UnaryOp::Cos => ptx_sys::ptx_tensor_cos_f32(inp, out, n, stream.raw()),
                    UnaryOp::Tan => ptx_sys::ptx_tensor_tan_f32(inp, out, n, stream.raw()),
                    UnaryOp::Tanh => ptx_sys::ptx_tensor_tanh_f32(inp, out, n, stream.raw()),
                    UnaryOp::Sinh => ptx_sys::ptx_tensor_sinh_f32(inp, out, n, stream.raw()),
                    UnaryOp::Cosh => ptx_sys::ptx_tensor_cosh_f32(inp, out, n, stream.raw()),
                    UnaryOp::Ceil => ptx_sys::ptx_tensor_ceil_f32(inp, out, n, stream.raw()),
                    UnaryOp::Floor => ptx_sys::ptx_tensor_floor_f32(inp, out, n, stream.raw()),
                    UnaryOp::Round => ptx_sys::ptx_tensor_round_f32(inp, out, n, stream.raw()),
                    UnaryOp::Sign => ptx_sys::ptx_tensor_sign_f32(inp, out, n, stream.raw()),
                    UnaryOp::Sqr => ptx_sys::ptx_tensor_sqr_f32(inp, out, n, stream.raw()),
                    UnaryOp::Recip => ptx_sys::ptx_tensor_recip_f32(inp, out, n, stream.raw()),
                    UnaryOp::Erf => ptx_sys::ptx_tensor_erf_f32(inp, out, n, stream.raw()),
                }
            },
            DType::F64 => unsafe {
                let inp = input.data_ptr_typed::<f64>();
                let out = output.data_ptr_typed::<f64>();

                match op {
                    UnaryOp::Neg => ptx_sys::ptx_tensor_neg_f64(inp, out, n, stream.raw()),
                    UnaryOp::Exp => ptx_sys::ptx_tensor_exp_f64(inp, out, n, stream.raw()),
                    UnaryOp::Log => ptx_sys::ptx_tensor_log_f64(inp, out, n, stream.raw()),
                    UnaryOp::Sqrt => ptx_sys::ptx_tensor_sqrt_f64(inp, out, n, stream.raw()),
                    UnaryOp::Tanh => ptx_sys::ptx_tensor_tanh_f64(inp, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} not supported for F64", op),
                    }),
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Unary ops not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Negate all elements.
    pub fn neg(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Neg)
    }

    /// Absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Abs)
    }

    /// Exponential (e^x).
    pub fn exp(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Exp)
    }

    /// Natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Log)
    }

    /// Square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sqrt)
    }

    /// Reciprocal square root (1/sqrt(x)).
    pub fn rsqrt(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Rsqrt)
    }

    /// Sine.
    pub fn sin(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sin)
    }

    /// Cosine.
    pub fn cos(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Cos)
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Tanh)
    }

    /// Ceiling.
    pub fn ceil(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Ceil)
    }

    /// Floor.
    pub fn floor(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Floor)
    }

    /// Round to nearest integer.
    pub fn round(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Round)
    }

    /// Square (x^2).
    pub fn sqr(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sqr)
    }

    /// Reciprocal (1/x).
    pub fn recip(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Recip)
    }

    /// Base-2 logarithm.
    pub fn log2(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Log2)
    }

    /// Base-10 logarithm.
    pub fn log10(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Log10)
    }

    /// Tangent.
    pub fn tan(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Tan)
    }

    /// Hyperbolic sine.
    pub fn sinh(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sinh)
    }

    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Cosh)
    }

    /// Sign function (-1, 0, or 1).
    pub fn sign(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sign)
    }

    /// Gauss error function.
    pub fn erf(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Erf)
    }

    /// Copy tensor data.
    pub fn copy(&self) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "copy only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_copy_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }
}

// ============================================================================
// Operator overloads
// ============================================================================

use std::ops;

impl ops::Neg for &Tensor {
    type Output = Result<Tensor>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}
