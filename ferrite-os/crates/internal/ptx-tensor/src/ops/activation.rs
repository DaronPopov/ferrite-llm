//! Activation functions.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error, increment_ops};

/// Activation function type.
#[derive(Debug, Clone, Copy)]
pub enum ActivationOp {
    Relu,
    Relu6,
    LeakyRelu,
    Elu,
    Selu,
    Gelu,
    Sigmoid,
    Silu,
    Softplus,
    Mish,
    GeluTanh,
    HardSwish,
    HardSigmoid,
}

impl Tensor {
    /// Generic activation function (no parameters).
    fn activation_op(&self, op: ActivationOp) -> Result<Tensor> {
        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let inp = input.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();

                match op {
                    ActivationOp::Relu => ptx_sys::ptx_tensor_relu_f32(inp, out, n, stream.raw()),
                    ActivationOp::Relu6 => ptx_sys::ptx_tensor_relu6_f32(inp, out, n, stream.raw()),
                    ActivationOp::Selu => ptx_sys::ptx_tensor_selu_f32(inp, out, n, stream.raw()),
                    ActivationOp::Gelu => ptx_sys::ptx_tensor_gelu_f32(inp, out, n, stream.raw()),
                    ActivationOp::Sigmoid => ptx_sys::ptx_tensor_sigmoid_f32(inp, out, n, stream.raw()),
                    ActivationOp::Silu => ptx_sys::ptx_tensor_silu_f32(inp, out, n, stream.raw()),
                    ActivationOp::Softplus => ptx_sys::ptx_tensor_softplus_f32(inp, out, n, stream.raw()),
                    ActivationOp::Mish => ptx_sys::ptx_tensor_mish_f32(inp, out, n, stream.raw()),
                    ActivationOp::GeluTanh => ptx_sys::ptx_tensor_gelu_tanh_f32(inp, out, n, stream.raw()),
                    ActivationOp::HardSwish => ptx_sys::ptx_tensor_hardswish_f32(inp, out, n, stream.raw()),
                    ActivationOp::HardSigmoid => ptx_sys::ptx_tensor_hardsigmoid_f32(inp, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} requires parameters", op),
                    }),
                }
            },
            DType::F64 => unsafe {
                let inp = input.data_ptr_typed::<f64>();
                let out = output.data_ptr_typed::<f64>();

                match op {
                    ActivationOp::Relu => ptx_sys::ptx_tensor_relu_f64(inp, out, n, stream.raw()),
                    ActivationOp::Gelu => ptx_sys::ptx_tensor_gelu_f64(inp, out, n, stream.raw()),
                    ActivationOp::Sigmoid => ptx_sys::ptx_tensor_sigmoid_f64(inp, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} not supported for F64", op),
                    }),
                }
            },
            DType::F16 => unsafe {
                let inp = input.data_ptr_typed::<ptx_sys::__half>();
                let out = output.data_ptr_typed::<ptx_sys::__half>();

                match op {
                    ActivationOp::Relu => ptx_sys::ptx_tensor_relu_f16(inp, out, n, stream.raw()),
                    ActivationOp::Gelu => ptx_sys::ptx_tensor_gelu_f16(inp, out, n, stream.raw()),
                    ActivationOp::Sigmoid => ptx_sys::ptx_tensor_sigmoid_f16(inp, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} not supported for F16", op),
                    }),
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Activation not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Relu)
    }

    /// ReLU6 activation: min(max(0, x), 6)
    pub fn relu6(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Relu6)
    }

    /// Leaky ReLU: max(alpha * x, x)
    pub fn leaky_relu(&self, alpha: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "leaky_relu only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_leaky_relu_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                alpha,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// ELU activation: x if x > 0 else alpha * (exp(x) - 1)
    pub fn elu(&self, alpha: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "elu only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_elu_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                alpha,
                stream.raw(),
            );
        }

        increment_ops();
        Ok(output)
    }

    /// SELU activation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    pub fn selu(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Selu)
    }

    /// GELU activation: x * Phi(x) where Phi is the standard normal CDF
    pub fn gelu(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Gelu)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Sigmoid)
    }

    /// SiLU/Swish activation: x * sigmoid(x)
    pub fn silu(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Silu)
    }

    /// Alias for SiLU.
    pub fn swish(&self) -> Result<Tensor> {
        self.silu()
    }

    /// Softplus activation: log(1 + exp(x))
    pub fn softplus(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Softplus)
    }

    /// Mish activation: x * tanh(softplus(x))
    pub fn mish(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::Mish)
    }

    /// GELU with tanh approximation (same as PyTorch's gelu(approximate='tanh')).
    pub fn gelu_tanh(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::GeluTanh)
    }

    /// HardSwish: x * min(max(x+3, 0), 6) / 6
    pub fn hardswish(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::HardSwish)
    }

    /// HardSigmoid: min(max(x/6 + 0.5, 0), 1)
    pub fn hardsigmoid(&self) -> Result<Tensor> {
        self.activation_op(ActivationOp::HardSigmoid)
    }
}
