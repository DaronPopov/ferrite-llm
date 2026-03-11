//! Backward implementations for unary operations.

use ptx_tensor::Tensor;
use ptx_runtime::Result;

/// Backward for neg: grad_input = -grad_out
pub fn neg_backward(grad_out: &Tensor) -> Result<Tensor> {
    grad_out.neg()
}

/// Backward for exp: grad_input = grad_out * exp(x) = grad_out * output
pub fn exp_backward(grad_out: &Tensor, output: &Tensor) -> Result<Tensor> {
    grad_out.mul(output)
}

/// Backward for log: grad_input = grad_out / x
pub fn log_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    grad_out.div(input)
}

/// Backward for sqrt: grad_input = grad_out * 0.5 / sqrt(x) = grad_out * 0.5 / output
pub fn sqrt_backward(grad_out: &Tensor, output: &Tensor) -> Result<Tensor> {
    let half = grad_out.mul_scalar(0.5)?;
    half.div(output)
}

/// Backward for tanh: grad_input = grad_out * (1 - tanh(x)^2) = grad_out * (1 - output^2)
pub fn tanh_backward(grad_out: &Tensor, output: &Tensor) -> Result<Tensor> {
    let tanh_sq = output.sqr()?;
    let one_minus = tanh_sq.affine(-1.0, 1.0)?;  // 1 - tanh^2
    grad_out.mul(&one_minus)
}

/// Backward for sin: grad_input = grad_out * cos(x)
pub fn sin_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let cos_x = input.cos()?;
    grad_out.mul(&cos_x)
}

/// Backward for cos: grad_input = -grad_out * sin(x)
pub fn cos_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let sin_x = input.sin()?;
    grad_out.neg()?.mul(&sin_x)
}

/// Backward for abs: grad_input = grad_out * sign(x)
/// Note: gradient at 0 is undefined, we use 0
pub fn abs_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    // sign(x) = x / |x| for x != 0
    let abs_x = input.abs()?;
    let sign = input.div(&abs_x.clamp(1e-8, f32::INFINITY)?)?;
    grad_out.mul(&sign)
}

/// Backward for sqr (x^2): grad_input = grad_out * 2 * x
pub fn sqr_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let two_x = input.mul_scalar(2.0)?;
    grad_out.mul(&two_x)
}

/// Backward for recip (1/x): grad_input = -grad_out / x^2
pub fn recip_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let x_sq = input.sqr()?;
    grad_out.neg()?.div(&x_sq)
}
