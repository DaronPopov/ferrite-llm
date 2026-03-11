//! Backward implementations for binary operations.
//!
//! These wrap the backward kernels from tensor_backward.cu.

use ptx_tensor::Tensor;
use ptx_runtime::Result;

/// Backward for add: grad_a = grad_out, grad_b = grad_out
pub fn add_backward(grad_out: &Tensor) -> Result<(Tensor, Tensor)> {
    let grad_a = grad_out.clone_tensor()?;
    let grad_b = grad_out.clone_tensor()?;
    Ok((grad_a, grad_b))
}

/// Backward for sub: grad_a = grad_out, grad_b = -grad_out
pub fn sub_backward(grad_out: &Tensor) -> Result<(Tensor, Tensor)> {
    let grad_a = grad_out.clone_tensor()?;
    let grad_b = grad_out.neg()?;
    Ok((grad_a, grad_b))
}

/// Backward for mul: grad_a = grad_out * b, grad_b = grad_out * a
pub fn mul_backward(grad_out: &Tensor, a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let grad_a = grad_out.mul(b)?;
    let grad_b = grad_out.mul(a)?;
    Ok((grad_a, grad_b))
}

/// Backward for div: grad_a = grad_out / b, grad_b = -grad_out * a / b^2
pub fn div_backward(grad_out: &Tensor, a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let grad_a = grad_out.div(b)?;

    // grad_b = -grad_out * a / b^2
    let b_sq = b.sqr()?;
    let grad_b = grad_out.neg()?.mul(a)?.div(&b_sq)?;

    Ok((grad_a, grad_b))
}
