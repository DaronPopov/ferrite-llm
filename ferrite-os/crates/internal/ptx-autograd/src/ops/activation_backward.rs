//! Backward implementations for activation functions.

use ptx_tensor::Tensor;
use ptx_runtime::Result;

/// Backward for ReLU: grad_input = grad_out * (x > 0)
pub fn relu_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    // Create mask: 1 where input > 0, 0 otherwise
    // We can do this by computing relu(input) / (relu(input) + epsilon)
    // But this is inefficient - ideally we'd have a dedicated backward kernel

    // Simple approach: clamp input to [0, inf], then divide by clamped value + epsilon
    let relu_out = input.clamp(0.0, f32::INFINITY)?;
    let mask = relu_out.div(&relu_out.clamp(1e-8, f32::INFINITY)?)?;
    grad_out.mul(&mask)
}

/// Backward for Sigmoid: grad_input = grad_out * sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_backward(grad_out: &Tensor, output: &Tensor) -> Result<Tensor> {
    // output = sigmoid(x)
    // grad = output * (1 - output)
    let one_minus = output.affine(-1.0, 1.0)?;  // 1 - sigmoid
    let grad_factor = output.mul(&one_minus)?;
    grad_out.mul(&grad_factor)
}

/// Backward for GELU (approximate)
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// The derivative is complex:
/// GELU'(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
/// where z = sqrt(2/pi) * (x + 0.044715 * x^3)
pub fn gelu_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    // Simplified approximation using sigmoid
    // GELU(x) ≈ x * sigmoid(1.702 * x)
    // GELU'(x) ≈ sigmoid(1.702x) + 1.702 * x * sigmoid(1.702x) * (1 - sigmoid(1.702x))

    let scale = 1.702f32;
    let scaled_x = input.mul_scalar(scale)?;
    let sig = scaled_x.sigmoid()?;

    // sigmoid(1.702x)
    let term1 = sig.clone();

    // 1.702 * x * sigmoid * (1 - sigmoid)
    let one_minus_sig = sig.affine(-1.0, 1.0)?;
    let term2 = input.mul_scalar(scale)?.mul(&sig)?.mul(&one_minus_sig)?;

    let grad_factor = term1.add(&term2)?;
    grad_out.mul(&grad_factor)
}

/// Backward for SiLU/Swish: grad_input = grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
pub fn silu_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let sig = input.sigmoid()?;
    let one_minus_sig = sig.affine(-1.0, 1.0)?;

    // sigmoid + x * sigmoid * (1 - sigmoid)
    let term2 = input.mul(&sig)?.mul(&one_minus_sig)?;
    let grad_factor = sig.add(&term2)?;

    grad_out.mul(&grad_factor)
}

/// Backward for Softplus: grad_input = grad_out * sigmoid(x)
pub fn softplus_backward(grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
    let sig = input.sigmoid()?;
    grad_out.mul(&sig)
}

/// Backward for Leaky ReLU: grad_input = grad_out * (1 if x > 0 else alpha)
pub fn leaky_relu_backward(grad_out: &Tensor, input: &Tensor, alpha: f32) -> Result<Tensor> {
    // mask = 1 where x > 0, alpha where x <= 0
    // We can compute this as: clamp(x, 0, inf) / (clamp(x, 0, inf) + eps) * (1 - alpha) + alpha
    let relu_out = input.clamp(0.0, f32::INFINITY)?;
    let positive_mask = relu_out.div(&relu_out.clamp(1e-8, f32::INFINITY)?)?;  // 0 or 1

    // mask = positive_mask * (1 - alpha) + alpha = positive_mask - alpha * positive_mask + alpha
    let grad_factor = positive_mask.affine(1.0 - alpha, alpha)?;
    grad_out.mul(&grad_factor)
}

/// Backward for ELU: grad_input = grad_out * (1 if x > 0 else alpha * exp(x))
pub fn elu_backward(grad_out: &Tensor, input: &Tensor, output: &Tensor, alpha: f32) -> Result<Tensor> {
    // For x > 0: grad = 1
    // For x <= 0: output = alpha * (exp(x) - 1), so exp(x) = output/alpha + 1
    //            grad = alpha * exp(x) = output + alpha

    // Create mask for positive values
    let relu_out = input.clamp(0.0, f32::INFINITY)?;
    let positive_mask = relu_out.div(&relu_out.clamp(1e-8, f32::INFINITY)?)?;

    // For negative: grad = output + alpha
    let negative_grad = output.add_scalar(alpha)?;

    // Combine: positive_mask * 1 + (1 - positive_mask) * negative_grad
    let negative_mask = positive_mask.affine(-1.0, 1.0)?;  // 1 - positive_mask
    let grad_factor = positive_mask.add(&negative_mask.mul(&negative_grad)?)?;

    grad_out.mul(&grad_factor)
}
