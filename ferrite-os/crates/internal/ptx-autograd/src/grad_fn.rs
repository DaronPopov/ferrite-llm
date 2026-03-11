//! Gradient functions for automatic differentiation.

use ptx_tensor::Tensor;
use ptx_runtime::Result;

/// Trait for gradient functions.
///
/// A gradient function computes the gradients of inputs given the gradient
/// of the output.
pub trait GradFn: Send + Sync {
    /// Compute gradients for inputs.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the output tensor
    /// * `saved` - Tensors saved during forward pass
    ///
    /// # Returns
    ///
    /// Vector of gradients for each input. `None` means the input doesn't
    /// require gradient.
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>>;

    /// Compute gradients while recording on the tape for higher-order derivatives.
    ///
    /// When `create_graph=true`, gradient computations must themselves be
    /// differentiable. This method performs the same backward computation
    /// through `Variable` ops so the operations are recorded on the tape.
    ///
    /// The default implementation delegates to the non-recording `backward`.
    /// Override this for gradient functions that need to support second derivatives.
    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        // Default: compute through raw tensors (no graph recording)
        let saved_tensors: Vec<Tensor> = saved.iter().map(|v| v.tensor().clone()).collect();
        let grads = self.backward(grad_output.tensor(), &saved_tensors)?;
        Ok(grads
            .into_iter()
            .map(|opt| opt.map(|t| crate::Variable::new(t, false)))
            .collect())
    }

    /// Get the name of this gradient function (for debugging).
    fn name(&self) -> &'static str;
}

// ============================================================================
// Identity (no-op)
// ============================================================================

pub struct IdentityBackward;

impl GradFn for IdentityBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.clone_tensor()?)])
    }

    fn name(&self) -> &'static str {
        "IdentityBackward"
    }
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Gradient for addition: d/da (a + b) = 1, d/db (a + b) = 1
pub struct AddBackward;

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![
            Some(grad_output.clone_tensor()?),
            Some(grad_output.clone_tensor()?),
        ])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        _saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        Ok(vec![
            Some(grad_output.clone()),
            Some(grad_output.clone()),
        ])
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient for subtraction: d/da (a - b) = 1, d/db (a - b) = -1
pub struct SubBackward;

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![
            Some(grad_output.clone_tensor()?),
            Some(grad_output.neg()?),
        ])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        _saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        Ok(vec![
            Some(grad_output.clone()),
            Some(grad_output.neg()?),
        ])
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Gradient for multiplication: d/da (a * b) = b, d/db (a * b) = a
/// Saved: [a, b]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let a = &saved[0];
        let b = &saved[1];
        Ok(vec![
            Some(grad_output.mul(b)?),  // d/da = grad * b
            Some(grad_output.mul(a)?),  // d/db = grad * a
        ])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let a = &saved[0];
        let b = &saved[1];
        Ok(vec![
            Some(grad_output.mul(b)?),
            Some(grad_output.mul(a)?),
        ])
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient for division: d/da (a / b) = 1/b, d/db (a / b) = -a/b^2
/// Saved: [a, b]
pub struct DivBackward;

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let a = &saved[0];
        let b = &saved[1];

        let grad_a = grad_output.div(b)?;

        let b_inv = b.recip()?;
        let grad_b = grad_output.neg()?.mul(a)?.mul(&b_inv)?.mul(&b_inv)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let a = &saved[0];
        let b = &saved[1];

        // d/da = grad / b  (recorded on tape)
        let grad_a = grad_output.div(b)?;

        // d/db = -grad * a / b^2  (recorded on tape)
        let neg_grad = grad_output.neg()?;
        let neg_grad_a = neg_grad.mul(a)?;
        let grad_b = neg_grad_a.div(&b.mul(b)?)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Gradient for negation: d/dx (-x) = -1
pub struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.neg()?)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        _saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        Ok(vec![Some(grad_output.neg()?)])
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient for exp: d/dx exp(x) = exp(x)
/// Saved: [output] (the exp(x) value)
pub struct ExpBackward;

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // exp(x)
        Ok(vec![Some(grad_output.mul(output)?)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let output = &saved[0];  // exp(x)
        Ok(vec![Some(grad_output.mul(output)?)])
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Gradient for log: d/dx log(x) = 1/x
/// Saved: [input]
pub struct LogBackward;

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        Ok(vec![Some(grad_output.div(input)?)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let input = &saved[0];
        Ok(vec![Some(grad_output.div(input)?)])
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Gradient for sqrt: d/dx sqrt(x) = 0.5 / sqrt(x)
/// Saved: [output] (the sqrt(x) value)
pub struct SqrtBackward;

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // sqrt(x)
        let grad_input = grad_output.mul_scalar(0.5)?.div(output)?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let output = &saved[0];  // sqrt(x)
        // grad * 0.5 / sqrt(x) — both mul and div are recorded
        let half = crate::Variable::leaf(Tensor::full(
            grad_output.shape(),
            0.5,
            grad_output.dtype(),
            grad_output.runtime(),
        )?);
        let scaled = grad_output.mul(&half)?;
        Ok(vec![Some(scaled.div(output)?)])
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient for tanh: d/dx tanh(x) = 1 - tanh(x)^2
/// Saved: [output] (the tanh(x) value)
pub struct TanhBackward;

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // tanh(x)
        let tanh_sq = output.sqr()?;
        let one_minus = tanh_sq.affine(-1.0, 1.0)?;  // 1 - tanh^2
        let grad_input = grad_output.mul(&one_minus)?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_create_graph(
        &self,
        grad_output: &crate::Variable,
        saved: &[crate::Variable],
    ) -> Result<Vec<Option<crate::Variable>>> {
        let output = &saved[0];  // tanh(x)
        // 1 - tanh^2: tanh_sq via mul, then sub from ones
        let tanh_sq = output.mul(output)?;
        let ones = crate::Variable::ones(
            output.shape(), output.dtype(), output.runtime(), false,
        )?;
        let one_minus = ones.sub(&tanh_sq)?;
        Ok(vec![Some(grad_output.mul(&one_minus)?)])
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ============================================================================
// Activation Gradients
// ============================================================================

/// Gradient for ReLU: d/dx relu(x) = 1 if x > 0 else 0
/// Saved: [input]
pub struct ReluBackward;

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        // grad * (input > 0)
        // We use clamp(0, 1) on the sign to get 0 or 1
        // Actually, we need to implement this properly with a mask
        // For now, use the input to compute the mask via affine
        // This is a workaround - proper implementation needs backward kernels
        let positive = input.clamp(0.0, f32::INFINITY)?;
        let clamped = positive.clamp(1e-8, f32::INFINITY)?;
        let mask = positive.div(&clamped)?;  // 0 or 1
        Ok(vec![Some(grad_output.mul(&mask)?)])
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

/// Gradient for Sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
/// Saved: [output] (the sigmoid(x) value)
pub struct SigmoidBackward;

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // sigmoid(x)
        // grad * sigmoid * (1 - sigmoid)
        let one_minus = output.affine(-1.0, 1.0)?;  // 1 - sigmoid
        let grad_input = grad_output.mul(output)?.mul(&one_minus)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

/// Gradient for GELU (approximate): d/dx gelu(x) ≈ complex expression
/// Saved: [input]
pub struct GeluBackward;

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        // Approximate GELU gradient using tanh approximation
        // This is a simplified version - proper implementation needs the backward kernel
        let input = &saved[0];

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // The derivative is complex, for now use finite difference approximation
        // or implement proper backward kernel in CUDA

        // Simplified: just use sigmoid approximation for now
        let sig = input.sigmoid()?;
        let grad_input = grad_output.mul(&sig)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

// ============================================================================
// Reduction Gradients
// ============================================================================

/// Gradient for sum: broadcast grad_output back to input shape
/// Saved: [input_shape as tensor - or we store shape separately]
pub struct SumBackward {
    pub input_shape: Vec<usize>,
    pub dim: i32,
    pub keepdim: bool,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let grad_input = crate::ops::reduction_backward::sum_backward(
            grad_output,
            &self.input_shape,
            self.dim,
            self.keepdim,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient for mean: similar to sum but scaled by 1/n
pub struct MeanBackward {
    pub input_shape: Vec<usize>,
    pub dim: i32,
    pub keepdim: bool,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let grad_input = crate::ops::reduction_backward::mean_backward(
            grad_output,
            &self.input_shape,
            self.dim,
            self.keepdim,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ============================================================================
// Softmax Gradient
// ============================================================================

/// Gradient for softmax
/// Saved: [output] (the softmax output)
pub struct SoftmaxBackward {
    pub dim: i32,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // softmax(x)

        // For softmax: d/dx softmax(x)_i = softmax(x)_i * (1_{i=j} - softmax(x)_j)
        // In practice: grad_input = output * (grad_output - sum(grad_output * output))

        // sum(grad * softmax) along dim
        let grad_times_out = grad_output.mul(output)?;
        let sum = grad_times_out.sum_keepdim(self.dim)?;

        // grad_input = softmax * (grad - sum)
        let grad_input = output.mul(&grad_output.sub(&sum)?)?;

        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ============================================================================
// Matmul Gradient
// ============================================================================

/// Gradient for matmul: C = A @ B
/// d/dA = grad @ B.T
/// d/dB = A.T @ grad
/// Saved: [A, B]
pub struct MatmulBackward;

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let a = &saved[0];
        let b = &saved[1];

        let (grad_a, grad_b) = crate::ops::matmul_backward::matmul_backward(grad_output, a, b)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}
