//! Backward implementations for matrix multiplication.

use ptx_tensor::Tensor;
use ptx_runtime::{Result, Error};

/// Backward for matrix multiplication C = A @ B
///
/// For C = A @ B where A: (M, K), B: (K, N), C: (M, N):
/// - dL/dA = dL/dC @ B^T  (M, N) @ (N, K) = (M, K)
/// - dL/dB = A^T @ dL/dC  (K, M) @ (M, N) = (K, N)
pub fn matmul_backward(
    grad_out: &Tensor,  // (M, N)
    a: &Tensor,         // (M, K)
    b: &Tensor,         // (K, N)
) -> Result<(Tensor, Tensor)> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    if a_ndim < 2 || b_ndim < 2 {
        return Err(Error::Internal {
            message: format!(
                "matmul_backward requires at least 2D tensors, got {}D and {}D",
                a_ndim, b_ndim
            ),
        });
    }

    // B^T: swap last two dims
    let b_t = b.transpose(b_ndim - 2, b_ndim - 1)?;
    // A^T: swap last two dims
    let a_t = a.transpose(a_ndim - 2, a_ndim - 1)?;

    // grad_a = grad_out @ B^T
    let grad_a = grad_out.matmul(&b_t.require_contiguous()?)?;
    // grad_b = A^T @ grad_out
    let grad_b = a_t.require_contiguous()?.matmul(grad_out)?;

    Ok((grad_a, grad_b))
}

/// Backward for batched matrix multiplication C[i] = A[i] @ B[i]
///
/// Same as matmul but applied per batch. The tensor matmul handles batching
/// automatically when inputs have 3+ dimensions.
pub fn bmm_backward(
    grad_out: &Tensor,  // (batch, M, N)
    a: &Tensor,         // (batch, M, K)
    b: &Tensor,         // (batch, K, N)
) -> Result<(Tensor, Tensor)> {
    // Batched matmul backward is the same as regular matmul backward;
    // the matmul kernel handles batch dims natively.
    matmul_backward(grad_out, a, b)
}

/// Backward for linear layer: y = x @ W^T + b
///
/// - dL/dx = dL/dy @ W
/// - dL/dW = dL/dy^T @ x = (x^T @ dL/dy)^T
/// - dL/db = sum(dL/dy, dim=0)
pub fn linear_backward(
    grad_out: &Tensor,      // (batch, out_features)
    input: &Tensor,         // (batch, in_features)
    weight: &Tensor,        // (out_features, in_features)
    bias_requires_grad: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    // dL/dx = dL/dy @ W  -- (batch, out_features) @ (out_features, in_features) = (batch, in_features)
    let grad_input = grad_out.matmul(weight)?;

    // dL/dW = dL/dy^T @ x  -- need to transpose grad_out from (batch, out_features) to (out_features, batch)
    // then matmul with input (batch, in_features) -> (out_features, in_features)
    let go_ndim = grad_out.ndim();
    let grad_out_t = grad_out.transpose(go_ndim - 2, go_ndim - 1)?.require_contiguous()?;
    let grad_weight = grad_out_t.matmul(input)?;

    // dL/db = sum(dL/dy, dim=0) if bias exists
    let grad_bias = if bias_requires_grad {
        Some(grad_out.sum_keepdim(0)?)
    } else {
        None
    };

    Ok((grad_input, grad_weight, grad_bias))
}
