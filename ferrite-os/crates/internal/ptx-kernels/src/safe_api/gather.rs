use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Gather elements from `input` along a dimension using i32 `indices`.
///
/// Tensor is viewed as `[outer, input_dim_size, inner]`; indices have shape
/// `[outer, idx_dim_size, inner]`.  Output has the same shape as indices.
pub fn gather(
    input: &GuardedBuffer,
    indices: &GuardedBuffer,
    output: &GuardedBuffer,
    outer: usize,
    input_dim_size: usize,
    idx_dim_size: usize,
    inner: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    let numel_in = outer
        .checked_mul(input_dim_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: input.size_bytes(),
        })?;
    if numel_in == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "gather_f32",
        });
    }

    let numel_idx = outer
        .checked_mul(idx_dim_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: indices.size_bytes(),
        })?;

    // Validate buffer capacities
    input.validate_capacity::<f32>(numel_in)?;
    input.revalidate()?;
    indices.validate_capacity::<i32>(numel_idx)?;
    indices.revalidate()?;
    output.validate_capacity::<f32>(numel_idx)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_gather_f32(
            input.as_ptr_typed::<f32>(),
            indices.as_ptr_typed::<i32>(),
            output.as_ptr_typed::<f32>(),
            outer,
            input_dim_size,
            idx_dim_size,
            inner,
            context.stream(),
        );
    }
    post_launch(context, "gather_f32")
}
