use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Select top-k values and their indices along a dimension.
///
/// Tensor is viewed as `[outer, dim_size, inner]`; selects `k` extreme
/// values along the middle axis.
///
/// - `values_out` shape: `[outer, k, inner]` (f32)
/// - `indices_out` shape: `[outer, k, inner]` (i32)
/// - `largest`: if true, selects the k largest; otherwise the k smallest.
pub fn topk(
    input: &GuardedBuffer,
    values_out: &GuardedBuffer,
    indices_out: &GuardedBuffer,
    outer: usize,
    dim_size: usize,
    inner: usize,
    k: usize,
    largest: bool,
    context: &KernelContext,
) -> GuardResult<()> {
    if k == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "topk_f32",
        });
    }

    let numel_in = outer
        .checked_mul(dim_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: input.size_bytes(),
        })?;
    if numel_in == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "topk_f32",
        });
    }

    let numel_out = outer
        .checked_mul(k)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: values_out.size_bytes(),
        })?;

    // Validate buffer capacities
    input.validate_capacity::<f32>(numel_in)?;
    input.revalidate()?;
    values_out.validate_capacity::<f32>(numel_out)?;
    values_out.revalidate()?;
    indices_out.validate_capacity::<i32>(numel_out)?;
    indices_out.revalidate()?;

    unsafe {
        crate::candle::candle_launch_topk_f32(
            input.as_ptr_typed::<f32>(),
            values_out.as_ptr_typed::<f32>(),
            indices_out.as_ptr_typed::<i32>(),
            outer,
            dim_size,
            inner,
            k,
            if largest { 1 } else { 0 },
            context.stream(),
        );
    }
    post_launch(context, "topk_f32")
}
