use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Inclusive cumulative sum along a dimension.
///
/// Tensor is viewed as `[outer, dim_size, inner]`; the scan runs along the
/// middle axis.  Output has the same shape as input.
pub fn cumsum(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    outer: usize,
    dim_size: usize,
    inner: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    let numel = outer
        .checked_mul(dim_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: input.size_bytes(),
        })?;
    if numel == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "cumsum_f32",
        });
    }

    input.validate_capacity::<f32>(numel)?;
    input.revalidate()?;
    output.validate_capacity::<f32>(numel)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_cumsum_f32(
            input.as_ptr_typed::<f32>(),
            output.as_ptr_typed::<f32>(),
            outer,
            dim_size,
            inner,
            context.stream(),
        );
    }
    post_launch(context, "cumsum_f32")
}
