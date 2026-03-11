use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Argsort: sort each row of a `[nrows, ncols]` matrix.
///
/// Output is u32 indices.
/// - `ascending`: if true, smallest first; if false, largest first.
pub fn argsort(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    nrows: usize,
    ncols: usize,
    ascending: bool,
    context: &KernelContext,
) -> GuardResult<()> {
    let numel = nrows.checked_mul(ncols).ok_or(GuardError::BufferTooSmall {
        required: usize::MAX,
        available: input.size_bytes(),
    })?;
    if numel == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "argsort_f32",
        });
    }

    input.validate_capacity::<f32>(numel)?;
    input.revalidate()?;
    output.validate_capacity::<u32>(numel)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_argsort_f32(
            input.as_ptr_typed::<f32>(),
            output.as_ptr_typed::<u32>(),
            nrows,
            ncols,
            if ascending { 1 } else { 0 },
            context.stream(),
        );
    }
    post_launch(context, "argsort_f32")
}
