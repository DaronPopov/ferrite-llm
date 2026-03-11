use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Element-wise conditional select: `out[i] = cond[i] ? true_val[i] : false_val[i]`
///
/// - `cond`: u8 buffer where non-zero means true
/// - `true_val`, `false_val`: f32 input buffers
/// - `output`: f32 output buffer
pub fn where_cond(
    cond: &GuardedBuffer,
    true_val: &GuardedBuffer,
    false_val: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    if numel == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "where_f32",
        });
    }

    cond.validate_capacity::<u8>(numel)?;
    cond.revalidate()?;
    true_val.validate_capacity::<f32>(numel)?;
    true_val.revalidate()?;
    false_val.validate_capacity::<f32>(numel)?;
    false_val.revalidate()?;
    output.validate_capacity::<f32>(numel)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_where_f32(
            cond.as_ptr_typed::<u8>(),
            true_val.as_ptr_typed::<f32>(),
            false_val.as_ptr_typed::<f32>(),
            output.as_ptr_typed::<f32>(),
            numel,
            context.stream(),
        );
    }
    post_launch(context, "where_f32")
}
