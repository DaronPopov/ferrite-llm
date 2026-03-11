use crate::guards::{GuardError, GuardResult, GuardedBuffer, KernelContext};

use super::launch::post_launch;

/// Index-select: gather slices along a dimension by index.
///
/// Tensor is viewed as `[left_size, src_dim_size, right_size]`; `ids` selects
/// from `src_dim_size`.  Output shape: `[left_size, ids_dim_size, right_size]`.
pub fn index_select(
    input: &GuardedBuffer,
    ids: &GuardedBuffer,
    output: &GuardedBuffer,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    let numel_in = left_size
        .checked_mul(src_dim_size)
        .and_then(|v| v.checked_mul(right_size))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: input.size_bytes(),
        })?;
    if numel_in == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "index_select_f32",
        });
    }

    let numel_out = left_size
        .checked_mul(ids_dim_size)
        .and_then(|v| v.checked_mul(right_size))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: output.size_bytes(),
        })?;

    input.validate_capacity::<f32>(numel_in)?;
    input.revalidate()?;
    ids.validate_capacity::<u32>(ids_dim_size)?;
    ids.revalidate()?;
    output.validate_capacity::<f32>(numel_out)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_index_select_f32(
            input.as_ptr_typed::<f32>(),
            ids.as_ptr_typed::<u32>(),
            output.as_ptr_typed::<f32>(),
            left_size,
            src_dim_size,
            ids_dim_size,
            right_size,
            context.stream(),
        );
    }
    post_launch(context, "index_select_f32")
}

/// Scatter-add: accumulate `src` values into `output` at positions given by `ids`.
///
/// Output must be pre-initialized (e.g. zeros).
/// Tensor is viewed as `[left_size, src_dim_size, right_size]` for src;
/// output dim is `dst_dim_size`.
pub fn scatter_add(
    ids: &GuardedBuffer,
    src: &GuardedBuffer,
    output: &GuardedBuffer,
    left_size: usize,
    src_dim_size: usize,
    dst_dim_size: usize,
    right_size: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    let numel_src = left_size
        .checked_mul(src_dim_size)
        .and_then(|v| v.checked_mul(right_size))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: src.size_bytes(),
        })?;
    if numel_src == 0 {
        return Err(GuardError::ZeroElements {
            kernel: "scatter_add_f32",
        });
    }

    let numel_out = left_size
        .checked_mul(dst_dim_size)
        .and_then(|v| v.checked_mul(right_size))
        .ok_or(GuardError::BufferTooSmall {
            required: usize::MAX,
            available: output.size_bytes(),
        })?;

    ids.validate_capacity::<u32>(numel_src)?;
    ids.revalidate()?;
    src.validate_capacity::<f32>(numel_src)?;
    src.revalidate()?;
    output.validate_capacity::<f32>(numel_out)?;
    output.revalidate()?;

    unsafe {
        crate::candle::candle_launch_scatter_add_f32(
            ids.as_ptr_typed::<u32>(),
            src.as_ptr_typed::<f32>(),
            output.as_ptr_typed::<f32>(),
            left_size,
            src_dim_size,
            dst_dim_size,
            right_size,
            context.stream(),
        );
    }
    post_launch(context, "scatter_add_f32")
}
