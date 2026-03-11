use crate::guards::{GuardResult, GuardedBuffer, KernelContext};

use super::launch::{binary_guard, post_launch};

type BinaryLauncher = unsafe extern "C" fn(
    usize,
    usize,
    *const usize,
    *const usize,
    *const f32,
    *const usize,
    *const f32,
    *mut f32,
    crate::cudaStream_t,
);

fn launch_binary(
    left: &GuardedBuffer,
    right: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
    kernel_name: &'static str,
    launcher: BinaryLauncher,
) -> GuardResult<()> {
    let guard = binary_guard(left, right, output, numel, context)?;
    let params = guard.kernel_params();
    unsafe {
        launcher(
            params.0, params.1, params.2, params.3, params.4, params.5, params.6, params.7,
            params.8,
        );
    }
    post_launch(context, kernel_name)
}

pub fn add(
    left: &GuardedBuffer,
    right: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_binary(
        left,
        right,
        output,
        numel,
        context,
        "badd_f32",
        crate::candle::candle_launch_badd_f32,
    )
}

pub fn mul(
    left: &GuardedBuffer,
    right: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_binary(
        left,
        right,
        output,
        numel,
        context,
        "bmul_f32",
        crate::candle::candle_launch_bmul_f32,
    )
}

pub fn sub(
    left: &GuardedBuffer,
    right: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_binary(
        left,
        right,
        output,
        numel,
        context,
        "bsub_f32",
        crate::candle::candle_launch_bsub_f32,
    )
}

pub fn div(
    left: &GuardedBuffer,
    right: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_binary(
        left,
        right,
        output,
        numel,
        context,
        "bdiv_f32",
        crate::candle::candle_launch_bdiv_f32,
    )
}
