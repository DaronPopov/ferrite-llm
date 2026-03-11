use crate::guards::{GuardResult, GuardedBuffer, KernelContext};

use super::launch::{post_launch, unary_guard};

fn launch_unary(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
    kernel_name: &'static str,
    launcher: unsafe extern "C" fn(
        usize,
        usize,
        *const usize,
        *const f32,
        *mut f32,
        crate::cudaStream_t,
    ),
) -> GuardResult<()> {
    let guard = unary_guard(input, output, numel, context)?;
    let params = guard.kernel_params();
    unsafe {
        launcher(params.0, params.1, params.2, params.3, params.4, params.5);
    }
    post_launch(context, kernel_name)
}

pub fn gelu(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "ugelu_f32", crate::candle::candle_launch_ugelu_f32)
}

pub fn relu(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "urelu_f32", crate::candle::candle_launch_urelu_f32)
}

pub fn silu(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "usilu_f32", crate::candle::candle_launch_usilu_f32)
}

pub fn sigmoid(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "usigmoid_f32", crate::candle::candle_launch_usigmoid_f32)
}

pub fn abs(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "uabs_f32", crate::candle::candle_launch_uabs_f32)
}

pub fn sqrt(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "usqrt_f32", crate::candle::candle_launch_usqrt_f32)
}

pub fn exp(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "uexp_f32", crate::candle::candle_launch_uexp_f32)
}

pub fn log(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "ulog_f32", crate::candle::candle_launch_ulog_f32)
}

pub fn tanh(
    input: &GuardedBuffer,
    output: &GuardedBuffer,
    numel: usize,
    context: &KernelContext,
) -> GuardResult<()> {
    launch_unary(input, output, numel, context, "utanh_f32", crate::candle::candle_launch_utanh_f32)
}
