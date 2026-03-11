use crate::guards::{
    BinaryOpGuard, GuardResult, GuardedBuffer, KernelContext, UnaryOpGuard,
};

pub(crate) fn unary_guard<'a>(
    input: &'a GuardedBuffer,
    output: &'a GuardedBuffer,
    numel: usize,
    context: &'a KernelContext,
) -> GuardResult<UnaryOpGuard<'a>> {
    UnaryOpGuard::new(input, output, numel, context)
}

pub(crate) fn binary_guard<'a>(
    left: &'a GuardedBuffer,
    right: &'a GuardedBuffer,
    output: &'a GuardedBuffer,
    numel: usize,
    context: &'a KernelContext,
) -> GuardResult<BinaryOpGuard<'a>> {
    BinaryOpGuard::new(left, right, output, numel, context)
}

/// Check for kernel launch errors, attributing any failure to the named kernel.
pub(crate) fn post_launch(context: &KernelContext, kernel: &'static str) -> GuardResult<()> {
    context.check_last_error_for(kernel)
}
