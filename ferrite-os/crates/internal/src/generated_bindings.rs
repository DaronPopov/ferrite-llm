// Auto-generated FFI bindings
// DO NOT EDIT - regenerate with KernelCompiler

#![allow(dead_code)]

use ptx_runtime::Stream;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Auto-generated FFI for: softplus

extern "C" {
    fn ptx_softplus_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// softplus activation (f32)
pub fn softplus(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_softplus_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sparseopintersection

extern "C" {
    fn ptx_sparseopintersection_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sparseopintersection activation (f32)
pub fn sparseopintersection(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sparseopintersection_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: averagepool2d

extern "C" {
    fn ptx_averagepool2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// averagepool2d activation (f32)
pub fn averagepool2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_averagepool2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hardsigmoid

extern "C" {
    fn ptx_hardsigmoid_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hardsigmoid activation (f32)
pub fn hardsigmoid(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hardsigmoid_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reduceargmin

extern "C" {
    fn ptx_reduceargmin_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reduceargmin activation (f32)
pub fn reduceargmin(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reduceargmin_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tanh

extern "C" {
    fn ptx_tanh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tanh activation (f32)
pub fn tanh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tanh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: atan

extern "C" {
    fn ptx_atan_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// atan activation (f32)
pub fn atan(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_atan_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: asin

extern "C" {
    fn ptx_asin_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// asin activation (f32)
pub fn asin(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_asin_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shifted_chebyshev_polynomial_w

extern "C" {
    fn ptx_shifted_chebyshev_polynomial_w_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shifted_chebyshev_polynomial_w activation (f32)
pub fn shifted_chebyshev_polynomial_w(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shifted_chebyshev_polynomial_w_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: pow

extern "C" {
    fn ptx_pow_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// pow activation (f32)
pub fn pow(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_pow_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: adaptiveaveragepooling

extern "C" {
    fn ptx_adaptiveaveragepooling_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// adaptiveaveragepooling activation (f32)
pub fn adaptiveaveragepooling(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_adaptiveaveragepooling_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensorfactories

extern "C" {
    fn ptx_tensorfactories_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensorfactories activation (f32)
pub fn tensorfactories(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensorfactories_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: adaptivemaxpooling2d

extern "C" {
    fn ptx_adaptivemaxpooling2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// adaptivemaxpooling2d activation (f32)
pub fn adaptivemaxpooling2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_adaptivemaxpooling2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: asinh

extern "C" {
    fn ptx_asinh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// asinh activation (f32)
pub fn asinh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_asinh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: embeddingbag

extern "C" {
    fn ptx_embeddingbag_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// embeddingbag activation (f32)
pub fn embeddingbag(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_embeddingbag_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: weightnorm

extern "C" {
    fn ptx_weightnorm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// weightnorm activation (f32)
pub fn weightnorm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_weightnorm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: unique

extern "C" {
    fn ptx_unique_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// unique activation (f32)
pub fn unique(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_unique_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: gcdlcm

extern "C" {
    fn ptx_gcdlcm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// gcdlcm activation (f32)
pub fn gcdlcm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_gcdlcm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fill

extern "C" {
    fn ptx_fill_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fill activation (f32)
pub fn fill(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fill_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: batchlinearalgebraeig

extern "C" {
    fn ptx_batchlinearalgebraeig_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// batchlinearalgebraeig activation (f32)
pub fn batchlinearalgebraeig(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_batchlinearalgebraeig_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sortimpl

extern "C" {
    fn ptx_sortimpl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sortimpl activation (f32)
pub fn sortimpl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sortimpl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: miscopss

extern "C" {
    fn ptx_miscopss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// miscopss activation (f32)
pub fn miscopss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_miscopss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: logicalopss

extern "C" {
    fn ptx_logicalopss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// logicalopss activation (f32)
pub fn logicalopss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_logicalopss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: scaled_modified_bessel_k1

extern "C" {
    fn ptx_scaled_modified_bessel_k1_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// scaled_modified_bessel_k1 activation (f32)
pub fn scaled_modified_bessel_k1(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_scaled_modified_bessel_k1_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachop

extern "C" {
    fn ptx_foreachop_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachop activation (f32)
pub fn foreachop(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachop_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensorcompare

extern "C" {
    fn ptx_tensorcompare_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensorcompare activation (f32)
pub fn tensorcompare(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensorcompare_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tan

extern "C" {
    fn ptx_tan_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tan activation (f32)
pub fn tan(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tan_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: indexing

extern "C" {
    fn ptx_indexing_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// indexing activation (f32)
pub fn indexing(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_indexing_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: gridsampler

extern "C" {
    fn ptx_gridsampler_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// gridsampler activation (f32)
pub fn gridsampler(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_gridsampler_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: softmax

extern "C" {
    fn ptx_softmax_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// softmax activation (f32)
pub fn softmax(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_softmax_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sinh

extern "C" {
    fn ptx_sinh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sinh activation (f32)
pub fn sinh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sinh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: replicationpadding

extern "C" {
    fn ptx_replicationpadding_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// replicationpadding activation (f32)
pub fn replicationpadding(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_replicationpadding_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: abs

extern "C" {
    fn ptx_abs_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// abs activation (f32)
pub fn abs(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_abs_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachopscalartensor

extern "C" {
    fn ptx_foreachopscalartensor_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachopscalartensor activation (f32)
pub fn foreachopscalartensor(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachopscalartensor_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: nllloss2d

extern "C" {
    fn ptx_nllloss2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// nllloss2d activation (f32)
pub fn nllloss2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_nllloss2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: acos

extern "C" {
    fn ptx_acos_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// acos activation (f32)
pub fn acos(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_acos_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fused_adam_impl

extern "C" {
    fn ptx_fused_adam_impl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fused_adam_impl activation (f32)
pub fn fused_adam_impl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fused_adam_impl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fused_adagrad_impl

extern "C" {
    fn ptx_fused_adagrad_impl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fused_adagrad_impl activation (f32)
pub fn fused_adagrad_impl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fused_adagrad_impl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: multilabelmargincriterion

extern "C" {
    fn ptx_multilabelmargincriterion_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// multilabelmargincriterion activation (f32)
pub fn multilabelmargincriterion(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_multilabelmargincriterion_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: nonzero

extern "C" {
    fn ptx_nonzero_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// nonzero activation (f32)
pub fn nonzero(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_nonzero_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fused_adamw_amsgrad_impl

extern "C" {
    fn ptx_fused_adamw_amsgrad_impl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fused_adamw_amsgrad_impl activation (f32)
pub fn fused_adamw_amsgrad_impl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fused_adamw_amsgrad_impl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: multimarginloss

extern "C" {
    fn ptx_multimarginloss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// multimarginloss activation (f32)
pub fn multimarginloss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_multimarginloss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: copysign

extern "C" {
    fn ptx_copysign_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// copysign activation (f32)
pub fn copysign(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_copysign_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachternaryop

extern "C" {
    fn ptx_foreachternaryop_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachternaryop activation (f32)
pub fn foreachternaryop(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachternaryop_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: divfloor

extern "C" {
    fn ptx_divfloor_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// divfloor activation (f32)
pub fn divfloor(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_divfloor_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: scattergather

extern "C" {
    fn ptx_scattergather_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// scattergather activation (f32)
pub fn scattergather(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_scattergather_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fused_adam_amsgrad_impl

extern "C" {
    fn ptx_fused_adam_amsgrad_impl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fused_adam_amsgrad_impl activation (f32)
pub fn fused_adam_amsgrad_impl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fused_adam_amsgrad_impl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sorting

extern "C" {
    fn ptx_sorting_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sorting activation (f32)
pub fn sorting(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sorting_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reduceaminmax

extern "C" {
    fn ptx_reduceaminmax_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reduceaminmax activation (f32)
pub fn reduceaminmax(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reduceaminmax_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: laguerre_polynomial_l

extern "C" {
    fn ptx_laguerre_polynomial_l_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// laguerre_polynomial_l activation (f32)
pub fn laguerre_polynomial_l(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_laguerre_polynomial_l_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachopscalarlist

extern "C" {
    fn ptx_foreachopscalarlist_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachopscalarlist activation (f32)
pub fn foreachopscalarlist(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachopscalarlist_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fractionalmaxpool2d

extern "C" {
    fn ptx_fractionalmaxpool2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fractionalmaxpool2d activation (f32)
pub fn fractionalmaxpool2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fractionalmaxpool2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: remainder

extern "C" {
    fn ptx_remainder_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// remainder activation (f32)
pub fn remainder(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_remainder_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: naiveconvolutiontranspose2d

extern "C" {
    fn ptx_naiveconvolutiontranspose2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// naiveconvolutiontranspose2d activation (f32)
pub fn naiveconvolutiontranspose2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_naiveconvolutiontranspose2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachoplist

extern "C" {
    fn ptx_foreachoplist_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachoplist activation (f32)
pub fn foreachoplist(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachoplist_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: col2im

extern "C" {
    fn ptx_col2im_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// col2im activation (f32)
pub fn col2im(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_col2im_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fractionalmaxpool3d

extern "C" {
    fn ptx_fractionalmaxpool3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fractionalmaxpool3d activation (f32)
pub fn fractionalmaxpool3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fractionalmaxpool3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reflectionpad

extern "C" {
    fn ptx_reflectionpad_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reflectionpad activation (f32)
pub fn reflectionpad(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reflectionpad_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: chebyshev_polynomial_t

extern "C" {
    fn ptx_chebyshev_polynomial_t_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// chebyshev_polynomial_t activation (f32)
pub fn chebyshev_polynomial_t(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_chebyshev_polynomial_t_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: pointwiseops

extern "C" {
    fn ptx_pointwiseops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// pointwiseops activation (f32)
pub fn pointwiseops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_pointwiseops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hardtanh

extern "C" {
    fn ptx_hardtanh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hardtanh activation (f32)
pub fn hardtanh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hardtanh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shifted_chebyshev_polynomial_u

extern "C" {
    fn ptx_shifted_chebyshev_polynomial_u_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shifted_chebyshev_polynomial_u activation (f32)
pub fn shifted_chebyshev_polynomial_u(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shifted_chebyshev_polynomial_u_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: lossctc

extern "C" {
    fn ptx_lossctc_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// lossctc activation (f32)
pub fn lossctc(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_lossctc_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: depthwiseconv2d

extern "C" {
    fn ptx_depthwiseconv2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// depthwiseconv2d activation (f32)
pub fn depthwiseconv2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_depthwiseconv2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: logcumsumexp

extern "C" {
    fn ptx_logcumsumexp_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// logcumsumexp activation (f32)
pub fn logcumsumexp(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_logcumsumexp_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsampletrilinear3d

extern "C" {
    fn ptx_upsampletrilinear3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsampletrilinear3d activation (f32)
pub fn upsampletrilinear3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsampletrilinear3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensortopk

extern "C" {
    fn ptx_tensortopk_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensortopk activation (f32)
pub fn tensortopk(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensortopk_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bucketization

extern "C" {
    fn ptx_bucketization_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bucketization activation (f32)
pub fn bucketization(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bucketization_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplenearest1d

extern "C" {
    fn ptx_upsamplenearest1d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplenearest1d activation (f32)
pub fn upsamplenearest1d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplenearest1d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: softshrink

extern "C" {
    fn ptx_softshrink_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// softshrink activation (f32)
pub fn softshrink(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_softshrink_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionlognormal

extern "C" {
    fn ptx_distributionlognormal_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionlognormal activation (f32)
pub fn distributionlognormal(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionlognormal_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: step

extern "C" {
    fn ptx_step_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// step activation (f32)
pub fn step(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_step_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: spherical_bessel_j0

extern "C" {
    fn ptx_spherical_bessel_j0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// spherical_bessel_j0 activation (f32)
pub fn spherical_bessel_j0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_spherical_bessel_j0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fusedadagrad

extern "C" {
    fn ptx_fusedadagrad_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fusedadagrad activation (f32)
pub fn fusedadagrad(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fusedadagrad_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachreduceop

extern "C" {
    fn ptx_foreachreduceop_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachreduceop activation (f32)
pub fn foreachreduceop(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachreduceop_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensormode

extern "C" {
    fn ptx_tensormode_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensormode activation (f32)
pub fn tensormode(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensormode_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: adaptiveaveragepooling3d

extern "C" {
    fn ptx_adaptiveaveragepooling3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// adaptiveaveragepooling3d activation (f32)
pub fn adaptiveaveragepooling3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_adaptiveaveragepooling3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reduceargmax

extern "C" {
    fn ptx_reduceargmax_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reduceargmax activation (f32)
pub fn reduceargmax(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reduceargmax_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionbernoulli

extern "C" {
    fn ptx_distributionbernoulli_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionbernoulli activation (f32)
pub fn distributionbernoulli(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionbernoulli_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: int4mm

extern "C" {
    fn ptx_int4mm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// int4mm activation (f32)
pub fn int4mm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_int4mm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: silu

extern "C" {
    fn ptx_silu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// silu activation (f32)
pub fn silu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_silu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hardswish

extern "C" {
    fn ptx_hardswish_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hardswish activation (f32)
pub fn hardswish(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hardswish_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachopscalar

extern "C" {
    fn ptx_foreachopscalar_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachopscalar activation (f32)
pub fn foreachopscalar(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachopscalar_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: ops

extern "C" {
    fn ptx_ops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// ops activation (f32)
pub fn ops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_ops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: complexs

extern "C" {
    fn ptx_complexs_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// complexs activation (f32)
pub fn complexs(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_complexs_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: compareeq

extern "C" {
    fn ptx_compareeq_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// compareeq activation (f32)
pub fn compareeq(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_compareeq_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: randperm

extern "C" {
    fn ptx_randperm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// randperm activation (f32)
pub fn randperm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_randperm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: dilatedmaxpool3d

extern "C" {
    fn ptx_dilatedmaxpool3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// dilatedmaxpool3d activation (f32)
pub fn dilatedmaxpool3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_dilatedmaxpool3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: scaledgroupmm

extern "C" {
    fn ptx_scaledgroupmm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// scaledgroupmm activation (f32)
pub fn scaledgroupmm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_scaledgroupmm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: im2col

extern "C" {
    fn ptx_im2col_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// im2col activation (f32)
pub fn im2col(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_im2col_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: mul

extern "C" {
    fn ptx_mul_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// mul activation (f32)
pub fn mul(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_mul_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: mixeddtypeslinear

extern "C" {
    fn ptx_mixeddtypeslinear_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// mixeddtypeslinear activation (f32)
pub fn mixeddtypeslinear(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_mixeddtypeslinear_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: logsigmoid

extern "C" {
    fn ptx_logsigmoid_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// logsigmoid activation (f32)
pub fn logsigmoid(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_logsigmoid_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: layer_norm_kernel

extern "C" {
    fn ptx_layer_norm_kernel_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// layer_norm_kernel activation (f32)
pub fn layer_norm_kernel(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_layer_norm_kernel_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reduceminvalues

extern "C" {
    fn ptx_reduceminvalues_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reduceminvalues activation (f32)
pub fn reduceminvalues(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reduceminvalues_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: chebyshev_polynomial_w

extern "C" {
    fn ptx_chebyshev_polynomial_w_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// chebyshev_polynomial_w activation (f32)
pub fn chebyshev_polynomial_w(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_chebyshev_polynomial_w_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shiftopss

extern "C" {
    fn ptx_shiftopss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shiftopss activation (f32)
pub fn shiftopss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shiftopss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: logs

extern "C" {
    fn ptx_logs_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// logs activation (f32)
pub fn logs(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_logs_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shape

extern "C" {
    fn ptx_shape_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shape activation (f32)
pub fn shape(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shape_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: dilatedmaxpool2d

extern "C" {
    fn ptx_dilatedmaxpool2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// dilatedmaxpool2d activation (f32)
pub fn dilatedmaxpool2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_dilatedmaxpool2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: modified_bessel_k0

extern "C" {
    fn ptx_modified_bessel_k0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// modified_bessel_k0 activation (f32)
pub fn modified_bessel_k0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_modified_bessel_k0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bessel_j1

extern "C" {
    fn ptx_bessel_j1_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bessel_j1 activation (f32)
pub fn bessel_j1(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bessel_j1_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionnormal

extern "C" {
    fn ptx_distributionnormal_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionnormal activation (f32)
pub fn distributionnormal(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionnormal_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reducemoment

extern "C" {
    fn ptx_reducemoment_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reducemoment activation (f32)
pub fn reducemoment(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reducemoment_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: maxunpooling

extern "C" {
    fn ptx_maxunpooling_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// maxunpooling activation (f32)
pub fn maxunpooling(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_maxunpooling_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: unfoldbackward

extern "C" {
    fn ptx_unfoldbackward_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// unfoldbackward activation (f32)
pub fn unfoldbackward(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_unfoldbackward_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: copy

extern "C" {
    fn ptx_copy_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// copy activation (f32)
pub fn copy(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_copy_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: rangefactories

extern "C" {
    fn ptx_rangefactories_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// rangefactories activation (f32)
pub fn rangefactories(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_rangefactories_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: compares

extern "C" {
    fn ptx_compares_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// compares activation (f32)
pub fn compares(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_compares_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cumprod

extern "C" {
    fn ptx_cumprod_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cumprod activation (f32)
pub fn cumprod(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cumprod_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplenearest2d

extern "C" {
    fn ptx_upsamplenearest2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplenearest2d activation (f32)
pub fn upsamplenearest2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplenearest2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reducenorm

extern "C" {
    fn ptx_reducenorm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reducenorm activation (f32)
pub fn reducenorm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reducenorm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: divtrunc

extern "C" {
    fn ptx_divtrunc_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// divtrunc activation (f32)
pub fn divtrunc(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_divtrunc_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sortstable

extern "C" {
    fn ptx_sortstable_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sortstable activation (f32)
pub fn sortstable(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sortstable_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: logaddexp

extern "C" {
    fn ptx_logaddexp_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// logaddexp activation (f32)
pub fn logaddexp(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_logaddexp_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: summaryops

extern "C" {
    fn ptx_summaryops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// summaryops activation (f32)
pub fn summaryops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_summaryops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: airy_ai

extern "C" {
    fn ptx_airy_ai_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// airy_ai activation (f32)
pub fn airy_ai(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_airy_ai_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fused_adamw_impl

extern "C" {
    fn ptx_fused_adamw_impl_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fused_adamw_impl activation (f32)
pub fn fused_adamw_impl(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fused_adamw_impl_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: specialops

extern "C" {
    fn ptx_specialops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// specialops activation (f32)
pub fn specialops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_specialops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: naivedilatedconvolution

extern "C" {
    fn ptx_naivedilatedconvolution_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// naivedilatedconvolution activation (f32)
pub fn naivedilatedconvolution(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_naivedilatedconvolution_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: divtrue

extern "C" {
    fn ptx_divtrue_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// divtrue activation (f32)
pub fn divtrue(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_divtrue_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: rreluwithnoise

extern "C" {
    fn ptx_rreluwithnoise_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// rreluwithnoise activation (f32)
pub fn rreluwithnoise(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_rreluwithnoise_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reducemaxvalues

extern "C" {
    fn ptx_reducemaxvalues_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reducemaxvalues activation (f32)
pub fn reducemaxvalues(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reducemaxvalues_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: modified_bessel_i1

extern "C" {
    fn ptx_modified_bessel_i1_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// modified_bessel_i1 activation (f32)
pub fn modified_bessel_i1(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_modified_bessel_i1_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bessel_y0

extern "C" {
    fn ptx_bessel_y0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bessel_y0 activation (f32)
pub fn bessel_y0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bessel_y0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shifted_chebyshev_polynomial_v

extern "C" {
    fn ptx_shifted_chebyshev_polynomial_v_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shifted_chebyshev_polynomial_v activation (f32)
pub fn shifted_chebyshev_polynomial_v(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shifted_chebyshev_polynomial_v_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sin

extern "C" {
    fn ptx_sin_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sin activation (f32)
pub fn sin(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sin_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fusedadam

extern "C" {
    fn ptx_fusedadam_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fusedadam activation (f32)
pub fn fusedadam(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fusedadam_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: naiveconvolutiontranspose3d

extern "C" {
    fn ptx_naiveconvolutiontranspose3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// naiveconvolutiontranspose3d activation (f32)
pub fn naiveconvolutiontranspose3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_naiveconvolutiontranspose3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: groupmm

extern "C" {
    fn ptx_groupmm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// groupmm activation (f32)
pub fn groupmm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_groupmm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: uniquecub

extern "C" {
    fn ptx_uniquecub_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// uniquecub activation (f32)
pub fn uniquecub(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_uniquecub_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: loss

extern "C" {
    fn ptx_loss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// loss activation (f32)
pub fn loss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_loss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributions

extern "C" {
    fn ptx_distributions_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributions activation (f32)
pub fn distributions(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributions_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionexponential

extern "C" {
    fn ptx_distributionexponential_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionexponential activation (f32)
pub fn distributionexponential(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionexponential_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: triangularops

extern "C" {
    fn ptx_triangularops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// triangularops activation (f32)
pub fn triangularops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_triangularops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: legendre_polynomial_p

extern "C" {
    fn ptx_legendre_polynomial_p_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// legendre_polynomial_p activation (f32)
pub fn legendre_polynomial_p(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_legendre_polynomial_p_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: scaled_modified_bessel_k0

extern "C" {
    fn ptx_scaled_modified_bessel_k0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// scaled_modified_bessel_k0 activation (f32)
pub fn scaled_modified_bessel_k0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_scaled_modified_bessel_k0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: flattenindices

extern "C" {
    fn ptx_flattenindices_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// flattenindices activation (f32)
pub fn flattenindices(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_flattenindices_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: rnn

extern "C" {
    fn ptx_rnn_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// rnn activation (f32)
pub fn rnn(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_rnn_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: modified_bessel_k1

extern "C" {
    fn ptx_modified_bessel_k1_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// modified_bessel_k1 activation (f32)
pub fn modified_bessel_k1(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_modified_bessel_k1_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: zeta

extern "C" {
    fn ptx_zeta_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// zeta activation (f32)
pub fn zeta(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_zeta_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fusedadamw

extern "C" {
    fn ptx_fusedadamw_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fusedadamw activation (f32)
pub fn fusedadamw(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fusedadamw_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: chebyshev_polynomial_u

extern "C" {
    fn ptx_chebyshev_polynomial_u_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// chebyshev_polynomial_u activation (f32)
pub fn chebyshev_polynomial_u(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_chebyshev_polynomial_u_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hermite_polynomial_he

extern "C" {
    fn ptx_hermite_polynomial_he_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hermite_polynomial_he activation (f32)
pub fn hermite_polynomial_he(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hermite_polynomial_he_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: adaptivemaxpooling3d

extern "C" {
    fn ptx_adaptivemaxpooling3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// adaptivemaxpooling3d activation (f32)
pub fn adaptivemaxpooling3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_adaptivemaxpooling3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionrandom

extern "C" {
    fn ptx_distributionrandom_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionrandom activation (f32)
pub fn distributionrandom(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionrandom_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: amps

extern "C" {
    fn ptx_amps_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// amps activation (f32)
pub fn amps(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_amps_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: igamma

extern "C" {
    fn ptx_igamma_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// igamma activation (f32)
pub fn igamma(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_igamma_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: miscbackwardopss

extern "C" {
    fn ptx_miscbackwardopss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// miscbackwardopss activation (f32)
pub fn miscbackwardopss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_miscbackwardopss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: normalization

extern "C" {
    fn ptx_normalization_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// normalization activation (f32)
pub fn normalization(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_normalization_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cudascalar

extern "C" {
    fn ptx_cudascalar_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cudascalar activation (f32)
pub fn cudascalar(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cudascalar_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cos

extern "C" {
    fn ptx_cos_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cos activation (f32)
pub fn cos(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cos_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: s

extern "C" {
    fn ptx_s_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// s activation (f32)
pub fn s(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_s_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: embeddingbackward

extern "C" {
    fn ptx_embeddingbackward_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// embeddingbackward activation (f32)
pub fn embeddingbackward(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_embeddingbackward_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: gelu

extern "C" {
    fn ptx_gelu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// gelu activation (f32)
pub fn gelu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_gelu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: gammas

extern "C" {
    fn ptx_gammas_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// gammas activation (f32)
pub fn gammas(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_gammas_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: recordstream

extern "C" {
    fn ptx_recordstream_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// recordstream activation (f32)
pub fn recordstream(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_recordstream_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: convolutionmm2d

extern "C" {
    fn ptx_convolutionmm2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// convolutionmm2d activation (f32)
pub fn convolutionmm2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_convolutionmm2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reducelogic

extern "C" {
    fn ptx_reducelogic_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reducelogic activation (f32)
pub fn reducelogic(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reducelogic_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: lerp

extern "C" {
    fn ptx_lerp_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// lerp activation (f32)
pub fn lerp(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_lerp_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fractions

extern "C" {
    fn ptx_fractions_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fractions activation (f32)
pub fn fractions(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fractions_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hermite_polynomial_h

extern "C" {
    fn ptx_hermite_polynomial_h_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hermite_polynomial_h activation (f32)
pub fn hermite_polynomial_h(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hermite_polynomial_h_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: modified_bessel_i0

extern "C" {
    fn ptx_modified_bessel_i0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// modified_bessel_i0 activation (f32)
pub fn modified_bessel_i0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_modified_bessel_i0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: atanh

extern "C" {
    fn ptx_atanh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// atanh activation (f32)
pub fn atanh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_atanh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: segmentreduce

extern "C" {
    fn ptx_segmentreduce_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// segmentreduce activation (f32)
pub fn segmentreduce(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_segmentreduce_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: fusedsgd

extern "C" {
    fn ptx_fusedsgd_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// fusedsgd activation (f32)
pub fn fusedsgd(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_fusedsgd_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: threshold

extern "C" {
    fn ptx_threshold_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// threshold activation (f32)
pub fn threshold(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_threshold_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplebicubic2d

extern "C" {
    fn ptx_upsamplebicubic2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplebicubic2d activation (f32)
pub fn upsamplebicubic2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplebicubic2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cross

extern "C" {
    fn ptx_cross_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cross activation (f32)
pub fn cross(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cross_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: renorm

extern "C" {
    fn ptx_renorm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// renorm activation (f32)
pub fn renorm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_renorm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bessel_j0

extern "C" {
    fn ptx_bessel_j0_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bessel_j0 activation (f32)
pub fn bessel_j0(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bessel_j0_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: embedding

extern "C" {
    fn ptx_embedding_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// embedding activation (f32)
pub fn embedding(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_embedding_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplenearest3d

extern "C" {
    fn ptx_upsamplenearest3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplenearest3d activation (f32)
pub fn upsamplenearest3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplenearest3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributioncauchy

extern "C" {
    fn ptx_distributioncauchy_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributioncauchy activation (f32)
pub fn distributioncauchy(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributioncauchy_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cosh

extern "C" {
    fn ptx_cosh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cosh activation (f32)
pub fn cosh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cosh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: index

extern "C" {
    fn ptx_index_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// index activation (f32)
pub fn index(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_index_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: repeat

extern "C" {
    fn ptx_repeat_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// repeat activation (f32)
pub fn repeat(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_repeat_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reduce

extern "C" {
    fn ptx_reduce_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reduce activation (f32)
pub fn reduce(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reduce_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: acosh

extern "C" {
    fn ptx_acosh_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// acosh activation (f32)
pub fn acosh(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_acosh_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: shifted_chebyshev_polynomial_t

extern "C" {
    fn ptx_shifted_chebyshev_polynomial_t_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// shifted_chebyshev_polynomial_t activation (f32)
pub fn shifted_chebyshev_polynomial_t(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_shifted_chebyshev_polynomial_t_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: chebyshev_polynomial_v

extern "C" {
    fn ptx_chebyshev_polynomial_v_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// chebyshev_polynomial_v activation (f32)
pub fn chebyshev_polynomial_v(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_chebyshev_polynomial_v_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distribution

extern "C" {
    fn ptx_distribution_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distribution activation (f32)
pub fn distribution(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distribution_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: complex

extern "C" {
    fn ptx_complex_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// complex activation (f32)
pub fn complex(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_complex_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: spectralops

extern "C" {
    fn ptx_spectralops_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// spectralops activation (f32)
pub fn spectralops(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_spectralops_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cumsum

extern "C" {
    fn ptx_cumsum_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cumsum activation (f32)
pub fn cumsum(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cumsum_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplebilinear2d

extern "C" {
    fn ptx_upsamplebilinear2d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplebilinear2d activation (f32)
pub fn upsamplebilinear2d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplebilinear2d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: validatecompressedindices

extern "C" {
    fn ptx_validatecompressedindices_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// validatecompressedindices activation (f32)
pub fn validatecompressedindices(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_validatecompressedindices_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: dropout

extern "C" {
    fn ptx_dropout_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// dropout activation (f32)
pub fn dropout(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_dropout_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: elu

extern "C" {
    fn ptx_elu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// elu activation (f32)
pub fn elu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_elu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: signs

extern "C" {
    fn ptx_signs_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// signs activation (f32)
pub fn signs(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_signs_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: multinomial

extern "C" {
    fn ptx_multinomial_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// multinomial activation (f32)
pub fn multinomial(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_multinomial_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: leakyrelu

extern "C" {
    fn ptx_leakyrelu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// leakyrelu activation (f32)
pub fn leakyrelu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_leakyrelu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bessel_y1

extern "C" {
    fn ptx_bessel_y1_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bessel_y1 activation (f32)
pub fn bessel_y1(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bessel_y1_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: indexutils

extern "C" {
    fn ptx_indexutils_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// indexutils activation (f32)
pub fn indexutils(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_indexutils_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: reducesumprod

extern "C" {
    fn ptx_reducesumprod_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// reducesumprod activation (f32)
pub fn reducesumprod(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_reducesumprod_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensortransformations

extern "C" {
    fn ptx_tensortransformations_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensortransformations activation (f32)
pub fn tensortransformations(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensortransformations_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: bitwiseopss

extern "C" {
    fn ptx_bitwiseopss_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// bitwiseopss activation (f32)
pub fn bitwiseopss(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_bitwiseopss_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: mish

extern "C" {
    fn ptx_mish_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// mish activation (f32)
pub fn mish(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_mish_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: hardshrink

extern "C" {
    fn ptx_hardshrink_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// hardshrink activation (f32)
pub fn hardshrink(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_hardshrink_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: depthwiseconv3d

extern "C" {
    fn ptx_depthwiseconv3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// depthwiseconv3d activation (f32)
pub fn depthwiseconv3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_depthwiseconv3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distance

extern "C" {
    fn ptx_distance_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distance activation (f32)
pub fn distance(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distance_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: cumminmax

extern "C" {
    fn ptx_cumminmax_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// cumminmax activation (f32)
pub fn cumminmax(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_cumminmax_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: upsamplelinear1d

extern "C" {
    fn ptx_upsamplelinear1d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// upsamplelinear1d activation (f32)
pub fn upsamplelinear1d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_upsamplelinear1d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sparsemm

extern "C" {
    fn ptx_sparsemm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sparsemm activation (f32)
pub fn sparsemm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sparsemm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: averagepool3d

extern "C" {
    fn ptx_averagepool3d_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// averagepool3d activation (f32)
pub fn averagepool3d(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_averagepool3d_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: maxminelementwise

extern "C" {
    fn ptx_maxminelementwise_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// maxminelementwise activation (f32)
pub fn maxminelementwise(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_maxminelementwise_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: glu

extern "C" {
    fn ptx_glu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// glu activation (f32)
pub fn glu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_glu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: group_norm_kernel

extern "C" {
    fn ptx_group_norm_kernel_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// group_norm_kernel activation (f32)
pub fn group_norm_kernel(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_group_norm_kernel_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: functionofamatrixutils

extern "C" {
    fn ptx_functionofamatrixutils_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// functionofamatrixutils activation (f32)
pub fn functionofamatrixutils(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_functionofamatrixutils_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: int8mm

extern "C" {
    fn ptx_int8mm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// int8mm activation (f32)
pub fn int8mm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_int8mm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: tensorshape

extern "C" {
    fn ptx_tensorshape_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// tensorshape activation (f32)
pub fn tensorshape(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_tensorshape_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: foreachpointwiseop

extern "C" {
    fn ptx_foreachpointwiseop_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// foreachpointwiseop activation (f32)
pub fn foreachpointwiseop(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_foreachpointwiseop_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: linearalgebra

extern "C" {
    fn ptx_linearalgebra_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// linearalgebra activation (f32)
pub fn linearalgebra(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_linearalgebra_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: sort

extern "C" {
    fn ptx_sort_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// sort activation (f32)
pub fn sort(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_sort_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: distributionuniform

extern "C" {
    fn ptx_distributionuniform_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// distributionuniform activation (f32)
pub fn distributionuniform(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_distributionuniform_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: prelu

extern "C" {
    fn ptx_prelu_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// prelu activation (f32)
pub fn prelu(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_prelu_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

// Auto-generated FFI for: rowwisescaledmm

extern "C" {
    fn ptx_rowwisescaledmm_f32(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: *mut libc::c_void
    );
}

/// rowwisescaledmm activation (f32)
pub fn rowwisescaledmm(
    input: *const f32,
    output: *mut f32,
    numel: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        ptx_rowwisescaledmm_f32(input, output, numel, stream.raw());
    }
    Ok(())
}

