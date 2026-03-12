use crate::capture::frame::Frame;
use crate::runtime::context::HasAllocator;
use crate::runtime::tensor::CpuTensor;
use crate::{HostTensor, LangError, Result};

/// Normalisation applied when converting u8 frames to f32 tensors.
#[derive(Clone, Debug)]
pub enum Normalize {
    /// Divide by 255.0 → [0, 1].
    UnitRange,
    /// ImageNet-style: (x/255.0 - mean) / std, per channel.
    ImageNet {
        mean: [f32; 3],
        std: [f32; 3],
    },
    /// Cast u8 → f32 without scaling.
    None,
}

impl Normalize {
    /// Standard ImageNet normalisation constants.
    pub fn imagenet() -> Self {
        Normalize::ImageNet {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

/// Convert a frame to a `HostTensor` suitable for `GpuLangRuntime::execute()`.
///
/// Performs a **fused HWC→CHW transpose + u8→f32 normalisation** in a single
/// pass.  Output shape: `[1, C, H, W]`.
pub fn frame_to_host_tensor(frame: &Frame, normalize: &Normalize) -> Result<HostTensor> {
    let (data, shape) = convert_hwc_to_chw_f32(frame, normalize)?;
    HostTensor::new(shape, data)
}

/// Convert a frame to a TLSF-backed `CpuTensor<f32>` (for `to_gpu()`).
///
/// Same fused HWC→CHW + normalise pass; output shape `[1, C, H, W]`.
pub fn frame_to_cpu_tensor(ctx: &(impl HasAllocator + ?Sized), frame: &Frame, normalize: &Normalize) -> Result<CpuTensor<f32>> {
    let (data, shape) = convert_hwc_to_chw_f32(frame, normalize)?;

    // Build via TLSF allocator for O(1) free on drop.
    CpuTensor::with_allocator(ctx, shape, |i| data[i])
}

/// Core conversion: HWC u8 → CHW f32, single pass.
fn convert_hwc_to_chw_f32(frame: &Frame, normalize: &Normalize) -> Result<(Vec<f32>, Vec<usize>)> {
    let w = frame.width();
    let h = frame.height();
    let c = frame.format().channels();
    let src = frame.as_slice();

    if src.len() < h * w * c {
        return Err(LangError::Capture {
            message: format!(
                "frame buffer too small for {}x{}x{}: got {} bytes",
                h, w, c, src.len()
            ),
        });
    }

    let numel = c * h * w;
    let mut out = vec![0.0f32; numel];

    match normalize {
        Normalize::UnitRange => {
            let scale = 1.0 / 255.0;
            for y in 0..h {
                let row_off = y * w * c;
                for x in 0..w {
                    let px = row_off + x * c;
                    for ch in 0..c {
                        // CHW layout: out[ch][y][x]
                        out[ch * h * w + y * w + x] = src[px + ch] as f32 * scale;
                    }
                }
            }
        }
        Normalize::ImageNet { mean, std } => {
            if c != 3 {
                return Err(LangError::Capture {
                    message: format!(
                        "ImageNet normalisation requires 3 channels, got {}",
                        c
                    ),
                });
            }
            for y in 0..h {
                let row_off = y * w * 3;
                for x in 0..w {
                    let px = row_off + x * 3;
                    for ch in 0..3 {
                        let val = src[px + ch] as f32 / 255.0;
                        out[ch * h * w + y * w + x] = (val - mean[ch]) / std[ch];
                    }
                }
            }
        }
        Normalize::None => {
            for y in 0..h {
                let row_off = y * w * c;
                for x in 0..w {
                    let px = row_off + x * c;
                    for ch in 0..c {
                        out[ch * h * w + y * w + x] = src[px + ch] as f32;
                    }
                }
            }
        }
    }

    let shape = vec![1, c, h, w];
    Ok((out, shape))
}
