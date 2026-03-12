use tch::{Device, Kind, Tensor};

use crate::capture::convert::Normalize;
use crate::capture::frame::Frame;
use crate::{LangError, Result};

/// Convert a `Frame` to a `tch::Tensor`.
///
/// Performs HWC→CHW transpose + normalisation, then moves to `device`.
/// Output shape: `[1, C, H, W]`.
pub fn frame_to_tch_tensor(frame: &Frame, normalize: &Normalize, device: Device) -> Result<Tensor> {
    let w = frame.width() as i64;
    let h = frame.height() as i64;
    let c = frame.format().channels() as i64;
    let src = frame.as_slice();

    // Build tensor from raw u8 data in HWC layout.
    let hwc = Tensor::from_data_size(src, &[h, w, c], Kind::Uint8);

    // HWC → CHW: permute(2,0,1), add batch dim, cast to f32.
    let chw = hwc
        .permute([2, 0, 1])
        .unsqueeze(0)
        .to_kind(Kind::Float);

    // Apply normalisation.
    let normed = match normalize {
        Normalize::UnitRange => chw / 255.0,
        Normalize::ImageNet { mean, std } => {
            if c != 3 {
                return Err(LangError::Capture {
                    message: format!(
                        "ImageNet normalisation requires 3 channels, got {}",
                        c
                    ),
                });
            }
            let mean_t = Tensor::from_slice(&mean[..]).view([1, 3, 1, 1]);
            let std_t = Tensor::from_slice(&std[..]).view([1, 3, 1, 1]);
            (chw / 255.0 - mean_t) / std_t
        }
        Normalize::None => chw,
    };

    Ok(normed.to_device(device))
}

/// Extension trait for converting `Frame` to `tch::Tensor`.
pub trait FrameToTensor {
    fn to_tch_tensor(&self, normalize: &Normalize, device: Device) -> Result<Tensor>;
}

impl FrameToTensor for Frame {
    fn to_tch_tensor(&self, normalize: &Normalize, device: Device) -> Result<Tensor> {
        frame_to_tch_tensor(self, normalize, device)
    }
}
