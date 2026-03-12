use crate::capture::frame::{Frame, FrameMeta, PixelFormat};
use crate::runtime::context::HasAllocator;
use crate::{LangError, Result};

/// Bilinear resize. Returns a new Frame with updated FrameMeta.
pub fn resize(ctx: &(impl HasAllocator + ?Sized), frame: &Frame, dst_w: u32, dst_h: u32) -> Result<Frame> {
    let src_w = frame.width();
    let src_h = frame.height();
    let channels = frame.format().channels();
    let src = frame.as_slice();

    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut dst = vec![0u8; dw * dh * channels];

    let x_ratio = src_w as f32 / dw as f32;
    let y_ratio = src_h as f32 / dh as f32;

    for dy in 0..dh {
        let sy = dy as f32 * y_ratio;
        let sy0 = (sy as usize).min(src_h.saturating_sub(1));
        let sy1 = (sy0 + 1).min(src_h.saturating_sub(1));
        let fy = sy - sy0 as f32;

        for dx in 0..dw {
            let sx = dx as f32 * x_ratio;
            let sx0 = (sx as usize).min(src_w.saturating_sub(1));
            let sx1 = (sx0 + 1).min(src_w.saturating_sub(1));
            let fx = sx - sx0 as f32;

            for c in 0..channels {
                let p00 = src[sy0 * src_w * channels + sx0 * channels + c] as f32;
                let p10 = src[sy0 * src_w * channels + sx1 * channels + c] as f32;
                let p01 = src[sy1 * src_w * channels + sx0 * channels + c] as f32;
                let p11 = src[sy1 * src_w * channels + sx1 * channels + c] as f32;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                dst[dy * dw * channels + dx * channels + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    let meta = FrameMeta {
        width: dw,
        height: dh,
        format: frame.format(),
        frame_index: frame.meta().frame_index,
        timestamp_us: frame.meta().timestamp_us,
    };

    Frame::from_bytes(ctx, meta, &dst)
}

/// Center crop a region from the frame.
pub fn crop(ctx: &(impl HasAllocator + ?Sized), frame: &Frame, x: u32, y: u32, w: u32, h: u32) -> Result<Frame> {
    let src_w = frame.width();
    let src_h = frame.height();
    let channels = frame.format().channels();
    let src = frame.as_slice();

    let cx = x as usize;
    let cy = y as usize;
    let cw = w as usize;
    let ch = h as usize;

    if cx + cw > src_w || cy + ch > src_h {
        return Err(LangError::Capture {
            message: format!(
                "crop region ({},{},{},{}) exceeds frame bounds ({}x{})",
                cx, cy, cw, ch, src_w, src_h
            ),
        });
    }

    let mut dst = vec![0u8; cw * ch * channels];
    for row in 0..ch {
        let src_off = (cy + row) * src_w * channels + cx * channels;
        let dst_off = row * cw * channels;
        dst[dst_off..dst_off + cw * channels]
            .copy_from_slice(&src[src_off..src_off + cw * channels]);
    }

    let meta = FrameMeta {
        width: cw,
        height: ch,
        format: frame.format(),
        frame_index: frame.meta().frame_index,
        timestamp_us: frame.meta().timestamp_us,
    };

    Frame::from_bytes(ctx, meta, &dst)
}

/// Letterbox: resize preserving aspect ratio, pad remaining area with `pad_value`.
pub fn letterbox(ctx: &(impl HasAllocator + ?Sized), frame: &Frame, dst_w: u32, dst_h: u32, pad_value: u8) -> Result<Frame> {
    let src_w = frame.width() as f32;
    let src_h = frame.height() as f32;
    let dw = dst_w as f32;
    let dh = dst_h as f32;

    let scale = (dw / src_w).min(dh / src_h);
    let new_w = (src_w * scale).round() as u32;
    let new_h = (src_h * scale).round() as u32;

    // Resize to fit
    let resized = resize(ctx, frame, new_w, new_h)?;

    // Create padded output
    let channels = frame.format().channels();
    let out_size = dst_w as usize * dst_h as usize * channels;
    let mut dst = vec![pad_value; out_size];

    // Center the resized image in the output
    let x_off = ((dst_w - new_w) / 2) as usize;
    let y_off = ((dst_h - new_h) / 2) as usize;
    let resized_data = resized.as_slice();
    let nw = new_w as usize;
    let nh = new_h as usize;
    let dw_usize = dst_w as usize;

    for row in 0..nh {
        let src_off = row * nw * channels;
        let dst_off = (y_off + row) * dw_usize * channels + x_off * channels;
        dst[dst_off..dst_off + nw * channels]
            .copy_from_slice(&resized_data[src_off..src_off + nw * channels]);
    }

    let meta = FrameMeta {
        width: dst_w as usize,
        height: dst_h as usize,
        format: frame.format(),
        frame_index: frame.meta().frame_index,
        timestamp_us: frame.meta().timestamp_us,
    };

    Frame::from_bytes(ctx, meta, &dst)
}

/// Convert pixel format (e.g. Rgb8 → Gray8, Bgr8 → Rgb8).
pub fn convert_format(ctx: &(impl HasAllocator + ?Sized), frame: &Frame, target: PixelFormat) -> Result<Frame> {
    let src_fmt = frame.format();
    if src_fmt == target {
        // No conversion needed — copy the frame
        let meta = frame.meta().clone();
        return Frame::from_bytes(ctx, meta, frame.as_slice());
    }

    let w = frame.width();
    let h = frame.height();
    let src = frame.as_slice();

    let dst_channels = target.channels();
    let mut dst = vec![0u8; w * h * dst_channels];

    match (src_fmt, target) {
        (PixelFormat::Rgb8, PixelFormat::Gray8)
        | (PixelFormat::Bgr8, PixelFormat::Gray8) => {
            // Luminance: 0.299*R + 0.587*G + 0.114*B
            let (r_idx, g_idx, b_idx) = if src_fmt == PixelFormat::Rgb8 {
                (0, 1, 2)
            } else {
                (2, 1, 0)
            };
            for i in 0..(w * h) {
                let off = i * 3;
                let r = src[off + r_idx] as f32;
                let g = src[off + g_idx] as f32;
                let b = src[off + b_idx] as f32;
                dst[i] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            }
        }
        (PixelFormat::Gray8, PixelFormat::Rgb8)
        | (PixelFormat::Gray8, PixelFormat::Bgr8) => {
            for i in 0..(w * h) {
                let v = src[i];
                let off = i * 3;
                dst[off] = v;
                dst[off + 1] = v;
                dst[off + 2] = v;
            }
        }
        (PixelFormat::Bgr8, PixelFormat::Rgb8)
        | (PixelFormat::Rgb8, PixelFormat::Bgr8) => {
            // Swap R and B channels
            for i in 0..(w * h) {
                let off = i * 3;
                dst[off] = src[off + 2];
                dst[off + 1] = src[off + 1];
                dst[off + 2] = src[off];
            }
        }
        _ => {
            return Err(LangError::Capture {
                message: format!("unsupported format conversion: {:?} → {:?}", src_fmt, target),
            });
        }
    }

    let meta = FrameMeta {
        width: w,
        height: h,
        format: target,
        frame_index: frame.meta().frame_index,
        timestamp_us: frame.meta().timestamp_us,
    };

    Frame::from_bytes(ctx, meta, &dst)
}
