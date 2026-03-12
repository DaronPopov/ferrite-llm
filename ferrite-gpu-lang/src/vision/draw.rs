use crate::capture::frame::Frame;
use super::bbox::{BoundingBox, Detection};
use super::tracker::Track;

/// Draw a horizontal line on the frame (in-place). Clips to frame bounds.
fn draw_hline(data: &mut [u8], width: usize, channels: usize, x1: i32, x2: i32, y: i32, height: i32, color: [u8; 3]) {
    if y < 0 || y >= height {
        return;
    }
    let x_start = x1.max(0) as usize;
    let x_end = (x2.min(width as i32 - 1) + 1) as usize;
    let row = y as usize;
    for x in x_start..x_end {
        let off = (row * width + x) * channels;
        for c in 0..channels.min(3) {
            data[off + c] = color[c];
        }
    }
}

/// Draw a vertical line on the frame (in-place). Clips to frame bounds.
fn draw_vline(data: &mut [u8], width: usize, channels: usize, x: i32, y1: i32, y2: i32, height: i32, color: [u8; 3]) {
    if x < 0 || x >= width as i32 {
        return;
    }
    let y_start = y1.max(0) as usize;
    let y_end = (y2.min(height as i32 - 1) + 1) as usize;
    let col = x as usize;
    for y in y_start..y_end {
        let off = (y * width + col) * channels;
        for c in 0..channels.min(3) {
            data[off + c] = color[c];
        }
    }
}

/// Draw a bounding box outline on the frame (in-place).
pub fn draw_bbox(frame: &mut Frame, bbox: &BoundingBox, color: [u8; 3], thickness: u32) {
    let w = frame.width();
    let h = frame.height();
    let channels = frame.format().channels();
    let data = frame.as_mut_slice();

    let x1 = bbox.x1 as i32;
    let y1 = bbox.y1 as i32;
    let x2 = bbox.x2 as i32;
    let y2 = bbox.y2 as i32;
    let t = thickness as i32;
    let hi = h as i32;

    // Draw rectangle with thickness
    for i in 0..t {
        // Top and bottom edges
        draw_hline(data, w, channels, x1, x2, y1 + i, hi, color);
        draw_hline(data, w, channels, x1, x2, y2 - i, hi, color);
        // Left and right edges
        draw_vline(data, w, channels, x1 + i, y1, y2, hi, color);
        draw_vline(data, w, channels, x2 - i, y1, y2, hi, color);
    }
}

/// Draw bounding boxes from detections with class-colored outlines.
pub fn draw_detections(frame: &mut Frame, detections: &[Detection], palette: &[[u8; 3]]) {
    for det in detections {
        let color = if palette.is_empty() {
            [0, 255, 0]
        } else {
            palette[det.class_id as usize % palette.len()]
        };
        draw_bbox(frame, &det.bbox, color, 2);
    }
}

/// Draw tracked objects with IDs rendered as colored marker blocks.
///
/// Each track gets a small colored marker in the top-left corner of its bounding
/// box, with the marker size encoding the track ID (visual distinction without
/// font rendering).
pub fn draw_tracks(frame: &mut Frame, tracks: &[Track], palette: &[[u8; 3]]) {
    let w = frame.width();
    let h = frame.height();
    let channels = frame.format().channels();

    for track in tracks {
        let color = if palette.is_empty() {
            [0, 255, 0]
        } else {
            palette[track.class_id as usize % palette.len()]
        };

        // Draw bounding box
        draw_bbox(frame, &track.bbox, color, 2);

        // Draw small ID marker: a filled square in the top-left corner
        // Size = 4x4 pixels, with brightness modulated by track ID
        let marker_size = 4usize;
        let mx = track.bbox.x1 as usize;
        let my = track.bbox.y1 as usize;

        // Use track ID to create a distinct marker color
        let id_r = ((track.id * 37) % 256) as u8;
        let id_g = ((track.id * 73) % 256) as u8;
        let id_b = ((track.id * 113) % 256) as u8;
        let marker_color = [id_r, id_g, id_b];

        let data = frame.as_mut_slice();
        for dy in 0..marker_size {
            for dx in 0..marker_size {
                let px = mx + dx;
                let py = my + dy;
                if px < w && py < h {
                    let off = (py * w + px) * channels;
                    for c in 0..channels.min(3) {
                        data[off + c] = marker_color[c];
                    }
                }
            }
        }
    }
}
