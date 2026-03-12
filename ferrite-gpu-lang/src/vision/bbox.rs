/// Axis-aligned bounding box in pixel coordinates (xyxy format).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            x1: x,
            y1: y,
            x2: x + w,
            y2: y + h,
        }
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Intersection over Union with another bounding box.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter = inter_w * inter_h;

        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// A single detection: bounding box + class + confidence.
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub class_id: u32,
    pub score: f32,
}

/// Non-maximum suppression (pure Rust, no external dependencies).
///
/// Sorts detections by descending score, then greedily keeps detections
/// that don't overlap too much with already-kept ones.
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    for det in detections.iter() {
        let dominated = keep.iter().any(|kept: &Detection| {
            det.bbox.iou(&kept.bbox) > iou_threshold
        });
        if !dominated {
            keep.push(det.clone());
        }
    }
    keep
}

/// Class-aware non-maximum suppression.
///
/// Only suppresses detections within the same class.
pub fn nms_class_aware(detections: &mut Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    for det in detections.iter() {
        let dominated = keep.iter().any(|kept: &Detection| {
            kept.class_id == det.class_id && det.bbox.iou(&kept.bbox) > iou_threshold
        });
        if !dominated {
            keep.push(det.clone());
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_basics() {
        let b = BoundingBox::new(10.0, 20.0, 50.0, 80.0);
        assert_eq!(b.width(), 40.0);
        assert_eq!(b.height(), 60.0);
        assert_eq!(b.area(), 2400.0);
        assert_eq!(b.center(), (30.0, 50.0));
    }

    #[test]
    fn bbox_from_xywh() {
        let b = BoundingBox::from_xywh(10.0, 20.0, 40.0, 60.0);
        assert_eq!(b.x1, 10.0);
        assert_eq!(b.y1, 20.0);
        assert_eq!(b.x2, 50.0);
        assert_eq!(b.y2, 80.0);
    }

    #[test]
    fn bbox_iou_perfect_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(20.0, 20.0, 30.0, 30.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn bbox_iou_partial() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        // intersection: 5x5=25, union: 100+100-25=175
        let expected = 25.0 / 175.0;
        assert!((a.iou(&b) - expected).abs() < 1e-6);
    }

    #[test]
    fn nms_removes_overlapping() {
        let mut dets = vec![
            Detection { bbox: BoundingBox::new(0.0, 0.0, 10.0, 10.0), class_id: 0, score: 0.9 },
            Detection { bbox: BoundingBox::new(1.0, 1.0, 11.0, 11.0), class_id: 0, score: 0.8 },
            Detection { bbox: BoundingBox::new(50.0, 50.0, 60.0, 60.0), class_id: 0, score: 0.7 },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 2); // first and third kept
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_non_overlapping() {
        let mut dets = vec![
            Detection { bbox: BoundingBox::new(0.0, 0.0, 10.0, 10.0), class_id: 0, score: 0.9 },
            Detection { bbox: BoundingBox::new(50.0, 50.0, 60.0, 60.0), class_id: 0, score: 0.8 },
        ];
        let kept = nms(&mut dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_class_aware_different_classes() {
        let mut dets = vec![
            Detection { bbox: BoundingBox::new(0.0, 0.0, 10.0, 10.0), class_id: 0, score: 0.9 },
            Detection { bbox: BoundingBox::new(1.0, 1.0, 11.0, 11.0), class_id: 1, score: 0.8 },
        ];
        // Different classes — both should survive even though boxes overlap
        let kept = nms_class_aware(&mut dets, 0.3);
        assert_eq!(kept.len(), 2);
    }
}
