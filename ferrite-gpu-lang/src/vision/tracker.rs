use super::bbox::{BoundingBox, Detection};

/// A tracked object with persistent identity across frames.
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub bbox: BoundingBox,
    pub score: f32,
    pub class_id: u32,
    /// Ticks since creation.
    pub age: u32,
    /// Consecutive matched ticks.
    pub hits: u32,
    /// Consecutive unmatched ticks.
    pub misses: u32,
}

/// Configuration for the multi-object tracker.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// IoU threshold for matching detections to tracks.
    pub iou_threshold: f32,
    /// Maximum consecutive misses before a track is pruned.
    pub max_misses: u32,
    /// Minimum hits before a track is considered confirmed.
    pub min_hits: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            max_misses: 5,
            min_hits: 3,
        }
    }
}

/// Greedy IoU-based multi-object tracker.
///
/// Matches incoming detections to existing tracks using IoU, updates matched
/// tracks, creates new tracks for unmatched detections, and prunes tracks
/// that have gone unmatched for too long.
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
    config: TrackerConfig,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            config,
        }
    }

    /// Run one tracking step: match detections to tracks, update state.
    /// Returns the full set of active tracks (including unconfirmed ones).
    pub fn update(&mut self, detections: &[Detection]) -> &[Track] {
        let n_dets = detections.len();
        let n_tracks = self.tracks.len();

        let mut matched_det = vec![false; n_dets];
        let mut matched_track = vec![false; n_tracks];

        // Greedy IoU matching: for each track, find best matching detection
        for ti in 0..n_tracks {
            let mut best_iou = 0.0f32;
            let mut best_di = None;

            for di in 0..n_dets {
                if matched_det[di] {
                    continue;
                }
                let iou = self.tracks[ti].bbox.iou(&detections[di].bbox);
                if iou > best_iou && iou > self.config.iou_threshold {
                    best_iou = iou;
                    best_di = Some(di);
                }
            }

            if let Some(di) = best_di {
                matched_det[di] = true;
                matched_track[ti] = true;

                // Update matched track with detection
                self.tracks[ti].bbox = detections[di].bbox;
                self.tracks[ti].score = detections[di].score;
                self.tracks[ti].class_id = detections[di].class_id;
                self.tracks[ti].age += 1;
                self.tracks[ti].hits += 1;
                self.tracks[ti].misses = 0;
            }
        }

        // Increment misses for unmatched tracks
        for ti in 0..n_tracks {
            if !matched_track[ti] {
                self.tracks[ti].age += 1;
                self.tracks[ti].misses += 1;
            }
        }

        // Prune dead tracks
        let max_misses = self.config.max_misses;
        self.tracks.retain(|t| t.misses <= max_misses);

        // Create new tracks from unmatched detections
        for (di, det) in detections.iter().enumerate() {
            if !matched_det[di] {
                self.tracks.push(Track {
                    id: self.next_id,
                    bbox: det.bbox,
                    score: det.score,
                    class_id: det.class_id,
                    age: 0,
                    hits: 1,
                    misses: 0,
                });
                self.next_id += 1;
            }
        }

        &self.tracks
    }

    /// All active tracks (both confirmed and unconfirmed).
    pub fn active_tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Only tracks with enough consecutive hits to be considered confirmed.
    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| t.hits >= self.config.min_hits)
            .collect()
    }

    /// Reset tracker state, clearing all tracks.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: u32) -> Detection {
        Detection {
            bbox: BoundingBox::new(x1, y1, x2, y2),
            class_id,
            score,
        }
    }

    #[test]
    fn tracker_creates_new_tracks() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(50.0, 50.0, 60.0, 60.0, 0.8, 1),
        ];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].id, 1);
        assert_eq!(tracks[1].id, 2);
    }

    #[test]
    fn tracker_matches_existing() {
        let mut tracker = Tracker::new(TrackerConfig::default());

        // Frame 1: create track
        let dets1 = vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0)];
        tracker.update(&dets1);
        assert_eq!(tracker.active_tracks().len(), 1);
        let id = tracker.active_tracks()[0].id;

        // Frame 2: slightly moved detection should match
        let dets2 = vec![det(1.0, 1.0, 11.0, 11.0, 0.85, 0)];
        tracker.update(&dets2);
        assert_eq!(tracker.active_tracks().len(), 1);
        assert_eq!(tracker.active_tracks()[0].id, id); // same track
        assert_eq!(tracker.active_tracks()[0].hits, 2);
    }

    #[test]
    fn tracker_prunes_dead_tracks() {
        let mut tracker = Tracker::new(TrackerConfig {
            max_misses: 2,
            ..TrackerConfig::default()
        });

        // Create track
        tracker.update(&[det(0.0, 0.0, 10.0, 10.0, 0.9, 0)]);
        assert_eq!(tracker.active_tracks().len(), 1);

        // Miss 1
        tracker.update(&[]);
        assert_eq!(tracker.active_tracks().len(), 1);
        assert_eq!(tracker.active_tracks()[0].misses, 1);

        // Miss 2
        tracker.update(&[]);
        assert_eq!(tracker.active_tracks().len(), 1);
        assert_eq!(tracker.active_tracks()[0].misses, 2);

        // Miss 3 → pruned (misses > max_misses=2)
        tracker.update(&[]);
        assert_eq!(tracker.active_tracks().len(), 0);
    }

    #[test]
    fn tracker_confirmed_tracks() {
        let mut tracker = Tracker::new(TrackerConfig {
            min_hits: 3,
            ..TrackerConfig::default()
        });

        let d = det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        tracker.update(&[d.clone()]);
        assert_eq!(tracker.confirmed_tracks().len(), 0);

        let d2 = det(1.0, 1.0, 11.0, 11.0, 0.9, 0);
        tracker.update(&[d2.clone()]);
        assert_eq!(tracker.confirmed_tracks().len(), 0);

        let d3 = det(2.0, 2.0, 12.0, 12.0, 0.9, 0);
        tracker.update(&[d3]);
        assert_eq!(tracker.confirmed_tracks().len(), 1);
    }

    #[test]
    fn tracker_reset() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[det(0.0, 0.0, 10.0, 10.0, 0.9, 0)]);
        assert_eq!(tracker.active_tracks().len(), 1);
        tracker.reset();
        assert_eq!(tracker.active_tracks().len(), 0);
    }
}
