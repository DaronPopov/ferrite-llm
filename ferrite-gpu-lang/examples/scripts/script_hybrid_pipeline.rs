#![cfg(feature = "capture")]

//! Hybrid Camera Pipeline
//!
//! End-to-end demo showing all three new modules working together:
//! - `sensor::CaptureThread` — background camera I/O
//! - `vision::Tracker` — multi-object tracking with pure Rust NMS
//! - `pipeline::Pipeline3` — typed three-stage pipeline scheduler
//!
//! Architecture:
//!   CaptureThread → [Preprocess] → [Detect] → [Track] → Results
//!     background       stage 1       stage 2     stage 3
//!
//! Required features: `capture`

use std::time::{Duration, Instant};

use ferrite_gpu_lang::capture::frame::Frame;
use ferrite_gpu_lang::sensor::{CameraAdapter, CaptureThread, SensorClock};
use ferrite_gpu_lang::vision::bbox::{BoundingBox, Detection};
use ferrite_gpu_lang::vision::tracker::{Track, Tracker, TrackerConfig};
use ferrite_gpu_lang::vision::draw;
use ferrite_gpu_lang::pipeline::stage::{Pipeline3, Stage, StageMetrics};
use ferrite_gpu_lang::{cpu, HasAllocator};

const NUM_FRAMES: usize = 100;
const DETECT_W: u32 = 320;
const DETECT_H: u32 = 240;

// ---------------------------------------------------------------------------
// Stage 1: Preprocess — letterbox frame to detection input size
// ---------------------------------------------------------------------------
struct PreprocessStage;

impl Stage for PreprocessStage {
    type Input = Frame;
    type Output = Frame;

    fn name(&self) -> &str { "preprocess" }

    fn process(&mut self, input: Frame) -> Option<Frame> {
        // Letterbox resize for detection
        cpu::<Option<Frame>, std::convert::Infallible, _>(|ctx| {
            Ok(ferrite_gpu_lang::vision::ops::letterbox(ctx, &input, DETECT_W, DETECT_H, 128).ok())
        }).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Stage 2: Detect — synthetic detections from frame brightness
// ---------------------------------------------------------------------------
struct DetectStage {
    frame_count: u64,
}

impl DetectStage {
    fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl Stage for DetectStage {
    type Input = Frame;
    type Output = Vec<Detection>;

    fn name(&self) -> &str { "detect" }

    fn process(&mut self, input: Frame) -> Option<Vec<Detection>> {
        self.frame_count += 1;

        // Synthetic detection: extract "objects" from pixel data.
        // In a real pipeline, this would run a neural network.
        let data = input.as_slice();
        let w = input.width() as f32;
        let h = input.height() as f32;
        let channels = input.format().channels();

        // Compute average brightness in quadrants
        let mut dets = Vec::new();
        let quadrants = [
            (0.0, 0.0, 0.5, 0.5),
            (0.5, 0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5, 1.0),
            (0.5, 0.5, 1.0, 1.0),
        ];

        for (qi, &(fx1, fy1, fx2, fy2)) in quadrants.iter().enumerate() {
            let x1 = (fx1 * w) as usize;
            let y1 = (fy1 * h) as usize;
            let x2 = (fx2 * w) as usize;
            let y2 = (fy2 * h) as usize;

            let mut sum = 0u64;
            let mut count = 0u64;
            for y in y1..y2 {
                for x in x1..x2 {
                    let off = (y * input.width() + x) * channels;
                    if off < data.len() {
                        sum += data[off] as u64;
                        count += 1;
                    }
                }
            }

            let brightness = if count > 0 { sum as f32 / count as f32 } else { 0.0 };
            // "Detect" objects in bright quadrants
            if brightness > 100.0 {
                dets.push(Detection {
                    bbox: BoundingBox::new(
                        fx1 * w + 10.0,
                        fy1 * h + 10.0,
                        fx2 * w - 10.0,
                        fy2 * h - 10.0,
                    ),
                    class_id: qi as u32,
                    score: brightness / 255.0,
                });
            }
        }

        // Apply NMS
        let mut filtered = ferrite_gpu_lang::vision::bbox::nms(&mut dets, 0.45);
        Some(filtered)
    }
}

// ---------------------------------------------------------------------------
// Stage 3: Track — multi-object tracker
// ---------------------------------------------------------------------------
struct TrackStage {
    tracker: Tracker,
}

impl TrackStage {
    fn new() -> Self {
        Self {
            tracker: Tracker::new(TrackerConfig {
                iou_threshold: 0.3,
                max_misses: 5,
                min_hits: 2,
            }),
        }
    }
}

impl Stage for TrackStage {
    type Input = Vec<Detection>;
    type Output = Vec<Track>;

    fn name(&self) -> &str { "track" }

    fn process(&mut self, input: Vec<Detection>) -> Option<Vec<Track>> {
        self.tracker.update(&input);
        Some(self.tracker.active_tracks().to_vec())
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== Hybrid Camera Pipeline ===");
    println!("frames={} detect={}x{}", NUM_FRAMES, DETECT_W, DETECT_H);

    // --- Start background capture ---
    let camera = CameraAdapter::default_webcam().map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Camera opened: {}", camera.info().name);

    let capture = CaptureThread::spawn(camera, 4);
    println!("Capture thread started");

    // Wait briefly for first frame
    std::thread::sleep(Duration::from_millis(200));

    // --- Build 3-stage pipeline ---
    let mut pipeline = Pipeline3::new(
        PreprocessStage,
        DetectStage::new(),
        TrackStage::new(),
        4, // ring capacity
    );

    let clock = SensorClock::new();
    let mut frames_processed = 0u64;
    let mut total_detections = 0u64;

    println!("\n--- Running pipeline ---");
    let run_start = Instant::now();

    for tick in 0..NUM_FRAMES {
        // Get latest frame from capture thread
        let sample = match capture.recv_timeout(Duration::from_millis(100)) {
            Some(s) => s,
            None => {
                eprintln!("  tick {}: no frame (timeout)", tick);
                continue;
            }
        };

        let frame = sample.data;
        let ts = sample.timestamp_us;

        // Run pipeline tick
        if let Some(tracks) = pipeline.tick(frame) {
            let confirmed: Vec<_> = tracks.iter().filter(|t| t.hits >= 2).collect();
            if tick % 20 == 0 {
                println!(
                    "  tick {:3} | ts={:.1}ms | tracks={} (confirmed={}) | latency={:.1}ms",
                    tick,
                    ts as f64 / 1000.0,
                    tracks.len(),
                    confirmed.len(),
                    clock.elapsed_ms() - (ts as f64 / 1000.0),
                );
            }
            total_detections += tracks.len() as u64;
        }

        frames_processed += 1;
    }

    let total_ms = run_start.elapsed().as_secs_f64() * 1000.0;
    let fps = frames_processed as f64 / (total_ms / 1000.0);

    // --- Pipeline stats ---
    let stats = pipeline.stats();

    // --- Shutdown ---
    capture.stop();

    // --- Print TLSF stats ---
    let tlsf_stats = cpu::<_, std::convert::Infallible, _>(|ctx| {
        Ok(ctx.allocator_stats())
    }).unwrap();

    // --- Results ---
    println!("\n--- Results ---");
    println!("RESULT script=hybrid_pipeline");
    println!("RESULT frames_processed={}", frames_processed);
    println!("RESULT total_time_ms={:.1}", total_ms);
    println!("RESULT fps={:.1}", fps);
    println!("RESULT pipeline_ticks={}", stats.total_ticks);
    println!("RESULT pipeline_results={}", stats.result_frames);
    println!("RESULT total_tracks_emitted={}", total_detections);

    println!("\n--- Per-stage latency ---");
    for m in &stats.stages {
        println!(
            "  {:<12} avg={:.1}us  min={}us  max={}us  invocations={}",
            m.name,
            m.avg_us(),
            if m.invocations > 0 { m.min_us } else { 0 },
            m.max_us,
            m.invocations
        );
    }

    println!("\n--- TLSF Stats ---");
    println!("  allocated_bytes={}", tlsf_stats.allocated_bytes);
    println!("  allocation_count={}", tlsf_stats.allocation_count);
    println!("  peak_bytes={}", tlsf_stats.peak_bytes);

    println!("\nHybrid pipeline: {:.1} FPS over {} frames.", fps, frames_processed);

    Ok(())
}
