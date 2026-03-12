#![cfg(feature = "torch")]

//! GPU-Resident Dataflow Pipeline
//!
//! 5-stage vision pipeline running on independent CUDA streams with ring buffer
//! interconnects. Demonstrates true GPU-side pipelining where different stages
//! process different frames simultaneously.
//!
//! Architecture:
//!   Frame N → [Preprocess] → [Backbone] → [DetectHead] → [NMS] → [Tracker] → Results
//!              stream 0       stream 1     stream 2      stream 3   stream 4
//!
//! Tick-based scheduling dispatches stages in REVERSE order (downstream first)
//! to prevent ring buffer overflow.

use anyhow::Result;
use aten_ptx::{
    get_fragmentation, init_pytorch_tlsf_ex, num_streams, print_stats, reset_torch_stream,
    set_torch_stream, sync_all_streams,
};
use ferrite_gpu_lang::cv::{iou_xyxy, nms_cpu, yolo_decode, NmsSpec, YoloDecodeSpec};
use ferrite_gpu_lang::dataflow::{PipelineStats, RingBuffer, StageMetrics};
use std::time::Instant;
use tch::{Device, Kind, Tensor};

const NUM_FRAMES: usize = 500;
const INPUT_H: i64 = 480;
const INPUT_W: i64 = 640;
const CROP_SIZE: i64 = 224;
const NUM_STAGES: usize = 5;
const RING_CAPACITY: usize = 4;

// ---------------------------------------------------------------------------
// Stage 0: Preprocess — synthetic [3,H,W] → [1,3,224,224]
// ---------------------------------------------------------------------------
struct Preprocess {
    mean: Tensor,
    std: Tensor,
}

impl Preprocess {
    fn new(device: Device) -> Self {
        let mean = Tensor::from_slice(&[0.485f32, 0.456, 0.406])
            .to_device(device)
            .view([1, 3, 1, 1]);
        let std = Tensor::from_slice(&[0.229f32, 0.224, 0.225])
            .to_device(device)
            .view([1, 3, 1, 1]);
        Self { mean, std }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [3, H, W] → unsqueeze → bilinear resize → normalize
        let x = input.unsqueeze(0); // [1, 3, H, W]
        let x = x.f_upsample_bilinear2d([CROP_SIZE, CROP_SIZE], false, None, None)
            .unwrap();
        (&x - &self.mean) / &self.std
    }
}

// ---------------------------------------------------------------------------
// Stage 1: Backbone — conv(3→16,s2) → relu → conv(16→32,s2) → relu →
//                     conv(32→64,s2) → relu → adaptive_avg_pool2d([7,7])
// ---------------------------------------------------------------------------
struct Backbone {
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
}

impl Backbone {
    fn new(device: Device) -> Self {
        let scale = |fan_in: i64| (2.0 / fan_in as f64).sqrt();
        Self {
            w1: Tensor::randn([16, 3, 3, 3], (Kind::Float, device)) * scale(3 * 3 * 3),
            w2: Tensor::randn([32, 16, 3, 3], (Kind::Float, device)) * scale(16 * 3 * 3),
            w3: Tensor::randn([64, 32, 3, 3], (Kind::Float, device)) * scale(32 * 3 * 3),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x
            .f_conv2d::<Tensor>(&self.w1, None, [2, 2], [1, 1], [1, 1], 1)
            .unwrap()
            .relu();
        let x = x
            .f_conv2d::<Tensor>(&self.w2, None, [2, 2], [1, 1], [1, 1], 1)
            .unwrap()
            .relu();
        let x = x
            .f_conv2d::<Tensor>(&self.w3, None, [2, 2], [1, 1], [1, 1], 1)
            .unwrap()
            .relu();
        x.adaptive_avg_pool2d([7, 7]) // [1, 64, 7, 7]
    }
}

// ---------------------------------------------------------------------------
// Stage 2: DetectHead — flatten → matmul → reshape [49, 6]
//   [1,64,7,7] → [1, 3136] → matmul [3136, 294] → [1, 294] → [49, 6]
// ---------------------------------------------------------------------------
struct DetectHead {
    proj_w: Tensor,
    proj_b: Tensor,
}

impl DetectHead {
    fn new(device: Device) -> Self {
        let in_features = 64 * 7 * 7; // 3136
        let out_features = 49 * 6; // 294
        let scale = (2.0 / in_features as f64).sqrt();
        Self {
            proj_w: Tensor::randn([in_features, out_features], (Kind::Float, device)) * scale,
            proj_b: Tensor::zeros([out_features], (Kind::Float, device)),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [1, 64, 7, 7]
        let flat = x.view([1, -1]); // [1, 3136]
        let out = flat.matmul(&self.proj_w) + &self.proj_b; // [1, 294]
        out.view([49, 6]) // [49, 6] — each row is [x, y, w, h, obj, cls]
    }
}

// ---------------------------------------------------------------------------
// Stage 3: NMS — yolo_decode + nms_cpu
// ---------------------------------------------------------------------------
struct NmsStage;

impl NmsStage {
    fn forward(&self, raw_dets: &Tensor) -> Tensor {
        // raw_dets: [49, 6] — treat as [x,y,w,h,obj_logit,cls_logit]
        let decoded = yolo_decode(
            raw_dets,
            YoloDecodeSpec {
                input_is_xywh: true,
                score_threshold: 0.01,
            },
        )
        .unwrap_or_else(|_| Tensor::zeros([0, 6], (Kind::Float, raw_dets.device())));

        if decoded.size()[0] == 0 {
            return decoded;
        }

        nms_cpu(
            &decoded,
            NmsSpec {
                iou_threshold: 0.45,
                score_threshold: 0.01,
                max_detections: 20,
                class_aware: false,
            },
        )
        .unwrap_or_else(|_| Tensor::zeros([0, 6], (Kind::Float, raw_dets.device())))
    }
}

// ---------------------------------------------------------------------------
// Stage 4: Tracker — greedy IoU matching with persistent track state
// ---------------------------------------------------------------------------
struct Track {
    id: u64,
    bbox: [f32; 4], // x1, y1, x2, y2
    score: f32,
    class_id: f32,
    age: u32,
    hits: u32,
}

struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
    max_age: u32,
    iou_threshold: f32,
}

impl Tracker {
    fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age: 5,
            iou_threshold: 0.3,
        }
    }

    fn forward(&mut self, dets: &Tensor, device: Device) -> Tensor {
        let n = dets.size()[0] as usize;

        // Extract detections to CPU
        let det_cpu = dets
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();
        let flat: Vec<f32> = if n > 0 {
            Vec::<f32>::try_from(det_cpu.f_view([-1]).unwrap()).unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut det_boxes: Vec<[f32; 6]> = Vec::with_capacity(n);
        for i in 0..n {
            let r = &flat[i * 6..(i + 1) * 6];
            det_boxes.push([r[0], r[1], r[2], r[3], r[4], r[5]]);
        }

        // Greedy IoU matching: match detections to existing tracks
        let mut matched_det = vec![false; n];
        let mut matched_track = vec![false; self.tracks.len()];

        for (ti, track) in self.tracks.iter().enumerate() {
            let track_box: [f32; 6] = [
                track.bbox[0],
                track.bbox[1],
                track.bbox[2],
                track.bbox[3],
                track.score,
                track.class_id,
            ];
            let mut best_iou = 0.0f32;
            let mut best_di = None;
            for (di, det) in det_boxes.iter().enumerate() {
                if matched_det[di] {
                    continue;
                }
                let iou = iou_xyxy(&track_box, det);
                if iou > best_iou && iou > self.iou_threshold {
                    best_iou = iou;
                    best_di = Some(di);
                }
            }
            if let Some(di) = best_di {
                matched_det[di] = true;
                matched_track[ti] = true;
            }
        }

        // Update matched tracks
        for (ti, track) in self.tracks.iter_mut().enumerate() {
            if matched_track[ti] {
                // Find which detection matched (repeat logic — simple approach)
                let track_box: [f32; 6] = [
                    track.bbox[0],
                    track.bbox[1],
                    track.bbox[2],
                    track.bbox[3],
                    track.score,
                    track.class_id,
                ];
                for det in &det_boxes {
                    let iou = iou_xyxy(&track_box, det);
                    if iou > self.iou_threshold {
                        track.bbox = [det[0], det[1], det[2], det[3]];
                        track.score = det[4];
                        track.class_id = det[5];
                        track.age = 0;
                        track.hits += 1;
                        break;
                    }
                }
            } else {
                track.age += 1;
            }
        }

        // Remove dead tracks
        self.tracks.retain(|t| t.age <= self.max_age);

        // Create new tracks from unmatched detections
        for (di, det) in det_boxes.iter().enumerate() {
            if !matched_det[di] {
                self.tracks.push(Track {
                    id: self.next_id,
                    bbox: [det[0], det[1], det[2], det[3]],
                    score: det[4],
                    class_id: det[5],
                    age: 0,
                    hits: 1,
                });
                self.next_id += 1;
            }
        }

        // Build output: [M, 7] — [x1, y1, x2, y2, score, class_id, track_id]
        if self.tracks.is_empty() {
            return Tensor::zeros([0, 7], (Kind::Float, device));
        }

        let mut out = Vec::with_capacity(self.tracks.len() * 7);
        for track in &self.tracks {
            out.push(track.bbox[0]);
            out.push(track.bbox[1]);
            out.push(track.bbox[2]);
            out.push(track.bbox[3]);
            out.push(track.score);
            out.push(track.class_id);
            out.push(track.id as f32);
        }

        Tensor::from_slice(&out)
            .view([-1, 7])
            .to_device(device)
    }

    fn active_count(&self) -> usize {
        self.tracks.len()
    }
}

// ---------------------------------------------------------------------------
// Sequential execution: all stages on stream 0, one frame at a time
// ---------------------------------------------------------------------------
fn run_sequential(
    device: Device,
    preprocess: &Preprocess,
    backbone: &Backbone,
    detect_head: &DetectHead,
    nms_stage: &NmsStage,
    tracker: &mut Tracker,
    frames: &[Tensor],
    metrics: &mut [StageMetrics; NUM_STAGES],
) -> u64 {
    let mut result_count = 0u64;

    for frame in frames {
        let t0 = Instant::now();
        let preprocessed = preprocess.forward(frame);
        metrics[0].record(t0.elapsed().as_micros());

        let t1 = Instant::now();
        let features = backbone.forward(&preprocessed);
        metrics[1].record(t1.elapsed().as_micros());

        let t2 = Instant::now();
        let raw_dets = detect_head.forward(&features);
        metrics[2].record(t2.elapsed().as_micros());

        let t3 = Instant::now();
        let nms_dets = nms_stage.forward(&raw_dets);
        metrics[3].record(t3.elapsed().as_micros());

        let t4 = Instant::now();
        let _tracked = tracker.forward(&nms_dets, device);
        metrics[4].record(t4.elapsed().as_micros());

        result_count += 1;
    }

    result_count
}

// ---------------------------------------------------------------------------
// Pipelined execution: 5 stages on 5 streams, tick-based reverse dispatch
// ---------------------------------------------------------------------------
fn run_pipelined(
    device: Device,
    preprocess: &Preprocess,
    backbone: &Backbone,
    detect_head: &DetectHead,
    nms_stage: &NmsStage,
    tracker: &mut Tracker,
    frames: &[Tensor],
    metrics: &mut [StageMetrics; NUM_STAGES],
) -> PipelineStats {
    let num_frames = frames.len();
    let warmup_ticks = (NUM_STAGES - 1) as u64; // 4 ticks to fill pipeline

    // Ring buffers between stages
    let mut buf_01 = RingBuffer::new(RING_CAPACITY); // preprocess → backbone
    let mut buf_12 = RingBuffer::new(RING_CAPACITY); // backbone → detect_head
    let mut buf_23 = RingBuffer::new(RING_CAPACITY); // detect_head → nms
    let mut buf_34 = RingBuffer::new(RING_CAPACITY); // nms → tracker

    let mut frame_idx = 0usize;
    let mut result_frames = 0u64;
    let total_ticks = num_frames + warmup_ticks as usize;

    let overall_start = Instant::now();

    for _tick in 0..total_ticks {
        // Dispatch in REVERSE order: downstream first, upstream last

        // Stage 4: Tracker (stream 4)
        if let Some(nms_out) = buf_34.pop() {
            set_torch_stream(4);
            let t = Instant::now();
            let _tracked = tracker.forward(&nms_out, device);
            metrics[4].record(t.elapsed().as_micros());
            result_frames += 1;
        }

        // Stage 3: NMS (stream 3)
        if let Some(raw_dets) = buf_23.pop() {
            set_torch_stream(3);
            let t = Instant::now();
            let nms_out = nms_stage.forward(&raw_dets);
            metrics[3].record(t.elapsed().as_micros());
            buf_34.push(nms_out);
        }

        // Stage 2: DetectHead (stream 2)
        if let Some(features) = buf_12.pop() {
            set_torch_stream(2);
            let t = Instant::now();
            let raw_dets = detect_head.forward(&features);
            metrics[2].record(t.elapsed().as_micros());
            buf_23.push(raw_dets);
        }

        // Stage 1: Backbone (stream 1)
        if let Some(preprocessed) = buf_01.pop() {
            set_torch_stream(1);
            let t = Instant::now();
            let features = backbone.forward(&preprocessed);
            metrics[1].record(t.elapsed().as_micros());
            buf_12.push(features);
        }

        // Stage 0: Preprocess (stream 0) — ingest new frame
        if frame_idx < num_frames {
            set_torch_stream(0);
            let t = Instant::now();
            let preprocessed = preprocess.forward(&frames[frame_idx]);
            metrics[0].record(t.elapsed().as_micros());
            buf_01.push(preprocessed);
            frame_idx += 1;
        }

        // Barrier: sync all streams before next tick
        sync_all_streams();
        reset_torch_stream();
    }

    let total_time_ms = overall_start.elapsed().as_secs_f64() * 1000.0;

    PipelineStats {
        total_ticks: total_ticks as u64,
        total_time_ms,
        warmup_ticks,
        result_frames,
        stages: Vec::new(), // metrics are tracked externally
    }
}

fn main() -> Result<()> {
    let device_id = 0i32;
    let device = Device::Cuda(device_id as usize);

    println!("=== GPU-Resident Dataflow Pipeline ===");
    println!(
        "frames={} input={}x{} crop={} stages={} ring_cap={}",
        NUM_FRAMES, INPUT_H, INPUT_W, CROP_SIZE, NUM_STAGES, RING_CAPACITY
    );

    // Init runtime with enough streams (at least 5 for the pipeline)
    init_pytorch_tlsf_ex(device_id, 0.70, 8).map_err(|e| anyhow::anyhow!("{}", e))?;
    let active_streams = num_streams();
    println!("PTX-OS streams: {}", active_streams);
    assert!(
        active_streams >= NUM_STAGES,
        "need at least {} streams, got {}",
        NUM_STAGES,
        active_streams
    );

    let _guard = tch::no_grad_guard();

    // --- Initialize persistent stage state ---
    println!("\n--- Initializing stages ---");
    let init_start = Instant::now();

    let preprocess = Preprocess::new(device);
    let backbone = Backbone::new(device);
    let detect_head = DetectHead::new(device);
    let nms_stage = NmsStage;

    sync_all_streams();
    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    let frag_after_init = get_fragmentation();
    println!("  Init: {:.1}ms, fragmentation={:.6}", init_ms, frag_after_init);

    // --- Generate synthetic frames ---
    println!("  Generating {} synthetic frames...", NUM_FRAMES);
    let frames: Vec<Tensor> = (0..NUM_FRAMES)
        .map(|_| Tensor::rand([3, INPUT_H, INPUT_W], (Kind::Float, device)))
        .collect();
    sync_all_streams();

    // =========================================================================
    // Mode 1: Sequential
    // =========================================================================
    println!("\n--- Mode 1: Sequential (single stream) ---");
    let mut seq_tracker = Tracker::new();
    let mut seq_metrics = [
        StageMetrics::new("preprocess"),
        StageMetrics::new("backbone"),
        StageMetrics::new("detect_head"),
        StageMetrics::new("nms"),
        StageMetrics::new("tracker"),
    ];

    set_torch_stream(0);
    let seq_start = Instant::now();
    let seq_results = run_sequential(
        device,
        &preprocess,
        &backbone,
        &detect_head,
        &nms_stage,
        &mut seq_tracker,
        &frames,
        &mut seq_metrics,
    );
    sync_all_streams();
    reset_torch_stream();
    let seq_time_ms = seq_start.elapsed().as_secs_f64() * 1000.0;
    let seq_fps = seq_results as f64 / (seq_time_ms / 1000.0);
    println!(
        "  Sequential: {} frames in {:.1}ms = {:.1} FPS",
        seq_results, seq_time_ms, seq_fps
    );

    // =========================================================================
    // Mode 2: Pipelined
    // =========================================================================
    println!("\n--- Mode 2: Pipelined (5 streams) ---");
    let mut pipe_tracker = Tracker::new();
    let mut pipe_metrics = [
        StageMetrics::new("preprocess"),
        StageMetrics::new("backbone"),
        StageMetrics::new("detect_head"),
        StageMetrics::new("nms"),
        StageMetrics::new("tracker"),
    ];

    let pipe_stats = run_pipelined(
        device,
        &preprocess,
        &backbone,
        &detect_head,
        &nms_stage,
        &mut pipe_tracker,
        &frames,
        &mut pipe_metrics,
    );
    let pipe_fps = pipe_stats.result_frames as f64 / (pipe_stats.total_time_ms / 1000.0);
    println!(
        "  Pipelined: {} frames in {:.1}ms = {:.1} FPS ({} ticks, {} warmup)",
        pipe_stats.result_frames, pipe_stats.total_time_ms, pipe_fps,
        pipe_stats.total_ticks, pipe_stats.warmup_ticks
    );

    let speedup = pipe_fps / seq_fps;
    let vram_after = get_fragmentation();
    let vram_stable = (vram_after - frag_after_init).abs() < 0.01;
    let active_tracks = pipe_tracker.active_count();

    // --- Per-stage latency ---
    println!("\n--- Per-stage latency (pipelined) ---");
    for m in &pipe_metrics {
        println!(
            "  {:<12} avg={:.1}us  min={}us  max={}us  invocations={}",
            m.name,
            m.avg_us(),
            if m.invocations > 0 { m.min_us } else { 0 },
            m.max_us,
            m.invocations
        );
    }

    // =========================================================================
    // RESULT output
    // =========================================================================
    println!("\n--- Results ---");
    println!("RESULT script=dataflow_pipeline");
    println!("RESULT num_frames={}", NUM_FRAMES);
    println!("RESULT input_size={}", CROP_SIZE);
    println!("RESULT stages={}", NUM_STAGES);
    println!("RESULT sequential_fps={:.1}", seq_fps);
    println!("RESULT pipeline_fps={:.1}", pipe_fps);
    println!("RESULT speedup={:.2}x", speedup);
    println!("RESULT pipeline_result_frames={}", pipe_stats.result_frames);
    println!("RESULT preprocess_avg_us={:.1}", pipe_metrics[0].avg_us());
    println!("RESULT backbone_avg_us={:.1}", pipe_metrics[1].avg_us());
    println!("RESULT detect_head_avg_us={:.1}", pipe_metrics[2].avg_us());
    println!("RESULT nms_avg_us={:.1}", pipe_metrics[3].avg_us());
    println!("RESULT tracker_avg_us={:.1}", pipe_metrics[4].avg_us());
    println!("RESULT fragmentation_after_init={:.6}", frag_after_init);
    println!("RESULT vram_after_pipeline={:.6}", vram_after);
    println!("RESULT vram_stable={}", vram_stable);
    println!("RESULT active_tracks={}", active_tracks);

    println!("\n--- TLSF Pool State ---");
    print_stats();

    println!(
        "\nKey insight: pipelined={:.1} FPS vs sequential={:.1} FPS → {:.2}x speedup.",
        pipe_fps, seq_fps, speedup
    );
    println!(
        "  {} frames processed with {} active tracks.",
        pipe_stats.result_frames, active_tracks
    );
    println!(
        "  VRAM stable: frag {:.6} → {:.6} (stable={}).",
        frag_after_init, vram_after, vram_stable
    );

    Ok(())
}
