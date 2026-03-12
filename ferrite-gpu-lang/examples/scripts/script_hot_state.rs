#![cfg(feature = "torch")]

//! Persistent GPU State Across Program Phases
//!
//! Load model weights in Phase 1 ("boot"), then run multiple independent inference
//! "jobs" in Phase 2 that reuse the same GPU-resident weights without reloading.
//! Simulates zero-cold-start deployment where the GPU state persists between
//! logical program invocations.
//!
//! What this demonstrates:
//! - Phase 1: expensive weight initialization (done once)
//! - Phase 2: multiple independent inference jobs sharing persistent weights
//! - Zero weight-loading between jobs — weights never leave GPU memory
//! - TLSF pool cleanly reclaims per-job intermediates while weights stay pinned
//! - Phase 1 load time >> Phase 2 per-job time (10-100x difference)

use anyhow::Result;
use aten_ptx::{
    get_fragmentation, init_pytorch_tlsf_ex, num_streams, print_stats, reset_torch_stream,
    set_torch_stream, sync_all_streams,
};
use std::time::Instant;
use tch::{Device, Kind, Tensor};

struct Config {
    hidden: i64,
    num_layers: usize,
    num_jobs: usize,
    batch_size: i64,
    streams: u32,
    wave_streams: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: 2048,
            num_layers: 6,
            num_jobs: 20,
            batch_size: 16,
            streams: 32,
            wave_streams: 8,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hidden" => {
                cfg.hidden = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--num-layers" => {
                cfg.num_layers = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--num-jobs" => {
                cfg.num_jobs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--batch-size" => {
                cfg.batch_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--streams" => {
                cfg.streams = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--wave-streams" => {
                cfg.wave_streams = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    cfg
}

fn main() -> Result<()> {
    let cfg = parse_args();
    let device_id = 0i32;
    let device = Device::Cuda(device_id as usize);

    println!("=== Persistent GPU State Across Phases ===");
    println!(
        "hidden={} layers={} jobs={} batch={} streams={} wave={}",
        cfg.hidden, cfg.num_layers, cfg.num_jobs, cfg.batch_size, cfg.streams, cfg.wave_streams
    );

    // --- Init runtime + TLSF pool ---
    init_pytorch_tlsf_ex(device_id, 0.70, cfg.streams).map_err(|e| anyhow::anyhow!("{}", e))?;

    let active_streams = num_streams();
    println!("PTX-OS streams: {}", active_streams);
    println!("script=hot_state");

    let _guard = tch::no_grad_guard();

    // =========================================================================
    // Phase 1 — "Boot": Load and initialize model weights (timed)
    // =========================================================================
    println!("\n--- Phase 1: Boot (weight initialization) ---");
    let boot_start = Instant::now();

    // Allocate num_layers weight matrices (hidden x hidden) + biases
    let weights: Vec<Tensor> = (0..cfg.num_layers)
        .map(|layer_idx| {
            // Synthetic "trained" values: Xavier-like initialization with layer-dependent scale
            let scale = 1.0 / (cfg.hidden as f64).sqrt();
            let w = Tensor::randn([cfg.hidden, cfg.hidden], (Kind::Float, device)) * scale;
            // Simulate "training" by adding a structured pattern per layer
            let pattern = Tensor::ones([cfg.hidden, cfg.hidden], (Kind::Float, device))
                * (0.01 * (layer_idx as f64 + 1.0));
            &w + &pattern
        })
        .collect();

    let biases: Vec<Tensor> = (0..cfg.num_layers)
        .map(|layer_idx| {
            Tensor::ones([cfg.hidden], (Kind::Float, device)) * (0.001 * (layer_idx as f64 + 1.0))
        })
        .collect();

    sync_all_streams();
    let boot_time = boot_start.elapsed();
    let frag_after_boot = get_fragmentation();

    println!(
        "  Boot complete: {} layers loaded in {:.1}ms",
        cfg.num_layers,
        boot_time.as_secs_f64() * 1000.0
    );
    println!("  Fragmentation after boot: {:.6}", frag_after_boot);

    // =========================================================================
    // Phase 2 — "Jobs": Run independent inference tasks (timed individually)
    // =========================================================================
    println!("\n--- Phase 2: Jobs (inference with persistent weights) ---");

    let mut job_latencies_ms: Vec<f64> = Vec::with_capacity(cfg.num_jobs);
    let mut job_frags: Vec<f64> = Vec::with_capacity(cfg.num_jobs);
    let mut output_checksums: Vec<f64> = Vec::with_capacity(cfg.num_jobs);

    let jobs_start = Instant::now();
    let mut job_base = 0usize;

    while job_base < cfg.num_jobs {
        let wave = (cfg.num_jobs - job_base).min(cfg.wave_streams);

        for local_idx in 0..wave {
            let job_idx = job_base + local_idx;
            let stream_id = job_idx % active_streams;
            set_torch_stream(stream_id);

            let job_start = Instant::now();

            // Generate new random input batch for this job
            let mut x = Tensor::randn(
                [cfg.batch_size, cfg.hidden],
                (Kind::Float, device),
            );

            // Forward pass through all layers (matmul → relu chain)
            // Weights are PERSISTENT — shared across all jobs, never reloaded
            for layer in 0..cfg.num_layers {
                x = x.matmul(&weights[layer])
                    .f_add(&biases[layer])
                    .unwrap()
                    .relu();
            }

            // Read output to CPU (simulating result extraction)
            let checksum = x.mean(Kind::Float).double_value(&[]);
            output_checksums.push(checksum);

            let job_latency = job_start.elapsed();
            job_latencies_ms.push(job_latency.as_secs_f64() * 1000.0);

            // Intermediates (x) drop here — O(1) TLSF free
            // Weights remain allocated — zero reload cost
        }

        sync_all_streams();
        reset_torch_stream();

        // Record fragmentation after each wave
        let frag = get_fragmentation();
        for _ in 0..wave {
            job_frags.push(frag);
        }

        job_base += wave;
    }

    let total_jobs_time = jobs_start.elapsed();

    // --- Report ---
    let boot_ms = boot_time.as_secs_f64() * 1000.0;
    let avg_job_ms = if job_latencies_ms.is_empty() {
        0.0
    } else {
        job_latencies_ms.iter().sum::<f64>() / job_latencies_ms.len() as f64
    };
    let min_job_ms = job_latencies_ms
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_job_ms = job_latencies_ms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let speedup = if avg_job_ms > 0.0 {
        boot_ms / avg_job_ms
    } else {
        0.0
    };

    let frag_after_jobs = get_fragmentation();
    let frag_stable = (frag_after_jobs - frag_after_boot).abs() < 0.01;

    // Print per-job detail for first few and last
    println!();
    for (i, (latency, checksum)) in job_latencies_ms
        .iter()
        .zip(output_checksums.iter())
        .enumerate()
    {
        if i < 3 || i >= cfg.num_jobs - 2 {
            println!(
                "  job {:>2}: {:.2}ms  checksum={:.6}  frag={:.6}",
                i, latency, checksum, job_frags[i]
            );
        } else if i == 3 {
            println!("  ...");
        }
    }

    println!("\n--- Results ---");
    println!("RESULT script=hot_state");
    println!("RESULT hidden={}", cfg.hidden);
    println!("RESULT num_layers={}", cfg.num_layers);
    println!("RESULT num_jobs={}", cfg.num_jobs);
    println!("RESULT batch_size={}", cfg.batch_size);
    println!("RESULT streams={}", active_streams);
    println!("RESULT wave_streams={}", cfg.wave_streams);
    println!("RESULT boot_time_ms={:.1}", boot_ms);
    println!("RESULT avg_job_time_ms={:.2}", avg_job_ms);
    println!("RESULT min_job_time_ms={:.2}", min_job_ms);
    println!("RESULT max_job_time_ms={:.2}", max_job_ms);
    println!(
        "RESULT total_jobs_time_ms={:.1}",
        total_jobs_time.as_secs_f64() * 1000.0
    );
    println!("RESULT speedup_ratio={:.1}x", speedup);
    println!("RESULT fragmentation_after_boot={:.6}", frag_after_boot);
    println!("RESULT fragmentation_after_jobs={:.6}", frag_after_jobs);
    println!("RESULT vram_stable={}", frag_stable);

    println!("\n--- TLSF Pool State ---");
    print_stats();

    println!("\nKey insight: boot={:.1}ms vs avg_job={:.2}ms → {:.1}x speedup.", boot_ms, avg_job_ms, speedup);
    println!("  A framework that reloads weights per-request wastes {:.0}% of its time on loading.",
        if speedup > 1.0 { (1.0 - 1.0 / speedup) * 100.0 } else { 0.0 });
    println!("  VRAM stable across all {} jobs: frag {:.6} → {:.6}.", cfg.num_jobs, frag_after_boot, frag_after_jobs);
    println!("  The OS keeps state hot — zero cold starts, zero reloads.");

    Ok(())
}
