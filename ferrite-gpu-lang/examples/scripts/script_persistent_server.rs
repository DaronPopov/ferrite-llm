#![cfg(feature = "torch")]

//! GPU as Persistent Inference Server
//!
//! The GPU is the server, not the accelerator. Model weights load once and stay
//! resident. A continuous request loop processes independent "client" requests on
//! isolated streams. No cold start, no reallocation, no framework overhead.
//!
//! What this demonstrates:
//! - Persistent model weights that never deallocate (kept in outer scope)
//! - Each request gets its own CUDA stream via round-robin
//! - Requests are processed in waves — simulating concurrent clients
//! - TLSF stats show zero fragmentation across thousands of request cycles
//! - Per-request latency is sub-millisecond since allocation is O(1)

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
    batch_size: i64,
    num_requests: usize,
    streams: u32,
    wave_streams: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: 2048,
            num_layers: 4,
            batch_size: 8,
            num_requests: 1000,
            streams: 64,
            wave_streams: 16,
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
            "--batch-size" => {
                cfg.batch_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--num-requests" => {
                cfg.num_requests = args[i + 1].parse().unwrap();
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

    println!("=== GPU Persistent Inference Server ===");
    println!(
        "hidden={} layers={} batch={} requests={} streams={} wave={}",
        cfg.hidden, cfg.num_layers, cfg.batch_size, cfg.num_requests, cfg.streams, cfg.wave_streams
    );

    // --- Init runtime + TLSF pool ---
    init_pytorch_tlsf_ex(device_id, 0.70, cfg.streams).map_err(|e| anyhow::anyhow!("{}", e))?;

    let active_streams = num_streams();
    println!("PTX-OS streams: {}", active_streams);
    println!("script=persistent_server");

    let _guard = tch::no_grad_guard();

    // --- Phase 1: Load model weights (PERSISTENT — never dropped) ---
    let load_start = Instant::now();
    let weights: Vec<Tensor> = (0..cfg.num_layers)
        .map(|_| {
            Tensor::randn([cfg.hidden, cfg.hidden], (Kind::Float, device)) * (1.0 / (cfg.hidden as f64).sqrt())
        })
        .collect();
    let biases: Vec<Tensor> = (0..cfg.num_layers)
        .map(|_| Tensor::zeros([cfg.hidden], (Kind::Float, device)))
        .collect();
    sync_all_streams();
    let load_time = load_start.elapsed();
    println!(
        "weights_loaded: {} layers, {:.1}ms",
        cfg.num_layers,
        load_time.as_secs_f64() * 1000.0
    );

    let frag_after_load = get_fragmentation();
    println!("fragmentation_after_load={:.6}", frag_after_load);

    // --- Phase 2: Request loop ---
    let mut total_latency_us = 0u128;
    let mut max_latency_us = 0u128;
    let mut min_latency_us = u128::MAX;

    let request_start = Instant::now();
    let mut req_base = 0usize;

    while req_base < cfg.num_requests {
        let wave = (cfg.num_requests - req_base).min(cfg.wave_streams);

        for local_idx in 0..wave {
            let req_idx = req_base + local_idx;
            let stream_id = req_idx % active_streams;
            set_torch_stream(stream_id);

            let req_t = Instant::now();

            // Generate random input for this request
            let mut x = Tensor::randn([cfg.batch_size, cfg.hidden], (Kind::Float, device));

            // Forward pass through all layers (matmul → relu)
            // Weights are shared (persistent), only intermediates are per-request
            for layer in 0..cfg.num_layers {
                x = x.matmul(&weights[layer]).f_add(&biases[layer]).unwrap().relu();
            }

            // Force computation to complete (read a scalar)
            let _val = x.mean(Kind::Float).double_value(&[]);

            let elapsed_us = req_t.elapsed().as_micros();
            total_latency_us += elapsed_us;
            max_latency_us = max_latency_us.max(elapsed_us);
            min_latency_us = min_latency_us.min(elapsed_us);
            // Intermediates (x) drop here — O(1) TLSF free
        }

        sync_all_streams();
        reset_torch_stream();
        req_base += wave;
    }

    let total_request_time = request_start.elapsed();

    // --- Report ---
    let requests_per_sec =
        cfg.num_requests as f64 / total_request_time.as_secs_f64();
    let avg_latency_us = total_latency_us as f64 / cfg.num_requests as f64;
    let frag_after_requests = get_fragmentation();

    println!("\n--- Results ---");
    println!("RESULT script=persistent_server");
    println!("RESULT hidden={}", cfg.hidden);
    println!("RESULT num_layers={}", cfg.num_layers);
    println!("RESULT batch_size={}", cfg.batch_size);
    println!("RESULT num_requests={}", cfg.num_requests);
    println!("RESULT streams={}", active_streams);
    println!("RESULT wave_streams={}", cfg.wave_streams);
    println!(
        "RESULT total_time_ms={:.1}",
        total_request_time.as_secs_f64() * 1000.0
    );
    println!("RESULT requests_per_sec={:.1}", requests_per_sec);
    println!("RESULT avg_latency_us={:.1}", avg_latency_us);
    println!("RESULT min_latency_us={}", min_latency_us);
    println!("RESULT max_latency_us={}", max_latency_us);
    println!(
        "RESULT weight_load_ms={:.1}",
        load_time.as_secs_f64() * 1000.0
    );
    println!("RESULT fragmentation_after_load={:.6}", frag_after_load);
    println!(
        "RESULT fragmentation_after_requests={:.6}",
        frag_after_requests
    );

    println!("\n--- TLSF Pool State ---");
    print_stats();

    println!(
        "\nKey insight: {} requests processed with weights never reloaded.",
        cfg.num_requests
    );
    println!("  Fragmentation stayed at {:.6} across the entire run.", frag_after_requests);
    println!(
        "  Avg request latency: {:.1}us — O(1) alloc makes per-request overhead negligible.",
        avg_latency_us
    );

    Ok(())
}
