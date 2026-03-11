//! PTX-OS dynamic-shape inference benchmark.
//!
//! Simulates variable batch size and sequence length (micro-batching / decoding).
//! Emphasizes allocator churn and fragmentation behavior.
//!
//! Run with:
//!   cargo run --release -p ptx-tensor --example bench_dynamic_shapes
//!
//! Tunables:
//!   PTX_BENCH_REQUESTS=1000
//!   PTX_BENCH_BATCH_MIN=1
//!   PTX_BENCH_BATCH_MAX=16
//!   PTX_BENCH_SEQ_MIN=16
//!   PTX_BENCH_SEQ_MAX=512
//!   PTX_BENCH_HIDDEN=256
//!   PTX_BENCH_SYNC_EVERY=10
//!   PTX_BENCH_PRINT_EVERY=100
//!   PTX_BENCH_MAX_STREAMS=32
//!   PTX_BENCH_POOL_FRACTION=0.70
//!   PTX_BENCH_RESERVE_VRAM_MB=256

use std::env;
use std::sync::Arc;
use std::time::Instant;

use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use ptx_tensor::{DType, Result, Tensor};

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        let span = max - min + 1;
        min + (self.next_u32() as usize % span)
    }
}

fn parse_usize_env(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_f32_env(key: &str, default: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let requests = parse_usize_env("PTX_BENCH_REQUESTS", 1000);
    let batch_min = parse_usize_env("PTX_BENCH_BATCH_MIN", 1);
    let batch_max = parse_usize_env("PTX_BENCH_BATCH_MAX", 16);
    let seq_min = parse_usize_env("PTX_BENCH_SEQ_MIN", 16);
    let seq_max = parse_usize_env("PTX_BENCH_SEQ_MAX", 512);
    let hidden = parse_usize_env("PTX_BENCH_HIDDEN", 256);
    let sync_every = parse_usize_env("PTX_BENCH_SYNC_EVERY", 10).max(1);
    let print_every = parse_usize_env("PTX_BENCH_PRINT_EVERY", 100).max(1);
    let max_streams = parse_usize_env("PTX_BENCH_MAX_STREAMS", 32).max(1) as u32;
    let pool_fraction = parse_f32_env("PTX_BENCH_POOL_FRACTION", 0.70);
    let reserve_vram_mb = parse_usize_env("PTX_BENCH_RESERVE_VRAM_MB", 256) as u64;

    println!("=== PTX-OS Dynamic-Shape Inference Benchmark ===");
    println!("requests: {}", requests);
    println!("batch: {}..{}", batch_min, batch_max);
    println!("seq:   {}..{}", seq_min, seq_max);
    println!("hidden: {}", hidden);
    println!("sync every: {}", sync_every);
    println!("print every: {}", print_every);
    println!("max streams: {}", max_streams);
    println!("pool fraction: {:.2}", pool_fraction);
    println!("reserve VRAM: {} MB", reserve_vram_mb);
    println!();

    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction,
        fixed_pool_size: 0,
        reserve_vram: reserve_vram_mb * 1024 * 1024,
        max_streams,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = Arc::new(PtxRuntime::with_stable_config(0, Some(runtime_cfg))?);
    let mut rng = Lcg::new(0xC0FFEE);

    let start = Instant::now();
    let mut total_elems: u64 = 0;
    let mut total_bytes: u64 = 0;

    for i in 1..=requests {
        let batch = rng.range_usize(batch_min, batch_max);
        let seq = rng.range_usize(seq_min, seq_max);
        let elems = (batch * seq * hidden) as u64;
        total_elems += elems;

        // Simulated inference: allocate + add + relu, then drop
        let a = Tensor::full(&[batch, seq, hidden], 1.0, DType::F32, &runtime)?;
        let b = Tensor::full(&[batch, seq, hidden], 2.0, DType::F32, &runtime)?;
        let c = a.add(&b)?;
        let _d = c.relu()?;

        // Approximate bytes touched (a,b,c,d) * 4 bytes per elem
        total_bytes += elems * 4 * 4;

        if i % sync_every == 0 {
            runtime.sync_all()?;
        }

        if i % print_every == 0 || i == requests {
            let stats = runtime.tlsf_stats();
            let max_alloc = runtime.max_allocatable();
            println!(
                "req {:>5}/{} | alloc {:.2} GB | free {:.2} GB | frag {:.2}% | max_alloc {:.2} GB",
                i,
                requests,
                stats.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                stats.free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                stats.fragmentation_ratio * 100.0,
                max_alloc as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
    }

    runtime.sync_all()?;
    let elapsed = start.elapsed().as_secs_f64();
    let gelem_s = (total_elems as f64) / elapsed / 1e9;
    let gb_s = (total_bytes as f64) / elapsed / 1e9;

    let final_stats = runtime.tlsf_stats();
    println!();
    println!("total time: {:.4} s", elapsed);
    println!("throughput: {:.3} GElem/s", gelem_s);
    println!("throughput: {:.3} GB/s (approx)", gb_s);
    println!(
        "final frag: {:.2}% | largest free {:.2} GB",
        final_stats.fragmentation_ratio * 100.0,
        final_stats.largest_free_block as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    Ok(())
}
