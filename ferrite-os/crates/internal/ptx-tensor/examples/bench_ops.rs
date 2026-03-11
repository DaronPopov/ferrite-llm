//! PTX-OS tensor op benchmark (elementwise).
//!
//! Run with:
//!   cargo run --release -p ptx-tensor --example bench_ops
//!
//! Tunables:
//!   PTX_BENCH_ELEMS=1048576
//!   PTX_BENCH_ITERS=200
//!   PTX_BENCH_WARMUP=10
//!   PTX_BENCH_MAX_STREAMS=32
//!   PTX_BENCH_POOL_FRACTION=0.70
//!   PTX_BENCH_RESERVE_VRAM_MB=256

use std::env;
use std::sync::Arc;
use std::time::Instant;

use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use ptx_tensor::{Tensor, DType, Result};

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
    let elems = parse_usize_env("PTX_BENCH_ELEMS", 1_048_576);
    let iters = parse_usize_env("PTX_BENCH_ITERS", 200);
    let warmup = parse_usize_env("PTX_BENCH_WARMUP", 10);
    let max_streams = parse_usize_env("PTX_BENCH_MAX_STREAMS", 32).max(1) as u32;
    let pool_fraction = parse_f32_env("PTX_BENCH_POOL_FRACTION", 0.70);
    let reserve_vram_mb = parse_usize_env("PTX_BENCH_RESERVE_VRAM_MB", 256) as u64;

    println!("=== PTX-OS Tensor Elementwise Benchmark ===");
    println!("elems: {}, iters: {}, warmup: {}", elems, iters, warmup);
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

    let a = Tensor::full(&[elems], 1.0, DType::F32, &runtime)?;
    let b = Tensor::full(&[elems], 2.0, DType::F32, &runtime)?;
    runtime.sync_all()?;

    // Warmup
    for _ in 0..warmup {
        let c = a.add(&b)?;
        let _d = c.relu()?;
    }
    runtime.sync_all()?;

    let start = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        let c = a.add(&b)?;
        let d = c.relu()?;
        last = Some(d);
    }
    runtime.sync_all()?;
    let elapsed = start.elapsed();
    drop(last);

    let total_ops = (iters as f64) * (elems as f64) * 2.0;
    let secs = elapsed.as_secs_f64();
    let gelem_s = (total_ops / secs) / 1e9;
    let ns_per_elem = (secs * 1e9) / total_ops;

    println!("total time: {:.4} s", secs);
    println!("throughput: {:.3} GElem/s (add + relu)", gelem_s);
    println!("latency: {:.3} ns/element", ns_per_elem);

    Ok(())
}
