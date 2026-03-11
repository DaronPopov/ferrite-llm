//! JITTER & LATENCY DISTRIBUTION BENCHMARK
//!
//! Tests timing consistency under massive concurrency:
//! - Kernel launch latency (p50, p95, p99, p99.9)
//! - Allocation jitter
//! - Stream interference
//! - Tail latency analysis
//!
//! Production systems need PREDICTABLE latency, not just fast average!

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, safe_api};
use std::time::{Instant, Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  📊 JITTER & LATENCY DISTRIBUTION BENCHMARK 📊             ║");
    println!("║                                                            ║");
    println!("║  Testing timing consistency under massive concurrency!    ║");
    println!("║  Production needs PREDICTABLE latency! ⚡                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.75;
    config.max_streams = 1024;
    config.quiet_init = false;

    println!("⚡ Initializing PTX-OS runtime...");
    let runtime = PtxRuntime::with_config(0, Some(config))?;
    println!("  ✓ TLSF Pool: {:.2} GB", runtime.tlsf_stats().total_pool_size as f64 / 1e9);
    println!();

    // Test 1: Allocation Jitter Under Load
    println!("═══════════════════════════════════════════════════════════");
    println!("🎯 TEST 1: ALLOCATION JITTER");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Measuring TLSF allocation latency distribution");
    println!("  Looking for consistent sub-microsecond performance");
    println!();

    test_allocation_jitter(&runtime)?;

    println!();

    // Test 2: Kernel Launch Jitter
    println!("═══════════════════════════════════════════════════════════");
    println!("🎯 TEST 2: KERNEL LAUNCH JITTER");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Measuring kernel launch latency distribution");
    println!("  Testing consistency across thousands of launches");
    println!();

    test_kernel_launch_jitter(&runtime)?;

    println!();

    // Test 3: Stream Interference
    println!("═══════════════════════════════════════════════════════════");
    println!("🎯 TEST 3: STREAM INTERFERENCE TEST");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Do streams affect each other's latency?");
    println!("  Comparing: 1 stream vs 100 streams vs 1000 streams");
    println!();

    test_stream_interference(&runtime)?;

    println!();

    // Test 4: End-to-End Latency Under Load
    println!("═══════════════════════════════════════════════════════════");
    println!("🎯 TEST 4: END-TO-END LATENCY");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Full pipeline: Allocate → Launch → Execute → Free");
    println!("  Measuring total latency under concurrent load");
    println!();

    test_end_to_end_latency(&runtime)?;

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🎉 JITTER ANALYSIS COMPLETE! 🎉                           ║");
    println!("║                                                            ║");
    println!("║  Check if your system has production-grade consistency!   ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Test TLSF allocation jitter
fn test_allocation_jitter(runtime: &PtxRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let num_samples = 10000;
    let alloc_size = 4 * 1024 * 1024; // 4MB

    println!("  Collecting {} allocation samples...", num_samples);

    let mut latencies = Vec::with_capacity(num_samples);

    unsafe {
        let runtime_ptr = runtime.raw();
        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);

        for _ in 0..num_samples {
            let start = Instant::now();
            let ptr = ptx_sys::gpu_hot_alloc_async(runtime_ptr, alloc_size, stream);
            let elapsed = start.elapsed();

            if ptr.is_null() {
                return Err("Allocation failed".into());
            }

            latencies.push(elapsed);
            ptx_sys::gpu_hot_free(runtime_ptr, ptr);
        }
    }

    print_latency_stats("TLSF Allocation", &latencies);

    Ok(())
}

/// Test kernel launch jitter
fn test_kernel_launch_jitter(runtime: &PtxRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let num_samples = 10000;
    let buffer_size = 1024 * 1024; // 1M elements
    let bytes = buffer_size * 4;

    println!("  Collecting {} kernel launch samples...", num_samples);

    let mut latencies = Vec::with_capacity(num_samples);

    unsafe {
        let runtime_ptr = runtime.raw();
        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);
        let ctx = KernelContext::new(runtime_ptr, stream)?;

        // Pre-allocate buffers to isolate kernel launch timing
        let input = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
        let output = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);

        if input.is_null() || output.is_null() {
            return Err("Pre-allocation failed".into());
        }

        let ig = GuardedBuffer::new(input, bytes, runtime_ptr)?;
        let og = GuardedBuffer::new(output, bytes, runtime_ptr)?;

        for _ in 0..num_samples {
            let start = Instant::now();
            safe_api::unary::relu(&ig, &og, buffer_size, &ctx)?;
            let elapsed = start.elapsed();

            latencies.push(elapsed);
        }

        ptx_sys::cudaStreamSynchronize(stream);
        ptx_sys::gpu_hot_free(runtime_ptr, input);
        ptx_sys::gpu_hot_free(runtime_ptr, output);
    }

    print_latency_stats("Kernel Launch", &latencies);

    Ok(())
}

/// Test stream interference
fn test_stream_interference(runtime: &PtxRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let num_samples = 1000;
    let buffer_size = 1024 * 1024;
    let bytes = buffer_size * 4;

    // Test with different stream counts
    let stream_counts = vec![1, 10, 100, 500, 1000];

    for num_streams in stream_counts {
        println!("  Testing with {} concurrent streams...", num_streams);

        let mut latencies = Vec::with_capacity(num_samples);

        unsafe {
            let runtime_ptr = runtime.raw();

            for sample in 0..num_samples {
                // Use different streams in rotation
                let stream_id = (sample % num_streams) as i32;
                let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, stream_id);
                let ctx = KernelContext::new(runtime_ptr, stream)?;

                let start = Instant::now();

                // Allocate
                let input = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
                let output = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);

                let ig = GuardedBuffer::new(input, bytes, runtime_ptr)?;
                let og = GuardedBuffer::new(output, bytes, runtime_ptr)?;

                // Launch
                safe_api::unary::relu(&ig, &og, buffer_size, &ctx)?;

                // Free
                ptx_sys::gpu_hot_free(runtime_ptr, input);
                ptx_sys::gpu_hot_free(runtime_ptr, output);

                let elapsed = start.elapsed();
                latencies.push(elapsed);
            }

            // Sync all used streams
            for stream_id in 0..num_streams.min(runtime.num_streams()) {
                let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, stream_id as i32);
                ptx_sys::cudaStreamSynchronize(stream);
            }
        }

        let stats = compute_latency_stats(&latencies);
        println!("    p50: {:>8.2?}  p95: {:>8.2?}  p99: {:>8.2?}  p99.9: {:>8.2?}",
                 stats.p50, stats.p95, stats.p99, stats.p999);
    }

    println!();
    println!("  📊 ANALYSIS:");
    println!("    If p99 stays consistent → No interference! ✅");
    println!("    If p99 increases → Streams interfere ❌");

    Ok(())
}

/// Test end-to-end latency
fn test_end_to_end_latency(runtime: &PtxRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let num_samples = 5000;
    let buffer_size = 1024 * 1024;
    let bytes = buffer_size * 4;
    let ns = runtime.num_streams();

    println!("  Collecting {} end-to-end samples...", num_samples);
    println!("  Pipeline: Alloc → Launch → Sync → Free");

    let mut latencies = Vec::with_capacity(num_samples);

    unsafe {
        let runtime_ptr = runtime.raw();

        for sample in 0..num_samples {
            let stream_id = (sample % ns) as i32;
            let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, stream_id);
            let ctx = KernelContext::new(runtime_ptr, stream)?;

            let start = Instant::now();

            // Full pipeline
            let input = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
            let output = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);

            let ig = GuardedBuffer::new(input, bytes, runtime_ptr)?;
            let og = GuardedBuffer::new(output, bytes, runtime_ptr)?;

            safe_api::unary::relu(&ig, &og, buffer_size, &ctx)?;

            ptx_sys::cudaStreamSynchronize(stream);

            ptx_sys::gpu_hot_free(runtime_ptr, input);
            ptx_sys::gpu_hot_free(runtime_ptr, output);

            let elapsed = start.elapsed();
            latencies.push(elapsed);
        }
    }

    print_latency_stats("End-to-End", &latencies);

    Ok(())
}

// Helper structures and functions

#[derive(Debug)]
struct LatencyStats {
    min: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    p999: Duration,
    max: Duration,
    mean: Duration,
    std_dev: f64,
}

fn compute_latency_stats(latencies: &[Duration]) -> LatencyStats {
    let mut sorted = latencies.to_vec();
    sorted.sort();

    let len = sorted.len();
    let min = sorted[0];
    let p50 = sorted[len * 50 / 100];
    let p95 = sorted[len * 95 / 100];
    let p99 = sorted[len * 99 / 100];
    let p999 = sorted[len * 999 / 1000];
    let max = sorted[len - 1];

    // Calculate mean
    let sum: Duration = sorted.iter().sum();
    let mean = sum / len as u32;

    // Calculate standard deviation
    let mean_ns = mean.as_nanos() as f64;
    let variance: f64 = sorted.iter()
        .map(|&d| {
            let diff = d.as_nanos() as f64 - mean_ns;
            diff * diff
        })
        .sum::<f64>() / len as f64;
    let std_dev = variance.sqrt();

    LatencyStats {
        min,
        p50,
        p95,
        p99,
        p999,
        max,
        mean,
        std_dev,
    }
}

fn print_latency_stats(name: &str, latencies: &[Duration]) {
    let stats = compute_latency_stats(latencies);

    println!();
    println!("  📊 {} Latency Distribution:", name);
    println!("    ─────────────────────────────────────────");
    println!("    Samples:  {}", latencies.len());
    println!("    Min:      {:>10.2?}", stats.min);
    println!("    p50:      {:>10.2?}", stats.p50);
    println!("    p95:      {:>10.2?}", stats.p95);
    println!("    p99:      {:>10.2?}", stats.p99);
    println!("    p99.9:    {:>10.2?}", stats.p999);
    println!("    Max:      {:>10.2?}", stats.max);
    println!("    Mean:     {:>10.2?}", stats.mean);
    println!("    Std Dev:  {:>10.2} ns", stats.std_dev);
    println!("    ─────────────────────────────────────────");

    // Analyze jitter
    let jitter_ratio = (stats.p99.as_nanos() as f64) / (stats.p50.as_nanos() as f64);
    println!("    Jitter (p99/p50): {:.2}x", jitter_ratio);

    if jitter_ratio < 2.0 {
        println!("    ✅ EXCELLENT! Very low jitter!");
    } else if jitter_ratio < 5.0 {
        println!("    ✅ GOOD! Acceptable jitter for production");
    } else if jitter_ratio < 10.0 {
        println!("    ⚠️  MODERATE jitter - investigate");
    } else {
        println!("    ❌ HIGH jitter - needs optimization");
    }

    // Tail latency analysis
    let tail_ratio = (stats.max.as_nanos() as f64) / (stats.p99.as_nanos() as f64);
    println!("    Tail (max/p99): {:.2}x", tail_ratio);

    if tail_ratio < 2.0 {
        println!("    ✅ Tail latencies well controlled!");
    } else if tail_ratio < 5.0 {
        println!("    ⚠️  Some outliers present");
    } else {
        println!("    ❌ Significant outliers detected!");
    }

    println!();
}
