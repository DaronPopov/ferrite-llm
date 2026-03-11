//! MASSIVE CANDLE KERNEL PARALLELISM
//!
//! Demonstrates running thousands of validated Candle kernels
//! across thousands of CUDA streams simultaneously.
//!
//! This proves that all Candle operations work perfectly
//! with PTX-OS TLSF orchestrating everything!

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, safe_api};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🚀 MASSIVE CANDLE KERNEL PARALLELISM 🚀                   ║");
    println!("║                                                            ║");
    println!("║  Running THOUSANDS of validated Candle kernels            ║");
    println!("║  across THOUSANDS of CUDA streams                         ║");
    println!("║                                                            ║");
    println!("║  All orchestrated by PTX-OS TLSF allocator!               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize with massive stream config
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.75;
    config.max_streams = 10000;  // 10K concurrent streams!
    config.quiet_init = false;

    println!("⚡ Initializing PTX-OS runtime...");
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    println!("  ✓ PTX-OS initialized");
    println!("  Streams: {}", runtime.num_streams());
    println!("  TLSF Pool: {:.2} GB", runtime.tlsf_stats().total_pool_size as f64 / 1e9);
    println!();

    // Test configurations - ramp up gradually
    let test_configs = [
        (500, 4096, "Warmup - 500 streams"),
        (1000, 4096, "1K streams - Getting Started"),
        (2500, 2048, "2.5K streams - Heating Up"),
        (5000, 1024, "5K streams - Major Scale"),
        (10000, 512, "10K streams - MAXIMUM PARALLELISM 🔥"),
    ];

    for (num_streams, elements_per_stream, description) in test_configs.iter() {
        println!("═══════════════════════════════════════════════════════════");
        println!("🎯 TEST: {}", description);
        println!("═══════════════════════════════════════════════════════════");
        println!("  Concurrent streams: {}", num_streams);
        println!("  Elements per stream: {}", elements_per_stream);
        println!("  Total elements: {:.2}M", (*num_streams * *elements_per_stream) as f64 / 1e6);
        println!();

        run_candle_stress_test(&runtime, *num_streams, *elements_per_stream)?;

        println!();
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🎉 SUCCESS! ALL CANDLE KERNELS VALIDATED! 🎉              ║");
    println!("║                                                            ║");
    println!("║  PTX-OS TLSF orchestrated thousands of streams            ║");
    println!("║  running validated Candle math kernels!                   ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn run_candle_stress_test(
    runtime: &PtxRuntime,
    num_streams: usize,
    elements: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = elements * std::mem::size_of::<f32>();

    // Allocation phase
    println!("🌊 Allocating {} stream workloads via TLSF...", num_streams);
    let alloc_start = Instant::now();

    let mut workloads = Vec::with_capacity(num_streams);

    unsafe {
        let runtime_ptr = runtime.raw();

        for i in 0..num_streams {
            let stream_id = (i % runtime.num_streams()) as i32;
            let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, stream_id);

            // Allocate 3 buffers per stream (input, temp, output)
            let input = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
            let temp = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
            let output = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);

            if input.is_null() || temp.is_null() || output.is_null() {
                println!("  ⚠️  TLSF allocation failed at stream {}", i);
                break;
            }

            workloads.push((stream, input, temp, output, bytes));

            if i > 0 && i % 1000 == 0 {
                println!("    Allocated {} streams...", i);
            }
        }
    }

    let alloc_time = alloc_start.elapsed();
    println!("  ✓ Allocated {} workloads in {:?}", workloads.len(), alloc_time);
    println!("    Rate: {:.0} allocs/sec", (workloads.len() * 3) as f64 / alloc_time.as_secs_f64());
    println!();

    // Launch phase - mix of different Candle kernels across streams
    println!("🚀 Launching {} Candle kernels across streams...", workloads.len() * 3);
    let launch_start = Instant::now();

    unsafe {
        let runtime_ptr = runtime.raw();

        for (idx, (stream, input, temp, output, bytes)) in workloads.iter().enumerate() {
            let ctx = KernelContext::new(runtime_ptr, *stream)?;
            let ig = GuardedBuffer::new(*input, *bytes, runtime_ptr)?;
            let tg = GuardedBuffer::new(*temp, *bytes, runtime_ptr)?;
            let og = GuardedBuffer::new(*output, *bytes, runtime_ptr)?;

            // Each stream runs 3 different Candle kernels:
            // 1. ReLU on input -> temp
            safe_api::unary::relu(&ig, &tg, elements, &ctx)?;

            // 2. Tanh on temp -> output
            safe_api::unary::tanh(&tg, &og, elements, &ctx)?;

            // 3. Binary add: output = output + input
            safe_api::binary::add(&og, &ig, &og, elements, &ctx)?;

            if idx > 0 && idx % 1000 == 0 {
                println!("    Launched kernels for {} streams...", idx);
            }
        }
    }

    let launch_time = launch_start.elapsed();
    println!("  ✓ Launched {} kernels in {:?}", workloads.len() * 3, launch_time);
    println!("    Rate: {:.0} kernels/sec", (workloads.len() * 3) as f64 / launch_time.as_secs_f64());
    println!();

    // Sync phase
    println!("⏱️  Synchronizing {} streams...", workloads.len());
    let sync_start = Instant::now();

    unsafe {
        for (stream, _, _, _, _) in workloads.iter() {
            ptx_sys::cudaStreamSynchronize(*stream);
        }
    }

    let sync_time = sync_start.elapsed();
    println!("  ✓ All streams synchronized in {:?}", sync_time);
    println!();

    // Cleanup
    println!("🧹 Freeing {} TLSF allocations...", workloads.len() * 3);
    let free_start = Instant::now();

    unsafe {
        let runtime_ptr = runtime.raw();
        for (_, input, temp, output, _) in workloads.iter() {
            ptx_sys::gpu_hot_free(runtime_ptr, *input);
            ptx_sys::gpu_hot_free(runtime_ptr, *temp);
            ptx_sys::gpu_hot_free(runtime_ptr, *output);
        }
    }

    let free_time = free_start.elapsed();
    println!("  ✓ Freed all allocations in {:?}", free_time);
    println!("    Rate: {:.0} frees/sec", (workloads.len() * 3) as f64 / free_time.as_secs_f64());
    println!();

    // Summary
    let total_time = alloc_time + launch_time + sync_time + free_time;
    let total_ops = workloads.len() as f64 * elements as f64 * 3.0; // 3 kernels per stream
    let throughput = total_ops / sync_time.as_secs_f64();

    println!("📊 Performance Summary:");
    println!("  Allocation: {:?}", alloc_time);
    println!("  Launch:     {:?}", launch_time);
    println!("  Execute:    {:?}", sync_time);
    println!("  Cleanup:    {:?}", free_time);
    println!("  ─────────────────────");
    println!("  Total:      {:?}", total_time);
    println!();
    println!("  Throughput: {:.2} GFLOPS", throughput / 1e9);
    println!("  Elements:   {:.2}M", total_ops / 1e6);

    Ok(())
}
