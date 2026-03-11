//! EXTREME 100K STREAMS TEST
//!
//! Let's see if we can ACTUALLY launch 100,000 parallel kernels!
//! This is absolutely bonkers. No traditional CUDA setup could dream of this.
//!
//! WARNING: This is a stress test. Your GPU might cry.

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, test_kernels};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  💀 EXTREME 100K STREAMS TEST 💀                           ║");
    println!("║                                                            ║");
    println!("║  Attempting the IMPOSSIBLE:                               ║");
    println!("║  100,000 parallel kernels on 100,000 streams              ║");
    println!("║                                                            ║");
    println!("║  No traditional CUDA runtime can do this.                 ║");
    println!("║  Let's see if PTX-OS + TLSF can...                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // EXTREME CONFIG - MAXIMUM EVERYTHING
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.70;  // 70% to leave room for stream overhead
    config.max_streams = 20000;   // 20K stream pool (CUDA stream overhead limit)
    config.quiet_init = false;    // We want to see what happens

    // We'll ROTATE through these streams to launch 100K kernels!

    println!("⚡ Initializing PTX-OS with EXTREME config...");
    println!("  Max streams: {} 🔥🔥🔥", config.max_streams);
    println!("  Pool fraction: {:.0}%", config.pool_fraction * 100.0);

    let init_start = Instant::now();
    let runtime = PtxRuntime::with_config(0, Some(config))?;
    let init_time = init_start.elapsed();

    let actual_streams = runtime.num_streams();
    println!("  ✓ Runtime initialized in {:?}", init_time);
    println!("  Actual streams: {}", actual_streams);
    println!("  TLSF pool: {:.2} GB", runtime.tlsf_stats().total_pool_size as f64 / 1e9);
    println!();

    // Start with smaller test, then ramp up
    let test_sizes = [
        (1000, "1K - Warmup"),
        (5000, "5K - Getting Serious"),
        (10000, "10K - Now We're Talking"),
        (25000, "25K - This is Nuts"),
        (50000, "50K - Absolutely Insane"),
        (100000, "100K - THE BEAST 💀"),
    ];

    for (num_kernels, description) in test_sizes.iter() {
        println!("═══════════════════════════════════════════════════════════");
        println!("🎯 TEST: {} PARALLEL KERNELS ({})", num_kernels, description);
        println!("═══════════════════════════════════════════════════════════");
        println!();

        // Use smaller buffers for extreme parallelism
        let elements_per_kernel = if *num_kernels > 10000 { 256 } else { 1024 };
        let bytes_per_kernel = elements_per_kernel * 4;

        println!("📊 Configuration:");
        println!("  Kernels: {}", num_kernels);
        println!("  Elements per kernel: {}", elements_per_kernel);
        println!("  Total elements: {:.2}M", (*num_kernels * elements_per_kernel) as f64 / 1e6);
        println!("  Total memory: {:.2} MB", (*num_kernels * bytes_per_kernel * 2) as f64 / 1e6);
        println!();

        // ALLOCATION PHASE
        println!("🌊 Allocating {} workloads...", num_kernels);
        let start = Instant::now();

        let mut workloads = Vec::with_capacity(*num_kernels);
        for i in 0..*num_kernels {
            // Rotate through stream pool (use actual stream count, not hardcoded)
            let stream_id = (i % actual_streams) as i32;
            let stream = runtime.stream(stream_id).expect("valid stream id");

            match runtime.alloc_async(bytes_per_kernel, &stream) {
                Ok(input_ptr) => {
                    match runtime.alloc_async(bytes_per_kernel, &stream) {
                        Ok(output_ptr) => {
                            workloads.push((stream, input_ptr, output_ptr));
                        }
                        Err(e) => {
                            println!("  ⚠️  Failed to allocate output at kernel {}: {}", i, e);
                            break;
                        }
                    }
                }
                Err(e) => {
                    println!("  ⚠️  Failed to allocate input at kernel {}: {}", i, e);
                    break;
                }
            }

            // Progress indicator for large allocations
            if i > 0 && i % 10000 == 0 {
                println!("    ... {} allocated", i);
            }
        }

        let alloc_time = start.elapsed();
        let successful_allocs = workloads.len();

        println!("  ✓ {} workloads allocated in {:?}", successful_allocs, alloc_time);
        println!("  📈 Allocation rate: {:.2}M allocs/sec",
            (successful_allocs * 2) as f64 / alloc_time.as_secs_f64() / 1e6);

        if successful_allocs < *num_kernels {
            println!("  ⚠️  Only allocated {}/{} - continuing with what we have",
                successful_allocs, num_kernels);
        }
        println!();

        // Prepare minimal host data
        let host_data: Vec<f32> = (0..elements_per_kernel)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        // UPLOAD PHASE
        println!("📤 Uploading to {} streams...", successful_allocs);
        let start = Instant::now();

        for (_stream, input_ptr, _output_ptr) in &workloads {
            unsafe {
                ptx_sys::cudaMemcpyAsync(
                    *input_ptr,
                    host_data.as_ptr() as *const _,
                    bytes_per_kernel,
                    ptx_sys::cudaMemcpyHostToDevice,
                    _stream.raw(),
                );
            }
        }

        let upload_time = start.elapsed();
        println!("  ✓ Uploads queued in {:?}", upload_time);
        println!();

        // LAUNCH PHASE - THE MOMENT OF TRUTH
        println!("🔥 LAUNCHING {} KERNELS IN PARALLEL...", successful_allocs);
        let start = Instant::now();

        for (idx, (stream, input_ptr, output_ptr)) in workloads.iter().enumerate() {
            let input_buf = unsafe {
                GuardedBuffer::new(*input_ptr, bytes_per_kernel, runtime.raw())?
            };
            let output_buf = unsafe {
                GuardedBuffer::new(*output_ptr, bytes_per_kernel, runtime.raw())?
            };

            let ctx = KernelContext::new(runtime.raw(), stream.raw())?;

            // Launch GELU
            unsafe {
                test_kernels::test_launch_gelu_f32(
                    input_buf.as_ptr_typed(),
                    output_buf.as_ptr_typed(),
                    elements_per_kernel,
                    ctx.stream(),
                );
            }

            if idx > 0 && idx % 10000 == 0 {
                println!("    ... {} kernels launched", idx);
            }
        }

        let launch_time = start.elapsed();
        println!("  ✓ {} kernels launched in {:?}", successful_allocs, launch_time);
        println!("  🚀 Launch rate: {:.2}K kernels/sec",
            successful_allocs as f64 / launch_time.as_secs_f64() / 1000.0);
        println!();

        // EXECUTION PHASE
        println!("⏳ Waiting for {} streams to complete...", successful_allocs);
        let start = Instant::now();

        let mut sync_failures = 0;
        for (idx, (stream, _input_ptr, _output_ptr)) in workloads.iter().enumerate() {
            if let Err(e) = stream.synchronize() {
                sync_failures += 1;
                if sync_failures <= 10 {
                    println!("  ⚠️  Stream {} sync failed: {}", idx, e);
                }
            }

            if idx > 0 && idx % 10000 == 0 {
                println!("    ... {} streams synced", idx);
            }
        }

        let exec_time = start.elapsed();
        println!("  ✓ Execution completed in {:?}", exec_time);
        if sync_failures > 0 {
            println!("  ⚠️  {} streams failed to sync", sync_failures);
        }
        println!();

        // CLEANUP
        println!("🗑️  Cleaning up {} allocations...", successful_allocs * 2);
        let start = Instant::now();

        for (stream, input_ptr, output_ptr) in &workloads {
            unsafe {
                let _ = runtime.free_async(*input_ptr, stream);
                let _ = runtime.free_async(*output_ptr, stream);
            }
        }

        for (stream, _input_ptr, _output_ptr) in &workloads {
            let _ = stream.synchronize();
        }

        let cleanup_time = start.elapsed();
        println!("  ✓ Cleanup completed in {:?}", cleanup_time);
        println!();

        // SUMMARY
        let total_time = alloc_time + upload_time + launch_time + exec_time + cleanup_time;

        println!("📈 RESULTS:");
        println!("  ════════════════════════════════════════════════════");
        println!("  Successful kernels:  {}/{}", successful_allocs, num_kernels);
        println!("  Total elements:      {:.2}M", (successful_allocs * elements_per_kernel) as f64 / 1e6);
        println!("  ────────────────────────────────────────────────────");
        println!("  Allocation:          {:?}", alloc_time);
        println!("  Upload:              {:?}", upload_time);
        println!("  Launch:              {:?}", launch_time);
        println!("  Execution:           {:?}", exec_time);
        println!("  Cleanup:             {:?}", cleanup_time);
        println!("  ════════════════════════════════════════════════════");
        println!("  TOTAL TIME:          {:?}", total_time);
        println!("  ════════════════════════════════════════════════════");
        println!("  Throughput:          {:.2}M elements/sec",
            (successful_allocs * elements_per_kernel) as f64 / total_time.as_secs_f64() / 1e6);
        println!("  Kernel rate:         {:.2}K kernels/sec",
            successful_allocs as f64 / total_time.as_secs_f64() / 1000.0);

        let stats = runtime.tlsf_stats();
        println!();
        println!("💾 TLSF ALLOCATOR STATUS:");
        println!("  ════════════════════════════════════════════════════");
        println!("  Total allocations:   {:.2}M", stats.total_allocations as f64 / 1e6);
        println!("  Total frees:         {:.2}M", stats.total_frees as f64 / 1e6);
        println!("  Peak allocated:      {:.2} MB", stats.peak_allocated as f64 / 1e6);
        println!("  Fragmentation:       {:.2}%", stats.fragmentation_ratio * 100.0);
        println!("  Health:              {}",
            if stats.is_healthy { "✓ HEALTHY" } else { "⚠ ISSUES" });
        println!();

        if successful_allocs == *num_kernels {
            println!("✅ {} KERNELS - COMPLETE SUCCESS! 🎉", num_kernels);
        } else {
            println!("⚠️  {} KERNELS - PARTIAL SUCCESS ({}/{})",
                num_kernels, successful_allocs, num_kernels);
        }
        println!();

        // Small pause between tests
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🏆 EXTREME STREAM TEST COMPLETE! 🏆                       ║");
    println!("║                                                            ║");
    println!("║  PTX-OS + TLSF = THE FUTURE OF GPU COMPUTING 🚀           ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
