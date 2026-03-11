//! Parallel Batch Processing Example
//!
//! Demonstrates processing multiple batches in parallel using:
//! - Multiple streams for concurrent execution
//! - Stream-ordered TLSF allocations per stream
//! - Independent kernel execution on each stream
//! - Real-world batched inference pattern

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, test_kernels};
use std::time::Instant;

struct Batch {
    #[allow(dead_code)]
    id: usize,
    stream_id: i32,
    input_ptr: *mut libc::c_void,
    output_ptr: *mut libc::c_void,
    size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Parallel Batch Processing");
    println!("==============================\n");

    // Initialize with multi-stream config
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.75;
    config.max_streams = 64;  // Lots of streams for parallelism
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    println!("✓ PTX-OS Runtime initialized");
    println!("  Streams: {}", runtime.num_streams());

    // Adapt batch count to available streams (daemon may limit via PTX_MAX_STREAMS)
    let num_batches: usize = std::cmp::min(8, runtime.num_streams());
    const BATCH_SIZE: usize = 4096;
    const BYTES: usize = BATCH_SIZE * std::mem::size_of::<f32>();

    println!("\n📋 Configuration:");
    println!("  Batches: {}", num_batches);
    println!("  Elements per batch: {}", BATCH_SIZE);
    println!("  Memory per batch: {:.2} MB", BYTES as f64 / 1e6);

    // Allocate all batches on different streams
    println!("\n🌊 Allocating {} batches across multiple streams...", num_batches);
    let start = Instant::now();

    let mut batches = Vec::new();
    for i in 0..num_batches {
        let stream = runtime.stream(i as i32)?;

        let input_ptr = runtime.alloc_async(BYTES, &stream)?;
        let output_ptr = runtime.alloc_async(BYTES, &stream)?;

        batches.push(Batch {
            id: i,
            stream_id: stream.id(),
            input_ptr,
            output_ptr,
            size: BATCH_SIZE,
        });
    }

    let alloc_time = start.elapsed();
    println!("  ✓ All batches allocated in {:?}", alloc_time);

    // Prepare host data for all batches
    println!("\n🔧 Preparing batch data...");
    let host_batches: Vec<Vec<f32>> = (0..num_batches)
        .map(|batch_id| {
            (0..BATCH_SIZE)
                .map(|i| ((batch_id * BATCH_SIZE + i) as f32) / 1000.0)
                .collect()
        })
        .collect();

    // Upload all batches to GPU (each on its own stream)
    println!("\n📤 Uploading all batches (parallel streams)...");
    let start = Instant::now();

    for (i, batch) in batches.iter().enumerate() {
        let stream = runtime.stream(batch.stream_id)?;
        unsafe {
            ptx_sys::cudaMemcpyAsync(
                batch.input_ptr,
                host_batches[i].as_ptr() as *const _,
                BYTES,
                ptx_sys::cudaMemcpyHostToDevice,
                stream.raw(),
            );
        }
    }

    let upload_time = start.elapsed();
    println!("  ✓ All uploads queued in {:?}", upload_time);

    // Process all batches in parallel
    println!("\n🚀 Processing {} batches in parallel...", num_batches);
    let start = Instant::now();

    for batch in &batches {
        let stream = runtime.stream(batch.stream_id)?;
        let context = KernelContext::new(runtime.raw(), stream.raw())?;

        let input_buf = unsafe {
            GuardedBuffer::new(batch.input_ptr, BYTES, runtime.raw())?
        };
        let output_buf = unsafe {
            GuardedBuffer::new(batch.output_ptr, BYTES, runtime.raw())?
        };

        // Launch kernel on this stream
        unsafe {
            test_kernels::test_launch_gelu_f32(
                input_buf.as_ptr_typed::<f32>(),
                output_buf.as_ptr_typed::<f32>(),
                batch.size,
                context.stream(),
            );
        }
    }

    // Wait for all streams to complete
    for batch in &batches {
        let stream = runtime.stream(batch.stream_id)?;
        stream.synchronize()?;
    }

    let process_time = start.elapsed();
    println!("  ✓ All batches processed in {:?}", process_time);
    println!("  ⚡ Throughput: {:.2} batches/sec",
        num_batches as f64 / process_time.as_secs_f64());

    // Download results from all batches
    println!("\n📥 Downloading results...");
    let start = Instant::now();

    let mut results: Vec<Vec<f32>> = vec![vec![0.0f32; BATCH_SIZE]; num_batches];

    for (i, batch) in batches.iter().enumerate() {
        let stream = runtime.stream(batch.stream_id)?;
        unsafe {
            ptx_sys::cudaMemcpyAsync(
                results[i].as_mut_ptr() as *mut _,
                batch.output_ptr,
                BYTES,
                ptx_sys::cudaMemcpyDeviceToHost,
                stream.raw(),
            );
        }
    }

    // Sync all streams
    for batch in &batches {
        let stream = runtime.stream(batch.stream_id)?;
        stream.synchronize()?;
    }

    let download_time = start.elapsed();
    println!("  ✓ Results downloaded in {:?}", download_time);

    // Cleanup all batches (stream-ordered)
    println!("\n🗑️  Cleaning up...");
    for batch in &batches {
        let stream = runtime.stream(batch.stream_id)?;
        unsafe {
            runtime.free_async(batch.input_ptr, &stream)?;
            runtime.free_async(batch.output_ptr, &stream)?;
        }
    }

    // Final sync
    for batch in &batches {
        let stream = runtime.stream(batch.stream_id)?;
        stream.synchronize()?;
    }

    // Display sample results
    println!("\n📊 Sample Results:");
    for batch_id in 0..std::cmp::min(3, num_batches) {
        println!("  Batch {}:", batch_id);
        for i in 0..5 {
            println!("    [{:3}] {:.6}", i, results[batch_id][i]);
        }
    }

    // Performance summary
    let total_elements = num_batches * BATCH_SIZE;
    let total_time = alloc_time + upload_time + process_time + download_time;

    println!("\n📈 Performance Summary:");
    println!("  Allocation:   {:?}", alloc_time);
    println!("  Upload:       {:?}", upload_time);
    println!("  Processing:   {:?}", process_time);
    println!("  Download:     {:?}", download_time);
    println!("  Total:        {:?}", total_time);
    println!("  Throughput:   {:.2} M elements/sec",
        total_elements as f64 / total_time.as_secs_f64() / 1e6);

    let stats = runtime.tlsf_stats();
    println!("\n📊 TLSF Final Stats:");
    println!("  Total allocs: {}", stats.total_allocations);
    println!("  Total frees:  {}", stats.total_frees);
    println!("  Peak usage:   {:.2} MB", stats.peak_allocated as f64 / 1e6);

    println!("\n✅ Parallel batch processing completed!");
    Ok(())
}
