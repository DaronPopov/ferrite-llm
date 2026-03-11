//! Multi-Stream Active Compute Stress Test
//!
//! Saturates ALL available CUDA streams with real concurrent GPU work.
//! Unlike other examples that create many streams but only use 1-2,
//! this dispatches independent GELU kernels across every stream and
//! measures actual concurrent throughput.
//!
//! Stream count and pool fraction come from env (PTX_MAX_STREAMS,
//! PTX_POOL_FRACTION) so the daemon's --streams=N flag controls this.

use ptx_runtime::PtxRuntime;
use ptx_kernels::test_kernels;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = PtxRuntime::new(0)?;
    let num_streams = runtime.num_streams();

    println!("Multi-Stream Active Compute");
    println!("===========================");
    println!("  Streams: {num_streams}");

    // Each stream gets its own batch of work — enough to keep the SM busy.
    // Scale elements per stream down as stream count grows to stay within
    // the TLSF pool.  Total TLSF needed = 2 * num_streams * elements * 4 bytes.
    let elements_per_stream: usize = if num_streams <= 64 {
        256 * 1024  // 1 MB/stream → 128 MB total at 64 streams
    } else if num_streams <= 256 {
        64 * 1024   // 256 KB/stream → 128 MB total at 256 streams
    } else if num_streams <= 1024 {
        16 * 1024   // 64 KB/stream → 128 MB total at 1024 streams
    } else {
        4 * 1024    // 16 KB/stream → 128 MB total at 4096 streams
    };
    let bytes_per_stream = elements_per_stream * std::mem::size_of::<f32>();
    let total_bytes = bytes_per_stream * num_streams * 2; // input + output per stream

    println!("  Elements/stream: {elements_per_stream}");
    println!("  Bytes/stream: {} KB", bytes_per_stream / 1024);
    println!("  Total VRAM needed: {:.1} MB", total_bytes as f64 / (1024.0 * 1024.0));
    println!();

    // Phase 1: Allocate input+output buffers on each stream (stream-ordered)
    let t_alloc = Instant::now();
    let mut inputs: Vec<*mut libc::c_void> = Vec::with_capacity(num_streams);
    let mut outputs: Vec<*mut libc::c_void> = Vec::with_capacity(num_streams);

    for i in 0..num_streams {
        let stream = runtime.stream(i as i32)?;
        let inp = runtime.alloc_async(bytes_per_stream, &stream)?;
        let out = runtime.alloc_async(bytes_per_stream, &stream)?;
        inputs.push(inp);
        outputs.push(out);
    }
    let alloc_elapsed = t_alloc.elapsed();
    println!("[alloc] {num_streams} x 2 buffers in {:.3}ms", alloc_elapsed.as_secs_f64() * 1000.0);

    // Phase 2: Upload host data to each stream (async, concurrent)
    let host_data: Vec<f32> = (0..elements_per_stream)
        .map(|i| (i as f32) / 1000.0)
        .collect();

    let t_upload = Instant::now();
    for i in 0..num_streams {
        let stream = runtime.stream(i as i32)?;
        unsafe {
            ptx_sys::cudaMemcpyAsync(
                inputs[i],
                host_data.as_ptr() as *const _,
                bytes_per_stream,
                ptx_sys::cudaMemcpyHostToDevice,
                stream.raw(),
            );
        }
    }
    let upload_elapsed = t_upload.elapsed();
    println!("[upload] {num_streams} async memcpy queued in {:.3}ms", upload_elapsed.as_secs_f64() * 1000.0);

    // Phase 3: Launch GELU kernels on ALL streams simultaneously
    let t_launch = Instant::now();
    for i in 0..num_streams {
        let stream = runtime.stream(i as i32)?;
        unsafe {
            test_kernels::test_launch_gelu_f32(
                inputs[i] as *const f32,
                outputs[i] as *mut f32,
                elements_per_stream,
                stream.raw(),
            );
        }
    }
    let launch_elapsed = t_launch.elapsed();
    println!("[launch] {num_streams} GELU kernels dispatched in {:.3}ms", launch_elapsed.as_secs_f64() * 1000.0);

    // Phase 4: Synchronize all streams — this is where real compute happens
    let t_sync = Instant::now();
    runtime.sync_all()?;
    let sync_elapsed = t_sync.elapsed();

    let total_elements = num_streams * elements_per_stream;
    let throughput = total_elements as f64 / sync_elapsed.as_secs_f64() / 1e9;
    println!("[sync] All {num_streams} streams completed in {:.3}ms ({throughput:.2} G elements/sec)",
        sync_elapsed.as_secs_f64() * 1000.0);

    // Phase 5: Download one stream's output to verify correctness
    let mut result = vec![0.0f32; elements_per_stream];
    unsafe {
        ptx_sys::cudaMemcpy(
            result.as_mut_ptr() as *mut _,
            outputs[0],
            bytes_per_stream,
            ptx_sys::cudaMemcpyDeviceToHost,
        );
    }

    // Verify GELU: for x=0.0, GELU(0)=0; for x>0, GELU(x) ≈ x
    let sample_in = 1.0f32;
    let expected_gelu = sample_in * 0.5 * (1.0 + (sample_in * 0.7978845608 * (1.0 + 0.044715 * sample_in * sample_in)).tanh());
    let actual = result[1000]; // input was 1000/1000 = 1.0
    let error = (actual - expected_gelu).abs();
    let correct = error < 0.01;
    println!("[verify] GELU(1.0) = {actual:.6} (expected {expected_gelu:.6}, err={error:.6}) {}",
        if correct { "OK" } else { "MISMATCH" });

    // Phase 6: Cleanup (stream-ordered free)
    let t_free = Instant::now();
    for i in 0..num_streams {
        let stream = runtime.stream(i as i32)?;
        unsafe {
            runtime.free_async(inputs[i], &stream)?;
            runtime.free_async(outputs[i], &stream)?;
        }
    }
    runtime.sync_all()?;
    runtime.poll_deferred(0);
    let free_elapsed = t_free.elapsed();
    println!("[free] {num_streams} x 2 deferred frees in {:.3}ms", free_elapsed.as_secs_f64() * 1000.0);

    // Summary
    let total_time = alloc_elapsed + upload_elapsed + launch_elapsed + sync_elapsed + free_elapsed;
    let stats = runtime.tlsf_stats();

    println!();
    println!("Summary");
    println!("=======");
    println!("  Active streams: {num_streams}");
    println!("  Total elements: {} ({:.1} M)", total_elements, total_elements as f64 / 1e6);
    println!("  Alloc:   {:.3}ms", alloc_elapsed.as_secs_f64() * 1000.0);
    println!("  Upload:  {:.3}ms", upload_elapsed.as_secs_f64() * 1000.0);
    println!("  Launch:  {:.3}ms", launch_elapsed.as_secs_f64() * 1000.0);
    println!("  Compute: {:.3}ms", sync_elapsed.as_secs_f64() * 1000.0);
    println!("  Free:    {:.3}ms", free_elapsed.as_secs_f64() * 1000.0);
    println!("  Total:   {:.3}ms", total_time.as_secs_f64() * 1000.0);
    println!("  Throughput: {throughput:.2} G elements/sec");
    println!("  TLSF peak: {:.1} MB", stats.peak_allocated as f64 / (1024.0 * 1024.0));
    println!("  TLSF allocs: {} frees: {}", stats.total_allocations, stats.total_frees);
    println!("  Correct: {correct}");

    if !correct {
        return Err("GELU verification failed".into());
    }

    Ok(())
}
