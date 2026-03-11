//! stream_compute — 10,000-Stream Monte Carlo Pipeline
//!
//! Proves: Mass-scale parallel computation with 10,000 software streams,
//! alloc_async/free_async, deferred free draining — impossible with standard CUDA
//! (32 hardware streams, cudaMalloc serializes).
//!
//! OS primitives exercised: 10K streams, alloc_async/free_async, poll_deferred,
//! VMM (pages, swap, pin), SHM, tensor ops (exp, sigmoid, reduce_sum, fill, mul_scalar).

use std::time::Instant;

use anyhow::Result;

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.50;
const MAX_STREAMS: u32 = 10_000;
const SAMPLES_PER_STREAM: usize = 256; // 256 f32 = 1 KB
const SAMPLE_BYTES: usize = SAMPLES_PER_STREAM * std::mem::size_of::<f32>();
const BATCH_SIZE: usize = 1000;

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();
    println!("=== STREAM COMPUTE ===");
    println!("10,000-stream Monte Carlo pipeline");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!("Config: pool_fraction={}, max_streams={}, batch_size={}", POOL_FRACTION, MAX_STREAMS, BATCH_SIZE);
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("stream_mc", 10);

    // Initialize VMM for overflow tracking
    let vmm = unsafe { platform::vmm_safe_init(&rt, 64 * 1024 * 1024)? }; // 64MB swap

    // SHM accumulator: 10000 f32 results
    let shm_size = MAX_STREAMS as usize * 4;
    let shm_ptr = unsafe { platform::shm_safe_alloc(&rt, "mc_results", shm_size)? };
    unsafe {
        ptx_sys::cudaMemset(shm_ptr, 0, shm_size);
    }

    // VMM pages for tracking overflow stats
    let vmm_page = unsafe {
        platform::vmm_safe_alloc_page(vmm, ptx_sys::VMM_FLAG_READ | ptx_sys::VMM_FLAG_WRITE)?
    };
    unsafe { ptx_sys::vmm_pin_page(vmm, vmm_page); }

    let mut epoch = 0u64;
    let mut total_streams_dispatched: u64 = 0;
    let mut total_deferred_drained: u64 = 0;

    let start = Instant::now();
    let deadline = std::time::Duration::from_secs(duration_secs);

    println!("Starting Monte Carlo epoch loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();
        epoch += 1;

        let mut epoch_ptrs: Vec<*mut libc::c_void> = Vec::with_capacity(MAX_STREAMS as usize);
        let mut epoch_streams: Vec<ptx_runtime::Stream> = Vec::with_capacity(MAX_STREAMS as usize);

        // Dispatch in batches of BATCH_SIZE
        let num_batches = (MAX_STREAMS as usize + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..num_batches {
            if start.elapsed() >= deadline {
                // Free any already allocated in this partial epoch
                for ptr in &epoch_ptrs {
                    let stream = rt.next_stream();
                    unsafe { let _ = rt.free_async(*ptr, &stream); }
                }
                epoch_ptrs.clear();
                break;
            }

            let batch_start = batch_idx * BATCH_SIZE;
            let batch_end = std::cmp::min(batch_start + BATCH_SIZE, MAX_STREAMS as usize);

            for stream_idx in batch_start..batch_end {
                let stream = rt.stream(stream_idx as i32)?;

                // alloc_async → fill → exp → sigmoid → mul_scalar → reduce_sum → free_async
                let ptr = rt.alloc_async(SAMPLE_BYTES, &stream)?;

                // Fill with seed value based on stream index and epoch
                let seed_val = ((stream_idx as f32 + 1.0) * 0.001) + (epoch as f32 * 0.0001);
                unsafe {
                    ptx_sys::ptx_tensor_fill_f32(
                        ptr as *mut f32,
                        SAMPLES_PER_STREAM,
                        seed_val,
                        stream.raw(),
                    );

                    // exp(x) — simulates Monte Carlo transformation
                    ptx_sys::ptx_tensor_exp_f32(
                        ptr as *mut f32,
                        ptr as *mut f32,
                        SAMPLES_PER_STREAM,
                        stream.raw(),
                    );

                    // sigmoid(x) — normalize to [0,1]
                    ptx_sys::ptx_tensor_sigmoid_f32(
                        ptr as *mut f32,
                        ptr as *mut f32,
                        SAMPLES_PER_STREAM,
                        stream.raw(),
                    );

                    // mul_scalar(x, 0.5) — scale
                    ptx_sys::ptx_tensor_mul_scalar_f32(
                        ptr as *mut f32,
                        0.5,
                        ptr as *mut f32,
                        SAMPLES_PER_STREAM,
                        stream.raw(),
                    );

                    // reduce_sum → dedicated SHM slot (one slot per stream)
                    let shm_slot = (shm_ptr as *mut f32).add(stream_idx);
                    ptx_sys::ptx_tensor_reduce_sum_f32(
                        ptr as *mut f32,
                        shm_slot,
                        1, SAMPLES_PER_STREAM, 1,
                        stream.raw(),
                    );
                }

                epoch_ptrs.push(ptr);
                epoch_streams.push(stream);
                total_streams_dispatched += 1;
            }

            // Poll deferred frees between batches
            rt.poll_deferred(100);
            total_deferred_drained += 1;
        }

        // sync_all after epoch
        rt.sync_all()?;

        // Read one deterministic SHM sample result for telemetry
        let sample_idx = ((epoch.wrapping_mul(7919)) % MAX_STREAMS as u64) as usize;
        let sample_src = unsafe { (shm_ptr as *mut f32).add(sample_idx) };
        let mut final_result: f32 = 0.0;
        unsafe {
            ptx_sys::cudaMemcpy(
                &mut final_result as *mut f32 as *mut libc::c_void,
                sample_src as *const libc::c_void,
                4,
                ptx_sys::cudaMemcpyDeviceToHost,
            );
        }

        // Free all async allocations
        for (i, ptr) in epoch_ptrs.iter().enumerate() {
            let stream = if i < epoch_streams.len() {
                epoch_streams[i]
            } else {
                rt.next_stream()
            };
            unsafe { let _ = rt.free_async(*ptr, &stream); }
        }
        rt.poll_deferred(1000);

        // VMM: swap out the tracking page periodically, then swap back in
        if epoch % 5 == 0 {
            unsafe {
                ptx_sys::vmm_unpin_page(vmm, vmm_page);
                ptx_sys::vmm_swap_out(vmm, vmm_page);
                ptx_sys::vmm_swap_in(vmm, vmm_page);
                ptx_sys::vmm_pin_page(vmm, vmm_page);
            }
        }

        // Telemetry
        if reporter.should_report() {
            let (resident, swapped, faults, evictions) = unsafe {
                platform::vmm_safe_get_stats(vmm)
            };
            reporter.report(&rt, &format!(
                "epoch={} | streams={} | deferred={} | result={:.4} | vmm: res={} swap={} faults={} evict={}",
                epoch, total_streams_dispatched, total_deferred_drained,
                final_result, resident, swapped, faults, evictions,
            ));
        }
    }

    println!("\n=== STREAM COMPUTE COMPLETE ===");
    println!("Total epochs: {}", epoch);
    println!("Total streams dispatched: {}", total_streams_dispatched);
    println!("Total deferred drained: {}", total_deferred_drained);
    println!("Duration: {:.1}s", reporter.elapsed().as_secs_f64());

    // Cleanup
    unsafe {
        ptx_sys::vmm_unpin_page(vmm, vmm_page);
        ptx_sys::vmm_free_page(vmm, vmm_page);
        ptx_sys::vmm_shutdown(vmm);
        platform::shm_safe_unlink(&rt, "mc_results", shm_ptr)?;
    }

    rt.sync_all()?;
    rt.poll_deferred(10_000);
    platform::assert_clean_exit(&rt);

    Ok(())
}
