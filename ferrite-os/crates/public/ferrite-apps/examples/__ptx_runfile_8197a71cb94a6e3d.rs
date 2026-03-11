// memory_churn_guard — async alloc/free churn with invariant checks.
//
// Stresses the allocator with variable-size stream-ordered churn while
// continuously validating pool integrity and leak-free cleanup behavior.

use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use rand::Rng;

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.45;
const MAX_STREAMS: u32 = 256;
const OPS_PER_CYCLE: usize = 384;
const INFLIGHT_TARGET: usize = 1200;
const MIN_ALLOC: usize = 256;
const MAX_ALLOC: usize = 256 * 1024;
const ALIGN: usize = 256;
const CYCLE_SLEEP_MS: u64 = 8;

#[derive(Clone, Copy)]
struct InFlight {
    ptr: *mut libc::c_void,
    stream: ptx_runtime::Stream,
    size: usize,
}

fn aligned_size(raw: usize) -> usize {
    let clamped = raw.clamp(MIN_ALLOC, MAX_ALLOC);
    ((clamped + ALIGN - 1) / ALIGN) * ALIGN
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();

    println!("=== MEMORY CHURN GUARD ===");
    println!("variable-size async alloc/free churn with invariant checks");
    if duration_secs == 0 {
        println!("Duration: infinite (DURATION=0)");
    } else {
        println!("Duration: {}", platform::format_duration(duration_secs));
    }
    println!(
        "Config: pool_fraction={} max_streams={} inflight_target={} ops/cycle={}",
        POOL_FRACTION, MAX_STREAMS, INFLIGHT_TARGET, OPS_PER_CYCLE
    );
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("churn_guard", 5);
    let mut rng = rand::thread_rng();

    let start = Instant::now();
    let deadline = if duration_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(duration_secs))
    };

    let mut inflight: Vec<InFlight> = Vec::with_capacity(INFLIGHT_TARGET + OPS_PER_CYCLE);
    let mut cycle: u64 = 0;
    let mut total_allocs: u64 = 0;
    let mut total_frees: u64 = 0;
    let mut bytes_allocated: u64 = 0;
    let mut bytes_freed: u64 = 0;

    while deadline.map(|d| start.elapsed() < d).unwrap_or(true) {
        rt.keepalive();
        cycle += 1;

        for _ in 0..OPS_PER_CYCLE {
            let stream = rt.next_stream();
            let size = aligned_size(rng.gen_range(MIN_ALLOC..=MAX_ALLOC));

            if !rt.can_allocate(size) {
                rt.poll_deferred(1000);
                if !rt.can_allocate(size) {
                    continue;
                }
            }

            let ptr = match rt.alloc_async(size, &stream) {
                Ok(ptr) => ptr,
                Err(_) => {
                    rt.poll_deferred(2000);
                    continue;
                }
            };

            // Touch allocation with a lightweight fill to ensure work is enqueued.
            unsafe {
                ptx_sys::ptx_tensor_fill_f32(
                    ptr as *mut f32,
                    size / std::mem::size_of::<f32>(),
                    (cycle % 97) as f32 * 0.001,
                    stream.raw(),
                );
            }

            inflight.push(InFlight { ptr, stream, size });
            total_allocs += 1;
            bytes_allocated += size as u64;

            while inflight.len() > INFLIGHT_TARGET {
                let idx = rng.gen_range(0..inflight.len());
                let evict = inflight.swap_remove(idx);
                unsafe {
                    rt.free_async(evict.ptr, &evict.stream)?;
                }
                total_frees += 1;
                bytes_freed += evict.size as u64;
            }
        }

        if cycle % 8 == 0 {
            rt.sync_all()?;
            rt.poll_deferred(5000);

            let health = rt.validate_pool();
            ensure!(health.is_valid, "pool invalid at cycle {}", cycle);
            ensure!(
                !health.has_corrupted_blocks,
                "pool corruption detected at cycle {}",
                cycle
            );
        }

        if reporter.should_report() {
            let tlsf = rt.tlsf_stats();
            reporter.report(
                &rt,
                &format!(
                    "cycle={} inflight={} allocs={} frees={} net={}MB",
                    cycle,
                    inflight.len(),
                    total_allocs,
                    total_frees,
                    (bytes_allocated.saturating_sub(bytes_freed)) as f64 / (1024.0 * 1024.0),
                ),
            );
            if tlsf.fragmentation_ratio > 0.25 {
                eprintln!(
                    "warning: elevated fragmentation ratio {:.6} at cycle {}",
                    tlsf.fragmentation_ratio, cycle
                );
            }
        }

        thread::sleep(Duration::from_millis(CYCLE_SLEEP_MS));
    }

    for alloc in inflight.drain(..) {
        unsafe {
            rt.free_async(alloc.ptr, &alloc.stream)?;
        }
        total_frees += 1;
        bytes_freed += alloc.size as u64;
    }

    rt.sync_all()?;
    rt.poll_deferred(20_000);

    let tlsf = rt.tlsf_stats();
    if tlsf.allocated_bytes > 256 {
        bail!(
            "post-drain allocated bytes too high: {} (expected <=256)",
            tlsf.allocated_bytes
        );
    }

    println!("\n=== CHURN GUARD COMPLETE ===");
    println!("cycles={} elapsed={:.2}s", cycle, start.elapsed().as_secs_f64());
    println!(
        "allocs={} frees={} bytes_in={} bytes_out={}",
        total_allocs,
        total_frees,
        platform::format_bytes(bytes_allocated as usize),
        platform::format_bytes(bytes_freed as usize),
    );

    platform::assert_clean_exit(&rt);
    Ok(())
}
