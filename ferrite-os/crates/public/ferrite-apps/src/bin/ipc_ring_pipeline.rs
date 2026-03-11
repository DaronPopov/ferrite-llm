//! ipc_ring_pipeline — SHM-backed producer/consumer ring with GPU payloads.
//!
//! Simulates two app endpoints sharing a ring buffer through SHM:
//! - producer computes scalar payloads on GPU and writes them into SHM slots,
//! - consumer reads SHM slots, transforms them on GPU, and records throughput.

use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.30;
const MAX_STREAMS: u32 = 16;
const SHM_NAME: &str = "ipc_ring_pipeline";
const RING_CAPACITY: usize = 128;
const WORK_TENSOR_LEN: usize = 2048;
const CYCLE_SLEEP_MS: u64 = 20;

const OFF_HEAD: usize = 0;
const OFF_TAIL: usize = 4;
const OFF_PRODUCED: usize = 8;
const OFF_CONSUMED: usize = 16;
const OFF_SLOTS: usize = 24;
const SHM_BYTES: usize = OFF_SLOTS + (RING_CAPACITY * std::mem::size_of::<f32>());

unsafe fn memcpy_checked(
    dst: *mut libc::c_void,
    src: *const libc::c_void,
    bytes: usize,
    kind: libc::c_int,
    label: &str,
) -> Result<()> {
    let rc = ptx_sys::cudaMemcpy(dst, src, bytes, kind);
    if rc != ptx_sys::cudaSuccess {
        bail!("cudaMemcpy failed ({label}): {}", rc);
    }
    Ok(())
}

unsafe fn write_ring_meta(
    base: *mut libc::c_void,
    head: u32,
    tail: u32,
    produced: u64,
    consumed: u64,
) -> Result<()> {
    memcpy_checked(
        (base as *mut u8).add(OFF_HEAD) as *mut libc::c_void,
        &head as *const u32 as *const libc::c_void,
        std::mem::size_of::<u32>(),
        ptx_sys::cudaMemcpyHostToDevice,
        "meta head",
    )?;
    memcpy_checked(
        (base as *mut u8).add(OFF_TAIL) as *mut libc::c_void,
        &tail as *const u32 as *const libc::c_void,
        std::mem::size_of::<u32>(),
        ptx_sys::cudaMemcpyHostToDevice,
        "meta tail",
    )?;
    memcpy_checked(
        (base as *mut u8).add(OFF_PRODUCED) as *mut libc::c_void,
        &produced as *const u64 as *const libc::c_void,
        std::mem::size_of::<u64>(),
        ptx_sys::cudaMemcpyHostToDevice,
        "meta produced",
    )?;
    memcpy_checked(
        (base as *mut u8).add(OFF_CONSUMED) as *mut libc::c_void,
        &consumed as *const u64 as *const libc::c_void,
        std::mem::size_of::<u64>(),
        ptx_sys::cudaMemcpyHostToDevice,
        "meta consumed",
    )?;
    Ok(())
}

unsafe fn slot_ptr(base: *mut libc::c_void, slot: usize) -> *mut libc::c_void {
    (base as *mut u8).add(OFF_SLOTS + slot * std::mem::size_of::<f32>()) as *mut libc::c_void
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();

    println!("=== IPC RING PIPELINE ===");
    println!("SHM-backed producer/consumer ring with GPU payloads");
    if duration_secs == 0 {
        println!("Duration: infinite (DURATION=0)");
    } else {
        println!("Duration: {}", platform::format_duration(duration_secs));
    }
    println!(
        "Config: pool_fraction={} max_streams={} ring_capacity={}",
        POOL_FRACTION, MAX_STREAMS, RING_CAPACITY
    );
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("ipc_ring", 5);

    let producer_ring = unsafe { platform::shm_safe_alloc(&rt, SHM_NAME, SHM_BYTES)? };
    let consumer_ring = unsafe { platform::shm_safe_open(&rt, SHM_NAME)? };
    unsafe {
        ptx_sys::cudaMemset(producer_ring, 0, SHM_BYTES);
    }

    let producer_stream = rt.stream(0)?;
    let consumer_stream = rt.stream(1)?;

    let work = rt.alloc(WORK_TENSOR_LEN * std::mem::size_of::<f32>())?;
    let producer_metric = rt.alloc(std::mem::size_of::<f32>())?;
    let consumer_metric = rt.alloc(std::mem::size_of::<f32>())?;

    let start = Instant::now();
    let deadline = if duration_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(duration_secs))
    };

    let mut cycle: u64 = 0;
    let mut head: usize = 0;
    let mut tail: usize = 0;
    let mut produced: u64 = 0;
    let mut consumed: u64 = 0;

    while deadline.map(|d| start.elapsed() < d).unwrap_or(true) {
        rt.keepalive();
        cycle += 1;

        // Producer phase: generate a scalar payload on GPU, enqueue into SHM ring.
        let phase = cycle as f32 * 0.025;
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(
                work.as_ptr_typed::<f32>(),
                WORK_TENSOR_LEN,
                0.0,
                0.01,
                producer_stream.raw(),
            );
            ptx_sys::ptx_tensor_add_scalar_f32(
                work.as_ptr_typed::<f32>(),
                phase,
                work.as_ptr_typed::<f32>(),
                WORK_TENSOR_LEN,
                producer_stream.raw(),
            );
            ptx_sys::ptx_tensor_sin_f32(
                work.as_ptr_typed::<f32>(),
                work.as_ptr_typed::<f32>(),
                WORK_TENSOR_LEN,
                producer_stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_mean_f32(
                work.as_ptr_typed::<f32>(),
                producer_metric.as_ptr_typed::<f32>(),
                1,
                WORK_TENSOR_LEN,
                1,
                producer_stream.raw(),
            );
        }
        producer_stream.synchronize()?;

        let ring_len = head.saturating_sub(tail);
        if ring_len < RING_CAPACITY {
            let slot = head % RING_CAPACITY;
            unsafe {
                memcpy_checked(
                    slot_ptr(producer_ring, slot),
                    producer_metric.as_ptr() as *const libc::c_void,
                    std::mem::size_of::<f32>(),
                    ptx_sys::cudaMemcpyDeviceToDevice,
                    "enqueue payload",
                )?;
            }
            head += 1;
            produced += 1;
        }

        // Consumer phase: dequeue payload, transform on GPU, and account throughput.
        if head > tail {
            let slot = tail % RING_CAPACITY;
            unsafe {
                memcpy_checked(
                    consumer_metric.as_ptr(),
                    slot_ptr(consumer_ring, slot) as *const libc::c_void,
                    std::mem::size_of::<f32>(),
                    ptx_sys::cudaMemcpyDeviceToDevice,
                    "dequeue payload",
                )?;
                ptx_sys::ptx_tensor_sigmoid_f32(
                    consumer_metric.as_ptr_typed::<f32>(),
                    consumer_metric.as_ptr_typed::<f32>(),
                    1,
                    consumer_stream.raw(),
                );
            }
            consumer_stream.synchronize()?;
            tail += 1;
            consumed += 1;
        }

        unsafe {
            write_ring_meta(
                producer_ring,
                head as u32,
                tail as u32,
                produced,
                consumed,
            )?;
        }

        if reporter.should_report() {
            let mut sample: f32 = 0.0;
            unsafe {
                memcpy_checked(
                    &mut sample as *mut f32 as *mut libc::c_void,
                    consumer_metric.as_ptr() as *const libc::c_void,
                    std::mem::size_of::<f32>(),
                    ptx_sys::cudaMemcpyDeviceToHost,
                    "sample metric",
                )?;
            }

            reporter.report(
                &rt,
                &format!(
                    "cycle={} produced={} consumed={} backlog={} sample={:.6}",
                    cycle,
                    produced,
                    consumed,
                    head.saturating_sub(tail),
                    sample,
                ),
            );
        }

        thread::sleep(Duration::from_millis(CYCLE_SLEEP_MS));
    }

    println!("\n=== IPC RING COMPLETE ===");
    println!("cycles={} elapsed={:.2}s", cycle, start.elapsed().as_secs_f64());
    println!(
        "produced={} consumed={} final_backlog={}",
        produced,
        consumed,
        head.saturating_sub(tail)
    );

    drop(consumer_metric);
    drop(producer_metric);
    drop(work);

    rt.sync_all()?;

    unsafe {
        platform::shm_safe_close(&rt, consumer_ring);
        platform::shm_safe_unlink(&rt, SHM_NAME, producer_ring)?;
    }

    rt.sync_all()?;
    rt.poll_deferred(10_000);
    platform::assert_clean_exit(&rt);

    Ok(())
}
