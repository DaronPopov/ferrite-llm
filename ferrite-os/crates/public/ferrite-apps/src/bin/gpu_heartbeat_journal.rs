//! gpu_heartbeat_journal — Keepalive + Tensor Heartbeat + VFS Journal.
//!
//! Runs as a lightweight daemon-facing workload that continuously:
//! 1) emits runtime keepalive,
//! 2) executes a small tensor heartbeat on GPU, and
//! 3) appends heartbeat records into a VFS journal file.

use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.25;
const MAX_STREAMS: u32 = 8;
const TENSOR_LEN: usize = 4096;
const TICK_SLEEP_MS: u64 = 250;
const JOURNAL_DIR: &str = "/journal";
const JOURNAL_FILE: &str = "/journal/heartbeat.log";

unsafe fn vfs_write_all(vfs: *mut ptx_sys::VFSState, fd: i32, bytes: &[u8]) -> Result<()> {
    let rc = ptx_sys::vfs_write(
        vfs,
        fd,
        bytes.as_ptr() as *const libc::c_void,
        bytes.len(),
    );
    if rc < 0 {
        bail!("vfs_write failed: {}", rc);
    }
    if rc as usize != bytes.len() {
        bail!("short vfs_write: wrote {} of {} bytes", rc, bytes.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();

    println!("=== GPU HEARTBEAT JOURNAL ===");
    println!("keepalive + tensor heartbeat + VFS journal append");
    if duration_secs == 0 {
        println!("Duration: infinite (DURATION=0)");
    } else {
        println!("Duration: {}", platform::format_duration(duration_secs));
    }
    println!(
        "Config: pool_fraction={} max_streams={} tensor_len={}",
        POOL_FRACTION, MAX_STREAMS, TENSOR_LEN
    );
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("heartbeat", 5);

    let vfs = unsafe { platform::vfs_safe_init(&rt)? };
    unsafe {
        platform::vfs_safe_mkdir(vfs, JOURNAL_DIR)?;
    }

    let fd = unsafe {
        platform::vfs_safe_open(
            vfs,
            JOURNAL_FILE,
            ptx_sys::VFS_O_CREAT | ptx_sys::VFS_O_TRUNC | ptx_sys::VFS_O_APPEND | ptx_sys::VFS_O_RDWR,
        )?
    };

    let stream = rt.next_stream();
    let tensor_bytes = TENSOR_LEN * std::mem::size_of::<f32>();
    let metric_bytes = std::mem::size_of::<f32>();

    let tensor = rt
        .alloc_async(tensor_bytes, &stream)
        .context("alloc heartbeat tensor")?;
    let metric = rt
        .alloc_async(metric_bytes, &stream)
        .context("alloc heartbeat metric")?;

    let start = Instant::now();
    let deadline = if duration_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(duration_secs))
    };

    let mut tick: u64 = 0;
    while deadline.map(|d| start.elapsed() < d).unwrap_or(true) {
        rt.keepalive();

        let phase = tick as f32 * 0.07;
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(tensor as *mut f32, TENSOR_LEN, 0.0, 0.01, stream.raw());
            ptx_sys::ptx_tensor_add_scalar_f32(
                tensor as *mut f32,
                phase,
                tensor as *mut f32,
                TENSOR_LEN,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_sin_f32(
                tensor as *mut f32,
                tensor as *mut f32,
                TENSOR_LEN,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_mean_f32(
                tensor as *mut f32,
                metric as *mut f32,
                1,
                TENSOR_LEN,
                1,
                stream.raw(),
            );
        }
        stream.synchronize()?;

        let mut mean: f32 = 0.0;
        unsafe {
            let rc = ptx_sys::cudaMemcpy(
                &mut mean as *mut f32 as *mut libc::c_void,
                metric as *const libc::c_void,
                metric_bytes,
                ptx_sys::cudaMemcpyDeviceToHost,
            );
            if rc != ptx_sys::cudaSuccess {
                bail!("cudaMemcpy D2H metric failed: {}", rc);
            }
        }

        let tlsf = rt.tlsf_stats();
        let line = format!(
            "tick={} mean={:.6} pool_used={} frag={:.6} watchdog={}\n",
            tick,
            mean,
            tlsf.allocated_bytes,
            tlsf.fragmentation_ratio,
            if rt.check_watchdog() { "TRIPPED" } else { "ok" },
        );
        unsafe { vfs_write_all(vfs, fd, line.as_bytes())?; }

        if reporter.should_report() {
            reporter.report(
                &rt,
                &format!("tick={} mean={:.6} journal={}", tick, mean, JOURNAL_FILE),
            );
        }

        tick += 1;
        thread::sleep(Duration::from_millis(TICK_SLEEP_MS));
    }

    println!("\n=== HEARTBEAT COMPLETE ===");
    println!("ticks={} elapsed={:.2}s", tick, start.elapsed().as_secs_f64());

    let cleanup_stream = rt.next_stream();
    unsafe {
        let _ = rt.free_async(metric, &cleanup_stream);
        let _ = rt.free_async(tensor, &cleanup_stream);
    }
    rt.sync_all()?;
    rt.poll_deferred(10_000);

    unsafe {
        let rc = ptx_sys::vfs_close(vfs, fd);
        if rc < 0 {
            eprintln!("warning: vfs_close failed: {}", rc);
        }
        let _ = platform::vfs_safe_unlink(vfs, JOURNAL_FILE);
        let _ = platform::vfs_safe_rmdir(vfs, JOURNAL_DIR);
        ptx_sys::vfs_shutdown(vfs);
    }

    platform::assert_clean_exit(&rt);
    Ok(())
}
