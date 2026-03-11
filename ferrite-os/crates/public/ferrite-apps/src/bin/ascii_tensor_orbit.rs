//! ascii_tensor_orbit — continuous daemon-friendly ASCII + tensor demo.
//!
//! Runs a lightweight frame loop that:
//! 1) executes GPU tensor ops each frame, and
//! 2) paints animated ASCII art to stdout.
//!
//! Discovery: appears in daemon `run-list` as
//! `ferrite-apps/src/bin/ascii_tensor_orbit.rs#main`.

use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.30;
const MAX_STREAMS: u32 = 4;
const TENSOR_LEN: usize = 2048;
const PREVIEW_LEN: usize = 64;
const FRAME_SLEEP_MS: u64 = 90;

fn shade(v: f32) -> char {
    const PALETTE: &[u8] = b" .:-=+*#%@";
    let clamped = v.clamp(0.0, 1.0);
    let idx = (clamped * (PALETTE.len() as f32 - 1.0)).round() as usize;
    PALETTE[idx] as char
}

fn preview_line(values: &[f32]) -> String {
    values.iter().map(|&v| shade(v)).collect()
}

fn spinner(frame: u64) -> char {
    match frame % 4 {
        0 => '|',
        1 => '/',
        2 => '-',
        _ => '\\',
    }
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();

    println!("=== ASCII TENSOR ORBIT ===");
    println!("daemon run-entry demo (continuous ASCII + GPU tensor math)");
    if duration_secs == 0 {
        println!("duration: infinite (DURATION=0)");
    } else {
        println!("duration: {}", platform::format_duration(duration_secs));
    }
    println!("tip: set DURATION=0 for endless mode");
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let stream = rt.next_stream();

    let tensor_bytes = TENSOR_LEN * std::mem::size_of::<f32>();
    let metric_bytes = std::mem::size_of::<f32>();

    let tensor = rt
        .alloc_async(tensor_bytes, &stream)
        .context("alloc frame tensor")?;
    let metric = rt
        .alloc_async(metric_bytes, &stream)
        .context("alloc frame metric")?;

    let start = Instant::now();
    let deadline = if duration_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(duration_secs))
    };

    print!("\x1B[2J");
    io::stdout().flush().ok();

    let mut frame: u64 = 0;
    loop {
        if let Some(max_dur) = deadline {
            if start.elapsed() >= max_dur {
                break;
            }
        }

        rt.keepalive();
        let phase = frame as f32 * 0.11;

        unsafe {
            ptx_sys::ptx_tensor_arange_f32(tensor as *mut f32, TENSOR_LEN, 0.0, 0.06, stream.raw());
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
            ptx_sys::ptx_tensor_sigmoid_f32(
                tensor as *mut f32,
                tensor as *mut f32,
                TENSOR_LEN,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_sum_f32(
                tensor as *mut f32,
                metric as *mut f32,
                1,
                TENSOR_LEN,
                1,
                stream.raw(),
            );
        }

        rt.sync_all().context("sync frame")?;

        let mut sum: f32 = 0.0;
        let mut preview = vec![0f32; PREVIEW_LEN];

        unsafe {
            ptx_sys::cudaMemcpy(
                &mut sum as *mut f32 as *mut libc::c_void,
                metric as *const libc::c_void,
                metric_bytes,
                ptx_sys::cudaMemcpyDeviceToHost,
            );
            ptx_sys::cudaMemcpy(
                preview.as_mut_ptr() as *mut libc::c_void,
                tensor as *const libc::c_void,
                PREVIEW_LEN * std::mem::size_of::<f32>(),
                ptx_sys::cudaMemcpyDeviceToHost,
            );
        }

        let mean = sum / TENSOR_LEN as f32;
        let orbit = preview_line(&preview);
        let mut ship = vec![' '; PREVIEW_LEN];
        ship[(frame as usize) % PREVIEW_LEN] = '^';
        let ship_line: String = ship.into_iter().collect();

        print!("\x1B[H");
        println!("ASCII Tensor Orbit  [{}]", spinner(frame));
        println!("frame={}  mean={:.5}  streams={} pool={:.2}", frame, mean, MAX_STREAMS, POOL_FRACTION);
        println!();
        println!("  {}", orbit);
        println!("  {}", ship_line);
        println!("  {}", orbit.chars().rev().collect::<String>());
        io::stdout().flush().ok();

        frame += 1;
        thread::sleep(Duration::from_millis(FRAME_SLEEP_MS));
    }

    println!("\ncompleted: frames={} elapsed={:.2}s", frame, start.elapsed().as_secs_f64());

    unsafe {
        let _ = rt.free_async(metric, &stream);
        let _ = rt.free_async(tensor, &stream);
    }
    rt.sync_all()?;
    rt.poll_deferred(10_000);
    platform::assert_clean_exit(&rt);

    Ok(())
}
