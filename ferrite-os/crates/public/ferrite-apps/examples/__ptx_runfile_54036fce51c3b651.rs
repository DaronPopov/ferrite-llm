// gpu_logwatch — GPU-Resident Log Stream Monitor
//
// Simulates a real-time log analytics service running entirely on GPU.
// Continuously ingests batches of simulated log metrics (latencies, error
// rates), performs rolling statistical analysis via GPU reductions, and
// emits anomaly alerts when values exceed adaptive thresholds.
//
// Demonstrates: long-running GPU service with continuous alloc/compute/free
// cycles, rolling windows, anomaly detection, and periodic reporting — all
// on GPU runtime primitives (TLSF, streams, tensor ops).

use std::time::{Duration, Instant};

use anyhow::Result;

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.40;
const MAX_STREAMS: u32 = 64;

/// Elements per log batch — each f32 represents a simulated request latency.
const BATCH_SIZE: usize = 8192;
const BATCH_BYTES: usize = BATCH_SIZE * std::mem::size_of::<f32>();

/// Rolling window: keep the last N batches alive for trend analysis.
const WINDOW_SIZE: usize = 8;

fn main() -> Result<()> {
    let duration_secs = std::env::args()
        .position(|a| a == "--duration")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(platform::get_duration_secs);

    let burst_every: u64 = std::env::args()
        .position(|a| a == "--burst-every")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    let report_interval_secs: u64 = std::env::args()
        .position(|a| a == "--report-interval")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    println!("=== GPU LOG STREAM MONITOR ===");
    println!("GPU-resident log analytics with rolling anomaly detection");
    println!(
        "Duration: {}  Burst every: {} cycles  Report interval: {}s",
        platform::format_duration(duration_secs),
        burst_every,
        report_interval_secs,
    );
    println!(
        "Config: batch_size={}  window={}  pool_fraction={}",
        BATCH_SIZE, WINDOW_SIZE, POOL_FRACTION,
    );
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("logwatch", report_interval_secs);

    // Scratch buffers for reductions (1 element each)
    let mean_buf = rt.alloc(4)?;
    let max_buf = rt.alloc(4)?;
    let min_buf = rt.alloc(4)?;

    // Rolling window of live GPU batches
    let mut window: Vec<ptx_runtime::GpuPtr> = Vec::with_capacity(WINDOW_SIZE);

    let mut cycle: u64 = 0;
    let mut total_ingested: u64 = 0;
    let mut total_anomalies: u64 = 0;
    let mut recent_anomalies: u64 = 0;
    let mut seed: u64 = 42;

    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);

    println!("Starting log ingest loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();
        cycle += 1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        let stream = rt.next_stream();
        let is_burst = burst_every > 0 && cycle % burst_every == 0;

        // 1. Allocate and populate a new log batch
        let batch = rt.alloc(BATCH_BYTES)?;
        unsafe {
            // Fill with normal-distributed simulated latencies (mean ~0, std ~1)
            ptx_sys::ptx_tensor_randn_f32(
                batch.as_ptr_typed::<f32>(),
                BATCH_SIZE,
                seed,
                stream.raw(),
            );
            // Shift to realistic latency range: mean ~5ms, std ~1ms
            ptx_sys::ptx_tensor_mul_scalar_f32(
                batch.as_ptr_typed::<f32>(),
                1.0,
                batch.as_ptr_typed::<f32>(),
                BATCH_SIZE,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_add_scalar_f32(
                batch.as_ptr_typed::<f32>(),
                5.0,
                batch.as_ptr_typed::<f32>(),
                BATCH_SIZE,
                stream.raw(),
            );
        }

        // On burst cycles, inject anomalous spikes (simulates traffic surge)
        if is_burst {
            unsafe {
                ptx_sys::ptx_tensor_mul_scalar_f32(
                    batch.as_ptr_typed::<f32>(),
                    10.0,
                    batch.as_ptr_typed::<f32>(),
                    BATCH_SIZE,
                    stream.raw(),
                );
            }
        }

        // 2. GPU-side analytics: compute batch statistics
        unsafe {
            ptx_sys::ptx_tensor_reduce_mean_f32(
                batch.as_ptr_typed::<f32>(),
                mean_buf.as_ptr_typed::<f32>(),
                1,
                BATCH_SIZE,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_max_f32(
                batch.as_ptr_typed::<f32>(),
                max_buf.as_ptr_typed::<f32>(),
                1,
                BATCH_SIZE,
                1,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_reduce_min_f32(
                batch.as_ptr_typed::<f32>(),
                min_buf.as_ptr_typed::<f32>(),
                1,
                BATCH_SIZE,
                1,
                stream.raw(),
            );
        }

        stream.synchronize()?;

        // 3. Read back statistics
        let mut mean_val: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        let mut min_val: f32 = 0.0;
        unsafe {
            mean_buf.copy_to_host(&mut mean_val as *mut f32 as *mut libc::c_void, 4)?;
            max_buf.copy_to_host(&mut max_val as *mut f32 as *mut libc::c_void, 4)?;
            min_buf.copy_to_host(&mut min_val as *mut f32 as *mut libc::c_void, 4)?;
        }

        total_ingested += BATCH_SIZE as u64;

        // 4. Anomaly detection: mean > 15ms is anomalous (normal ~5ms)
        let anomaly = mean_val > 15.0;
        if anomaly {
            total_anomalies += 1;
            recent_anomalies += 1;
            println!(
                "  ANOMALY [{}] mean={:.2} max={:.2} min={:.2} (burst={})",
                cycle, mean_val, max_val, min_val, is_burst,
            );
        }

        // 5. Manage rolling window: add new batch, evict oldest
        window.push(batch);
        if window.len() > WINDOW_SIZE {
            let _evicted = window.remove(0); // GpuPtr drop frees TLSF
        }

        // 6. Periodic reporting
        if reporter.should_report() {
            let pool = rt.tlsf_stats();
            let elapsed = start.elapsed().as_secs();
            let rate = if elapsed > 0 {
                total_ingested / elapsed
            } else {
                total_ingested
            };
            reporter.report(
                &rt,
                &format!(
                    "ingest={} rate={}/s batches={} anomalies={} (recent={}) window={} pool_used={:.1}MB",
                    platform::format_bytes(total_ingested as usize * 4),
                    rate,
                    cycle,
                    total_anomalies,
                    recent_anomalies,
                    window.len(),
                    pool.allocated_bytes as f64 / (1024.0 * 1024.0),
                ),
            );
            recent_anomalies = 0;
        }

        // Brief yield for OS scheduling
        std::thread::sleep(Duration::from_millis(20));
    }

    let elapsed = reporter.elapsed();
    let rate = if elapsed.as_secs() > 0 {
        total_ingested / elapsed.as_secs()
    } else {
        total_ingested
    };

    println!("\n=== GPU LOGWATCH COMPLETE ===");
    println!("Total ingested: {} events ({} batches)", total_ingested, cycle);
    println!("Throughput: {}/s", rate);
    println!("Anomalies detected: {}", total_anomalies);
    println!(
        "Duration: {:.1}s",
        elapsed.as_secs_f64(),
    );

    // Cleanup: drop window batches + scratch buffers
    window.clear();
    drop(mean_buf);
    drop(max_buf);
    drop(min_buf);

    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}
