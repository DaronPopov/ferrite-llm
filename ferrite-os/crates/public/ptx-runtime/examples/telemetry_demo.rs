//! Telemetry and Observability Demo
//!
//! Demonstrates how to use Ferrite-OS telemetry features:
//! - Structured logging with tracing
//! - Metrics collection
//! - Performance monitoring
//! - Log levels and filtering

use ptx_runtime::{PtxRuntime, telemetry};
use std::time::Duration;
use std::thread;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize telemetry (do this once at app startup)
    telemetry::init();

    // Alternatively, log to files:
    // telemetry::init_with_file("./logs")?;

    tracing::info!("🔍 Telemetry Demo Starting");
    tracing::info!("   Set RUST_LOG=trace to see detailed logs");
    tracing::info!("   Set RUST_LOG=debug to see debug logs");
    tracing::info!("   Set RUST_LOG=info for production (default)");
    println!();

    // Initialize runtime (will log initialization details)
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.6;
    config.max_streams = 16;
    config.quiet_init = false;

    tracing::info!("Initializing Ferrite-OS runtime...");
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    let stats = runtime.tlsf_stats();
    tracing::info!(
        pool_size_gb = stats.total_pool_size as f64 / 1e9,
        "Runtime initialized successfully"
    );
    println!();

    // Perform some operations (all logged and metriced)
    tracing::info!("Running workload with telemetry...");

    for iteration in 0..5 {
        tracing::debug!(iteration, "Starting iteration");

        let mut allocations = Vec::new();

        // Allocate memory (logged at TRACE level)
        for i in 0..100 {
            let size = 1024 * (i + 1);
            match runtime.alloc(size) {
                Ok(ptr) => allocations.push(ptr),
                Err(e) => {
                    tracing::error!(
                        size,
                        iteration,
                        error = ?e,
                        "Allocation failed"
                    );
                }
            }
        }

        tracing::debug!(
            iteration,
            allocations = allocations.len(),
            "Allocations complete"
        );

        // Sync (logged and metriced)
        runtime.sync_all()?;

        // Free memory (automatic on drop, logged at TRACE)
        drop(allocations);

        tracing::debug!(iteration, "Iteration complete");

        thread::sleep(Duration::from_millis(100));
    }

    println!();
    tracing::info!("Workload complete, collecting metrics...");
    println!();

    // Get metrics snapshot
    let metrics = telemetry::metrics().snapshot();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    TELEMETRY REPORT                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    println!("📊 Operation Counts:");
    println!("   Allocations: {}", metrics.allocations);
    println!("   Deallocations: {}", metrics.deallocations);
    println!("   Allocation failures: {}", metrics.allocation_failures);
    println!("   Stream syncs: {}", metrics.stream_synchronizations);
    println!();

    println!("💾 Memory Statistics:");
    println!("   Total allocated: {:.2} MB",
             metrics.total_bytes_allocated as f64 / 1e6);
    println!("   Total freed: {:.2} MB",
             metrics.total_bytes_freed as f64 / 1e6);
    println!("   Net allocated: {:.2} MB",
             metrics.net_allocated_bytes() as f64 / 1e6);
    println!();

    println!("✅ Success Rates:");
    println!("   Allocation success: {:.2}%",
             metrics.allocation_success_rate() * 100.0);
    println!();

    // Get TLSF stats
    let tlsf_stats = runtime.tlsf_stats();
    println!("🏥 TLSF Pool Health:");
    println!("   Current allocated: {:.2} MB",
             tlsf_stats.allocated_bytes as f64 / 1e6);
    println!("   Utilization: {:.1}%", tlsf_stats.utilization_percent);
    println!("   Fragmentation: {:.6}%", tlsf_stats.fragmentation_ratio * 100.0);
    println!();

    tracing::info!("✅ Telemetry demo complete");

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   HOW TO USE TELEMETRY                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("1. Log Levels (set RUST_LOG environment variable):");
    println!("   RUST_LOG=error   - Only errors");
    println!("   RUST_LOG=warn    - Warnings and errors");
    println!("   RUST_LOG=info    - Info, warnings, errors (default)");
    println!("   RUST_LOG=debug   - Debug info + above");
    println!("   RUST_LOG=trace   - Everything (very verbose)");
    println!();
    println!("2. Module-specific logging:");
    println!("   RUST_LOG=ptx_runtime=debug  - Debug runtime only");
    println!("   RUST_LOG=ptx_runtime::memory=trace - Trace memory ops");
    println!();
    println!("3. Metrics in your code:");
    println!("   use ptx_runtime::telemetry;");
    println!("   let metrics = telemetry::metrics().snapshot();");
    println!("   println!(\"Allocations: {{}}\", metrics.allocations);");
    println!();
    println!("4. Performance timing:");
    println!("   let _timer = telemetry::OpTimer::new(\"my_operation\");");
    println!("   // ... operation runs ...");
    println!("   // Timer logs duration when dropped");
    println!();

    Ok(())
}
