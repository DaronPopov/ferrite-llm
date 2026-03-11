use std::sync::Arc;

use anyhow::{Context, Result};
use ptx_runtime::PtxRuntime;
use ptx_sys::GPUHotConfig;

pub fn init_runtime(pool_fraction: f32, max_streams: u32) -> Result<Arc<PtxRuntime>> {
    let config = GPUHotConfig {
        pool_fraction,
        max_streams,
        enable_leak_detection: true,
        enable_pool_health: true,
        warning_threshold: 0.9,
        quiet_init: true,
        ..GPUHotConfig::default()
    };

    let rt = PtxRuntime::with_config(0, Some(config))
        .context("Failed to initialize PTX-OS runtime")?;

    rt.set_watchdog(30_000);
    rt.keepalive();

    Ok(Arc::new(rt))
}

pub fn get_duration_secs() -> u64 {
    std::env::var("DURATION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}
