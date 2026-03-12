use anyhow::Result;

#[cfg(feature = "tlsf-alloc")]
use std::sync::OnceLock;

#[cfg(feature = "tlsf-alloc")]
static TLSF_RUNTIME: OnceLock<ptx_runtime::PtxRuntime> = OnceLock::new();

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| !matches!(value.as_str(), "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
}

#[cfg(feature = "tlsf-alloc")]
fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.parse().ok()
}

#[cfg(feature = "tlsf-alloc")]
fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

#[cfg(feature = "tlsf-alloc")]
fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|value| match value.as_str() {
        "1" | "true" | "True" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "False" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    })
}

#[cfg(feature = "tlsf-alloc")]
fn prefer_orin_unified_memory() -> bool {
    env_enabled("FERRITE_TLSF_PREFER_ORIN_UM") || std::env::consts::ARCH == "aarch64"
}

#[cfg(feature = "tlsf-alloc")]
fn allocator_requested() -> bool {
    env_enabled("FERRITE_TLSF_ALLOC")
}

#[cfg(feature = "tlsf-alloc")]
fn configured_pool_fraction() -> f32 {
    env_f32("FERRITE_TLSF_POOL_FRACTION")
        .filter(|value| *value > 0.0 && *value <= 1.0)
        .unwrap_or(0.75)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_fixed_pool_size_bytes() -> usize {
    let mib = env_usize("FERRITE_TLSF_POOL_MB");
    let bytes = env_usize("FERRITE_TLSF_POOL_BYTES");
    bytes.or_else(|| mib.map(|value| value * 1024 * 1024)).unwrap_or(0)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_reserve_vram_bytes() -> usize {
    let mib = env_usize("FERRITE_TLSF_RESERVE_VRAM_MB");
    let bytes = env_usize("FERRITE_TLSF_RESERVE_VRAM_BYTES");
    bytes.or_else(|| mib.map(|value| value * 1024 * 1024))
        .unwrap_or(256 * 1024 * 1024)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_min_pool_size_bytes() -> usize {
    env_usize("FERRITE_TLSF_MIN_POOL_MB")
        .map(|value| value * 1024 * 1024)
        .or_else(|| env_usize("FERRITE_TLSF_MIN_POOL_BYTES"))
        .unwrap_or(512 * 1024 * 1024)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_max_pool_size_bytes() -> usize {
    env_usize("FERRITE_TLSF_MAX_POOL_MB")
        .map(|value| value * 1024 * 1024)
        .or_else(|| env_usize("FERRITE_TLSF_MAX_POOL_BYTES"))
        .unwrap_or(0)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_enable_pool_health() -> bool {
    env_bool("FERRITE_TLSF_ENABLE_POOL_HEALTH").unwrap_or(false)
}

#[cfg(feature = "tlsf-alloc")]
fn configured_warning_threshold() -> f32 {
    env_f32("FERRITE_TLSF_WARNING_THRESHOLD")
        .filter(|value| *value > 0.0 && *value <= 1.0)
        .unwrap_or(0.98)
}

#[cfg(feature = "tlsf-alloc")]
pub fn maybe_enable_tlsf_allocator(device_id: i32) -> Result<()> {
    if !allocator_requested() {
        return Ok(());
    }

    if TLSF_RUNTIME.get().is_some() {
        return Ok(());
    }

    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = configured_pool_fraction();
    config.fixed_pool_size = configured_fixed_pool_size_bytes();
    config.min_pool_size = configured_min_pool_size_bytes();
    config.max_pool_size = configured_max_pool_size_bytes();
    config.reserve_vram = configured_reserve_vram_bytes();
    config.prefer_orin_unified_memory = prefer_orin_unified_memory();
    config.quiet_init = !env_enabled("FERRITE_TLSF_VERBOSE");
    config.enable_pool_health = configured_enable_pool_health();
    config.warning_threshold = configured_warning_threshold();

    let runtime = ptx_runtime::PtxRuntime::with_config(device_id, Some(config))
        .map_err(|e| anyhow::anyhow!("Failed to initialize TLSF allocator runtime: {e}"))?;
    runtime.enable_hooks(env_enabled("FERRITE_TLSF_VERBOSE"));

    let _ = TLSF_RUNTIME.set(runtime);
    tracing::info!(
        device_id,
        pool_fraction = config.pool_fraction,
        fixed_pool_size = config.fixed_pool_size,
        min_pool_size = config.min_pool_size,
        max_pool_size = config.max_pool_size,
        reserve_vram = config.reserve_vram,
        prefer_orin_unified_memory = config.prefer_orin_unified_memory,
        "Ferrite TLSF allocator enabled"
    );
    Ok(())
}

#[cfg(not(feature = "tlsf-alloc"))]
pub fn maybe_enable_tlsf_allocator(_device_id: i32) -> Result<()> {
    if env_enabled("FERRITE_TLSF_ALLOC") {
        tracing::warn!(
            "FERRITE_TLSF_ALLOC was set, but ferrite was built without the `tlsf-alloc` feature"
        );
    }
    Ok(())
}
