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
fn prefer_orin_unified_memory() -> bool {
    env_enabled("FERRITE_TLSF_PREFER_ORIN_UM") || std::env::consts::ARCH == "aarch64"
}

#[cfg(feature = "tlsf-alloc")]
fn allocator_requested() -> bool {
    env_enabled("FERRITE_TLSF_ALLOC")
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
    config.prefer_orin_unified_memory = prefer_orin_unified_memory();
    config.quiet_init = !env_enabled("FERRITE_TLSF_VERBOSE");

    let runtime = ptx_runtime::PtxRuntime::with_config(device_id, Some(config))
        .map_err(|e| anyhow::anyhow!("Failed to initialize TLSF allocator runtime: {e}"))?;
    runtime.enable_hooks(env_enabled("FERRITE_TLSF_VERBOSE"));

    let _ = TLSF_RUNTIME.set(runtime);
    tracing::info!(
        device_id,
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
