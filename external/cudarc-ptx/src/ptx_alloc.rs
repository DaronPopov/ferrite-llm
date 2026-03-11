//! PTX-OS TLSF Allocator Integration
//!
//! This module provides the TLSF allocator backend for cudarc.
//! When the `ptx-alloc` feature is enabled, all device allocations
//! go through PTX-OS TLSF instead of stock cudaMalloc.

#[cfg(feature = "ptx-alloc")]
use ptx_runtime::{global_runtime, GpuPtr, PtxRuntime};
#[cfg(feature = "ptx-alloc")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "ptx-alloc")]
use std::collections::HashMap;

use crate::driver::sys::CUdeviceptr;

#[cfg(feature = "ptx-alloc")]
static GLOBAL_RUNTIME: OnceLock<Arc<ptx_runtime::PtxRuntime>> = OnceLock::new();
#[cfg(feature = "ptx-alloc")]
static PTR_MAP: OnceLock<Mutex<HashMap<CUdeviceptr, Arc<GpuPtr>>>> = OnceLock::new();

/// Initialize PTX runtime (lazy, thread-safe)
#[cfg(feature = "ptx-alloc")]
fn get_or_init_runtime() -> Arc<PtxRuntime> {
    let runtime = GLOBAL_RUNTIME.get_or_init(|| {
        let runtime = global_runtime()
            .unwrap_or_else(|e| panic!("[cudarc-ptx] FATAL: Failed to initialize TLSF runtime: {:?}", e));
        runtime.export_for_hook();
        runtime.enable_hooks(false);
        eprintln!("[cudarc-ptx] ✓ TLSF allocator attached to global PTX runtime");
        runtime
    });

    PTR_MAP.get_or_init(|| Mutex::new(HashMap::new()));
    Arc::clone(runtime)
}

/// Allocate via TLSF (replaces cuMemAlloc)
#[cfg(feature = "ptx-alloc")]
pub unsafe fn tlsf_malloc(num_bytes: usize) -> Result<CUdeviceptr, crate::driver::result::DriverError> {
    let runtime = get_or_init_runtime();

    match runtime.alloc(num_bytes) {
        Ok(gpu_ptr) => {
            let raw_ptr = gpu_ptr.as_ptr() as CUdeviceptr;
            let gpu_ptr_arc = Arc::new(gpu_ptr);

            // Track pointer for proper cleanup
            if let Some(map) = PTR_MAP.get() {
                if let Ok(mut ptr_map) = map.lock() {
                    ptr_map.insert(raw_ptr, gpu_ptr_arc);
                }
            }

            Ok(raw_ptr)
        }
        Err(e) => {
            eprintln!("[cudarc-ptx] TLSF alloc failed ({} bytes): {:?}", num_bytes, e);
            Err(crate::driver::result::DriverError(crate::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))
        }
    }
}

/// Free via TLSF (replaces cuMemFree)
#[cfg(feature = "ptx-alloc")]
pub unsafe fn tlsf_free(ptr: CUdeviceptr) -> Result<(), crate::driver::result::DriverError> {
    if ptr == 0 {
        return Ok(()); // Null pointer, nothing to free
    }

    if let Some(map) = PTR_MAP.get() {
        if let Ok(mut ptr_map) = map.lock() {
            if let Some(gpu_ptr_arc) = ptr_map.remove(&ptr) {
                drop(gpu_ptr_arc); // GpuPtr Drop will free via TLSF
                return Ok(());
            }
        }
    }

    eprintln!("[cudarc-ptx] WARNING: Attempted to free unknown pointer {:?}", ptr);
    Err(crate::driver::result::DriverError(crate::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE))
}

/// Get TLSF statistics
#[cfg(feature = "ptx-alloc")]
pub fn get_tlsf_stats() -> Option<ptx_sys::TLSFPoolStats> {
    GLOBAL_RUNTIME.get().map(|runtime| runtime.tlsf_stats())
}

/// Print TLSF health report
#[cfg(feature = "ptx-alloc")]
pub fn print_tlsf_health() {
    if let Some(stats) = get_tlsf_stats() {
        eprintln!("\n[cudarc-ptx] TLSF Health Report:");
        eprintln!("  Pool size:       {:.2} GB", stats.total_pool_size as f64 / 1024.0 / 1024.0 / 1024.0);
        eprintln!("  Allocated:       {:.2} MB", stats.allocated_bytes as f64 / 1024.0 / 1024.0);
        eprintln!("  Peak:            {:.2} MB", stats.peak_allocated as f64 / 1024.0 / 1024.0);
        eprintln!("  Fragmentation:   {:.6}", stats.fragmentation_ratio);
        eprintln!("  Utilization:     {:.2}%\n", stats.utilization_percent);
    }
}

// Fallback: Use stock CUDA when ptx-alloc not enabled
#[cfg(not(feature = "ptx-alloc"))]
pub unsafe fn tlsf_malloc(num_bytes: usize) -> Result<CUdeviceptr, crate::driver::result::DriverError> {
    crate::driver::result::malloc_sync_original(num_bytes)
}

#[cfg(not(feature = "ptx-alloc"))]
pub unsafe fn tlsf_free(ptr: CUdeviceptr) -> Result<(), crate::driver::result::DriverError> {
    crate::driver::result::free_sync_original(ptr)
}

#[cfg(not(feature = "ptx-alloc"))]
pub fn print_tlsf_health() {
    eprintln!("[cudarc-ptx] TLSF not enabled (compile with --features ptx-alloc)");
}
