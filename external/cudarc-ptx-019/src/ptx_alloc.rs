#[cfg(feature = "ptx-alloc")]
use ptx_runtime::{global_runtime, GpuPtr, PtxRuntime};
#[cfg(feature = "ptx-alloc")]
use std::collections::HashMap;
#[cfg(feature = "ptx-alloc")]
use std::sync::{Arc, Mutex, OnceLock};

use crate::driver::sys::CUdeviceptr;

#[cfg(feature = "ptx-alloc")]
static GLOBAL_RUNTIME: OnceLock<Arc<ptx_runtime::PtxRuntime>> = OnceLock::new();
#[cfg(feature = "ptx-alloc")]
static PTR_MAP: OnceLock<Mutex<HashMap<CUdeviceptr, Arc<GpuPtr>>>> = OnceLock::new();

#[cfg(feature = "ptx-alloc")]
fn get_or_init_runtime() -> Arc<PtxRuntime> {
    let runtime = GLOBAL_RUNTIME.get_or_init(|| {
        let runtime = global_runtime()
            .unwrap_or_else(|e| panic!("[cudarc-ptx-019] FATAL: Failed to initialize TLSF runtime: {e:?}"));
        runtime.export_for_hook();
        runtime.enable_hooks(false);
        eprintln!("[cudarc-ptx-019] TLSF allocator attached to global PTX runtime");
        runtime
    });

    PTR_MAP.get_or_init(|| Mutex::new(HashMap::new()));
    Arc::clone(runtime)
}

#[cfg(feature = "ptx-alloc")]
pub unsafe fn tlsf_malloc(
    num_bytes: usize,
) -> Result<CUdeviceptr, crate::driver::result::DriverError> {
    let runtime = get_or_init_runtime();

    match runtime.alloc(num_bytes) {
        Ok(gpu_ptr) => {
            let raw_ptr = gpu_ptr.as_ptr() as CUdeviceptr;
            let gpu_ptr_arc = Arc::new(gpu_ptr);

            if let Some(map) = PTR_MAP.get() {
                if let Ok(mut ptr_map) = map.lock() {
                    ptr_map.insert(raw_ptr, gpu_ptr_arc);
                }
            }

            Ok(raw_ptr)
        }
        Err(_) => Err(crate::driver::result::DriverError(
            crate::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        )),
    }
}

#[cfg(feature = "ptx-alloc")]
pub unsafe fn tlsf_free(
    ptr: CUdeviceptr,
) -> Result<(), crate::driver::result::DriverError> {
    if ptr == 0 {
        return Ok(());
    }

    if let Some(map) = PTR_MAP.get() {
        if let Ok(mut ptr_map) = map.lock() {
            if let Some(gpu_ptr_arc) = ptr_map.remove(&ptr) {
                drop(gpu_ptr_arc);
                return Ok(());
            }
        }
    }

    Err(crate::driver::result::DriverError(
        crate::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
    ))
}

#[cfg(not(feature = "ptx-alloc"))]
pub unsafe fn tlsf_malloc(
    num_bytes: usize,
) -> Result<CUdeviceptr, crate::driver::result::DriverError> {
    crate::driver::result::malloc_sync_original(num_bytes)
}

#[cfg(not(feature = "ptx-alloc"))]
pub unsafe fn tlsf_free(
    ptr: CUdeviceptr,
) -> Result<(), crate::driver::result::DriverError> {
    crate::driver::result::free_sync_original(ptr)
}
