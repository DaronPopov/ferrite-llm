use std::ffi::CString;

use anyhow::{Context, Result, bail};
use ptx_runtime::PtxRuntime;

pub unsafe fn shm_safe_alloc(rt: &PtxRuntime, name: &str, size: usize) -> Result<*mut libc::c_void> {
    let c_name = CString::new(name).context("Invalid SHM name")?;
    let ptr = ptx_sys::gpu_hot_shm_alloc(rt.raw(), c_name.as_ptr(), size);
    if ptr.is_null() {
        bail!("shm_alloc({}) failed", name);
    }
    Ok(ptr)
}

pub unsafe fn shm_safe_open(rt: &PtxRuntime, name: &str) -> Result<*mut libc::c_void> {
    let c_name = CString::new(name).context("Invalid SHM name")?;
    let ptr = ptx_sys::gpu_hot_shm_open(rt.raw(), c_name.as_ptr());
    if ptr.is_null() {
        bail!("shm_open({}) failed", name);
    }
    Ok(ptr)
}

pub unsafe fn shm_safe_close(rt: &PtxRuntime, ptr: *mut libc::c_void) {
    ptx_sys::gpu_hot_shm_close(rt.raw(), ptr);
}

pub unsafe fn shm_safe_unlink(rt: &PtxRuntime, name: &str, ptr: *mut libc::c_void) -> Result<()> {
    // Close first (frees TLSF allocation), then unlink (removes name)
    ptx_sys::gpu_hot_shm_close(rt.raw(), ptr);
    let c_name = CString::new(name).context("Invalid SHM name")?;
    ptx_sys::gpu_hot_shm_unlink(rt.raw(), c_name.as_ptr());
    Ok(())
}
