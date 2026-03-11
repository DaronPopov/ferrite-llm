use std::ffi::CString;

use anyhow::{Context, Result, bail};
use ptx_runtime::PtxRuntime;

pub unsafe fn vfs_safe_init(rt: &PtxRuntime) -> Result<*mut ptx_sys::VFSState> {
    let vfs = ptx_sys::vfs_init(rt.raw());
    if vfs.is_null() {
        bail!("VFS init failed");
    }
    Ok(vfs)
}

pub unsafe fn vfs_safe_mkdir(vfs: *mut ptx_sys::VFSState, path: &str) -> Result<()> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let rc = ptx_sys::vfs_mkdir(vfs, c_path.as_ptr(), 0o755);
    if rc < 0 {
        bail!("vfs_mkdir({}) failed: {}", path, rc);
    }
    Ok(())
}

pub unsafe fn vfs_safe_open(vfs: *mut ptx_sys::VFSState, path: &str, flags: u32) -> Result<i32> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let fd = ptx_sys::vfs_open(vfs, c_path.as_ptr(), flags);
    if fd < 0 {
        bail!("vfs_open({}) failed: {}", path, fd);
    }
    Ok(fd)
}

pub unsafe fn vfs_safe_create_tensor(
    vfs: *mut ptx_sys::VFSState,
    path: &str,
    shape: &[i32],
    dtype: i32,
) -> Result<i32> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let mut shape_buf: Vec<i32> = shape.to_vec();
    let fd = ptx_sys::vfs_create_tensor(
        vfs,
        c_path.as_ptr(),
        shape_buf.as_mut_ptr(),
        shape.len() as i32,
        dtype,
    );
    if fd < 0 {
        bail!("vfs_create_tensor({}) failed: {}", path, fd);
    }
    Ok(fd)
}

pub unsafe fn vfs_safe_mmap_tensor(
    vfs: *mut ptx_sys::VFSState,
    path: &str,
) -> Result<*mut libc::c_void> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let ptr = ptx_sys::vfs_mmap_tensor(vfs, c_path.as_ptr());
    if ptr.is_null() {
        bail!("vfs_mmap_tensor({}) returned null", path);
    }
    Ok(ptr)
}

pub unsafe fn vfs_safe_sync_tensor(vfs: *mut ptx_sys::VFSState, path: &str) -> Result<()> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let rc = ptx_sys::vfs_sync_tensor(vfs, c_path.as_ptr());
    if rc < 0 {
        bail!("vfs_sync_tensor({}) failed: {}", path, rc);
    }
    Ok(())
}

pub unsafe fn vfs_safe_unlink(vfs: *mut ptx_sys::VFSState, path: &str) -> Result<()> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let rc = ptx_sys::vfs_unlink(vfs, c_path.as_ptr());
    if rc < 0 {
        bail!("vfs_unlink({}) failed: {}", path, rc);
    }
    Ok(())
}

pub unsafe fn vfs_safe_rmdir(vfs: *mut ptx_sys::VFSState, path: &str) -> Result<()> {
    let c_path = CString::new(path).context("Invalid VFS path")?;
    let rc = ptx_sys::vfs_rmdir(vfs, c_path.as_ptr());
    if rc < 0 {
        bail!("vfs_rmdir({}) failed: {}", path, rc);
    }
    Ok(())
}
