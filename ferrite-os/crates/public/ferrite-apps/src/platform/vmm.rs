use anyhow::{Result, bail};
use ptx_runtime::PtxRuntime;

pub unsafe fn vmm_safe_init(rt: &PtxRuntime, swap_size: usize) -> Result<*mut ptx_sys::VMMState> {
    let vmm = ptx_sys::vmm_init(rt.raw(), swap_size);
    if vmm.is_null() {
        bail!("VMM init failed");
    }
    Ok(vmm)
}

pub unsafe fn vmm_safe_alloc_page(
    vmm: *mut ptx_sys::VMMState,
    flags: u32,
) -> Result<*mut libc::c_void> {
    let page = ptx_sys::vmm_alloc_page(vmm, flags);
    if page.is_null() {
        bail!("vmm_alloc_page failed");
    }
    Ok(page)
}

pub unsafe fn vmm_safe_get_stats(vmm: *mut ptx_sys::VMMState) -> (u64, u64, u64, u64) {
    let (mut resident, mut swapped, mut faults, mut evictions) = (0u64, 0u64, 0u64, 0u64);
    ptx_sys::vmm_get_stats(vmm, &mut resident, &mut swapped, &mut faults, &mut evictions);
    (resident, swapped, faults, evictions)
}
