use super::*;

impl PtxRuntime {
    /// Get runtime statistics.
    pub fn stats(&self) -> ptx_sys::GPUHotStats {
        let mut hot = ptx_sys::GPUHotStats::default();
        unsafe { ptx_sys::gpu_hot_get_stats(self.inner.raw, &mut hot) };
        hot
    }

    /// Get TLSF pool statistics.
    pub fn tlsf_stats(&self) -> ptx_sys::TLSFPoolStats {
        let mut stats = ptx_sys::TLSFPoolStats::default();
        unsafe { ptx_sys::gpu_hot_get_tlsf_stats(self.inner.raw, &mut stats) };
        stats
    }

    /// Validate the TLSF pool and get a health report.
    pub fn validate_pool(&self) -> ptx_sys::TLSFHealthReport {
        let mut report = ptx_sys::TLSFHealthReport::default();
        unsafe { ptx_sys::gpu_hot_validate_tlsf_pool(self.inner.raw, &mut report) };
        report
    }

    /// Print the pool memory map (for debugging).
    pub fn print_pool_map(&self) {
        unsafe { ptx_sys::gpu_hot_print_pool_map(self.inner.raw) }
    }

    /// Export the runtime pointer for LD_PRELOAD hook integration.
    ///
    /// Sets PTX_RUNTIME_PTR environment variable so the CUDA intercept hook
    /// can reuse this runtime instead of creating a competing one.
    ///
    /// **Call this before any CUDA operations that should go through TLSF.**
    pub fn export_for_hook(&self) {
        let mut raw_void: *mut libc::c_void = std::ptr::null_mut();
        let st = unsafe { ptx_sys::ptx_stable_get_hot_runtime(self.inner.stable, &mut raw_void) };
        if st != ptx_sys::PTXStableStatus::Ok || raw_void.is_null() {
            tracing::warn!(status = st as i32, "Failed to export runtime pointer for hook");
            return;
        }
        let ptr_str = format!("{:x}", raw_void as usize);
        std::env::set_var("PTX_RUNTIME_PTR", ptr_str);
        tracing::info!(ptr = ?self.inner.raw, "Runtime pointer exported for LD_PRELOAD hook");
    }

    // ========================================================================
    // Context Management
    // ========================================================================

    /// Get the raw CUcontext pointer captured during runtime initialization.
    ///
    /// Returns `None` if no context was captured (e.g., driver API unavailable).
    pub fn context(&self) -> Option<ptx_sys::CUcontext> {
        let mut ctx: *mut libc::c_void = std::ptr::null_mut();
        let st = unsafe { ptx_sys::ptx_stable_get_context(self.inner.stable, &mut ctx) };
        if st != ptx_sys::PTXStableStatus::Ok {
            return None;
        }
        if ctx.is_null() { None } else { Some(ctx) }
    }

    /// Export the CUcontext pointer as `PTX_CONTEXT_PTR` environment variable.
    ///
    /// External consumers (e.g., cudarc) can read this to share the same context.
    pub fn export_context(&self) {
        unsafe { ptx_sys::gpu_hot_export_context(self.inner.raw) };
        tracing::info!("CUcontext exported via PTX_CONTEXT_PTR");
    }

    /// Get context hook statistics, if the hook is loaded.
    ///
    /// Returns `None` if the context hook library is not present (weak symbol).
    pub fn context_stats(&self) -> Option<ptx_sys::PTXContextStats> {
        let mut stats = ptx_sys::PTXContextStats::default();
        unsafe { ptx_sys::ptx_context_hook_get_stats(&mut stats) };
        Some(stats)
    }

    // ========================================================================
    // Pool Management
    // ========================================================================

    /// Run defragmentation on the memory pool.
    pub fn defragment(&self) {
        unsafe { ptx_sys::gpu_hot_defragment_pool(self.inner.raw) }
    }

    /// Set the warning threshold for pool utilization.
    pub fn set_warning_threshold(&self, threshold: f32) {
        unsafe { ptx_sys::gpu_hot_set_warning_threshold(self.inner.raw, threshold) }
    }

    /// Enable or disable automatic defragmentation.
    pub fn set_auto_defrag(&self, enable: bool) {
        unsafe { ptx_sys::gpu_hot_set_auto_defrag(self.inner.raw, enable) }
    }

    /// Attempt to expand the TLSF memory pool.
    ///
    /// DEFERRED: Pool expansion requires allocating a new CUDA memory region
    /// and managing multiple disjoint pools within the TLSF bitmap structure.
    /// This is not yet implemented in the C core.
    ///
    /// **Mitigation:** Size the initial pool via `GPUHotConfig::pool_fraction`
    /// or `GPUHotConfig::fixed_pool_size` at init time.
    pub fn expand_pool(&self, _additional_bytes: usize) -> Result<()> {
        Err(Error::NotSupported {
            message: "TLSF pool expansion not yet implemented. Size pool at init via GPUHotConfig.".to_string(),
        })
    }

    /// Attempt to shrink the TLSF memory pool.
    ///
    /// DEFERRED: Pool shrinking requires relocating live allocations then
    /// returning the tail region to CUDA, which needs pointer-update
    /// cooperation from all consumers.
    ///
    /// **Mitigation:** Use `defragment()` to coalesce free blocks.
    pub fn shrink_pool(&self) -> Result<()> {
        Err(Error::NotSupported {
            message: "TLSF pool shrinking not yet implemented. Use defragment() to coalesce free blocks.".to_string(),
        })
    }

    /// Attempt to compact the TLSF memory pool.
    ///
    /// DEFERRED: True compaction (moving live allocations) requires a
    /// pointer-forwarding or handle-indirection layer. `defragment()` already
    /// coalesces adjacent free blocks, which covers the common case.
    pub fn compact_pool(&self) -> Result<()> {
        Err(Error::NotSupported {
            message: "TLSF pool compaction not yet implemented. Use defragment() to coalesce adjacent free blocks.".to_string(),
        })
    }

    // ========================================================================
    // Watchdog
    // ========================================================================

    /// Set the watchdog timeout.
    pub fn set_watchdog(&self, timeout_ms: i32) {
        unsafe { ptx_sys::gpu_hot_set_watchdog(self.inner.raw, timeout_ms) }
    }

    /// Check if the watchdog has tripped.
    pub fn check_watchdog(&self) -> bool {
        unsafe { ptx_sys::gpu_hot_check_watchdog(self.inner.raw) }
    }

    /// Reset the watchdog.
    pub fn reset_watchdog(&self) {
        unsafe { ptx_sys::gpu_hot_reset_watchdog(self.inner.raw) }
    }

    // ========================================================================
    // Keepalive
    // ========================================================================

    /// Send a keepalive signal to prevent GPU idle timeout.
    pub fn keepalive(&self) {
        unsafe { ptx_sys::gpu_hot_keepalive(self.inner.raw) }
    }

    // ========================================================================
    // System State
    // ========================================================================

    /// Get the system state snapshot.
    pub fn system_snapshot(&self) -> ptx_sys::GPUHotSystemSnapshot {
        let mut snapshot = ptx_sys::GPUHotSystemSnapshot::default();
        unsafe { ptx_sys::gpu_hot_get_system_snapshot(self.inner.raw, &mut snapshot) };
        snapshot
    }

    /// Flush the task queue.
    pub fn flush_task_queue(&self) {
        unsafe { ptx_sys::gpu_hot_flush_task_queue(self.inner.raw) }
    }

    // ========================================================================
    // Per-Owner Memory Tracking
    // ========================================================================

    /// Get per-owner memory usage statistics.
    pub fn owner_stats(&self) -> ptx_sys::TLSFOwnerStats {
        let mut stats = ptx_sys::TLSFOwnerStats::default();
        unsafe { ptx_sys::gpu_hot_get_owner_stats(self.inner.raw, &mut stats) };
        stats
    }

    // ========================================================================
    // VMM Bridge
    // ========================================================================

    /// Set the VMM reference for eviction-on-pressure.
    ///
    /// # Safety
    /// The VMM pointer must remain valid for the lifetime of the runtime.
    pub unsafe fn set_vmm(&self, vmm: *mut ptx_sys::VMMState) {
        ptx_sys::gpu_hot_set_vmm(self.inner.raw, vmm);
    }

    // ========================================================================
    // Allocation Event Log
    // ========================================================================

    /// Get the allocation event ring buffer snapshot.
    pub fn alloc_events(&self) -> ptx_sys::TLSFEventRing {
        let mut ring = ptx_sys::TLSFEventRing::default();
        unsafe { ptx_sys::gpu_hot_get_alloc_events(self.inner.raw, &mut ring) };
        ring
    }

    // ========================================================================
    // Task Submission
    // ========================================================================

    /// Submit a task to the persistent kernel's task queue.
    ///
    /// Returns the task ID on success, or -1 on failure.
    pub fn submit_task(&self, opcode: u32, priority: u32, args: &mut [*mut libc::c_void; 8]) -> i32 {
        unsafe { ptx_sys::ptx_os_submit_task(self.inner.raw, opcode, priority, args.as_mut_ptr()) }
    }

    // ========================================================================
    // Persistent Kernel & CUDA Dynamic Parallelism
    // ========================================================================

    /// Boot the persistent OS kernel on the GPU.
    ///
    /// The persistent kernel runs an infinite scheduling loop that polls
    /// the task queue and dispatches work via CUDA Dynamic Parallelism.
    /// This must be called before `submit_task` or `cdp_test_recursive`.
    pub fn boot_persistent_kernel(&self) {
        unsafe { ptx_sys::ptx_os_boot_persistent_kernel(self.inner.raw) }
    }

    /// Run the CDP recursive self-scheduling test.
    ///
    /// Submits a kernel that, upon execution by the persistent scheduler,
    /// re-enqueues itself back into the task queue — creating a fully
    /// autonomous GPU-side execution loop:
    ///
    /// ```text
    /// Host submit → Scheduler → CDP child → device_submit → Scheduler → ...
    /// ```
    ///
    /// Returns the number of iterations completed, or an error.
    pub fn cdp_test_recursive(&self, iterations: i32) -> Result<i32> {
        let result = unsafe { ptx_sys::ptx_cdp_test_recursive(self.inner.raw, iterations) };
        if result < 0 {
            Err(Error::Internal {
                message: format!("CDP recursive test failed with code {}", result),
            })
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::error::Error;

    #[test]
    fn test_expand_pool_is_deferred() {
        // expand_pool returns NotSupported until pool expansion is implemented
        // We can't call it without a runtime, so verify the error type matches
        let err = Error::NotSupported {
            message: "TLSF pool expansion not yet implemented. Size pool at init via GPUHotConfig.".to_string(),
        };
        assert!(matches!(err, Error::NotSupported { .. }));
    }

    #[test]
    fn test_shrink_pool_is_deferred() {
        let err = Error::NotSupported {
            message: "TLSF pool shrinking not yet implemented. Use defragment() to coalesce free blocks.".to_string(),
        };
        assert!(matches!(err, Error::NotSupported { .. }));
    }

    #[test]
    fn test_compact_pool_is_deferred() {
        let err = Error::NotSupported {
            message: "TLSF pool compaction not yet implemented. Use defragment() to coalesce adjacent free blocks.".to_string(),
        };
        assert!(matches!(err, Error::NotSupported { .. }));
    }
}
