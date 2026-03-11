use super::*;

impl PtxRuntime {
    /// Initialize the PTX-OS runtime on the given device.
    ///
    /// Uses default configuration (all available VRAM minus headroom).
    pub fn new(device_id: i32) -> Result<Self> {
        tracing::info!(device_id, "Initializing PTX-OS runtime");
        Self::with_config(device_id, None)
    }

    /// Initialize the runtime with custom configuration.
    pub fn with_config(device_id: i32, config: Option<ptx_sys::GPUHotConfig>) -> Result<Self> {
        let cfg = config.unwrap_or_default();
        let mut flags = ptx_sys::PTX_STABLE_CONFIG_DEFAULT;
        if cfg.prefer_orin_unified_memory {
            flags |= ptx_sys::PTX_STABLE_CONFIG_PREFER_ORIN_UM;
        }
        if cfg.use_managed_pool {
            flags |= ptx_sys::PTX_STABLE_CONFIG_USE_MANAGED_POOL;
        }
        let stable_cfg = ptx_sys::PTXStableConfig {
            struct_size: std::mem::size_of::<ptx_sys::PTXStableConfig>() as u32,
            abi_version: ptx_sys::PTX_STABLE_ABI_VERSION,
            flags,
            device_id,
            pool_fraction: cfg.pool_fraction,
            fixed_pool_size: cfg.fixed_pool_size as u64,
            reserve_vram: cfg.reserve_vram as u64,
            max_streams: cfg.max_streams,
            quiet_init: u8::from(cfg.quiet_init),
            enable_leak_detection: u8::from(cfg.enable_leak_detection),
            enable_pool_health: u8::from(cfg.enable_pool_health),
            _reserved0: 0,
        };
        Self::with_stable_config(device_id, Some(stable_cfg))
    }

    /// Initialize the runtime with the versioned stable ABI config.
    pub fn with_stable_config(device_id: i32, config: Option<ptx_sys::PTXStableConfig>) -> Result<Self> {
        let _timer = crate::telemetry::OpTimer::new("runtime_init");
        let mut stable_cfg = config.unwrap_or_default();
        stable_cfg.device_id = device_id;

        tracing::debug!(
            device_id,
            pool_fraction = stable_cfg.pool_fraction,
            max_streams = stable_cfg.max_streams,
            abi_version = stable_cfg.abi_version,
            "Initializing with stable config"
        );

        let mut stable_raw: *mut ptx_sys::PTXStableRuntime = std::ptr::null_mut();
        let init_status = unsafe { ptx_sys::ptx_stable_init(&stable_cfg, &mut stable_raw) };
        if init_status != ptx_sys::PTXStableStatus::Ok || stable_raw.is_null() {
            tracing::error!(device_id, status = init_status as i32, "Stable runtime initialization failed");
            return Err(Error::InitFailed { device_id });
        }

        let mut raw_void: *mut libc::c_void = std::ptr::null_mut();
        let raw_status = unsafe { ptx_sys::ptx_stable_get_hot_runtime(stable_raw, &mut raw_void) };
        if raw_status != ptx_sys::PTXStableStatus::Ok || raw_void.is_null() {
            let _ = unsafe { ptx_sys::ptx_stable_release(stable_raw) };
            return Err(Error::stable(raw_status));
        }
        let raw = raw_void as *mut ptx_sys::GPUHotRuntime;

        let mut runtime_stats = ptx_sys::GPUHotStats::default();
        unsafe { ptx_sys::gpu_hot_get_stats(raw, &mut runtime_stats) };
        // stream_poll_count = min(actual_num_streams, 1024) — reflects the real
        // stream count created by the C runtime (which applies env overrides).
        // active_streams only counts streams with pending work (0 at init).
        let num_streams = if runtime_stats.stream_poll_count > 0 {
            runtime_stats.stream_poll_count as u32
        } else if stable_cfg.max_streams > 0 {
            stable_cfg.max_streams
        } else {
            1
        };

        tracing::info!(device_id, num_streams, "Runtime initialized successfully");

        let inner = Arc::new(PtxRuntimeInner { stable: stable_raw, raw });

        // Create stream pool with configured number of streams
        let streams: Vec<Stream> = (0..num_streams as i32)
            .map(|id| {
                let stream = unsafe { ptx_sys::gpu_hot_get_stream(raw, id) };
                Stream::new(stream, id)
            })
            .collect();

        Ok(Self {
            inner,
            device: Device::new(device_id),
            streams: StreamPool::new(streams),
            cublas: Mutex::new(None),
        })
    }

    /// Get the device this runtime is associated with.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get a reference to the inner runtime (for creating GpuPtrs).
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> Arc<PtxRuntimeInner> {
        Arc::clone(&self.inner)
    }

    /// Get the raw runtime pointer.
    ///
    /// # Safety
    ///
    /// This pointer must not outlive the PtxRuntime instance.
    /// Callers must ensure they don't use the pointer after the runtime is dropped.
    pub fn raw(&self) -> *mut ptx_sys::GPUHotRuntime {
        self.inner.raw
    }

    /// Enable CUDA allocation hooks to route all cudaMalloc calls through TLSF.
    ///
    /// This enables interception of CUDA allocations made by external libraries
    /// (like Candle) so they use the TLSF allocator instead of cudaMalloc.
    ///
    /// Call this immediately after runtime initialization if you want to marshal
    /// all CUDA allocations through TLSF.
    pub fn enable_hooks(&self, verbose: bool) {
        unsafe {
            ptx_sys::ptx_hook_init(self.inner.raw, verbose);
        }
        tracing::info!("CUDA allocation hooks enabled");
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /// Allocate GPU memory.
    ///
    /// Returns a smart pointer that automatically frees the memory when dropped.
    pub fn alloc(&self, size: usize) -> Result<GpuPtr> {
        let _timer = crate::telemetry::OpTimer::new("gpu_alloc");

        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let status = unsafe { ptx_sys::ptx_stable_alloc(self.inner.stable, size, &mut ptr) };
        if status != ptx_sys::PTXStableStatus::Ok {
            crate::telemetry::metrics().record_allocation_failure();
            tracing::warn!(size, status = status as i32, "Stable allocation failed");
            return Err(Error::stable(status));
        }

        match unsafe { GpuPtr::new(ptr, size, Arc::clone(&self.inner)) } {
            Ok(gpu_ptr) => {
                crate::telemetry::metrics().record_allocation(size);
                tracing::trace!(size, ptr = ?ptr, "GPU allocation successful");
                Ok(gpu_ptr)
            }
            Err(e) => {
                crate::telemetry::metrics().record_allocation_failure();
                tracing::warn!(size, error = ?e, "GPU allocation failed");
                Err(e)
            }
        }
    }

    /// Free GPU memory that was allocated by this runtime.
    pub fn free(&self, ptr: GpuPtr) -> Result<()> {
        drop(ptr);
        Ok(())
    }

    /// Allocate GPU memory (async, stream-ordered).
    ///
    /// The allocation itself is immediate, but the returned pointer should be
    /// used and freed on the same stream to preserve ordering guarantees.
    pub fn alloc_async(&self, size: usize, stream: &Stream) -> Result<*mut libc::c_void> {
        let ptr = unsafe { ptx_sys::gpu_hot_alloc_async(self.inner.raw, size, stream.raw()) };
        if ptr.is_null() {
            Err(Error::AllocationFailed { size })
        } else {
            Ok(ptr)
        }
    }

    /// Free GPU memory asynchronously on a given stream.
    ///
    /// This defers the actual free until all prior work on the stream completes.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `ptr` is not used after this call (no use-after-free)
    /// - `ptr` is freed exactly once (no double-free)
    /// - All GPU operations using `ptr` on other streams have completed
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidPointer` if `ptr` is null.
    /// Returns `Error::InvalidPointerOwnership` if `ptr` was not allocated by this runtime.
    pub unsafe fn free_async(&self, ptr: *mut libc::c_void, stream: &Stream) -> Result<()> {
        if ptr.is_null() {
            return Err(Error::InvalidPointer);
        }
        if !self.owns_ptr(ptr) {
            return Err(Error::InvalidPointerOwnership {
                ptr_debug: format!("{:?}", ptr),
            });
        }

        // SAFETY: Caller must uphold documented preconditions
        ptx_sys::gpu_hot_free_async(self.inner.raw, ptr, stream.raw());
        Ok(())
    }

    /// Poll and release completed deferred frees.
    ///
    /// `max_drain` of 0 drains all completed events.
    pub fn poll_deferred(&self, max_drain: i32) {
        unsafe { ptx_sys::gpu_hot_poll_deferred(self.inner.raw, max_drain) }
    }

    /// Check if a size can be allocated.
    pub fn can_allocate(&self, size: usize) -> bool {
        unsafe { ptx_sys::gpu_hot_can_allocate(self.inner.raw, size) }
    }

    /// Get the maximum size that can be allocated.
    pub fn max_allocatable(&self) -> usize {
        unsafe { ptx_sys::gpu_hot_get_max_allocatable(self.inner.raw) }
    }

    /// Check if a pointer belongs to this runtime.
    pub fn owns_ptr(&self, ptr: *mut libc::c_void) -> bool {
        let mut owned = false;
        let st = unsafe { ptx_sys::ptx_stable_owns_ptr(self.inner.stable, ptr, &mut owned) };
        st == ptx_sys::PTXStableStatus::Ok && owned
    }

    // ========================================================================
    // Tenant-Aware Memory Allocation
    // ========================================================================

    /// Allocate GPU memory on behalf of a specific tenant.
    ///
    /// The returned `GpuPtr` tracks the owning tenant ID. The caller is
    /// responsible for updating the tenant's VRAM usage counters (typically
    /// done through the scheduler's `TenantRegistry`).
    ///
    /// This method does NOT perform quota checks itself -- that is the
    /// scheduler's responsibility at admission time. This separation keeps
    /// the hot allocation path fast and avoids coupling the allocator to the
    /// scheduler.
    pub fn alloc_for_tenant(&self, tenant_id: u64, size: usize) -> Result<GpuPtr> {
        let _timer = crate::telemetry::OpTimer::new("gpu_alloc_tenant");

        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let status = unsafe { ptx_sys::ptx_stable_alloc(self.inner.stable, size, &mut ptr) };
        if status != ptx_sys::PTXStableStatus::Ok {
            crate::telemetry::metrics().record_allocation_failure();
            tracing::warn!(
                tenant_id,
                size,
                status = status as i32,
                "Tenant allocation failed"
            );
            return Err(Error::stable(status));
        }

        match unsafe { GpuPtr::new_for_tenant(ptr, size, Arc::clone(&self.inner), tenant_id) } {
            Ok(gpu_ptr) => {
                crate::telemetry::metrics().record_allocation(size);
                tracing::trace!(
                    tenant_id,
                    size,
                    ptr = ?ptr,
                    "Tenant GPU allocation successful"
                );
                Ok(gpu_ptr)
            }
            Err(e) => {
                crate::telemetry::metrics().record_allocation_failure();
                tracing::warn!(
                    tenant_id,
                    size,
                    error = ?e,
                    "Tenant GPU allocation failed"
                );
                Err(e)
            }
        }
    }
}
