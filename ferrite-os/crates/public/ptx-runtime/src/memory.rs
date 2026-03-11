//! GPU memory management with RAII.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{Error, Result};

/// Tracks total bytes leaked due to failed GPU frees (observable for health monitoring).
static LEAKED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Returns the total number of GPU bytes leaked due to failed free operations.
pub fn leaked_bytes() -> usize {
    LEAKED_BYTES.load(Ordering::Relaxed)
}

/// A smart pointer to GPU memory with automatic deallocation.
///
/// When a `GpuPtr` is dropped, it automatically frees the GPU memory
/// through the PTX-OS runtime. Optionally tracks which tenant owns this
/// allocation for multi-tenant VRAM accounting.
pub struct GpuPtr {
    ptr: NonNull<libc::c_void>,
    size: usize,
    runtime: Arc<PtxRuntimeInner>,
    /// The tenant that owns this allocation (None for legacy untracked allocations).
    tenant_id: Option<u64>,
}

// Internal runtime reference for GpuPtr
pub(crate) struct PtxRuntimeInner {
    pub(crate) stable: *mut ptx_sys::PTXStableRuntime,
    pub(crate) raw: *mut ptx_sys::GPUHotRuntime,
}

impl Drop for PtxRuntimeInner {
    fn drop(&mut self) {
        if !self.stable.is_null() {
            unsafe {
                let status = ptx_sys::ptx_stable_release(self.stable);
                if status != ptx_sys::PTXStableStatus::Ok {
                    LEAKED_BYTES.fetch_add(0, Ordering::Relaxed); // mark runtime leak event
                    tracing::error!(
                        status = status as i32,
                        leaked_bytes = LEAKED_BYTES.load(Ordering::Relaxed),
                        "Failed to release stable runtime — runtime resources leaked"
                    );
                }
            }
        }
    }
}

// Safety: The PTX-OS runtime is thread-safe
unsafe impl Send for PtxRuntimeInner {}
unsafe impl Sync for PtxRuntimeInner {}

impl GpuPtr {
    /// Create a new GpuPtr from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is a valid GPU pointer allocated by the given runtime
    /// - `size` is the actual size of the allocation
    pub(crate) unsafe fn new(
        ptr: *mut libc::c_void,
        size: usize,
        runtime: Arc<PtxRuntimeInner>,
    ) -> Result<Self> {
        NonNull::new(ptr)
            .map(|ptr| Self { ptr, size, runtime, tenant_id: None })
            .ok_or(Error::AllocationFailed { size })
    }

    /// Create a new GpuPtr from a raw pointer with tenant ownership tracking.
    ///
    /// # Safety
    ///
    /// Same as [`GpuPtr::new`]. Additionally, the caller must ensure that
    /// tenant VRAM accounting has been updated before calling this.
    pub(crate) unsafe fn new_for_tenant(
        ptr: *mut libc::c_void,
        size: usize,
        runtime: Arc<PtxRuntimeInner>,
        tenant_id: u64,
    ) -> Result<Self> {
        NonNull::new(ptr)
            .map(|ptr| Self { ptr, size, runtime, tenant_id: Some(tenant_id) })
            .ok_or(Error::AllocationFailed { size })
    }

    /// Get the tenant ID that owns this allocation, if tenant-tracked.
    pub fn tenant_id(&self) -> Option<u64> {
        self.tenant_id
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr.as_ptr()
    }

    /// Get the raw pointer as a typed pointer.
    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    /// Get the size of the allocation in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if the pointer is valid.
    pub fn is_valid(&self) -> bool {
        let mut owned = false;
        let st = unsafe {
            ptx_sys::ptx_stable_owns_ptr(self.runtime.stable, self.ptr.as_ptr(), &mut owned)
        };
        st == ptx_sys::PTXStableStatus::Ok && owned
    }

    /// Copy data from host to this GPU memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` has at least `size` bytes.
    pub unsafe fn copy_from_host(&self, data: *const libc::c_void, size: usize) -> Result<()> {
        if size > self.size {
            return Err(Error::Internal {
                message: format!("Copy size {} exceeds allocation size {}", size, self.size),
            });
        }
        let err = ptx_sys::cudaMemcpy(
            self.ptr.as_ptr(),
            data,
            size,
            ptx_sys::cudaMemcpyHostToDevice,
        );
        Error::check_cuda(err)
    }

    /// Copy data from this GPU memory to host.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` has at least `size` bytes allocated.
    pub unsafe fn copy_to_host(&self, data: *mut libc::c_void, size: usize) -> Result<()> {
        if size > self.size {
            return Err(Error::Internal {
                message: format!("Copy size {} exceeds allocation size {}", size, self.size),
            });
        }
        let err = ptx_sys::cudaMemcpy(
            data,
            self.ptr.as_ptr(),
            size,
            ptx_sys::cudaMemcpyDeviceToHost,
        );
        Error::check_cuda(err)
    }

    /// Copy data from another GPU pointer to this one.
    pub fn copy_from_device(&self, src: &GpuPtr) -> Result<()> {
        if src.size > self.size {
            return Err(Error::Internal {
                message: format!(
                    "Source size {} exceeds destination size {}",
                    src.size, self.size
                ),
            });
        }
        let err = unsafe {
            ptx_sys::cudaMemcpy(
                self.ptr.as_ptr(),
                src.ptr.as_ptr(),
                src.size,
                ptx_sys::cudaMemcpyDeviceToDevice,
            )
        };
        Error::check_cuda(err)
    }

    /// Set all bytes in the allocation to a value.
    pub fn memset(&self, value: i32) -> Result<()> {
        let err = unsafe { ptx_sys::cudaMemset(self.ptr.as_ptr(), value, self.size) };
        Error::check_cuda(err)
    }

    /// Zero out the allocation.
    pub fn zero(&self) -> Result<()> {
        self.memset(0)
    }
}

impl Drop for GpuPtr {
    fn drop(&mut self) {
        // SAFETY: ptr must not have been freed yet
        // Invariant: Rust's ownership guarantees Drop is called exactly once
        // Invariant: ptr was allocated by same runtime
        crate::telemetry::metrics().record_deallocation(self.size);

        tracing::trace!(
            size = self.size,
            ptr = ?self.ptr.as_ptr(),
            "Freeing GPU memory"
        );

        unsafe {
            let status = ptx_sys::ptx_stable_free(self.runtime.stable, self.ptr.as_ptr());
            if status != ptx_sys::PTXStableStatus::Ok {
                LEAKED_BYTES.fetch_add(self.size, Ordering::Relaxed);
                tracing::error!(
                    status = status as i32,
                    ptr = ?self.ptr.as_ptr(),
                    size = self.size,
                    total_leaked = LEAKED_BYTES.load(Ordering::Relaxed),
                    "Stable free failed during GpuPtr drop — memory leaked"
                );
            }
        }
    }
}

// Safety: GPU pointers can be sent between threads
unsafe impl Send for GpuPtr {}

// Safety: GPU pointers can be shared between threads (read-only access is safe)
unsafe impl Sync for GpuPtr {}

/// A view into GPU memory without ownership.
///
/// This is useful for creating tensor views that share storage.
#[derive(Clone)]
pub struct GpuSlice {
    ptr: *mut libc::c_void,
    size: usize,
    // Keep the parent allocation alive
    _parent: Arc<GpuPtr>,
}

impl GpuSlice {
    /// Create a new slice from a GpuPtr.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - offset + size overflows usize
    /// - offset + size exceeds parent size
    pub fn new(parent: Arc<GpuPtr>, offset: usize, size: usize) -> Result<Self> {
        // SAFETY: Check for overflow before performing arithmetic
        let end = offset.checked_add(size).ok_or_else(|| Error::Internal {
            message: format!(
                "Slice bounds overflow: offset={}, size={} (would overflow usize)",
                offset, size
            ),
        })?;

        if end > parent.size() {
            return Err(Error::Internal {
                message: format!(
                    "Slice [{}, {}) out of bounds for size {}",
                    offset,
                    end,
                    parent.size()
                ),
            });
        }

        // SAFETY: Bounds checked above, arithmetic cannot overflow
        // Parent pointer is valid, offset is within bounds
        let ptr = unsafe { (parent.as_ptr() as *mut u8).add(offset) as *mut libc::c_void };
        Ok(Self {
            ptr,
            size,
            _parent: parent,
        })
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

// Safety: Same as GpuPtr
unsafe impl Send for GpuSlice {}
unsafe impl Sync for GpuSlice {}
