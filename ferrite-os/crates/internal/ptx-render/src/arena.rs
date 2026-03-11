//! GPU scratch arena — bump allocator over a single TLSF allocation.
//!
//! **Level 0** of the render stack. All per-frame GPU temporaries (clip-space
//! vertices, screen positions, etc.) are sub-allocated from one big GPU buffer
//! that was allocated once from the TLSF pool at init time.
//!
//! Per-frame cost is zero: `reset()` just sets the offset back to 0.
//! No TLSF alloc/free per frame. No fragmentation. No contention.

use std::sync::Arc;

use ptx_runtime::{GpuPtr, PtxRuntime, Result};

/// GPU-side bump allocator. One TLSF allocation, many sub-allocations.
pub struct ScratchArena {
    /// The single backing allocation from the TLSF pool.
    /// Kept alive for RAII — Drop frees the GPU memory.
    _gpu_mem: GpuPtr,
    /// Base device pointer (cached from gpu_mem).
    base: *mut u8,
    /// Total size in bytes.
    total_bytes: usize,
    /// Current bump offset in bytes.
    offset: usize,
}

// ScratchArena holds a GpuPtr which is Send. The raw pointer `base` is derived
// from the GpuPtr and is only accessed via &mut self, so this is safe.
unsafe impl Send for ScratchArena {}

/// A typed slice into GPU memory within the scratch arena.
///
/// No ownership — the arena owns all memory. The slice is invalidated on
/// `arena.reset()`.
#[derive(Debug, Clone, Copy)]
pub struct GpuSlice<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> GpuSlice<T> {
    /// Raw pointer for passing to CUDA kernels.
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

// GpuSlice is just a raw pointer + length into device memory.
unsafe impl<T> Send for GpuSlice<T> {}
unsafe impl<T> Sync for GpuSlice<T> {}

const ARENA_ALIGN: usize = 256; // CUDA recommended alignment

impl ScratchArena {
    /// Allocate a scratch arena of `size_bytes` from the TLSF pool.
    ///
    /// This is the **only** TLSF allocation the render stack makes per
    /// render context. Everything else is bump-allocated from here.
    pub fn new(size_bytes: usize, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let gpu_mem = runtime.alloc(size_bytes)?;
        let base = gpu_mem.as_ptr() as *mut u8;
        Ok(Self {
            _gpu_mem: gpu_mem,
            base,
            total_bytes: size_bytes,
            offset: 0,
        })
    }

    /// Sub-allocate `count` elements of type `T` from the arena.
    ///
    /// Returns `None` if the arena is exhausted (caller should use a
    /// larger arena). Alignment is CUDA-friendly (256 bytes).
    pub fn alloc<T>(&mut self, count: usize) -> Option<GpuSlice<T>> {
        let bytes = count * std::mem::size_of::<T>();
        if bytes == 0 {
            return Some(GpuSlice {
                ptr: self.base as *mut T,
                len: 0,
            });
        }

        let aligned = (self.offset + ARENA_ALIGN - 1) & !(ARENA_ALIGN - 1);
        let end = aligned + bytes;
        if end > self.total_bytes {
            return None;
        }

        let ptr = unsafe { self.base.add(aligned) } as *mut T;
        self.offset = end;
        Some(GpuSlice { ptr, len: count })
    }

    /// Reset the arena for the next frame. Zero cost.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Bytes currently used.
    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.total_bytes - self.offset
    }

    /// Total arena capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.total_bytes
    }
}
