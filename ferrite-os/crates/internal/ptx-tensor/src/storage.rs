//! GPU storage for tensor data.

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, GpuPtr, Result, Error};
use crate::dtype::DType;

/// GPU storage for tensor data.
///
/// Storage represents a contiguous block of GPU memory that can be shared
/// between multiple tensors (for views).
pub struct Storage {
    /// The GPU memory pointer.
    ptr: Arc<GpuPtr>,
    /// Size in bytes.
    len: usize,
    /// Data type of elements.
    dtype: DType,
    /// Runtime reference.
    runtime: Arc<PtxRuntime>,
}

impl Storage {
    /// Create new storage with uninitialized data.
    pub fn new(len: usize, dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let size_bytes = len.checked_mul(dtype.size_bytes()).ok_or_else(|| Error::Internal {
            message: format!("storage size overflow: {} elements * {} bytes", len, dtype.size_bytes()),
        })?;
        let ptr = runtime.alloc(size_bytes)?;
        Ok(Self {
            ptr: Arc::new(ptr),
            len,
            dtype,
            runtime: Arc::clone(runtime),
        })
    }

    /// Create new storage initialized to zero.
    pub fn zeros(len: usize, dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let storage = Self::new(len, dtype, runtime)?;
        storage.ptr.zero()?;
        Ok(storage)
    }

    /// Create storage from host data.
    ///
    /// # Safety
    ///
    /// The caller must ensure the data pointer and length are valid.
    pub unsafe fn from_host<T: bytemuck::Pod>(
        data: &[T],
        dtype: DType,
        runtime: &Arc<PtxRuntime>,
    ) -> Result<Self> {
        let storage = Self::new(data.len(), dtype, runtime)?;
        storage.ptr.copy_from_host(
            data.as_ptr() as *const libc::c_void,
            data.len() * std::mem::size_of::<T>(),
        )?;
        Ok(storage)
    }

    /// Get the raw GPU pointer.
    pub fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr.as_ptr()
    }

    /// Get the typed GPU pointer.
    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    /// Get the length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * self.dtype.size_bytes()
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the runtime reference.
    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        &self.runtime
    }

    /// Copy data to host.
    ///
    /// Synchronizes all GPU streams before copying to ensure all pending
    /// kernel writes are visible.
    pub fn to_host<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        self.runtime.sync_all()?;
        let mut data = vec![bytemuck::Zeroable::zeroed(); self.len];
        unsafe {
            self.ptr.copy_to_host(
                data.as_mut_ptr() as *mut libc::c_void,
                self.len * std::mem::size_of::<T>(),
            )?;
        }
        Ok(data)
    }

    /// Copy data from another storage.
    pub fn copy_from(&self, src: &Storage) -> Result<()> {
        if src.len != self.len || src.dtype != self.dtype {
            return Err(Error::Internal {
                message: "Storage size or dtype mismatch".to_string(),
            });
        }
        self.ptr.copy_from_device(&src.ptr)
    }

    /// Clone the storage (creates new GPU allocation).
    pub fn deep_clone(&self) -> Result<Self> {
        let new_storage = Self::new(self.len, self.dtype, &self.runtime)?;
        new_storage.copy_from(self)?;
        Ok(new_storage)
    }

    /// Create storage from an existing GPU pointer.
    ///
    /// This does not allocate or copy; it wraps the provided allocation.
    pub fn from_gpu_ptr(
        ptr: Arc<GpuPtr>,
        len: usize,
        dtype: DType,
        runtime: &Arc<PtxRuntime>,
    ) -> Self {
        Self {
            ptr,
            len,
            dtype,
            runtime: Arc::clone(runtime),
        }
    }

    /// Check if this storage is the same allocation as another.
    pub fn same_storage(&self, other: &Storage) -> bool {
        Arc::ptr_eq(&self.ptr, &other.ptr)
    }

    /// Fill storage with a value.
    pub fn fill_f32(&self, value: f32) -> Result<()> {
        if self.dtype != DType::F32 {
            return Err(Error::Internal {
                message: "fill_f32 requires F32 dtype".to_string(),
            });
        }
        let stream = self.runtime.next_stream();
        unsafe {
            ptx_sys::ptx_tensor_fill_f32(
                self.as_ptr_typed::<f32>(),
                self.len,
                value,
                stream.raw(),
            );
        }
        Ok(())
    }
}

impl Clone for Storage {
    /// Clone shares the underlying storage (shallow clone).
    fn clone(&self) -> Self {
        Self {
            ptr: Arc::clone(&self.ptr),
            len: self.len,
            dtype: self.dtype,
            runtime: Arc::clone(&self.runtime),
        }
    }
}

// Safety: Storage is thread-safe (GPU memory access is synchronized by CUDA)
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}
