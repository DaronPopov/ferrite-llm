use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::runtime::context::{HasAllocator, HasRuntime};
use crate::{GpuCtx, LangError, Result};
use ptx_kernels::{GuardedBuffer, KernelContext};
use ptx_kernels::safe_api::unary;
use ptx_runtime::GpuPtr;

enum CpuStorage<T> {
    Vec(Vec<T>),
    Tlsf(TlsfStorage<T>),
}

struct TlsfStorage<T> {
    alloc: Arc<crate::cpu_tlsf::CpuTlsf>,
    ptr: NonNull<T>,
    len: usize,
    bytes: usize,
    align: usize,
}

impl<T> Drop for TlsfStorage<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr/align` come from the same allocator on allocation.
        let _ = unsafe {
            self.alloc
                .deallocate(self.ptr.cast::<u8>(), self.bytes, self.align)
        };
    }
}

pub struct CpuTensor<T> {
    shape: Vec<usize>,
    storage: CpuStorage<T>,
}

impl<T: Copy> Clone for CpuTensor<T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            storage: CpuStorage::Vec(self.data().to_vec()),
        }
    }
}

impl<T> std::fmt::Debug for CpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuTensor")
            .field("shape", &self.shape)
            .field("len", &self.len())
            .finish()
    }
}

impl<T> CpuTensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        validate_shape(&shape)?;
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(LangError::InputLenMismatch {
                index: 0,
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            shape,
            storage: CpuStorage::Vec(data),
        })
    }

    pub fn with_allocator<F>(ctx: &(impl HasAllocator + ?Sized), shape: Vec<usize>, mut init: F) -> Result<Self>
    where
        T: Copy,
        F: FnMut(usize) -> T,
    {
        validate_shape(&shape)?;
        let len: usize = shape.iter().product();
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| LangError::Transfer {
                message: "cpu tensor byte size overflow".to_string(),
            })?;

        let align = std::mem::align_of::<T>();
        let raw = ctx.allocator().allocate(bytes.max(1), align)?;
        let ptr = raw.cast::<T>();

        // SAFETY: `ptr` points to `len * size_of::<T>()` writable bytes from allocator.
        unsafe {
            for i in 0..len {
                ptr.as_ptr().add(i).write(init(i));
            }
        }

        Ok(Self {
            shape,
            storage: CpuStorage::Tlsf(TlsfStorage {
                alloc: Arc::clone(ctx.allocator()),
                ptr,
                len,
                bytes,
                align,
            }),
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        match &self.storage {
            CpuStorage::Vec(v) => v.as_slice(),
            CpuStorage::Tlsf(t) => {
                // SAFETY: storage is initialized for `len` elements in constructor.
                unsafe { slice::from_raw_parts(t.ptr.as_ptr(), t.len) }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn into_inner(self) -> (Vec<usize>, Vec<T>)
    where
        T: Copy,
    {
        let shape = self.shape;
        let data = match self.storage {
            CpuStorage::Vec(v) => v,
            CpuStorage::Tlsf(t) => {
                // SAFETY: initialized for `len` elements.
                unsafe { slice::from_raw_parts(t.ptr.as_ptr(), t.len).to_vec() }
            }
        };
        (shape, data)
    }
}

pub struct GpuTensor<T> {
    shape: Vec<usize>,
    numel: usize,
    ptr: GpuPtr,
    _marker: PhantomData<T>,
}

impl<T> GpuTensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn as_ptr_typed(&self) -> *mut T {
        self.ptr.as_ptr_typed::<T>()
    }

    pub fn bytes_len(&self) -> usize {
        self.numel * size_of::<T>()
    }

    pub(crate) fn from_parts(shape: Vec<usize>, numel: usize, ptr: GpuPtr) -> Self {
        Self {
            shape,
            numel,
            ptr,
            _marker: PhantomData,
        }
    }

    pub(crate) fn gpu_ptr(&self) -> &GpuPtr {
        &self.ptr
    }
}

impl GpuTensor<f32> {
    pub fn relu(&self, ctx: &(impl HasRuntime + ?Sized)) -> Result<GpuTensor<f32>> {
        let runtime = ctx.gpu_runtime().runtime();
        let out = runtime.alloc(self.bytes_len())?;
        let stream = runtime.next_stream();
        let runtime_ptr = runtime.raw();
        let kctx = KernelContext::new(runtime_ptr, stream.raw())?;
        let inp_guard = unsafe { GuardedBuffer::new(self.ptr.as_ptr(), self.bytes_len(), runtime_ptr)? };
        let out_guard = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
        unary::relu(&inp_guard, &out_guard, self.numel, &kctx)?;
        stream.synchronize()?;
        Ok(GpuTensor::from_parts(self.shape.clone(), self.numel, out))
    }
}

pub trait ToGpu<T> {
    fn to_gpu(&self, ctx: &GpuCtx) -> Result<GpuTensor<T>>;
}

pub trait ToCpu<T> {
    fn to_cpu(&self) -> Result<CpuTensor<T>>;
}

impl<T: Copy> ToGpu<T> for CpuTensor<T> {
    fn to_gpu(&self, ctx: &GpuCtx) -> Result<GpuTensor<T>> {
        let numel = self.len();
        let bytes = numel
            .checked_mul(size_of::<T>())
            .ok_or_else(|| LangError::Transfer {
                message: "byte size overflow during to_gpu".to_string(),
            })?;

        let ptr = ctx.runtime().runtime().alloc(bytes.max(1))?;
        // SAFETY: self.data points to a valid contiguous host buffer of `bytes` length.
        unsafe {
            ptr.copy_from_host(self.data().as_ptr() as *const libc::c_void, bytes)?;
        }

        Ok(GpuTensor {
            shape: self.shape.clone(),
            numel,
            ptr,
            _marker: PhantomData,
        })
    }
}

impl<T: Copy + Default> ToCpu<T> for GpuTensor<T> {
    fn to_cpu(&self) -> Result<CpuTensor<T>> {
        let mut out = vec![T::default(); self.numel];
        let bytes = self
            .numel
            .checked_mul(size_of::<T>())
            .ok_or_else(|| LangError::Transfer {
                message: "byte size overflow during to_cpu".to_string(),
            })?;

        // SAFETY: `out` is a valid writable host buffer of `bytes` length.
        unsafe {
            self.ptr
                .copy_to_host(out.as_mut_ptr() as *mut libc::c_void, bytes)?;
        }

        CpuTensor::new(self.shape.clone(), out)
    }
}

fn validate_shape(shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return Err(LangError::InvalidShape(shape.to_vec()));
    }
    Ok(())
}
