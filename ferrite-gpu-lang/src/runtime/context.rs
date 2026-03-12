use std::mem::size_of;
use std::sync::Arc;

use crate::cpu_tlsf::{global_cpu_tlsf, CpuTlsf, CpuTlsfStats};
use crate::runtime::tensor::{CpuTensor, GpuTensor};
use crate::{GpuLangRuntime, LangError, Result};

// ── Traits ───────────────────────────────────────────────────────

pub trait HasAllocator {
    fn allocator(&self) -> &Arc<CpuTlsf>;
    fn allocator_stats(&self) -> CpuTlsfStats {
        self.allocator().stats()
    }
}

pub trait HasRuntime {
    fn gpu_runtime(&self) -> &GpuLangRuntime;
}

// ── CpuCtx ──────────────────────────────────────────────────────

pub struct CpuCtx {
    allocator: Arc<CpuTlsf>,
}

impl CpuCtx {
    pub fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }

    pub fn allocator_stats(&self) -> CpuTlsfStats {
        self.allocator.stats()
    }
}

impl HasAllocator for CpuCtx {
    fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }
}

// ── GpuCtx ──────────────────────────────────────────────────────

pub struct GpuCtx {
    runtime: GpuLangRuntime,
}

impl GpuCtx {
    pub fn runtime(&self) -> &GpuLangRuntime {
        &self.runtime
    }

    #[cfg(feature = "torch")]
    pub fn cv(&self) -> crate::cv::CvBuilder {
        crate::cv::CvBuilder::new()
    }
}

impl HasRuntime for GpuCtx {
    fn gpu_runtime(&self) -> &GpuLangRuntime {
        &self.runtime
    }
}

// ── FerCtx ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct FerStats {
    pub cpu: CpuTlsfStats,
    pub gpu: ptx_runtime::TLSFPoolStats,
}

pub struct FerCtx {
    allocator: Arc<CpuTlsf>,
    runtime: GpuLangRuntime,
}

impl HasAllocator for FerCtx {
    fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }
}

impl HasRuntime for FerCtx {
    fn gpu_runtime(&self) -> &GpuLangRuntime {
        &self.runtime
    }
}

impl FerCtx {
    pub fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }

    pub fn allocator_stats(&self) -> CpuTlsfStats {
        self.allocator.stats()
    }

    pub fn runtime(&self) -> &GpuLangRuntime {
        &self.runtime
    }

    pub fn stats(&self) -> FerStats {
        FerStats {
            cpu: self.allocator.stats(),
            gpu: self.runtime.runtime().tlsf_stats(),
        }
    }

    pub fn gpu_hot_stats(&self) -> ptx_runtime::GPUHotStats {
        self.runtime.runtime().stats()
    }

    pub fn gpu_health(&self) -> ptx_runtime::TLSFHealthReport {
        self.runtime.runtime().validate_pool()
    }

    pub fn gpu_defragment(&self) {
        self.runtime.runtime().defragment()
    }

    pub fn gpu_set_auto_defrag(&self, enable: bool) {
        self.runtime.runtime().set_auto_defrag(enable)
    }

    pub fn cpu_to_gpu<T: Copy>(&self, tensor: &CpuTensor<T>) -> Result<GpuTensor<T>> {
        let numel = tensor.len();
        let bytes = numel
            .checked_mul(size_of::<T>())
            .ok_or_else(|| LangError::Transfer {
                message: "byte size overflow during cpu_to_gpu".to_string(),
            })?;

        let ptr = self.runtime.runtime().alloc(bytes.max(1))?;
        unsafe {
            ptr.copy_from_host(tensor.data().as_ptr() as *const libc::c_void, bytes)?;
        }

        Ok(GpuTensor::from_parts(
            tensor.shape().to_vec(),
            numel,
            ptr,
        ))
    }

    pub fn gpu_to_cpu<T: Copy + Default>(&self, tensor: &GpuTensor<T>) -> Result<CpuTensor<T>> {
        let mut out = vec![T::default(); tensor.numel()];
        let bytes = tensor
            .numel()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| LangError::Transfer {
                message: "byte size overflow during gpu_to_cpu".to_string(),
            })?;

        unsafe {
            tensor
                .gpu_ptr()
                .copy_to_host(out.as_mut_ptr() as *mut libc::c_void, bytes)?;
        }

        CpuTensor::new(tensor.shape().to_vec(), out)
    }

    #[cfg(feature = "torch")]
    pub fn cv(&self) -> crate::cv::CvBuilder {
        crate::cv::CvBuilder::new()
    }
}

// ── Factory functions ───────────────────────────────────────────

pub fn cpu<T, E, F>(f: F) -> std::result::Result<T, E>
where
    F: FnOnce(&CpuCtx) -> std::result::Result<T, E>,
{
    let ctx = CpuCtx {
        allocator: Arc::clone(global_cpu_tlsf()),
    };
    f(&ctx)
}

pub fn gpu<T, F>(device_id: i32, f: F) -> Result<T>
where
    F: FnOnce(&GpuCtx) -> Result<T>,
{
    let ctx = GpuCtx {
        runtime: GpuLangRuntime::new(device_id)?,
    };
    f(&ctx)
}

#[cfg(feature = "torch")]
pub fn gpu_anyhow<T, F>(device_id: i32, f: F) -> anyhow::Result<T>
where
    F: FnOnce(&GpuCtx) -> anyhow::Result<T>,
{
    let ctx = GpuCtx {
        runtime: GpuLangRuntime::new(device_id).map_err(anyhow::Error::new)?,
    };
    f(&ctx)
}

pub fn fer<T, F>(device_id: i32, f: F) -> Result<T>
where
    F: FnOnce(&FerCtx) -> Result<T>,
{
    let ctx = FerCtx {
        allocator: Arc::clone(global_cpu_tlsf()),
        runtime: GpuLangRuntime::new(device_id)?,
    };
    f(&ctx)
}

#[cfg(feature = "torch")]
pub fn fer_anyhow<T, F>(device_id: i32, f: F) -> anyhow::Result<T>
where
    F: FnOnce(&FerCtx) -> anyhow::Result<T>,
{
    let ctx = FerCtx {
        allocator: Arc::clone(global_cpu_tlsf()),
        runtime: GpuLangRuntime::new(device_id).map_err(anyhow::Error::new)?,
    };
    f(&ctx)
}
