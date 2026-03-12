use std::alloc::Layout;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use crate::{LangError, Result};

type HostTlsf = rlsf::Tlsf<'static, u32, u32, 20, 16>;

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuTlsfStats {
    pub arena_bytes: usize,
    pub alloc_calls: u64,
    pub free_calls: u64,
    pub failed_alloc_calls: u64,
    pub current_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
}

pub struct CpuTlsf {
    tlsf: Mutex<HostTlsf>,
    stats: Mutex<CpuTlsfStats>,
}

impl CpuTlsf {
    fn new(arena_bytes: usize) -> Self {
        let mut tlsf = HostTlsf::new();

        // Leak once for process lifetime to satisfy TLSF pool lifetime.
        let pool: &'static mut [MaybeUninit<u8>] =
            Box::leak(vec![MaybeUninit::<u8>::uninit(); arena_bytes].into_boxed_slice());
        let _ = tlsf.insert_free_block(pool);

        Self {
            tlsf: Mutex::new(tlsf),
            stats: Mutex::new(CpuTlsfStats {
                arena_bytes,
                ..CpuTlsfStats::default()
            }),
        }
    }

    pub fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(bytes.max(1), align.max(1)).map_err(|e| {
            LangError::Transfer {
                message: format!("invalid layout for cpu tlsf alloc: {e}"),
            }
        })?;

        let mut tlsf = self.tlsf.lock().map_err(|_| LangError::Transfer {
            message: "cpu tlsf mutex poisoned".to_string(),
        })?;

        let ptr = tlsf.allocate(layout).ok_or_else(|| {
            let mut st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
            st.failed_alloc_calls += 1;
            LangError::Transfer {
                message: format!("cpu tlsf allocation failed (bytes={bytes}, align={align})"),
            }
        })?;

        let mut st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        st.alloc_calls += 1;
        st.current_allocated_bytes = st.current_allocated_bytes.saturating_add(bytes);
        st.peak_allocated_bytes = st.peak_allocated_bytes.max(st.current_allocated_bytes);

        Ok(ptr)
    }

    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> Result<()> {
        let mut tlsf = self.tlsf.lock().map_err(|_| LangError::Transfer {
            message: "cpu tlsf mutex poisoned".to_string(),
        })?;
        tlsf.deallocate(ptr, align.max(1));

        let mut st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        st.free_calls += 1;
        st.current_allocated_bytes = st.current_allocated_bytes.saturating_sub(bytes);
        Ok(())
    }

    pub fn stats(&self) -> CpuTlsfStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }
}

pub fn global_cpu_tlsf() -> &'static Arc<CpuTlsf> {
    static CELL: OnceLock<Arc<CpuTlsf>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes = std::env::var("FERRITE_CPU_TLSF_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256 * 1024 * 1024);
        Arc::new(CpuTlsf::new(bytes.max(1024 * 1024)))
    })
}
