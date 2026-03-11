mod clean_exit;
mod format;
mod runtime;
mod shm;
mod telemetry;
mod vfs;
mod vmm;

pub use clean_exit::assert_clean_exit;
pub use format::{format_bytes, format_duration};
pub use runtime::{get_duration_secs, init_runtime};
pub use shm::{shm_safe_alloc, shm_safe_close, shm_safe_open, shm_safe_unlink};
pub use telemetry::TelemetryReporter;
pub use vfs::{
    vfs_safe_create_tensor, vfs_safe_init, vfs_safe_mkdir, vfs_safe_mmap_tensor, vfs_safe_open,
    vfs_safe_rmdir, vfs_safe_sync_tensor, vfs_safe_unlink,
};
pub use vmm::{vmm_safe_alloc_page, vmm_safe_get_stats, vmm_safe_init};
