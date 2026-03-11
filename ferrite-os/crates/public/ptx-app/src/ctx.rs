//! Application context — the main handle passed to the user closure.

use std::path::Path;
use std::sync::Arc;

use humansize::{format_size, BINARY};
use indicatif::{ProgressBar, ProgressStyle};
use serde::de::DeserializeOwned;
use serde::Serialize;

use ptx_runtime::{PtxRuntime, Stream, TLSFPoolStats};
use ptx_tensor::{DType, Tensor};

use crate::checkpoint::CheckpointStore;
use crate::emit::Emitter;
use crate::error::AppError;
use crate::tensor_factory::TensorBuilder;

/// Memory pool statistics exposed to the user.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub utilization: f32,
    pub fragmentation: f32,
    pub healthy: bool,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} / {} ({:.1}% used, {} free{})",
            format_size(self.allocated_bytes, BINARY),
            format_size(self.total_bytes, BINARY),
            self.utilization,
            format_size(self.free_bytes, BINARY),
            if self.healthy { "" } else { ", UNHEALTHY" },
        )
    }
}

impl From<TLSFPoolStats> for PoolStats {
    fn from(s: TLSFPoolStats) -> Self {
        Self {
            total_bytes: s.total_pool_size,
            allocated_bytes: s.allocated_bytes,
            free_bytes: s.free_bytes,
            utilization: s.utilization_percent,
            fragmentation: s.fragmentation_ratio,
            healthy: s.is_healthy,
        }
    }
}

/// The application context passed to the user closure in [`FerApp::run`].
///
/// Provides tensor creation, stream access, memory stats, event emission,
/// checkpoint persistence, progress bars, memory-mapped file loading,
/// and an escape hatch to the raw runtime.
pub struct Ctx {
    runtime: Arc<PtxRuntime>,
    emitter: Emitter,
    checkpoint: CheckpointStore,
}

impl Ctx {
    pub(crate) fn new(
        runtime: Arc<PtxRuntime>,
        emitter: Emitter,
        checkpoint: CheckpointStore,
    ) -> Self {
        Self {
            runtime,
            emitter,
            checkpoint,
        }
    }

    // ── Tensor creation ──────────────────────────────────────────────

    /// Start building a tensor with the given shape and dtype.
    ///
    /// Finalize with `.zeros()`, `.ones()`, `.randn()`, `.fill(val)`, or
    /// `.from_slice(data)`.
    pub fn tensor(&self, shape: &[usize], dtype: DType) -> Result<TensorBuilder, AppError> {
        Ok(TensorBuilder::new(
            shape.to_vec(),
            dtype,
            Arc::clone(&self.runtime),
        ))
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor, AppError> {
        Tensor::zeros(shape, dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor filled with ones.
    pub fn ones(&self, shape: &[usize], dtype: DType) -> Result<Tensor, AppError> {
        Tensor::ones(shape, dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor filled with a constant value.
    pub fn full(&self, shape: &[usize], val: f32, dtype: DType) -> Result<Tensor, AppError> {
        Tensor::full(shape, val, dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a tensor from host data.
    pub fn from_slice<T: bytemuck::Pod>(
        &self,
        data: &[T],
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, AppError> {
        Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
    }

    /// Create a 1-D tensor with evenly spaced values.
    pub fn arange(&self, start: f32, end: f32, step: f32) -> Result<Tensor, AppError> {
        Tensor::arange(start, end, step, &self.runtime).map_err(AppError::from)
    }

    // ── Memory-mapped file loading ───────────────────────────────────

    /// Load a binary file directly into a GPU tensor via memory mapping.
    ///
    /// The file is memory-mapped (zero-copy from disk to virtual memory),
    /// then uploaded to the GPU in a single transfer. This avoids the
    /// double-copy of `fs::read()` + `from_slice()`.
    ///
    /// The file must contain exactly `shape.product() * dtype.size_bytes()`
    /// bytes of raw data (no header).
    pub fn mmap_load(
        &self,
        path: impl AsRef<Path>,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, AppError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| AppError::App {
            message: format!("cannot open {}: {}", path.display(), e),
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| AppError::App {
            message: format!("cannot mmap {}: {}", path.display(), e),
        })?;

        let elem_count: usize = shape.iter().product();
        let expected_bytes = elem_count * dtype.size_bytes();
        if mmap.len() != expected_bytes {
            return Err(AppError::ValidationError {
                message: format!(
                    "file {} is {} but shape {:?} x {:?} requires {}",
                    path.display(),
                    format_size(mmap.len(), BINARY),
                    shape,
                    dtype,
                    format_size(expected_bytes, BINARY),
                ),
            });
        }

        // Reinterpret the mmap'd bytes as the target type and upload.
        match dtype {
            DType::F32 => {
                let data: &[f32] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::F64 => {
                let data: &[f64] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::I32 => {
                let data: &[i32] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::U8 => {
                let data: &[u8] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::U32 => {
                let data: &[u32] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::I8 => {
                let data: &[i8] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::I16 => {
                let data: &[i16] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            DType::I64 => {
                let data: &[i64] = bytemuck::cast_slice(&mmap);
                Tensor::from_slice(data, shape, dtype, &self.runtime).map_err(AppError::from)
            }
            _ => Err(AppError::ValidationError {
                message: format!("mmap_load not supported for {:?}", dtype),
            }),
        }
    }

    // ── Streams ──────────────────────────────────────────────────────

    /// Get the next stream in round-robin order.
    pub fn stream(&self) -> Stream {
        self.runtime.next_stream()
    }

    /// Get a specific stream by ID.
    pub fn stream_by_id(&self, id: i32) -> Result<Stream, AppError> {
        self.runtime.stream(id).map_err(AppError::from)
    }

    // ── Memory ───────────────────────────────────────────────────────

    /// Get current memory pool statistics.
    ///
    /// The returned `PoolStats` implements `Display` for human-readable
    /// output (e.g. `"384 MiB / 1.5 GiB (25.6% used, 1.1 GiB free)"`).
    pub fn pool_stats(&self) -> PoolStats {
        self.runtime.tlsf_stats().into()
    }

    /// Check if the pool can satisfy an allocation of `bytes`.
    pub fn can_allocate(&self, bytes: usize) -> bool {
        self.runtime.can_allocate(bytes)
    }

    // ── Progress bars ────────────────────────────────────────────────

    /// Create a progress bar for a loop with a known total.
    ///
    /// ```no_run
    /// # use ptx_app::AppError;
    /// # fn example(ctx: &ptx_app::Ctx) -> Result<(), AppError> {
    /// let pb = ctx.progress(1000, "training");
    /// for epoch in 0..1000 {
    ///     // ... GPU work ...
    ///     pb.inc(1);
    /// }
    /// pb.finish_with_message("done");
    /// # Ok(())
    /// # }
    /// ```
    pub fn progress(&self, total: u64, task_name: &str) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix} [{bar:40.cyan/blue}] {pos}/{len} ({eta} remaining)"
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_prefix(task_name.to_string());
        pb
    }

    /// Create a spinner for an operation with unknown duration.
    ///
    /// ```no_run
    /// # use ptx_app::AppError;
    /// # fn example(ctx: &ptx_app::Ctx) -> Result<(), AppError> {
    /// let sp = ctx.spinner("loading model weights");
    /// // ... long operation ...
    /// sp.finish_with_message("loaded");
    /// # Ok(())
    /// # }
    /// ```
    pub fn spinner(&self, message: &str) -> ProgressBar {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&[".", "..", "...", "....", ""]),
        );
        sp.set_message(message.to_string());
        sp.enable_steady_tick(std::time::Duration::from_millis(120));
        sp
    }

    // ── Events ───────────────────────────────────────────────────────

    /// Emit a named event to the daemon event stream.
    pub fn emit(&self, name: &str, payload: &impl Serialize) {
        self.emitter.emit(name, payload);
    }

    /// Emit a log message through the event stream.
    pub fn log(&self, msg: &str) {
        self.emitter.log(msg);
    }

    // ── Checkpoint ───────────────────────────────────────────────────

    /// Save a checkpoint with the given label.
    pub fn checkpoint(&self, label: &str, state: &impl Serialize) -> Result<(), AppError> {
        self.checkpoint.save(label, state)
    }

    /// Restore a previously saved checkpoint.
    ///
    /// Returns `Ok(None)` if no checkpoint exists for the label.
    pub fn restore<T: DeserializeOwned>(&self, label: &str) -> Result<Option<T>, AppError> {
        self.checkpoint.restore(label)
    }

    // ── Escape hatch ─────────────────────────────────────────────────

    /// Get a reference to the underlying PTX runtime.
    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        &self.runtime
    }
}
