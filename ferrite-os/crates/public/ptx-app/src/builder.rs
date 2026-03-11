//! FerApp builder — the main entry point for GPU compute applications.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use ptx_runtime::PtxRuntime;
use ptx_sys::GPUHotConfig;

use crate::checkpoint::CheckpointStore;
use crate::ctx::Ctx;
use crate::daemon_client::DaemonClient;
use crate::emit::Emitter;
use crate::error::AppError;
use crate::resource::Priority;
use crate::restart::Restart;

/// High-level builder for daemon-native GPU compute applications.
///
/// # Example
///
/// ```no_run
/// use ptx_app::{FerApp, DType};
///
/// FerApp::new("my-app")
///     .pool_fraction(0.4)
///     .streams(8)
///     .run(|ctx| {
///         let t = ctx.tensor(&[1024, 1024], DType::F32)?.randn()?;
///         ctx.emit("result", &"done");
///         Ok(())
///     })
///     .expect("app failed");
/// ```
pub struct FerApp {
    name: String,
    pool_fraction: f32,
    streams: u32,
    device: i32,
    restart: Restart,
    priority: Priority,
    tenant_id: Option<u64>,
    daemon_socket: Option<String>,
    checkpoint_dir: Option<String>,
}

impl FerApp {
    /// Create a new FerApp builder with the given application name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pool_fraction: 0.30,
            streams: 8,
            device: 0,
            restart: Restart::default(),
            priority: Priority::default(),
            tenant_id: None,
            daemon_socket: None,
            checkpoint_dir: None,
        }
    }

    /// Set the fraction of GPU VRAM to allocate for the TLSF pool.
    ///
    /// Must be in the range (0.0, 1.0]. Default: 0.30.
    pub fn pool_fraction(mut self, f: f32) -> Self {
        self.pool_fraction = f;
        self
    }

    /// Set the number of CUDA streams to create. Default: 8.
    pub fn streams(mut self, count: u32) -> Self {
        self.streams = count;
        self
    }

    /// Set the CUDA device ID. Default: 0.
    pub fn device(mut self, id: i32) -> Self {
        self.device = id;
        self
    }

    /// Set the restart policy. Default: never.
    pub fn restart(mut self, r: Restart) -> Self {
        self.restart = r;
        self
    }

    /// Set the job priority. Default: Normal.
    pub fn priority(mut self, p: Priority) -> Self {
        self.priority = p;
        self
    }

    /// Set the tenant ID for multi-tenant scheduling.
    pub fn tenant(mut self, id: u64) -> Self {
        self.tenant_id = Some(id);
        self
    }

    /// Set an explicit daemon socket path.
    ///
    /// Default: auto-detect from `FERRITE_SOCKET`/`FERRITE_DAEMON_SOCKET`,
    /// then daemon-style per-user defaults. Falls back to `/tmp/ferrite.sock`
    /// for legacy compatibility.
    pub fn daemon_socket(mut self, path: &str) -> Self {
        self.daemon_socket = Some(path.to_string());
        self
    }

    /// Set the checkpoint directory.
    ///
    /// Default: platform data dir (e.g. `~/.local/share/ferrite/checkpoints/<name>` on Linux).
    pub fn checkpoint_dir(mut self, path: &str) -> Self {
        self.checkpoint_dir = Some(path.to_string());
        self
    }

    /// Run the application.
    ///
    /// ## Execution flow
    ///
    /// 1. Validate builder parameters.
    /// 2. If `FERRITE_JOB_ID` is set, we are the daemon-spawned process — skip
    ///    to step 4.
    /// 3. Otherwise, connect to the daemon and submit ourselves as a durable
    ///    job. The daemon will re-spawn us with `FERRITE_JOB_ID` set.
    /// 4. Initialize `PtxRuntime` with the declared configuration.
    /// 5. Build `Ctx`.
    /// 6. Call the user closure inside `catch_unwind`.
    /// 7. Sync GPU, emit completion event, return result.
    pub fn run<F>(self, f: F) -> Result<(), AppError>
    where
        F: FnOnce(&Ctx) -> Result<(), AppError> + Send + 'static,
    {
        self.validate()?;

        // Step 2: Check if we are daemon-spawned.
        let is_daemon_spawned = std::env::var("FERRITE_JOB_ID").is_ok();

        if !is_daemon_spawned {
            // Step 3: Try to submit to daemon. If daemon is unavailable,
            // run directly (standalone mode).
            if self.try_daemon_submit()? {
                return Ok(());
            }
            tracing::info!(
                app = %self.name,
                "daemon not available, running in standalone mode"
            );
        }

        // Step 4: Init runtime.
        let config = GPUHotConfig {
            pool_fraction: self.pool_fraction,
            max_streams: self.streams,
            ..GPUHotConfig::default()
        };
        let runtime = Arc::new(
            PtxRuntime::with_config(self.device, Some(config))
                .map_err(|e| AppError::Runtime(e))?,
        );

        // Step 5: Build Ctx.
        let daemon_client = self.make_daemon_client();
        let emitter = Emitter::new(
            self.name.clone(),
            self.tenant_id,
            daemon_client,
        );
        let checkpoint_dir = self.resolve_checkpoint_dir();
        let checkpoint = CheckpointStore::new(&checkpoint_dir)?;
        let ctx = Ctx::new(Arc::clone(&runtime), emitter, checkpoint);

        ctx.log(&format!(
            "app '{}' started (pool: {})",
            self.name, ctx.pool_stats()
        ));

        // Step 6: Call user closure inside catch_unwind.
        let result = panic::catch_unwind(AssertUnwindSafe(|| f(&ctx)));

        // Step 7: Sync GPU and report.
        if let Err(e) = runtime.sync_all() {
            tracing::warn!(error = %e, "failed to sync GPU after user closure");
        }

        match result {
            Ok(Ok(())) => {
                ctx.emit("app_completed", &serde_json::json!({
                    "app_name": self.name,
                    "status": "success",
                }));
                Ok(())
            }
            Ok(Err(app_err)) => {
                ctx.emit("app_completed", &serde_json::json!({
                    "app_name": self.name,
                    "status": "error",
                    "message": app_err.to_string(),
                }));
                Err(app_err)
            }
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                ctx.emit("app_completed", &serde_json::json!({
                    "app_name": self.name,
                    "status": "panic",
                    "message": msg,
                }));
                Err(AppError::Panic { message: msg })
            }
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn validate(&self) -> Result<(), AppError> {
        if self.name.is_empty() {
            return Err(AppError::ValidationError {
                message: "app name must not be empty".to_string(),
            });
        }
        if self
            .name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '-' && c != '_')
        {
            return Err(AppError::ValidationError {
                message: format!(
                    "app name '{}' contains invalid characters (use alphanumeric, '-', '_')",
                    self.name
                ),
            });
        }
        if self.pool_fraction <= 0.0 || self.pool_fraction > 1.0 {
            return Err(AppError::ValidationError {
                message: format!(
                    "pool_fraction must be in (0, 1], got {}",
                    self.pool_fraction
                ),
            });
        }
        if self.streams < 1 {
            return Err(AppError::ValidationError {
                message: "streams must be >= 1".to_string(),
            });
        }
        Ok(())
    }

    /// Try to submit this app to the daemon as a durable job.
    ///
    /// Returns `Ok(true)` if the job was submitted (caller should exit).
    /// Returns `Ok(false)` if the daemon is unavailable (run standalone).
    fn try_daemon_submit(&self) -> Result<bool, AppError> {
        let client = match &self.daemon_socket {
            Some(path) => DaemonClient::new(path),
            None => DaemonClient::auto_detect(),
        };

        if !client.socket_exists() {
            return Ok(false);
        }

        let exe = std::env::current_exe().map_err(|e| AppError::App {
            message: format!("cannot determine current executable: {}", e),
        })?;

        let args: Vec<String> = std::env::args().skip(1).collect();

        match client.submit_job(&exe.to_string_lossy(), &args) {
            Ok(job_id) => {
                tracing::info!(
                    app = %self.name,
                    job_id,
                    "submitted to daemon — daemon will spawn the process"
                );
                Ok(true)
            }
            Err(AppError::DaemonUnavailable { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn make_daemon_client(&self) -> Option<DaemonClient> {
        let client = match &self.daemon_socket {
            Some(path) => DaemonClient::new(path),
            None => DaemonClient::auto_detect(),
        };
        if client.socket_exists() {
            Some(client)
        } else {
            None
        }
    }

    fn resolve_checkpoint_dir(&self) -> String {
        if let Some(dir) = &self.checkpoint_dir {
            return dir.clone();
        }

        // Use platform-correct data directory:
        //   Linux:   $XDG_DATA_HOME/ferrite/checkpoints/<name>  (~/.local/share/...)
        //   macOS:   ~/Library/Application Support/ferrite/checkpoints/<name>
        if let Some(dirs) = directories::ProjectDirs::from("", "", "ferrite") {
            return dirs
                .data_dir()
                .join("checkpoints")
                .join(&self.name)
                .to_string_lossy()
                .to_string();
        }

        // Fallback if platform dirs unavailable.
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/.local/share/ferrite/checkpoints/{}", home, self.name)
    }
}
