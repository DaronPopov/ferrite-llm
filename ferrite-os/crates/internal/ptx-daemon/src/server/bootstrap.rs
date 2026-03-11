use std::io;
use std::sync::Arc;

use ferrite_platform::ipc::{IpcListener, Endpoint};
use ptx_runtime::PtxRuntime;
use tracing::{error, info, warn};

use crate::config::DaemonConfig;
use crate::job_store::JobStore;
use crate::pid::PidFile;
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;
use crate::supervisor::JobSupervisor;

pub(super) fn should_use_tui(config: &DaemonConfig) -> bool {
    !config.headless && ferrite_platform::tty::stdout_is_tty()
}

pub(super) fn init_logging(config: &DaemonConfig, use_tui: bool) {
    if use_tui {
        let log_dir = config.log_dir.clone().unwrap_or_else(|| {
            ferrite_platform::paths::temp_dir().to_string_lossy().to_string()
        });
        let file_appender = tracing_appender::rolling::daily(&log_dir, "ferrite-daemon.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        if let Err(e) = tracing_subscriber::fmt()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .try_init()
        {
            eprintln!("ferrite-daemon logging init skipped: {}", e);
        }
    } else if let Some(log_dir) = &config.log_dir {
        let file_appender = tracing_appender::rolling::daily(log_dir, "ferrite-daemon.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        if let Err(e) = tracing_subscriber::fmt()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .try_init()
        {
            eprintln!("ferrite-daemon logging init skipped: {}", e);
        }
    } else {
        if let Err(e) = tracing_subscriber::fmt()
            .with_target(true)
            .with_thread_ids(true)
            .try_init()
        {
            eprintln!("ferrite-daemon logging init skipped: {}", e);
        }
    }
}

pub(super) fn create_pid_file(config: &DaemonConfig) -> io::Result<PidFile> {
    PidFile::create(std::path::Path::new(&config.pid_file)).map_err(|e| {
        error!("Failed to create PID file: {}", e);
        e
    })
}

pub(super) fn prepare_listener(config: &DaemonConfig) -> io::Result<IpcListener> {
    let endpoint = Endpoint::new(&config.socket_path);
    IpcListener::bind(&endpoint)
}

pub(super) fn init_runtime(config: &DaemonConfig) -> io::Result<Arc<PtxRuntime>> {
    info!("Initializing GPU runtime on device {}", config.device_id);

    let mut runtime_config = ptx_sys::GPUHotConfig::default();
    runtime_config.max_streams = config.max_streams;
    runtime_config.pool_fraction = config.pool_fraction;
    runtime_config.prefer_orin_unified_memory = config.prefer_orin_unified_memory;
    runtime_config.use_managed_pool = config.use_managed_pool;
    runtime_config.enable_leak_detection = config.enable_leak_detection;

    let runtime = Arc::new(
        PtxRuntime::with_config(config.device_id, Some(runtime_config)).map_err(|e| {
            error!("Failed to initialize runtime: {:?}", e);
            io::Error::new(
                io::ErrorKind::Other,
                format!("Runtime init failed: {:?}", e),
            )
        })?,
    );

    info!(
        "Runtime initialized with {} streams, pool={:.0}%, orin_um={}, managed_pool={}",
        config.max_streams,
        config.pool_fraction * 100.0,
        config.prefer_orin_unified_memory,
        config.use_managed_pool
    );

    // Export runtime/context handles from the daemon process environment so
    // spawned runner subprocesses can inherit the same execution context.
    runtime.export_for_hook();
    runtime.export_context();
    if std::env::var("PTX_RUNTIME_PTR").is_err() {
        warn!("PTX_RUNTIME_PTR is not set after runtime export; runner subprocesses may initialize a separate pool");
    }

    Ok(runtime)
}

pub(super) fn boot_kernel_if_requested(config: &DaemonConfig, runtime: &Arc<PtxRuntime>) {
    if config.boot_kernel {
        info!("Booting persistent kernel");
        unsafe { ptx_sys::ptx_os_boot_persistent_kernel(runtime.raw()) };
    }
}

pub(super) fn build_state(
    runtime: Arc<PtxRuntime>,
    config: DaemonConfig,
) -> io::Result<(Arc<DaemonState>, Arc<parking_lot::Mutex<ScriptRunner>>)> {
    let fallback_dir = ferrite_platform::paths::fallback_job_state_dir();

    // Initialize the durable job supervisor.
    let job_store = match JobStore::new(&config.jobs.state_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("failed to create job store at {}: {}; using fallback", config.jobs.state_dir, e);
            JobStore::new(&fallback_dir).map_err(|fallback_err| {
                io::Error::other(format!(
                    "failed to create job store at {} ({}) and fallback {} ({})",
                    config.jobs.state_dir, e, fallback_dir, fallback_err
                ))
            })?
        }
    };

    let (supervisor, reconcile_diags) = match JobSupervisor::new(job_store) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("job supervisor init failed: {}; starting with empty supervisor", e);
            let fallback_store = JobStore::new(&fallback_dir).map_err(|fallback_err| {
                io::Error::other(format!(
                    "job supervisor init failed ({}) and fallback store {} creation failed ({})",
                    e, fallback_dir, fallback_err
                ))
            })?;
            // Create an empty supervisor with no recovered jobs.
            (crate::supervisor::JobSupervisor::empty(fallback_store), Vec::new())
        }
    };

    for diag in &reconcile_diags {
        ptx_runtime::telemetry::emit_diag(diag);
    }

    let state = Arc::new(DaemonState::new(runtime.clone(), config, supervisor));
    let mut script_runner = ScriptRunner::new(runtime);

    // Enable AOT disk cache under the working directory
    let cache_dir = std::env::current_dir()
        .unwrap_or_else(|_| ".".into())
        .join(".ferrite-cache")
        .join("jit");
    if let Err(e) = script_runner.enable_disk_cache(cache_dir) {
        tracing::warn!("failed to enable JIT disk cache: {}", e);
    }

    let runner = Arc::new(parking_lot::Mutex::new(script_runner));
    Ok((state, runner))
}
