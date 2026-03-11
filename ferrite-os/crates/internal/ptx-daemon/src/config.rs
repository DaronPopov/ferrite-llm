use std::env;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Configuration for the durable job subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobsConfig {
    /// Directory where job state files are persisted.
    #[serde(default = "default_jobs_state_dir")]
    pub state_dir: String,

    /// Default restart policy: "never", "on-failure", or "always".
    #[serde(default = "default_jobs_restart_policy")]
    pub default_restart_policy: String,

    /// Maximum retry attempts for the default restart policy.
    #[serde(default = "default_jobs_max_retries")]
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds.
    #[serde(default = "default_jobs_backoff_initial_ms")]
    pub backoff_initial_ms: u64,

    /// Maximum backoff delay in milliseconds.
    #[serde(default = "default_jobs_backoff_max_ms")]
    pub backoff_max_ms: u64,

    /// Backoff multiplier.
    #[serde(default = "default_jobs_backoff_multiplier")]
    pub backoff_multiplier: f64,
}

fn default_jobs_state_dir() -> String {
    ferrite_platform::paths::default_jobs_state_dir()
}
fn default_jobs_restart_policy() -> String {
    "never".to_string()
}
fn default_jobs_max_retries() -> u32 {
    3
}
fn default_jobs_backoff_initial_ms() -> u64 {
    1000
}
fn default_jobs_backoff_max_ms() -> u64 {
    30000
}
fn default_jobs_backoff_multiplier() -> f64 {
    2.0
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            state_dir: default_jobs_state_dir(),
            default_restart_policy: default_jobs_restart_policy(),
            max_retries: default_jobs_max_retries(),
            backoff_initial_ms: default_jobs_backoff_initial_ms(),
            backoff_max_ms: default_jobs_backoff_max_ms(),
            backoff_multiplier: default_jobs_backoff_multiplier(),
        }
    }
}

impl JobsConfig {
    /// Build a `RestartPolicy` from the config values.
    pub fn to_restart_policy(&self) -> ptx_runtime::job::RestartPolicy {
        use ptx_runtime::job::{BackoffConfig, RestartPolicy};
        let backoff = BackoffConfig {
            initial_delay_ms: self.backoff_initial_ms,
            max_delay_ms: self.backoff_max_ms,
            multiplier: self.backoff_multiplier,
        };
        match self.default_restart_policy.as_str() {
            "on-failure" => RestartPolicy::OnFailure {
                max_retries: self.max_retries,
                backoff,
            },
            "always" => RestartPolicy::Always {
                max_retries: self.max_retries,
                backoff,
            },
            _ => RestartPolicy::Never,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// GPU device ID
    #[serde(default)]
    pub device_id: i32,

    /// Unix socket path
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    /// PID file path
    #[serde(default = "default_pid_path")]
    pub pid_file: String,

    /// Maximum concurrent clients
    #[serde(default = "default_max_clients")]
    pub max_clients: usize,

    /// Maximum streams
    #[serde(default = "default_max_streams")]
    pub max_streams: u32,

    /// VRAM pool fraction
    #[serde(default = "default_pool_fraction")]
    pub pool_fraction: f32,

    /// Prefer Orin-specific unified-memory scheduler path.
    #[serde(default)]
    pub prefer_orin_unified_memory: bool,

    /// Use managed-memory TLSF backing pool.
    #[serde(default)]
    pub use_managed_pool: bool,

    /// Keepalive interval (milliseconds)
    #[serde(default = "default_keepalive_ms")]
    pub keepalive_ms: u64,

    /// Watch metrics interval (milliseconds)
    #[serde(default = "default_watch_ms")]
    pub watch_ms: u64,

    /// Enable watch mode
    #[serde(default)]
    pub watch_enabled: bool,

    /// Enable leak detection
    #[serde(default)]
    pub enable_leak_detection: bool,

    /// Boot persistent kernel
    #[serde(default)]
    pub boot_kernel: bool,

    /// Log directory
    #[serde(default = "default_log_dir")]
    pub log_dir: Option<String>,

    /// Client timeout (seconds)
    #[serde(default = "default_client_timeout")]
    pub client_timeout_secs: u64,

    /// Optional directory containing managed app binaries
    #[serde(default = "default_apps_bin_dir")]
    pub apps_bin_dir: Option<String>,

    /// Skip TUI, run original headless behavior
    #[serde(default)]
    pub headless: bool,

    /// GPU device name for TUI header display
    #[serde(default)]
    pub gpu_name: Option<String>,

    /// TUI theme variant ("default" or "high-contrast")
    #[serde(default)]
    pub theme: Option<String>,

    /// Control plane configuration.
    #[serde(default)]
    pub control_plane: ControlPlaneConfig,

    /// Durable job subsystem configuration.
    #[serde(default)]
    pub jobs: JobsConfig,

    /// Multi-tenant scheduler configuration.
    #[serde(default)]
    pub scheduler: SchedulerConfig,

    /// Single-pool strict mode: deny heavy GPU runs from child processes.
    #[serde(default)]
    pub single_pool_strict: bool,
}

/// Configuration for the OS control plane and policy enforcement layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlaneConfig {
    /// Maximum number of audit entries to retain in memory.
    #[serde(default = "default_audit_max_entries")]
    pub audit_max_entries: usize,

    /// Event stream ring buffer capacity.
    #[serde(default = "default_event_stream_buffer")]
    pub event_stream_buffer: usize,

    /// Default policy mode: "permissive" (allow-by-default) or "strict" (deny unknown).
    #[serde(default = "default_policy_mode")]
    pub default_policy: String,

    /// Whether the scheduler TUI panel is enabled.
    #[serde(default = "default_enable_scheduler_tui")]
    pub enable_scheduler_tui: bool,
}

fn default_audit_max_entries() -> usize {
    10_000
}
fn default_event_stream_buffer() -> usize {
    1_000
}
fn default_policy_mode() -> String {
    "permissive".to_string()
}
fn default_enable_scheduler_tui() -> bool {
    true
}

impl Default for ControlPlaneConfig {
    fn default() -> Self {
        Self {
            audit_max_entries: default_audit_max_entries(),
            event_stream_buffer: default_event_stream_buffer(),
            default_policy: default_policy_mode(),
            enable_scheduler_tui: default_enable_scheduler_tui(),
        }
    }
}

/// Configuration for the multi-tenant GPU scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling policy: "fair-share" or "fifo".
    #[serde(default = "default_scheduler_policy")]
    pub policy: String,

    /// Default VRAM quota for new tenants (bytes). 0 means unlimited.
    #[serde(default)]
    pub default_max_vram_bytes: u64,

    /// Default maximum streams per tenant. 0 means unlimited.
    #[serde(default)]
    pub default_max_streams: u64,

    /// Default maximum concurrent jobs per tenant. 0 means unlimited.
    #[serde(default)]
    pub default_max_concurrent_jobs: u64,

    /// Default runtime budget in milliseconds per tenant. 0 means unlimited.
    #[serde(default)]
    pub default_max_runtime_budget_ms: u64,

    /// Maximum number of jobs that can be queued in the scheduler.
    #[serde(default = "default_scheduler_max_queued_jobs")]
    pub max_queued_jobs: usize,

    /// Maximum queued jobs per tenant. 0 means no per-tenant limit.
    #[serde(default)]
    pub max_queued_jobs_per_tenant: usize,
}

fn default_scheduler_policy() -> String {
    "fair-share".to_string()
}

fn default_scheduler_max_queued_jobs() -> usize {
    4096
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: default_scheduler_policy(),
            default_max_vram_bytes: 0,
            default_max_streams: 0,
            default_max_concurrent_jobs: 0,
            default_max_runtime_budget_ms: 0,
            max_queued_jobs: default_scheduler_max_queued_jobs(),
            max_queued_jobs_per_tenant: 0,
        }
    }
}

impl SchedulerConfig {
    /// Convert this daemon config into the runtime's `SchedulerConfig`.
    pub fn to_runtime_config(&self) -> ptx_runtime::scheduler::SchedulerConfig {
        use ptx_runtime::scheduler::TenantQuotas;

        let quotas = TenantQuotas {
            max_vram_bytes: if self.default_max_vram_bytes == 0 {
                u64::MAX
            } else {
                self.default_max_vram_bytes
            },
            max_streams: if self.default_max_streams == 0 {
                u64::MAX
            } else {
                self.default_max_streams
            },
            max_concurrent_jobs: if self.default_max_concurrent_jobs == 0 {
                u64::MAX
            } else {
                self.default_max_concurrent_jobs
            },
            max_runtime_budget_ms: if self.default_max_runtime_budget_ms == 0 {
                u64::MAX
            } else {
                self.default_max_runtime_budget_ms
            },
        };

        ptx_runtime::scheduler::SchedulerConfig {
            policy: self.policy.clone(),
            default_tenant_quotas: quotas,
            max_queued_jobs: self.max_queued_jobs,
            max_queued_jobs_per_tenant: self.max_queued_jobs_per_tenant,
        }
    }
}

fn default_socket_path() -> String {
    ferrite_platform::paths::default_socket_addr()
}

fn default_pid_path() -> String {
    ferrite_platform::paths::default_pid_path()
}

fn default_max_clients() -> usize {
    32
}
fn default_max_streams() -> u32 {
    128
}
fn default_pool_fraction() -> f32 {
    0.25
}
fn default_keepalive_ms() -> u64 {
    5000
}
fn default_watch_ms() -> u64 {
    1000
}
fn default_log_dir() -> Option<String> {
    None
}
fn default_client_timeout() -> u64 {
    30
}
fn default_apps_bin_dir() -> Option<String> {
    None
}

fn env_bool_enabled(key: &str) -> bool {
    matches!(
        env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            socket_path: default_socket_path(),
            pid_file: default_pid_path(),
            max_clients: default_max_clients(),
            max_streams: default_max_streams(),
            pool_fraction: default_pool_fraction(),
            prefer_orin_unified_memory: false,
            use_managed_pool: false,
            keepalive_ms: default_keepalive_ms(),
            watch_ms: default_watch_ms(),
            watch_enabled: false,
            enable_leak_detection: false,
            boot_kernel: false,
            log_dir: default_log_dir(),
            client_timeout_secs: default_client_timeout(),
            apps_bin_dir: default_apps_bin_dir(),
            headless: false,
            gpu_name: None,
            theme: None,
            control_plane: ControlPlaneConfig::default(),
            jobs: JobsConfig::default(),
            scheduler: SchedulerConfig::default(),
            single_pool_strict: false,
        }
    }
}

impl DaemonConfig {
    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("TOML parse error: {}", e),
            )
        })
    }

    pub fn merge_from_env(&mut self) {
        if let Ok(val) = env::var("FERRITE_DEVICE") {
            if let Ok(id) = val.parse() {
                self.device_id = id;
            }
        }
        if let Ok(val) = env::var("FERRITE_SOCKET") {
            self.socket_path = val;
        } else if let Ok(val) = env::var("FERRITE_DAEMON_SOCKET") {
            // Backward-compatible alias used by older clients.
            self.socket_path = val;
        }
        if let Ok(val) = env::var("FERRITE_PID_FILE") {
            self.pid_file = val;
        }
        if let Ok(val) = env::var("FERRITE_MAX_STREAMS") {
            if let Ok(streams) = val.parse() {
                self.max_streams = streams;
            }
        }
        if env_bool_enabled("FERRITE_PREFER_ORIN_UM") {
            self.prefer_orin_unified_memory = true;
        }
        if env_bool_enabled("FERRITE_MANAGED_POOL") {
            self.use_managed_pool = true;
        }
        if env::var("FERRITE_BOOT_KERNEL").is_ok() {
            self.boot_kernel = true;
        }
        if env::var("FERRITE_WATCH").is_ok() {
            self.watch_enabled = true;
        }
        if let Ok(val) = env::var("FERRITE_APPS_BIN_DIR") {
            self.apps_bin_dir = Some(val);
        }
        if let Ok(val) = env::var("FERRITE_GPU_NAME") {
            self.gpu_name = Some(val);
        }
        if env::var("FERRITE_HEADLESS").is_ok() {
            self.headless = true;
        }
        if let Ok(val) = env::var("FERRITE_THEME") {
            self.theme = Some(val);
        }
        if env::var("FERRITE_SINGLE_POOL_STRICT").ok().as_deref() == Some("1") {
            self.single_pool_strict = true;
        }
    }
}
