mod files_editor;
pub mod run_state;
mod shell_input;
mod telemetry;
mod ui_effects;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use ratatui::layout::Rect;
use tachyonfx::Effect;

use crate::events::LogEntry;
use super::editor::{KeyMap, MacroEngine, MarkStore, RegisterFile, SearchState};
use super::fxscript::FxScriptConfig;
use super::workspace::PendingConfirm;
use super::workspace::tree::FileTree;
use super::profiling::ProfileRing;
use self::run_state::{RunConfig, RunOutputLine, RunStatus};

const MAX_LOG_ENTRIES: usize = 200;
const MAX_HISTORY: usize = 50;
const SPARKLINE_LEN: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Processes,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Shell,
    Files,
    Scheduler,
    RunOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiDensity {
    Auto,
    Compact,
    Balanced,
    Comfortable,
}

impl UiDensity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Compact => "compact",
            Self::Balanced => "balanced",
            Self::Comfortable => "comfortable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesFocus {
    Tree,
    Editor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditorMode {
    Normal,
    Insert,
    Visual,
    Command,
}

pub struct UndoEntry {
    pub lines: Vec<String>,
    pub cursor_line: usize,
    pub cursor_col: usize,
}

/// Pending operator for operator-pending mode (d{motion}, c{motion}, y{motion}).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PendingOperator {
    Delete,
    Change,
    Yank,
    Indent,
    Dedent,
}

/// State for f/F/t/T find char motions.
#[derive(Debug, Clone, Copy)]
pub struct FindCharState {
    pub ch: char,
    pub forward: bool,
    pub till: bool,
}

/// A buffer entry in the multi-buffer list.
#[allow(dead_code)]
pub struct BufferEntry {
    pub path: Option<PathBuf>,
    pub lines: Vec<String>,
    pub cursor_line: usize,
    pub cursor_col: usize,
    pub scroll: usize,
    pub dirty: bool,
    pub undo_stack: Vec<UndoEntry>,
    pub redo_stack: Vec<UndoEntry>,
}

#[derive(Debug, Clone)]
pub struct AppInfo {
    pub name: String,
}

pub struct PipelineStage {
    pub name: String,
    pub offset_ms: u128,
    pub duration_ms: u128,
}

pub struct PipelineState {
    #[allow(dead_code)]
    pub name: String,
    pub stages: Vec<PipelineStage>,
    pub total_ms: u128,
    pub timestamp: Instant,
}

pub struct TensorViz {
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub histogram: Vec<u32>,
    pub samples: Vec<f32>,
    pub timestamp: Instant,
}

/// Per-owner VRAM usage snapshot (merged with quota config each tick).
#[derive(Clone, Debug)]
pub struct OwnerUsage {
    pub owner_id: u32,
    pub label: String,
    pub allocated_bytes: u64,
    pub block_count: u32,
    pub soft_limit: u64, // 0 = unlimited
    pub over_quota: bool,
}

/// Persistent quota configuration for an owner.
#[derive(Clone, Debug)]
pub struct OwnerQuotaConfig {
    pub label: String,
    pub soft_limit: u64, // bytes, 0 = unlimited
}

// ── scheduler view types ──────────────────────────────────────────

/// Which sub-panel is active in the scheduler dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerViewMode {
    Queue,
    Tenants,
    Policy,
}

/// A summary of a queued/running job for the TUI.
#[derive(Debug, Clone)]
pub struct JobSummary {
    pub job_id: u64,
    pub tenant_id: u64,
    pub priority: i32,
    pub state: String,
    pub wait_secs: u64,
}

/// A summary of a tenant for the TUI.
#[derive(Debug, Clone)]
pub struct TenantSummary {
    pub tenant_id: u64,
    pub label: String,
    pub vram_used: u64,
    pub vram_limit: u64,
    pub streams_used: u32,
    pub streams_limit: u32,
    pub active_jobs: u32,
}

/// A recent policy decision for the TUI.
#[derive(Debug, Clone)]
pub struct PolicyDecisionSummary {
    pub elapsed_secs: f64,
    pub tenant_id: u64,
    pub action: String,
    pub allowed: bool,
    pub reason: Option<String>,
}

pub struct TuiState {
    // ── display mode ─────────────────────────────────────────
    pub sysmon_enabled: bool,
    pub ui_mode: UiMode,
    pub detail_mode: bool,
    pub ui_density: UiDensity,

    // ── header ───────────────────────────────────────────────
    pub device_name: String,
    pub running: bool,
    pub uptime_secs: u64,
    pub active_clients: u64,
    pub total_ops: u64,
    pub vram_used: u64,
    pub vram_total: u64,
    pub vram_percent: f32,

    // ── smoothed display values (lerped toward actual — no jitter) ──
    pub display_gpu_util: f32,
    pub display_vram_pct: f32,

    // ── vram pool (TLSF allocator — single source of truth) ─
    pub pool_used: u64,          // tlsf.allocated_bytes — live
    pub pool_total: u64,         // tlsf.total_pool_size — static (pool reservation)
    pub pool_percent: f32,       // tlsf.utilization_percent — computed by allocator
    pub pool_peak: u64,          // tlsf.peak_allocated — high water mark
    pub pool_frag: f32,          // tlsf.fragmentation_ratio — live
    pub pool_blocks: u32,        // tlsf.allocated_blocks — live
    pub pool_free_blocks: u32,   // tlsf.free_blocks — live
    pub pool_largest_free: u64,  // tlsf.largest_free_block — live
    pub pool_healthy: bool,      // tlsf.is_healthy — damped
    pub pool_needs_defrag: bool, // tlsf.needs_defrag — damped
    pool_defrag_hold: u8,        // hysteresis counter — hold state for N ticks
    pool_healthy_hold: u8,       // hysteresis counter
    pub pool_allocs: u64,        // tlsf.total_allocations — lifetime counter
    pub pool_frees: u64,         // tlsf.total_frees — lifetime counter

    // ── runtime (from stable API) ────────────────────────────
    pub gpu_util: f32,       // stats.gpu_utilization — driver-dependent, may be 0
    pub mem_util: f32,       // hardware memory controller utilization (%)
    pub stream_count: usize, // runtime.num_streams() — static (configured)
    pub watchdog: bool,      // stats.watchdog_tripped — damped
    watchdog_hold: u8,       // hysteresis counter
    pub avg_latency_us: f32,
    pub hardware_poll_valid: bool,

    // ── NVML telemetry (valid when hardware_poll_valid) ───
    pub temperature_c: i32,
    pub power_w: f32,
    pub sm_clock_mhz: u32,
    pub mem_clock_mhz: u32,

    // ── processes ────────────────────────────────────────────
    pub processes: Vec<AppInfo>,

    // ── log ──────────────────────────────────────────────────
    pub log: VecDeque<LogEntry>,
    /// Screen rect of the log body (set by layout each frame, for click hit-testing).
    pub log_body_area: Rect,

    // ── animation ────────────────────────────────────────────
    pub tick_count: u64,
    pub heartbeat_phase: f64,
    /// Wave phase for GPU utilization display — advances faster under load.
    pub wave_phase_gpu: f64,
    /// Wave phase for VRAM utilization display — advances faster under pressure.
    pub wave_phase_vram: f64,

    // ── focus ────────────────────────────────────────────────
    pub focus: Panel,
    pub log_scroll: usize,
    /// Vertical section offset for sysmon viewport in short terminals.
    pub sysmon_section_scroll: usize,
    /// Max valid section offset for current frame geometry.
    pub sysmon_section_max_scroll: usize,

    // ── jit ──────────────────────────────────────────────────
    pub script_cache_len: usize,

    // ── sparkline history ────────────────────────────────────
    pub gpu_util_history: VecDeque<f32>,
    pub vram_history: VecDeque<f32>,
    pub ops_rate_history: VecDeque<f64>,
    pub gflops_history: VecDeque<f32>,
    pub ops_rate_hw: f64,
    pub ops_rate_valid: bool,
    pub gflops_hw: f32,
    pub gflops_valid: bool,

    // ── stream activity ─────────────────────────────────────
    pub stream_activity: Vec<VecDeque<bool>>,

    // ── pipeline visualization ──────────────────────────────
    pub last_pipeline: Option<PipelineState>,

    // ── tensor visualization ────────────────────────────────
    pub last_tensor: Option<TensorViz>,

    // ── tachyonfx effects ─────────────────────────────────────
    pub effects: Vec<(&'static str, Effect)>,
    pub last_frame: Instant,
    pub fx_script: Option<FxScriptConfig>,
    pub fx_script_label: Option<String>,
    pub plot3d_socket: PathBuf,
    pub plot3d_child: Option<Child>,
    pub plot3d_scene: String,
    pub plot3d_last_push: Option<Instant>,
    pub plot3d_ipc_dev_ptr: usize,
    pub plot3d_ipc_bytes: usize,
    pub plot3d_ipc_handle_hex: Option<String>,

    // ── shell input ──────────────────────────────────────────
    pub input: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,

    // ── workspace / path guard ────────────────────────────────
    pub current_dir: PathBuf,
    #[allow(dead_code)]
    pub status_message: Option<(String, Instant)>,
    pub pending_confirm: Option<PendingConfirm>,

    // ── file explorer/editor ─────────────────────────────────
    pub files_focus: FilesFocus,
    pub workspace_root: PathBuf,
    pub file_tree: FileTree,
    pub file_list_offset: usize,
    pub file_cursor: usize,
    pub open_file: Option<PathBuf>,
    pub file_lines: Vec<String>,
    pub file_scroll: usize,
    pub file_cursor_line: usize,
    pub file_cursor_col: usize,
    pub file_dirty: bool,
    pub selection_anchor: Option<(usize, usize)>,
    pub selection_head: Option<(usize, usize)>,
    pub selecting: bool,
    pub clipboard: String,
    pub files_tree_area: Rect,
    pub files_editor_area: Rect,

    // ── vim editor state ──────────────────────────────────────
    pub editor_mode: EditorMode,
    pub editor_cmdline: String,
    pub editor_cmd_cursor: usize,
    pub file_hscroll: usize,
    pub preferred_col: usize,
    pub pending_key: Option<char>,
    pub undo_stack: Vec<UndoEntry>,
    pub redo_stack: Vec<UndoEntry>,

    // ── vim-grade editor engine (Plan C) ─────────────────────
    pub keymap: KeyMap,
    pub search: SearchState,
    pub registers: RegisterFile,
    pub marks: MarkStore,
    pub macro_engine: MacroEngine,
    /// Pending operator for operator-pending mode (d/c/y + motion).
    pub pending_operator: Option<PendingOperator>,
    /// Last f/F/t/T find character and direction for ; and , repeat.
    pub last_find_char: Option<FindCharState>,
    /// Multi-buffer: list of open buffers.
    pub buffers: Vec<BufferEntry>,
    /// Index of the active buffer in `buffers`.
    pub active_buffer: usize,
    /// Visible editor height (rows) — set by layout each frame.
    pub editor_visible_rows: usize,
    /// Repeat count prefix (e.g. 5j = move 5 lines down).
    pub count_prefix: Option<usize>,
    /// Pending key sequence being collected for multi-key bindings.
    pub pending_keys: Vec<super::editor::keymap::KeyPress>,

    // ── stability test ──────────────────────────────────────
    pub stability_running: Arc<AtomicBool>,

    // ── allocation event rate ──────────────────────────────
    pub alloc_rate: f64,
    last_event_count: u32,
    pub alloc_rate_history: VecDeque<f32>,

    // ── run state (Plan B) ─────────────────────────────────
    /// UI mode to return to when leaving RunOutput view.
    pub run_output_return_mode: UiMode,
    pub run_status: RunStatus,
    pub run_config: RunConfig,
    pub run_output: VecDeque<RunOutputLine>,
    pub run_output_scroll: usize,
    pub run_start_time: Option<Instant>,
    pub run_elapsed_ms: u64,
    pub run_cancel_flag: Arc<AtomicBool>,
    pub run_target_file: Option<PathBuf>,
    pub last_run_file: Option<PathBuf>,
    pub show_run_output: bool,

    // ── agent state (Plan C) ───────────────────────────────
    pub audit_trail: VecDeque<super::agent::protocol::AuditEntry>,
    pub agent_lock: Option<String>,
    pub agent_checkpoints: HashMap<String, super::agent::checkpoint::CheckpointData>,
    pub agent_modified_files: HashSet<PathBuf>,

    // ── kernel profiling ─────────────────────────────────
    pub kernel_profiles: ProfileRing,

    // ── vram quota enforcement ────────────────────────────
    pub vram_owners: Vec<OwnerUsage>,
    pub vram_quota_configs: BTreeMap<u32, OwnerQuotaConfig>,

    // ── scheduler / control plane (Plan-C) ───────────────
    /// Which scheduler sub-panel is active.
    pub scheduler_view_mode: SchedulerViewMode,
    /// Selected row index in the active scheduler sub-panel.
    pub scheduler_selected_index: usize,
    /// Snapshot of the current job queue for rendering.
    pub scheduler_queue_snapshot: Vec<JobSummary>,
    /// Snapshot of tenant summaries for rendering.
    pub scheduler_tenant_snapshot: Vec<TenantSummary>,
    /// Recent policy decisions for the policy sub-panel.
    pub scheduler_policy_decisions: VecDeque<PolicyDecisionSummary>,
    /// Whether the scheduler queue is paused (mirror of DaemonState).
    pub scheduler_paused: bool,
}

impl TuiState {
    pub fn new(device_name: String, stream_count: usize, workspace_root: PathBuf) -> Self {
        let current_dir = workspace_root.clone();
        let mut this = Self {
            sysmon_enabled: false,
            ui_mode: UiMode::Shell,
            detail_mode: false,
            ui_density: UiDensity::Auto,
            device_name,
            running: true,
            uptime_secs: 0,
            active_clients: 0,
            total_ops: 0,
            vram_used: 0,
            vram_total: 0,
            vram_percent: 0.0,
            display_gpu_util: 0.0,
            display_vram_pct: 0.0,

            pool_used: 0,
            pool_total: 0,
            pool_percent: 0.0,
            pool_peak: 0,
            pool_frag: 0.0,
            pool_blocks: 0,
            pool_free_blocks: 0,
            pool_largest_free: 0,
            pool_healthy: true,
            pool_needs_defrag: false,
            pool_defrag_hold: 0,
            pool_healthy_hold: 0,
            pool_allocs: 0,
            pool_frees: 0,

            gpu_util: 0.0,
            mem_util: 0.0,
            stream_count,
            watchdog: false,
            watchdog_hold: 0,
            avg_latency_us: 0.0,
            hardware_poll_valid: false,
            temperature_c: 0,
            power_w: 0.0,
            sm_clock_mhz: 0,
            mem_clock_mhz: 0,

            processes: Vec::new(),
            log: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            log_body_area: Rect::default(),
            tick_count: 0,
            heartbeat_phase: 0.0,
            wave_phase_gpu: 0.0,
            wave_phase_vram: 0.0,
            focus: Panel::Log,
            log_scroll: 0,
            sysmon_section_scroll: 0,
            sysmon_section_max_scroll: 0,
            script_cache_len: 0,
            gpu_util_history: VecDeque::with_capacity(SPARKLINE_LEN),
            vram_history: VecDeque::with_capacity(SPARKLINE_LEN),
            ops_rate_history: VecDeque::with_capacity(SPARKLINE_LEN),
            gflops_history: VecDeque::with_capacity(SPARKLINE_LEN),
            ops_rate_hw: 0.0,
            ops_rate_valid: false,
            gflops_hw: 0.0,
            gflops_valid: false,
            stream_activity: (0..stream_count)
                .map(|_| VecDeque::with_capacity(SPARKLINE_LEN))
                .collect(),
            last_pipeline: None,
            last_tensor: None,
            effects: Vec::new(),
            last_frame: Instant::now(),
            fx_script: None,
            fx_script_label: None,
            plot3d_socket: PathBuf::from("/tmp/ferrite-renderd.sock"),
            plot3d_child: None,
            plot3d_scene: "wave".to_string(),
            plot3d_last_push: None,
            plot3d_ipc_dev_ptr: 0,
            plot3d_ipc_bytes: 0,
            plot3d_ipc_handle_hex: None,
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            current_dir: current_dir.clone(),
            status_message: None,
            pending_confirm: None,
            files_focus: FilesFocus::Tree,
            workspace_root: workspace_root.clone(),
            file_tree: FileTree::new(workspace_root),
            file_list_offset: 0,
            file_cursor: 0,
            open_file: None,
            file_lines: vec![String::new()],
            file_scroll: 0,
            file_cursor_line: 0,
            file_cursor_col: 0,
            file_dirty: false,
            selection_anchor: None,
            selection_head: None,
            selecting: false,
            clipboard: String::new(),
            files_tree_area: Rect::default(),
            files_editor_area: Rect::default(),
            editor_mode: EditorMode::Normal,
            editor_cmdline: String::new(),
            editor_cmd_cursor: 0,
            file_hscroll: 0,
            preferred_col: 0,
            pending_key: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),

            keymap: KeyMap::new(),
            search: SearchState::new(),
            registers: RegisterFile::new(),
            marks: MarkStore::new(),
            macro_engine: MacroEngine::new(),
            pending_operator: None,
            last_find_char: None,
            buffers: Vec::new(),
            active_buffer: 0,
            editor_visible_rows: 24,
            count_prefix: None,
            pending_keys: Vec::new(),

            stability_running: Arc::new(AtomicBool::new(false)),
            alloc_rate: 0.0,
            last_event_count: 0,
            alloc_rate_history: VecDeque::with_capacity(SPARKLINE_LEN),

            run_output_return_mode: UiMode::Shell,
            run_status: RunStatus::Idle,
            run_config: RunConfig::default(),
            run_output: VecDeque::with_capacity(500),
            run_output_scroll: 0,
            run_start_time: None,
            run_elapsed_ms: 0,
            run_cancel_flag: Arc::new(AtomicBool::new(false)),
            run_target_file: None,
            last_run_file: None,
            show_run_output: false,

            audit_trail: VecDeque::with_capacity(200),
            agent_lock: None,
            agent_checkpoints: HashMap::new(),
            agent_modified_files: HashSet::new(),

            kernel_profiles: ProfileRing::new(),

            vram_owners: Vec::new(),
            vram_quota_configs: BTreeMap::new(),

            scheduler_view_mode: SchedulerViewMode::Queue,
            scheduler_selected_index: 0,
            scheduler_queue_snapshot: Vec::new(),
            scheduler_tenant_snapshot: Vec::new(),
            scheduler_policy_decisions: VecDeque::with_capacity(200),
            scheduler_paused: false,
        };
        if let Ok(cfg) = crate::tui::fxscript::parse_script(crate::tui::fxscript::DEFAULT_SCRIPT) {
            this.fx_script = Some(cfg);
            this.fx_script_label = Some("builtin:hq".to_string());
        }
        this.reload_file_entries();
        this
    }

}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or_else(|| s.len())
}
