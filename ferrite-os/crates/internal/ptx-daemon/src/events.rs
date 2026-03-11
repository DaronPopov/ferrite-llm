use std::fmt;
use std::time::Instant;

use crossterm::event::{KeyEvent, MouseEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogCategory {
    Sys,
    Jit,
    Run,
    App,
    Err,
}

impl fmt::Display for LogCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogCategory::Sys => write!(f, "sys"),
            LogCategory::Jit => write!(f, "jit"),
            LogCategory::Run => write!(f, "run"),
            LogCategory::App => write!(f, "app"),
            LogCategory::Err => write!(f, "err"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: Instant,
    pub category: LogCategory,
    pub message: String,
    /// When set, this entry is clickable — clicking it writes this command to the input.
    pub action: Option<String>,
}

impl LogEntry {
    pub fn new(category: LogCategory, message: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            category,
            message: message.into(),
            action: None,
        }
    }

    /// Attach a click action — the command that gets written to the input when clicked.
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct DaemonDiagnostic {
    pub component: &'static str,
    pub status: &'static str,
    pub code: &'static str,
    pub summary: String,
    pub remediation: String,
}

impl DaemonDiagnostic {
    #[allow(dead_code)]
    pub fn new(
        component: &'static str,
        status: &'static str,
        code: &'static str,
        summary: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            component,
            status,
            code,
            summary: summary.into(),
            remediation: remediation.into(),
        }
    }
}

pub enum DaemonEvent {
    Tick,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Paste(String),
    ClientHandled {
        command: String,
        success: bool,
    },
    AppEvent {
        app_name: String,
        message: String,
    },
    Log(LogEntry),
    /// A log entry with an optional click-action attached (for clickable TUI entries).
    LogAction(LogEntry),
    TensorResult {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
    PipelineResult {
        name: String,
        compile_ms: u128,
        exec_ms: u128,
    },
    #[allow(dead_code)]
    Diagnostic(DaemonDiagnostic),
    #[allow(dead_code)]
    Shutdown,

    // ── run events (Plan B) ────────────────────────────────
    RunStarted {
        #[allow(dead_code)]
        file: String,
    },
    RunOutput {
        line: String,
        is_error: bool,
    },
    RunFinished {
        success: bool,
        elapsed_ms: u64,
    },
    RunTimeout {
        elapsed_ms: u64,
    },

    // ── profiling events ─────────────────────────────────
    KernelProfiled {
        compile_ms: u64,
        exec_ms: u64,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        node_count: usize,
        total_elements: usize,
        success: bool,
        source_tag: String,
        vram_before: u64,
        vram_after: u64,
    },

    // ── agent events (Plan C) ──────────────────────────────
    #[allow(dead_code)]
    AgentCommand(crate::tui::agent::protocol::AgentCommand),
    AgentCommandSync(
        crate::tui::agent::protocol::AgentCommand,
        std::sync::mpsc::Sender<crate::tui::agent::protocol::AgentResponse>,
    ),

    // ── scheduler / control plane events ──────────────────
    /// A scheduler command was executed (for TUI log updates).
    #[allow(dead_code)]
    SchedulerCommandExecuted {
        command: String,
        success: bool,
    },
    /// A policy decision was made (for TUI policy view).
    #[allow(dead_code)]
    PolicyDecisionMade {
        tenant_id: u64,
        action: String,
        allowed: bool,
        reason: Option<String>,
    },
    /// The scheduler queue pause state changed.
    #[allow(dead_code)]
    SchedulerPauseChanged {
        paused: bool,
    },

    // ── durable job events (Plan-B) ──────────────────────────
    /// A new durable job was submitted.
    #[allow(dead_code)]
    JobSubmitted {
        job_id: u64,
        name: String,
    },
    /// A durable job process was started.
    #[allow(dead_code)]
    JobStarted {
        job_id: u64,
        name: String,
        pid: u32,
    },
    /// A durable job was stopped (cancelled) by the operator.
    #[allow(dead_code)]
    JobStopped {
        job_id: u64,
        name: String,
        reason: String,
    },
    /// A durable job is being restarted after failure.
    #[allow(dead_code)]
    JobRestarted {
        job_id: u64,
        name: String,
        attempt: u32,
        delay_ms: u64,
    },
    /// A durable job failed.
    #[allow(dead_code)]
    JobFailed {
        job_id: u64,
        name: String,
        reason: String,
    },
    /// A durable job was recovered from persisted state on boot.
    #[allow(dead_code)]
    JobRecovered {
        job_id: u64,
        name: String,
        state: String,
    },
}
