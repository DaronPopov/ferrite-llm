use std::env;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use ptx_runner::{
    discover_run_entries, parse_run_entry_request, parse_run_file_request,
    prepare_run_entry_command, prepare_run_file_command,
};
use tracing::{info, warn};

use crate::config::DaemonConfig;
use crate::event_stream::SchedulerEvent;
use crate::policy::decision::{PolicyContext, PolicyDecision};
use crate::state::{DaemonState, ManagedApp, MANAGED_APPS};

/// Evaluate policy for a command.  Returns `Some(json)` if denied, `None` if allowed.
/// Records the decision in the audit log and emits an event.
///
/// Denial responses use the standardized [`DenialPayload`] format so that
/// every policy rejection—regardless of command path—is machine-readable.
fn evaluate_policy(state: &DaemonState, action: &str, resource: &str) -> Option<String> {
    let ctx = PolicyContext::new(None, action, resource);
    let decision = {
        let mut engine = state.policy_engine.lock();
        engine.evaluate(&ctx)
    };

    // Emit to event stream
    {
        let (dec_str, reason, remediation) = match &decision {
            PolicyDecision::Allow => ("Allow".to_string(), None, None),
            PolicyDecision::Deny { reason, remediation } => (
                "Deny".to_string(),
                Some(reason.code().to_string()),
                Some(remediation.clone()),
            ),
        };
        let mut es = state.event_stream.lock();
        es.emit(SchedulerEvent::PolicyDecision {
            tenant_id: ctx.tenant_id.unwrap_or(0),
            action: action.to_string(),
            resource: resource.to_string(),
            decision: dec_str,
            reason,
            remediation,
        });
    }

    decision.to_denial_json(action, resource)
}

fn to_json_line<T: serde::Serialize>(value: &T) -> io::Result<String> {
    serde_json::to_string(value)
        .map(|json| format!("{json}\n"))
        .map_err(|e| io::Error::other(format!("serialize response: {e}")))
}

#[derive(Debug)]
struct RunRunnerResult {
    stdout: String,
    stderr: String,
    success: bool,
    exit_code: Option<i32>,
    elapsed_ms: u64,
}

/// Per-command overrides for child process resource limits.
#[derive(Debug, Clone, Default)]
struct RunOverrides {
    /// Override child stream count (PTX_MAX_STREAMS).
    streams: Option<u32>,
    /// Override child pool fraction (PTX_POOL_FRACTION).
    pool: Option<f32>,
}

#[derive(Debug)]
struct ParsedRunFileArgs {
    path: String,
    entry: Option<String>,
    passthrough: Vec<String>,
    overrides: RunOverrides,
}

#[derive(Debug)]
struct ParsedRunEntryArgs {
    entry_id: String,
    passthrough: Vec<String>,
    overrides: RunOverrides,
}

#[derive(Debug)]
enum StreamKind {
    Stdout,
    Stderr,
}

#[derive(Debug)]
struct StreamLine {
    kind: StreamKind,
    line: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunOrchestratorProfile {
    Safe,
    Balanced,
    Stress,
}

impl RunOrchestratorProfile {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Balanced => "balanced",
            Self::Stress => "stress",
        }
    }

    pub(crate) fn max_streams(self) -> u32 {
        match self {
            Self::Safe => 4,
            Self::Balanced => 8,
            Self::Stress => 16,
        }
    }

    pub(crate) fn pool_fraction(self) -> f32 {
        match self {
            Self::Safe => 0.30,
            Self::Balanced => 0.50,
            Self::Stress => 0.70,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunExecutionMode {
    /// External process with bounded runtime (non-single-pool-safe).
    ExternalProcess,
    /// Denied: target requires in-daemon execution not yet available.
    DeniedStrictMode,
}

fn classify_execution_mode(target: &str, strict: bool) -> RunExecutionMode {
    if !strict {
        return RunExecutionMode::ExternalProcess;
    }
    let t = target.to_ascii_lowercase();
    let heavy = t.contains("bench")
        || t.contains("stress")
        || t.contains("jitter")
        || t.contains("latency")
        || t.contains("training")
        || t.contains("inference");
    if heavy {
        RunExecutionMode::DeniedStrictMode
    } else {
        RunExecutionMode::ExternalProcess
    }
}

pub(crate) fn profile_env_pairs(profile: RunOrchestratorProfile) -> [(&'static str, &'static str); 2] {
    match profile {
        RunOrchestratorProfile::Safe => [("PTX_MAX_STREAMS", "4"), ("PTX_POOL_FRACTION", "0.30")],
        RunOrchestratorProfile::Balanced => [("PTX_MAX_STREAMS", "8"), ("PTX_POOL_FRACTION", "0.50")],
        RunOrchestratorProfile::Stress => [("PTX_MAX_STREAMS", "16"), ("PTX_POOL_FRACTION", "0.70")],
    }
}

fn parse_profile_override(value: &str) -> Option<RunOrchestratorProfile> {
    match value.trim().to_ascii_lowercase().as_str() {
        "safe" => Some(RunOrchestratorProfile::Safe),
        "balanced" | "default" => Some(RunOrchestratorProfile::Balanced),
        "stress" => Some(RunOrchestratorProfile::Stress),
        _ => None,
    }
}

fn profile_chain(initial: RunOrchestratorProfile) -> Vec<RunOrchestratorProfile> {
    match initial {
        RunOrchestratorProfile::Safe => vec![RunOrchestratorProfile::Safe],
        RunOrchestratorProfile::Balanced => vec![
            RunOrchestratorProfile::Balanced,
            RunOrchestratorProfile::Safe,
        ],
        RunOrchestratorProfile::Stress => vec![
            RunOrchestratorProfile::Stress,
            RunOrchestratorProfile::Balanced,
            RunOrchestratorProfile::Safe,
        ],
    }
}

pub(crate) fn is_oom_like_failure(stdout: &str, stderr: &str, exit_code: Option<i32>) -> bool {
    if matches!(exit_code, Some(137 | 139)) {
        return true;
    }
    let mut haystack = String::with_capacity(stdout.len() + stderr.len() + 1);
    haystack.push_str(stdout);
    haystack.push('\n');
    haystack.push_str(stderr);
    let haystack = haystack.to_ascii_lowercase();

    let needles = [
        "out of memory",
        "cudaerrormemoryallocation",
        "cuda_error_out_of_memory",
        "allocation failed",
        "failed to allocate",
        "failed: out of memory",
        "cuda stream create failed",
        "cudastreamcreatewithpriority failed",
    ];

    needles.iter().any(|n| haystack.contains(n))
}

pub(crate) fn compute_orchestrator_profiles(
    tlsf: &ptx_sys::TLSFPoolStats,
    hot: &ptx_sys::GPUHotStats,
    target: &str,
) -> Result<Vec<RunOrchestratorProfile>, String> {
    const MIB: usize = 1024 * 1024;

    let pool_free = tlsf.free_bytes;
    let vram_free = hot.vram_free as usize;
    let util = tlsf.utilization_percent as f64;

    if util >= 98.0 || pool_free < 96 * MIB || vram_free < 128 * MIB {
        return Err(format!(
            "run admission denied: daemon under memory pressure (pool util {:.1}%, pool free {:.1} MiB, vram free {:.1} MiB)",
            util,
            pool_free as f64 / (MIB as f64),
            vram_free as f64 / (MIB as f64),
        ));
    }

    if let Ok(raw) = env::var("FERRITE_RUN_PROFILE") {
        if let Some(profile) = parse_profile_override(&raw) {
            return Ok(profile_chain(profile));
        }
    }

    let t = target.to_ascii_lowercase();
    let stress_hint = t.contains("bench")
        || t.contains("stress")
        || t.contains("jitter")
        || t.contains("latency");

    let initial = if util >= 85.0 || pool_free < 512 * MIB || vram_free < 768 * MIB {
        RunOrchestratorProfile::Safe
    } else if stress_hint && util < 65.0 && pool_free > 1536 * MIB && vram_free > 2048 * MIB {
        RunOrchestratorProfile::Stress
    } else {
        RunOrchestratorProfile::Balanced
    };

    Ok(profile_chain(initial))
}

fn emit_scheduler_event(state: &DaemonState, event: SchedulerEvent) {
    let mut es = state.event_stream.lock();
    es.emit(event);
}

pub(crate) fn find_workspace_root(start: PathBuf) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(contents) = std::fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return Some(cur);
                }
            }
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

pub(crate) fn resolve_workspace_root() -> io::Result<PathBuf> {
    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = find_workspace_root(cwd) {
            return Ok(root);
        }
    }

    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = find_workspace_root(manifest_root) {
        return Ok(root);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Unable to locate workspace root for ptx-runner",
    ))
}

fn split_passthrough<'a>(args: &'a [&'a str]) -> (&'a [&'a str], &'a [&'a str]) {
    if let Some(idx) = args.iter().position(|arg| *arg == "--") {
        (&args[..idx], &args[idx + 1..])
    } else {
        (args, &[])
    }
}

/// Parse `--streams=N` and `--pool=F` from a flag list, returning overrides
/// and the remaining flags that weren't consumed.
fn parse_run_overrides(flags: &[&str]) -> (RunOverrides, Vec<String>) {
    let mut overrides = RunOverrides::default();
    let mut remaining = Vec::new();

    for &flag in flags {
        if let Some(val) = flag.strip_prefix("--streams=") {
            if let Ok(n) = val.parse::<u32>() {
                overrides.streams = Some(n);
            } else {
                remaining.push(flag.to_string());
            }
        } else if let Some(val) = flag.strip_prefix("--pool=") {
            if let Ok(f) = val.parse::<f32>() {
                overrides.pool = Some(f);
            } else {
                remaining.push(flag.to_string());
            }
        } else {
            remaining.push(flag.to_string());
        }
    }
    (overrides, remaining)
}

fn parse_run_file_args(args: &[&str]) -> Result<ParsedRunFileArgs, String> {
    if args.is_empty() {
        return Err("usage: run-file <path> [--entry <name>] [--streams=N] [--pool=F] [-- <args...>]".to_string());
    }
    let (head, tail) = split_passthrough(args);
    let path = head[0].to_string();
    let mut entry = None;
    let mut other_flags: Vec<&str> = Vec::new();
    let mut idx = 1usize;
    while idx < head.len() {
        match head[idx] {
            "--entry" => {
                let Some(value) = head.get(idx + 1) else {
                    return Err("missing value for --entry".to_string());
                };
                entry = Some((*value).to_string());
                idx += 2;
            }
            flag => {
                other_flags.push(flag);
                idx += 1;
            }
        }
    }

    let (overrides, unknown) = parse_run_overrides(&other_flags);
    if let Some(unk) = unknown.first() {
        return Err(format!(
            "unknown flag '{unk}' (expected --entry, --streams=N, --pool=F, or --)"
        ));
    }

    Ok(ParsedRunFileArgs {
        path,
        entry,
        passthrough: tail.iter().map(|arg| (*arg).to_string()).collect(),
        overrides,
    })
}

fn parse_run_entry_args(args: &[&str]) -> Result<ParsedRunEntryArgs, String> {
    if args.is_empty() {
        return Err("usage: run-entry <entry-id> [--streams=N] [--pool=F] [-- <args...>]".to_string());
    }
    let (head, tail) = split_passthrough(args);
    if head.is_empty() {
        return Err("usage: run-entry <entry-id> [--streams=N] [--pool=F] [-- <args...>]".to_string());
    }
    let entry_id = head[0].to_string();
    let (overrides, unknown) = parse_run_overrides(&head[1..]);
    if let Some(unk) = unknown.first() {
        return Err(format!(
            "unknown flag '{unk}' (expected --streams=N, --pool=F, or --)"
        ));
    }
    Ok(ParsedRunEntryArgs {
        entry_id,
        passthrough: tail.iter().map(|arg| (*arg).to_string()).collect(),
        overrides,
    })
}

fn run_runner_once_with_events(
    state: &DaemonState,
    request_id: u64,
    mode: &str,
    target: &str,
    entry: Option<&str>,
    runner_args: &[String],
    overrides: &RunOverrides,
    profile: RunOrchestratorProfile,
    attempt: usize,
    total_attempts: usize,
) -> io::Result<RunRunnerResult> {
    let workspace_root = resolve_workspace_root()?;
    let repo_root = workspace_root
        .parent()
        .unwrap_or(workspace_root.as_path())
        .to_path_buf();

    let mut prepared = match mode {
        "run-file" => {
            let mut req =
                parse_run_file_request(runner_args.get(1..).unwrap_or(&[])).map_err(io::Error::other)?;
            if req.path.is_relative() {
                req.path = workspace_root.join(req.path);
            }
            prepare_run_file_command(&workspace_root, &repo_root, req).map_err(io::Error::other)?
        }
        "run-entry" => {
            let req =
                parse_run_entry_request(runner_args.get(1..).unwrap_or(&[])).map_err(io::Error::other)?;
            prepare_run_entry_command(&workspace_root, &repo_root, req).map_err(io::Error::other)?
        }
        _ => {
            return Err(io::Error::other(format!(
                "unsupported runner mode '{mode}' for in-process execution"
            )))
        }
    };

    let cmd = prepared.command_mut();
    cmd.env("FERRITE_DAEMON_SOCKET", &state.config.socket_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Refresh exported context handle so subprocesses can share the GPU context.
    state.runtime.export_context();

    // Signal daemon-client mode: the child creates its own bounded runtime
    // instead of importing the daemon's host-side pointer (which is invalid
    // across process boundaries).  PTX_MAX_STREAMS and PTX_POOL_FRACTION
    // from the profile will be enforced by apply_env_overrides() in the
    // C runtime init path.
    cmd.env("PTX_DAEMON_CLIENT", "1");

    // NOTE: We intentionally do NOT propagate PTX_SINGLE_POOL_STRICT to
    // child processes.  The daemon already enforces strict mode at the Rust
    // level (denying heavy targets before spawning).  Light targets that
    // pass the daemon's classifier are allowed to create bounded pools —
    // that's the expected transitional behavior.  Setting the C guard here
    // would block ALL children from initializing any pool, breaking the
    // light-target path.

    for key in ["PTX_CONTEXT_PTR", "PTX_STREAM_PTR"] {
        if let Ok(value) = env::var(key) {
            cmd.env(key, value);
        }
    }

    for (key, value) in profile_env_pairs(profile) {
        cmd.env(key, value);
    }

    // Per-command overrides (--streams=N, --pool=F) take highest precedence.
    // Daemon-level env overrides are second priority.  Profile defaults are lowest.
    if let Some(streams) = overrides.streams {
        cmd.env("PTX_MAX_STREAMS", streams.to_string());
    } else if let Ok(v) = env::var("FERRITE_CHILD_MAX_STREAMS") {
        cmd.env("PTX_MAX_STREAMS", &v);
    }
    if let Some(pool) = overrides.pool {
        cmd.env("PTX_POOL_FRACTION", pool.to_string());
    } else if let Ok(v) = env::var("FERRITE_CHILD_POOL_FRACTION") {
        cmd.env("PTX_POOL_FRACTION", &v);
    }

    // Orchestrator profiles keep subprocess footprint bounded via env overrides.
    emit_scheduler_event(
        state,
        SchedulerEvent::RunStderrChunk {
            request_id,
            chunk: format!(
                "[orchestrator] attempt {}/{} profile={} streams={} pool_fraction={:.2}",
                attempt,
                total_attempts,
                profile.as_str(),
                profile.max_streams(),
                profile.pool_fraction()
            ),
        },
    );

    let cmdline = format!(
        "{:?} [attempt={}/{} profile={}]",
        cmd,
        attempt,
        total_attempts,
        profile.as_str()
    );
    emit_scheduler_event(
        state,
        SchedulerEvent::RunRequestAccepted {
            request_id,
            mode: mode.to_string(),
            target: target.to_string(),
            entry: entry.map(|value| value.to_string()),
            args: runner_args.to_vec(),
        },
    );
    emit_scheduler_event(
        state,
        SchedulerEvent::RunBuildStarted {
            request_id,
            command: cmdline,
        },
    );

    let run_start = Instant::now();
    let mut child = cmd.spawn()?;
    emit_scheduler_event(state, SchedulerEvent::RunStarted { request_id });

    let Some(stdout_reader) = child.stdout.take() else {
        return Err(io::Error::other("failed to capture runner stdout"));
    };
    let Some(stderr_reader) = child.stderr.take() else {
        return Err(io::Error::other("failed to capture runner stderr"));
    };

    let (tx, rx) = mpsc::channel::<StreamLine>();

    {
        let tx_out = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout_reader);
            for line in reader.lines().map_while(Result::ok) {
                if tx_out
                    .send(StreamLine {
                        kind: StreamKind::Stdout,
                        line,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    {
        let tx_err = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr_reader);
            for line in reader.lines().map_while(Result::ok) {
                if tx_err
                    .send(StreamLine {
                        kind: StreamKind::Stderr,
                        line,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(tx);

    let mut stdout = String::new();
    let mut stderr = String::new();
    for item in rx {
        match item.kind {
            StreamKind::Stdout => {
                if !stdout.is_empty() {
                    stdout.push('\n');
                }
                stdout.push_str(&item.line);
                emit_scheduler_event(
                    state,
                    SchedulerEvent::RunStdoutChunk {
                        request_id,
                        chunk: item.line,
                    },
                );
            }
            StreamKind::Stderr => {
                if !stderr.is_empty() {
                    stderr.push('\n');
                }
                stderr.push_str(&item.line);
                emit_scheduler_event(
                    state,
                    SchedulerEvent::RunStderrChunk {
                        request_id,
                        chunk: item.line,
                    },
                );
            }
        }
    }

    let status = child.wait()?;
    let elapsed_ms = run_start.elapsed().as_millis() as u64;
    let success = status.success();
    let exit_code = status.code();

    emit_scheduler_event(
        state,
        SchedulerEvent::RunBuildFinished {
            request_id,
            success,
            elapsed_ms,
        },
    );
    emit_scheduler_event(
        state,
        SchedulerEvent::RunFinished {
            request_id,
            success,
            exit_code,
            elapsed_ms,
        },
    );

    Ok(RunRunnerResult {
        stdout,
        stderr,
        success,
        exit_code,
        elapsed_ms,
    })
}

fn run_runner_command_with_events(
    state: &DaemonState,
    request_id: u64,
    mode: &str,
    target: &str,
    entry: Option<&str>,
    runner_args: &[String],
    overrides: &RunOverrides,
) -> io::Result<RunRunnerResult> {
    // Single-pool strict mode: classify and potentially deny heavy targets
    let exec_mode = classify_execution_mode(target, state.config.single_pool_strict);
    emit_scheduler_event(
        state,
        SchedulerEvent::RunExecutionModeSelected {
            request_id,
            mode: match exec_mode {
                RunExecutionMode::ExternalProcess => "external-process".to_string(),
                RunExecutionMode::DeniedStrictMode => "denied-strict-mode".to_string(),
            },
            strict: state.config.single_pool_strict,
            target: target.to_string(),
        },
    );

    if exec_mode == RunExecutionMode::DeniedStrictMode {
        let reason = format!(
            "single-pool strict mode: target '{}' classified as heavy GPU workload; \
             in-daemon execution not yet available",
            target
        );
        emit_scheduler_event(
            state,
            SchedulerEvent::SinglePoolDenial {
                request_id,
                target: target.to_string(),
                reason: reason.clone(),
            },
        );
        return Ok(RunRunnerResult {
            stdout: String::new(),
            stderr: reason.clone(),
            success: false,
            exit_code: None,
            elapsed_ms: 0,
        });
    }

    let profiles = if mode == "run-list" {
        vec![RunOrchestratorProfile::Safe]
    } else {
        let tlsf = state.runtime.tlsf_stats();
        let hot = state.runtime.stats();
        compute_orchestrator_profiles(&tlsf, &hot, target).map_err(io::Error::other)?
    };

    let total_attempts = profiles.len().max(1);
    let mut aggregate_stdout = String::new();
    let mut aggregate_stderr = String::new();
    let mut total_elapsed_ms = 0u64;
    let mut last_result: Option<RunRunnerResult> = None;

    for (idx, profile) in profiles.into_iter().enumerate() {
        let attempt = idx + 1;
        let result = run_runner_once_with_events(
            state,
            request_id,
            mode,
            target,
            entry,
            runner_args,
            overrides,
            profile,
            attempt,
            total_attempts,
        )?;
        total_elapsed_ms += result.elapsed_ms;

        if !result.stdout.is_empty() {
            if !aggregate_stdout.is_empty() {
                aggregate_stdout.push('\n');
            }
            aggregate_stdout.push_str(&format!(
                "## attempt {}/{} [{}]\n{}",
                attempt,
                total_attempts,
                profile.as_str(),
                result.stdout
            ));
        }
        if !result.stderr.is_empty() {
            if !aggregate_stderr.is_empty() {
                aggregate_stderr.push('\n');
            }
            aggregate_stderr.push_str(&format!(
                "## attempt {}/{} [{}]\n{}",
                attempt,
                total_attempts,
                profile.as_str(),
                result.stderr
            ));
        }

        let oom_like = is_oom_like_failure(&result.stdout, &result.stderr, result.exit_code);
        let success = result.success;
        last_result = Some(result);

        if success {
            break;
        }

        if oom_like && attempt < total_attempts {
            emit_scheduler_event(
                state,
                SchedulerEvent::RunStderrChunk {
                    request_id,
                    chunk: "[orchestrator] OOM-like failure detected; retrying with safer profile".to_string(),
                },
            );
            continue;
        }
        break;
    }

    let mut final_result =
        last_result.ok_or_else(|| io::Error::other("orchestrator did not execute any run attempts"))?;
    final_result.elapsed_ms = total_elapsed_ms.max(final_result.elapsed_ms);
    if !aggregate_stdout.is_empty() {
        final_result.stdout = aggregate_stdout;
    }
    if !aggregate_stderr.is_empty() {
        final_result.stderr = aggregate_stderr;
    }
    Ok(final_result)
}

pub fn cleanup_exited_apps(state: &DaemonState) {
    let mut apps = state.apps.lock();
    apps.retain(|_, app| match app.child.try_wait() {
        Ok(Some(status)) => {
            info!(
                app_id = app.id,
                app = %app.name,
                code = ?status.code(),
                "Managed app exited"
            );
            false
        }
        Ok(None) => true,
        Err(e) => {
            warn!(app_id = app.id, app = %app.name, error = %e, "Failed polling app state");
            false
        }
    });
}

pub fn resolve_app_target(config: &DaemonConfig, app: &str) -> PathBuf {
    if let Some(dir) = &config.apps_bin_dir {
        return Path::new(dir).join(app);
    }
    PathBuf::from(app)
}

pub fn handle_ping() -> io::Result<String> {
    Ok("{\"ok\":true,\"message\":\"pong\"}\n".to_string())
}

pub fn handle_status(state: &DaemonState) -> io::Result<String> {
    let tlsf = state.runtime.tlsf_stats();
    let response = serde_json::json!({
        "ok": true,
        "pool_total": tlsf.total_pool_size,
        "allocated": tlsf.allocated_bytes,
        "free": tlsf.free_bytes,
        "utilization": tlsf.utilization_percent,
        "fragmentation": tlsf.fragmentation_ratio,
        "healthy": tlsf.is_healthy,
        "single_pool_strict": state.config.single_pool_strict,
    });
    to_json_line(&response)
}

pub fn handle_stats(state: &DaemonState) -> io::Result<String> {
    let stats = state.runtime.stats();
    let response = serde_json::json!({
        "ok": true,
        "vram_allocated": stats.vram_allocated,
        "vram_used": stats.vram_used,
        "vram_free": stats.vram_free,
        "gpu_utilization": stats.gpu_utilization,
        "active_streams": stats.active_streams,
        "registered_kernels": stats.registered_kernels,
        "total_ops": stats.total_ops,
    });
    to_json_line(&response)
}

pub fn handle_metrics(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let stats = state.runtime.stats();
    let tlsf = state.runtime.tlsf_stats();
    let uptime = state.start_time.elapsed().as_secs();
    let active = state.active_clients.load(Ordering::Relaxed);
    let total = state.total_requests.load(Ordering::Relaxed);
    let failed = state.failed_requests.load(Ordering::Relaxed);
    let running_apps = state.apps.lock().len();

    let response = serde_json::json!({
        "ok": true,
        "uptime_secs": uptime,
        "active_clients": active,
        "running_apps": running_apps,
        "total_requests": total,
        "failed_requests": failed,
        "vram_allocated": stats.vram_allocated,
        "vram_used": stats.vram_used,
        "vram_free": stats.vram_free,
        "gpu_util": stats.gpu_utilization,
        "active_streams": stats.active_streams,
        "total_ops": stats.total_ops,
        "pool_total": tlsf.total_pool_size,
        "pool_allocated": tlsf.allocated_bytes,
        "pool_free": tlsf.free_bytes,
        "pool_util": tlsf.utilization_percent,
        "fragmentation": tlsf.fragmentation_ratio,
        "single_pool_strict": state.config.single_pool_strict,
    });
    to_json_line(&response)
}

pub fn handle_snapshot(state: &DaemonState) -> io::Result<String> {
    let snap = state.runtime.system_snapshot();
    let response = serde_json::json!({
        "ok": true,
        "total_ops": snap.total_ops,
        "active_processes": snap.active_processes,
        "active_tasks": snap.active_tasks,
        "vram_used": snap.total_vram_used,
        "watchdog_alert": snap.watchdog_alert,
        "queue_head": snap.queue_head,
        "queue_tail": snap.queue_tail,
    });
    to_json_line(&response)
}

pub fn handle_health(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let tlsf = state.runtime.tlsf_stats();
    let uptime = state.start_time.elapsed().as_secs();
    let active = state.active_clients.load(Ordering::Relaxed);
    let running_apps = state.apps.lock().len();

    let healthy = tlsf.is_healthy
        && active < state.config.max_clients as u64
        && tlsf.utilization_percent < 95.0;

    let response = serde_json::json!({
        "ok": true,
        "healthy": healthy,
        "uptime_secs": uptime,
        "pool_healthy": tlsf.is_healthy,
        "active_clients": active,
        "running_apps": running_apps,
        "max_clients": state.config.max_clients,
    });
    to_json_line(&response)
}

pub fn handle_keepalive(state: &DaemonState) -> io::Result<String> {
    state.runtime.keepalive();
    Ok("{\"ok\":true,\"message\":\"keepalive sent\"}\n".to_string())
}

pub fn handle_apps(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let apps = state.apps.lock();
    let running: Vec<serde_json::Value> = apps
        .values()
        .map(|app| {
            serde_json::json!({
                "id": app.id,
                "name": app.name,
                "pid": app.child.id(),
                "uptime_secs": app.started_at.elapsed().as_secs(),
                "args": app.args,
            })
        })
        .collect();
    let response = serde_json::json!({
        "ok": true,
        "running": running,
        "managed_apps": MANAGED_APPS,
        "apps_bin_dir": state.config.apps_bin_dir,
    });
    to_json_line(&response)
}

pub fn handle_app_start(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: app-start <app> [args...]\"}\n".to_string());
    }
    let app = args[0];

    // Policy enforcement: deny if not allowed
    if let Some(denial) = evaluate_policy(state, "app-start", app) {
        return Ok(denial);
    }

    if !MANAGED_APPS.iter().any(|x| *x == app) {
        let response = serde_json::json!({
            "error": "app not allowed",
            "app": app,
            "allowed": MANAGED_APPS,
        });
        return to_json_line(&response);
    }

    let app_args: Vec<String> = args.iter().skip(1).map(|s| s.to_string()).collect();
    let target = resolve_app_target(&state.config, app);
    let mut cmd = Command::new(&target);
    cmd.args(&app_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            let response = serde_json::json!({
                "error": "spawn failed",
                "app": app,
                "target": target,
                "message": e.to_string(),
            });
            return to_json_line(&response);
        }
    };

    let pid = child.id();
    let app_id = state.next_app_id.fetch_add(1, Ordering::Relaxed);
    let managed = ManagedApp {
        id: app_id,
        name: app.to_string(),
        args: app_args,
        started_at: Instant::now(),
        child,
    };
    state.apps.lock().insert(app_id, managed);
    let response = serde_json::json!({
        "ok": true,
        "message": "app started",
        "id": app_id,
        "name": app,
        "pid": pid,
    });
    to_json_line(&response)
}

pub fn handle_app_stop(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: app-stop <id|name>\"}\n".to_string());
    }
    let selector = args[0];

    // Policy enforcement: deny if not allowed
    if let Some(denial) = evaluate_policy(state, "app-stop", selector) {
        return Ok(denial);
    }

    let mut apps = state.apps.lock();
    let target_id = if let Ok(id) = selector.parse::<u64>() {
        Some(id)
    } else {
        apps.iter()
            .find(|(_, app)| app.name == selector)
            .map(|(id, _)| *id)
    };

    let Some(id) = target_id else {
        let response = serde_json::json!({
            "error": "app not found",
            "selector": selector,
        });
        return to_json_line(&response);
    };

    let Some(mut app) = apps.remove(&id) else {
        let response = serde_json::json!({
            "error": "app registry inconsistent",
            "selector": selector,
            "id": id,
        });
        return to_json_line(&response);
    };
    let name = app.name.clone();
    let pid = app.child.id();

    let mut stop_err: Option<String> = None;
    if let Err(e) = app.child.kill() {
        stop_err = Some(e.to_string());
    } else if let Err(e) = app.child.wait() {
        stop_err = Some(e.to_string());
    }

    let response = if let Some(err) = stop_err {
        serde_json::json!({
            "ok": false,
            "message": "failed to stop app cleanly",
            "id": id,
            "name": name,
            "pid": pid,
            "error": err,
        })
    } else {
        serde_json::json!({
            "ok": true,
            "message": "app stopped",
            "id": id,
            "name": name,
            "pid": pid,
        })
    };
    to_json_line(&response)
}

pub fn handle_run_file(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    let parsed = match parse_run_file_args(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            let response = serde_json::json!({ "error": message });
            return to_json_line(&response);
        }
    };

    if let Some(denial) = evaluate_policy(state, "run-file", &parsed.path) {
        return Ok(denial);
    }

    let request_id = state.next_run_id.fetch_add(1, Ordering::Relaxed);
    let mut runner_args = vec!["run-file".to_string(), parsed.path.clone()];
    if let Some(entry) = &parsed.entry {
        runner_args.push("--entry".to_string());
        runner_args.push(entry.clone());
    }
    if !parsed.passthrough.is_empty() {
        runner_args.push("--".to_string());
        runner_args.extend(parsed.passthrough.clone());
    }

    match run_runner_command_with_events(
        state,
        request_id,
        "run-file",
        &parsed.path,
        parsed.entry.as_deref(),
        &runner_args,
        &parsed.overrides,
    ) {
        Ok(result) => {
            let response = serde_json::json!({
                "ok": result.success,
                "request_id": request_id,
                "command": "run-file",
                "path": parsed.path,
                "entry": parsed.entry,
                "args": parsed.passthrough,
                "exit_code": result.exit_code,
                "elapsed_ms": result.elapsed_ms,
                "stdout": result.stdout,
                "stderr": result.stderr,
            });
            to_json_line(&response)
        }
        Err(error) => {
            emit_scheduler_event(
                state,
                SchedulerEvent::RunError {
                    request_id,
                    message: error.to_string(),
                },
            );
            let response = serde_json::json!({
                "ok": false,
                "request_id": request_id,
                "command": "run-file",
                "path": parsed.path,
                "entry": parsed.entry,
                "args": parsed.passthrough,
                "error": error.to_string(),
            });
            to_json_line(&response)
        }
    }
}

pub fn handle_run_entry(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    let parsed = match parse_run_entry_args(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            let response = serde_json::json!({ "error": message });
            return to_json_line(&response);
        }
    };

    if let Some(denial) = evaluate_policy(state, "run-entry", &parsed.entry_id) {
        return Ok(denial);
    }

    let request_id = state.next_run_id.fetch_add(1, Ordering::Relaxed);
    let mut runner_args = vec!["run-entry".to_string(), parsed.entry_id.clone()];
    if !parsed.passthrough.is_empty() {
        runner_args.push("--".to_string());
        runner_args.extend(parsed.passthrough.clone());
    }

    match run_runner_command_with_events(
        state,
        request_id,
        "run-entry",
        &parsed.entry_id,
        None,
        &runner_args,
        &parsed.overrides,
    ) {
        Ok(result) => {
            let response = serde_json::json!({
                "ok": result.success,
                "request_id": request_id,
                "command": "run-entry",
                "entry_id": parsed.entry_id,
                "args": parsed.passthrough,
                "streams_override": parsed.overrides.streams,
                "pool_override": parsed.overrides.pool,
                "exit_code": result.exit_code,
                "elapsed_ms": result.elapsed_ms,
                "stdout": result.stdout,
                "stderr": result.stderr,
            });
            to_json_line(&response)
        }
        Err(error) => {
            emit_scheduler_event(
                state,
                SchedulerEvent::RunError {
                    request_id,
                    message: error.to_string(),
                },
            );
            let response = serde_json::json!({
                "ok": false,
                "request_id": request_id,
                "command": "run-entry",
                "entry_id": parsed.entry_id,
                "args": parsed.passthrough,
                "streams_override": parsed.overrides.streams,
                "pool_override": parsed.overrides.pool,
                "error": error.to_string(),
            });
            to_json_line(&response)
        }
    }
}

pub fn handle_run_list(state: &DaemonState) -> io::Result<String> {
    if let Some(denial) = evaluate_policy(state, "run-list", "workspace") {
        return Ok(denial);
    }

    let request_id = state.next_run_id.fetch_add(1, Ordering::Relaxed);
    emit_scheduler_event(
        state,
        SchedulerEvent::RunRequestAccepted {
            request_id,
            mode: "run-list".to_string(),
            target: "workspace".to_string(),
            entry: None,
            args: vec!["run-list".to_string()],
        },
    );
    emit_scheduler_event(
        state,
        SchedulerEvent::RunBuildStarted {
            request_id,
            command: "in-process run-list scan".to_string(),
        },
    );

    let started = Instant::now();
    let workspace_root = match resolve_workspace_root() {
        Ok(root) => root,
        Err(error) => {
            emit_scheduler_event(
                state,
                SchedulerEvent::RunError {
                    request_id,
                    message: error.to_string(),
                },
            );
            let response = serde_json::json!({
                "ok": false,
                "request_id": request_id,
                "command": "run-list",
                "error": error.to_string(),
            });
            return to_json_line(&response);
        }
    };

    match discover_run_entries(&workspace_root) {
        Ok(entries) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            emit_scheduler_event(
                state,
                SchedulerEvent::RunBuildFinished {
                    request_id,
                    success: true,
                    elapsed_ms,
                },
            );
            emit_scheduler_event(
                state,
                SchedulerEvent::RunFinished {
                    request_id,
                    success: true,
                    exit_code: Some(0),
                    elapsed_ms,
                },
            );
            let payload = serde_json::json!({
                "ok": true,
                "count": entries.len(),
                "entries": entries,
            });
            to_json_line(&payload)
        }
        Err(error) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            emit_scheduler_event(
                state,
                SchedulerEvent::RunBuildFinished {
                    request_id,
                    success: false,
                    elapsed_ms,
                },
            );
            emit_scheduler_event(
                state,
                SchedulerEvent::RunFinished {
                    request_id,
                    success: false,
                    exit_code: Some(1),
                    elapsed_ms,
                },
            );
            emit_scheduler_event(
                state,
                SchedulerEvent::RunError {
                    request_id,
                    message: error.clone(),
                },
            );
            let response = serde_json::json!({
                "ok": false,
                "request_id": request_id,
                "command": "run-list",
                "error": error,
            });
            to_json_line(&response)
        }
    }
}

/// Handle `ferrite-stop` — stop all running programs without shutting down the daemon.
///
/// Stops all managed apps and all active durable jobs.
pub fn handle_ferrite_stop(state: &DaemonState) -> io::Result<String> {
    // Stop all managed apps
    let stopped_apps = {
        let mut apps = state.apps.lock();
        let count = apps.len();
        for (_, app) in apps.iter_mut() {
            let _ = app.child.kill();
            let _ = app.child.wait();
        }
        apps.clear();
        count
    };

    // Stop all active durable jobs
    let stopped_jobs = {
        let mut supervisor = state.supervisor.lock();
        supervisor.stop_all("stopped by ferrite-stop")
    };

    let response = serde_json::json!({
        "ok": true,
        "message": "all programs stopped",
        "stopped_apps": stopped_apps,
        "stopped_jobs": stopped_jobs,
    });
    to_json_line(&response)
}

pub fn handle_help() -> io::Result<String> {
    let help = r#"{
  "ok": true,
  "commands": {
    "ping": "Test daemon connectivity",
    "status": "Get TLSF pool status",
    "stats": "Get GPU runtime statistics",
    "metrics": "Get comprehensive metrics",
    "snapshot": "Get system snapshot",
    "health": "Get health check",
    "keepalive": "Send keepalive signal",
    "apps": "List managed app processes",
    "app-start <app> [args...]": "Start a managed app binary",
    "app-stop <id|name>": "Stop a managed app process",
    "run-file <path> [--entry <name>] [--streams=N] [--pool=F] [-- <args...>]": "Run a Rust file (--streams/--pool override child limits)",
    "run-entry <entry-id> [--streams=N] [--pool=F] [-- <args...>]": "Run entry by ID (--streams/--pool override child limits; heavy targets denied in strict mode)",
    "run-list": "List discoverable Rust entries in workspace",
    "scheduler <subcommand>": "Scheduler control (queue-status, tenants, pause, resume, stats, policies, policy)",
    "audit-query [--tenant ID] [--last N]": "Query control plane audit log",
    "events-stream": "Subscribe to real-time event stream (JSON lines)",
    "job-submit <cmd> [args...]": "Submit a durable job",
    "job-stop <id> [reason]": "Cancel a durable job",
    "job-status <id>": "Show status of a durable job",
    "job-list": "List all durable jobs",
    "job-history <id>": "Show state transition history for a job",
    "ferrite-stop": "Stop all running programs (apps, jobs) without shutting down",
    "shutdown": "Shutdown daemon",
    "help": "Show this help"
  }
}
"#;
    Ok(help.to_string())
}

// ── scheduler / control plane handlers ────────────────────────────

/// Handle the `scheduler` meta-command: dispatches to scheduler_commands.
pub fn handle_scheduler_command(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    use crate::scheduler_commands::{self, SchedulerCommand};

    let Some(cmd) = SchedulerCommand::parse_from_args(args) else {
        return Ok("{\"error\":\"unknown scheduler subcommand — try: scheduler queue-status\"}\n".to_string());
    };

    let mut engine = state.policy_engine.lock();
    let mut es = state.event_stream.lock();
    let mut sched = state.scheduler.lock();
    let mut paused = state.scheduler_paused.load(Ordering::Relaxed);
    let result = scheduler_commands::handle_scheduler_command(
        &cmd, &mut engine, &mut paused, &mut sched, Some(&mut es),
    );
    state.scheduler_paused.store(paused, Ordering::Relaxed);
    Ok(result)
}

/// Handle `scheduler-status`: quick overview of queue depth, active jobs, tenants.
pub fn handle_scheduler_status(state: &DaemonState) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let paused = state.scheduler_paused.load(Ordering::Relaxed);
    let sched = state.scheduler.lock();
    let snap = sched.state_snapshot();
    let response = serde_json::json!({
        "ok": true,
        "paused": paused,
        "rule_count": engine.rule_count(),
        "audit_entries": engine.audit_log().len(),
        "queue_depth": snap.queue_depth,
        "active_jobs": snap.active_jobs,
    });
    to_json_line(&response)
}

/// Handle `scheduler-queue`: list queued/running jobs.
pub fn handle_scheduler_queue(state: &DaemonState) -> io::Result<String> {
    let paused = state.scheduler_paused.load(Ordering::Relaxed);
    let sched = state.scheduler.lock();

    let queued: Vec<serde_json::Value> = sched.dispatcher().queued_jobs().iter().map(|job| {
        serde_json::json!({
            "job_id": job.id.0,
            "tenant_id": job.tenant_id.0,
            "state": job.state.to_string(),
            "priority": format!("{:?}", job.priority),
            "vram_estimate": job.estimated_vram_bytes,
        })
    }).collect();

    let active: Vec<serde_json::Value> = sched.dispatcher().active_records().iter().map(|r| {
        serde_json::json!({
            "job_id": r.job_id.0,
            "tenant_id": r.tenant_id.0,
            "stream_id": r.stream_id,
            "vram_reserved": r.estimated_vram_bytes,
            "running_ms": r.started_at.elapsed().as_millis() as u64,
        })
    }).collect();

    let total = queued.len() + active.len();
    let response = serde_json::json!({
        "ok": true,
        "paused": paused,
        "queued": queued,
        "active": active,
        "count": total,
    });
    to_json_line(&response)
}

/// Handle `scheduler-policy`: show active policies and recent decisions.
pub fn handle_scheduler_policy(state: &DaemonState) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let rules: Vec<serde_json::Value> = engine
        .list_rules()
        .into_iter()
        .map(|(name, desc)| serde_json::json!({"name": name, "description": desc}))
        .collect();

    let recent: Vec<serde_json::Value> = engine
        .audit_log()
        .query_recent(20)
        .into_iter()
        .map(|e| serde_json::json!({
            "action": e.action,
            "resource": e.resource,
            "decision": e.decision,
            "reason": e.reason,
            "tenant_id": e.tenant_id,
            "timestamp_secs": e.elapsed_secs,
        }))
        .collect();

    let response = serde_json::json!({
        "ok": true,
        "rules": rules,
        "recent_decisions": recent,
    });
    to_json_line(&response)
}

/// Handle `audit-query [--tenant ID] [--last N]`.
pub fn handle_audit_query(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let mut tenant_filter: Option<u64> = None;
    let mut limit: usize = 50;

    // Parse flags
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--tenant" => {
                if i + 1 < args.len() {
                    tenant_filter = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--last" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(50);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Emit audit query event
    {
        let mut es = state.event_stream.lock();
        es.emit(crate::event_stream::SchedulerEvent::AuditQuery {
            tenant_filter,
            result_count: 0, // updated below
        });
    }

    let entries: Vec<serde_json::Value> = if let Some(tid) = tenant_filter {
        engine.audit_log().query_by_tenant(tid, limit)
    } else {
        engine.audit_log().query_recent(limit)
    }
    .into_iter()
    .map(|e| serde_json::json!({
        "timestamp_secs": e.elapsed_secs,
        "tenant_id": e.tenant_id,
        "action": e.action,
        "resource": e.resource,
        "decision": e.decision,
        "reason": e.reason,
        "source_ip": e.source_ip,
    }))
    .collect();

    let response = serde_json::json!({
        "ok": true,
        "count": entries.len(),
        "tenant_filter": tenant_filter,
        "entries": entries,
    });
    to_json_line(&response)
}

/// Handle `events-stream`: return recent events as JSON lines.
pub fn handle_events_stream(state: &DaemonState) -> io::Result<String> {
    let es = state.event_stream.lock();
    let output = es.export_jsonl();
    if output.is_empty() {
        Ok("{\"ok\":true,\"message\":\"no events\"}\n".to_string())
    } else {
        Ok(format!("{}\n", output))
    }
}

// ── FerApp event handler ──────────────────────────────────────────

/// Handle `app-event <json>` -- ingest an application event from a FerApp.
///
/// Payload format:
/// ```json
/// {"app_name":"...", "event_name":"...", "payload":"...", "tenant_id": null}
/// ```
pub fn handle_app_event(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    let json_str = args.join(" ");
    if json_str.is_empty() {
        return Ok("{\"error\":\"usage: app-event {\\\"app_name\\\":\\\"...\\\", ...}\"}\n".to_string());
    }

    #[derive(serde::Deserialize)]
    struct AppEventPayload {
        app_name: String,
        event_name: String,
        payload: String,
        tenant_id: Option<u64>,
    }

    let parsed: AppEventPayload = match serde_json::from_str(&json_str) {
        Ok(p) => p,
        Err(e) => {
            let response = serde_json::json!({
                "error": "invalid app-event payload",
                "message": e.to_string(),
            });
            return to_json_line(&response);
        }
    };

    emit_scheduler_event(
        state,
        SchedulerEvent::AppEvent {
            app_name: parsed.app_name.clone(),
            event_name: parsed.event_name.clone(),
            payload: parsed.payload,
            tenant_id: parsed.tenant_id,
        },
    );

    let response = serde_json::json!({
        "ok": true,
        "message": "event accepted",
        "app_name": parsed.app_name,
        "event_name": parsed.event_name,
    });
    to_json_line(&response)
}

// ── durable job command handlers (Plan-B) ─────────────────────────

/// Handle `job-submit <command> [args...]` -- submit a new durable job.
pub fn handle_job_submit(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-submit <command> [args...]\"}\n".to_string());
    }

    let command = args[0].to_string();

    // Policy enforcement: deny if not allowed
    if let Some(denial) = evaluate_policy(state, "job-submit", &command) {
        return Ok(denial);
    }

    let job_args: Vec<String> = args.iter().skip(1).map(|s| s.to_string()).collect();
    let name = command.rsplit('/').next().unwrap_or(&command).to_string();
    let policy = state.config.jobs.to_restart_policy();

    let job = ptx_runtime::job::DurableJob::new(name.clone(), command, job_args, policy);
    let mut supervisor = state.supervisor.lock();
    match supervisor.submit(job) {
        Ok(id) => {
            let response = serde_json::json!({
                "ok": true,
                "message": "job submitted",
                "job_id": id.raw(),
                "name": name,
            });
            to_json_line(&response)
        }
        Err(e) => {
            let response = serde_json::json!({
                "error": "job submit failed",
                "message": e.to_string(),
            });
            to_json_line(&response)
        }
    }
}

/// Handle `job-stop <id>` -- cancel a durable job.
pub fn handle_job_stop(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-stop <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-stop requires a numeric job ID\"}\n".to_string());
        }
    };

    // Policy enforcement: deny if not allowed
    if let Some(denial) = evaluate_policy(state, "job-stop", &format!("job:{}", id)) {
        return Ok(denial);
    }

    let reason = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "stopped by operator".to_string()
    };

    let mut supervisor = state.supervisor.lock();
    match supervisor.stop(ptx_runtime::job::DurableJobId(id), reason) {
        Ok(()) => {
            let response = serde_json::json!({
                "ok": true,
                "message": "job stopped",
                "job_id": id,
            });
            to_json_line(&response)
        }
        Err(e) => {
            let response = serde_json::json!({
                "error": "job stop failed",
                "message": e.to_string(),
            });
            to_json_line(&response)
        }
    }
}

/// Handle `job-status <id>` -- show status of a specific durable job.
pub fn handle_job_status(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-status <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-status requires a numeric job ID\"}\n".to_string());
        }
    };

    let supervisor = state.supervisor.lock();
    match supervisor.status(ptx_runtime::job::DurableJobId(id)) {
        Some(job) => {
            let response = serde_json::json!({
                "ok": true,
                "job_id": job.id.raw(),
                "name": job.name,
                "command": job.command,
                "args": job.args,
                "state": job.state().to_string(),
                "failure_count": job.failure_count,
                "last_failure": job.last_failure,
                "process_pid": job.process_pid,
            });
            to_json_line(&response)
        }
        None => {
            let response = serde_json::json!({
                "error": "job not found",
                "job_id": id,
            });
            to_json_line(&response)
        }
    }
}

/// Handle `job-list` -- list all durable jobs.
pub fn handle_job_list(state: &DaemonState) -> io::Result<String> {
    let supervisor = state.supervisor.lock();
    let jobs: Vec<serde_json::Value> = supervisor
        .list()
        .iter()
        .map(|job| {
            serde_json::json!({
                "job_id": job.id.raw(),
                "name": job.name,
                "state": job.state().to_string(),
                "failure_count": job.failure_count,
                "process_pid": job.process_pid,
            })
        })
        .collect();

    let response = serde_json::json!({
        "ok": true,
        "count": jobs.len(),
        "jobs": jobs,
    });
    to_json_line(&response)
}

/// Handle `job-history <id>` -- show full state transition history for a job.
pub fn handle_job_history(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-history <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-history requires a numeric job ID\"}\n".to_string());
        }
    };

    let supervisor = state.supervisor.lock();
    match supervisor.status(ptx_runtime::job::DurableJobId(id)) {
        Some(job) => {
            let transitions: Vec<serde_json::Value> = job
                .state_machine
                .history()
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "from": t.from_state.to_string(),
                        "to": t.to_state.to_string(),
                        "timestamp": format!("{:?}", t.timestamp),
                        "reason": t.reason,
                    })
                })
                .collect();

            let response = serde_json::json!({
                "ok": true,
                "job_id": job.id.raw(),
                "name": job.name,
                "current_state": job.state().to_string(),
                "transition_count": transitions.len(),
                "transitions": transitions,
            });
            to_json_line(&response)
        }
        None => {
            let response = serde_json::json!({
                "error": "job not found",
                "job_id": id,
            });
            to_json_line(&response)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_execution_mode_permissive() {
        // When strict=false, all targets get ExternalProcess
        assert_eq!(
            classify_execution_mode("jitter_benchmark", false),
            RunExecutionMode::ExternalProcess
        );
        assert_eq!(
            classify_execution_mode("stress_test", false),
            RunExecutionMode::ExternalProcess
        );
        assert_eq!(
            classify_execution_mode("my_app", false),
            RunExecutionMode::ExternalProcess
        );
        assert_eq!(
            classify_execution_mode("training_loop", false),
            RunExecutionMode::ExternalProcess
        );
    }

    #[test]
    fn test_classify_execution_mode_strict_heavy() {
        // Heavy targets are denied in strict mode
        let heavy_targets = [
            "jitter_benchmark",
            "ptx-runtime/examples/jitter_benchmark.rs#main",
            "stress_test",
            "latency_probe",
            "bench_alloc",
            "training_loop",
            "inference_server",
        ];
        for target in heavy_targets {
            assert_eq!(
                classify_execution_mode(target, true),
                RunExecutionMode::DeniedStrictMode,
                "expected DeniedStrictMode for heavy target '{}'",
                target,
            );
        }
    }

    #[test]
    fn test_classify_execution_mode_strict_light() {
        // Non-heavy targets are allowed even in strict mode
        let light_targets = [
            "my_app",
            "hello_world",
            "matrix_multiply",
            "ptx-runtime/examples/basic.rs#main",
        ];
        for target in light_targets {
            assert_eq!(
                classify_execution_mode(target, true),
                RunExecutionMode::ExternalProcess,
                "expected ExternalProcess for light target '{}'",
                target,
            );
        }
    }

    #[test]
    fn test_is_oom_like_failure_exit_codes() {
        assert!(is_oom_like_failure("", "", Some(137)));
        assert!(is_oom_like_failure("", "", Some(139)));
        assert!(!is_oom_like_failure("", "", Some(1)));
        assert!(!is_oom_like_failure("", "", Some(0)));
    }

    #[test]
    fn test_is_oom_like_failure_messages() {
        assert!(is_oom_like_failure("", "out of memory", None));
        assert!(is_oom_like_failure("allocation failed", "", None));
        assert!(is_oom_like_failure("", "CUDA stream create failed", None));
        assert!(!is_oom_like_failure("all good", "no errors", None));
    }
}
