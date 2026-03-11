use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use ptx_runner::{discover_run_entries, parse_run_entry_request, parse_run_file_request, prepare_run_entry_command, prepare_run_file_command, PreparedRunCommand};

use crate::commands as daemon_commands;
use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;
use crate::tui::state::run_state::{RunOutputLine, RunProfile, RunStatus};
use crate::tui::state::TuiState;

pub fn cmd_run(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    if state.run_status != RunStatus::Idle {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            "already running — use 'stop' first",
        ));
        return;
    }

    let script = if state.file_dirty || state.open_file.is_some() {
        state.file_lines.join("\n")
    } else {
        state.push_log(LogEntry::new(LogCategory::Err, "no file open to run"));
        return;
    };

    if script.trim().is_empty() {
        state.push_log(LogEntry::new(LogCategory::Err, "empty file — nothing to run"));
        return;
    }

    state.run_output.clear();
    state.run_output_scroll = 0;
    state.run_status = RunStatus::Compiling;
    state.run_start_time = Some(std::time::Instant::now());
    state.run_elapsed_ms = 0;
    state.run_cancel_flag.store(false, Ordering::Relaxed);
    state.show_run_output = true;

    let file_path = state.open_file.clone();
    state.run_target_file = file_path.clone();
    state.last_run_file = file_path;

    state.push_log(LogEntry::new(LogCategory::Jit, "compiling..."));
    state.run_output.push_back(RunOutputLine::new(
        LogCategory::Jit,
        "compiling...",
    ));

    // Switch to full-screen run output view
    state.enter_run_output();

    let runner = Arc::clone(runner);
    let tx = tx.clone();
    let cancel = Arc::clone(&state.run_cancel_flag);
    let timeout_secs = state.run_config.timeout_secs;

    std::thread::spawn(move || {
        // Snapshot VRAM before execution for profiling
        let vram_before = {
            let r = runner.lock();
            r.pool_used_bytes()
        };

        let compile_start = std::time::Instant::now();

        // Compile
        let compile_result = {
            let mut r = runner.lock();
            r.compile(&script)
        };

        let compile_ms = compile_start.elapsed().as_millis() as u64;

        match compile_result {
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("compile error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: compile_ms,
                }).ok();
                return;
            }
            Ok(()) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("compiled in {}ms", compile_ms),
                    is_error: false,
                }).ok();
            }
        }

        // Check cancel between compile and execute
        if cancel.load(Ordering::Relaxed) {
            tx.send(DaemonEvent::RunOutput {
                line: "cancelled before execution".into(),
                is_error: true,
            }).ok();
            tx.send(DaemonEvent::RunFinished {
                success: false,
                elapsed_ms: compile_ms,
            }).ok();
            return;
        }

        tx.send(DaemonEvent::RunStarted {
            file: String::new(),
        }).ok();

        // Grab program info before execution for profiling
        let program_info = {
            let r = runner.lock();
            r.inspect_last()
        };

        // Execute
        let exec_start = std::time::Instant::now();

        // Timeout watchdog — tracks whether the timeout actually fired
        let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_timeout = Arc::clone(&cancel);
        let timed_out_flag = Arc::clone(&timed_out);
        let tx_timeout = tx.clone();
        let timeout_handle = std::thread::spawn(move || {
            let deadline = std::time::Duration::from_secs(timeout_secs);
            let start = std::time::Instant::now();
            while start.elapsed() < deadline {
                if cancel_timeout.load(Ordering::Relaxed) {
                    return; // Cancelled before timeout — not a timeout
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            // Timeout fired
            timed_out_flag.store(true, Ordering::Relaxed);
            cancel_timeout.store(true, Ordering::Relaxed);
            let elapsed = start.elapsed().as_millis() as u64;
            tx_timeout.send(DaemonEvent::RunOutput {
                line: format!("TIMEOUT after {}s — execution killed", timeout_secs),
                is_error: true,
            }).ok();
            tx_timeout.send(DaemonEvent::RunTimeout {
                elapsed_ms: elapsed,
            }).ok();
        });

        let exec_result = {
            let r = runner.lock();
            r.execute_last()
        };

        // Snapshot VRAM after execution for profiling delta
        let vram_after = {
            let r = runner.lock();
            r.pool_used_bytes()
        };

        cancel.store(true, Ordering::Relaxed); // Signal timeout thread to stop
        let _ = timeout_handle.join();

        let exec_ms = exec_start.elapsed().as_millis() as u64;
        let total_ms = compile_ms + exec_ms;

        let was_timeout = timed_out.load(Ordering::Relaxed);
        let success = exec_result.is_ok() && !was_timeout;

        match exec_result {
            Ok(result) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!(
                        "executed in {}ms — output shape {:?}, {} values",
                        exec_ms,
                        result.shape,
                        result.data.len()
                    ),
                    is_error: false,
                }).ok();

                // Send first few output values
                let preview: Vec<String> = result.data.iter().take(16).map(|v| format!("{:.4}", v)).collect();
                tx.send(DaemonEvent::RunOutput {
                    line: format!("output: [{}{}]",
                        preview.join(", "),
                        if result.data.len() > 16 { ", ..." } else { "" }
                    ),
                    is_error: false,
                }).ok();

                // Also send tensor result for visualization
                tx.send(DaemonEvent::TensorResult {
                    shape: result.shape,
                    data: result.data,
                }).ok();

                tx.send(DaemonEvent::RunFinished {
                    success: true,
                    elapsed_ms: total_ms,
                }).ok();
            }
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("execution error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: total_ms,
                }).ok();
            }
        }

        // Emit profiling event
        if let Some(info) = program_info {
            let tag = script.chars().take(40).collect::<String>();
            tx.send(DaemonEvent::KernelProfiled {
                compile_ms,
                exec_ms,
                input_shapes: info.input_shapes,
                output_shape: info.output_shape,
                node_count: info.node_count,
                total_elements: info.total_elements,
                success,
                source_tag: tag,
                vram_before,
                vram_after,
            }).ok();
        }
    });
}

pub fn cmd_stop_run(state: &mut TuiState) {
    if state.run_status == RunStatus::Idle {
        state.push_log(LogEntry::new(LogCategory::Sys, "nothing running"));
        return;
    }
    state.run_cancel_flag.store(true, Ordering::Relaxed);
    state.push_log(LogEntry::new(LogCategory::Sys, "stop signal sent"));
}

pub fn cmd_rerun(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    if let Some(path) = state.last_run_file.clone() {
        if state.open_file.as_ref() != Some(&path) {
            match state.open_file_path(path) {
                Ok(()) => {}
                Err(e) => {
                    state.push_log(LogEntry::new(LogCategory::Err, e));
                    return;
                }
            }
        }
    }
    cmd_run(state, runner, tx);
}

pub fn cmd_args(state: &mut TuiState, args: &[&str]) {
    state.run_config.args = args.iter().map(|s| s.to_string()).collect();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("run args: {:?}", state.run_config.args),
    ));
}

pub fn cmd_profile(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        Some("debug") => {
            state.run_config.profile = RunProfile::Debug;
            state.push_log(LogEntry::new(LogCategory::Sys, "profile: debug"));
        }
        Some("release") => {
            state.run_config.profile = RunProfile::Release;
            state.push_log(LogEntry::new(LogCategory::Sys, "profile: release"));
        }
        _ => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "profile: {:?} — usage: profile [debug|release]",
                    state.run_config.profile
                ),
            ));
        }
    }
}

pub fn cmd_timeout(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        Some(val) => {
            match val.parse::<u64>() {
                Ok(secs) if secs > 0 => {
                    state.run_config.timeout_secs = secs;
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("execution timeout: {}s", secs),
                    ));
                }
                _ => {
                    state.push_log(LogEntry::new(
                        LogCategory::Err,
                        "usage: timeout <seconds> (must be positive integer)",
                    ));
                }
            }
        }
        None => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("execution timeout: {}s — usage: timeout <seconds>", state.run_config.timeout_secs),
            ));
        }
    }
}

// ── daemon-backed runner commands ─────────────────────────────────

/// Run a .rs file through the daemon's in-process runner pipeline.
///
/// Usage: `run-file [path] [--entry <name>] [-- <args...>]`
///
/// When path is omitted, uses the currently open file.
pub fn cmd_run_file(
    state: &mut TuiState,
    daemon: &Arc<DaemonState>,
    tx: &Sender<DaemonEvent>,
    args: &[&str],
) {
    if !matches!(state.run_status, RunStatus::Idle | RunStatus::Succeeded | RunStatus::Failed | RunStatus::Timeout) {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            "already running — use '/stop' first",
        ));
        return;
    }

    // Parse arguments: [path] [--entry <name>] [-- <passthrough...>]
    let (file_path, entry, passthrough) = match parse_run_file_tui_args(args, state) {
        Ok(parsed) => parsed,
        Err(msg) => {
            state.push_log(LogEntry::new(LogCategory::Err, msg));
            return;
        }
    };

    if file_path.extension().and_then(|e| e.to_str()) != Some("rs") {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            format!("not a .rs file: {}", file_path.display()),
        ));
        return;
    }

    let profiles = match daemon_commands::compute_orchestrator_profiles(
        &daemon.runtime.tlsf_stats(),
        &daemon.runtime.stats(),
        &file_path.to_string_lossy(),
    ) {
        Ok(p) => p,
        Err(msg) => {
            state.push_log(LogEntry::new(LogCategory::Err, msg));
            return;
        }
    };

    // Reset run state
    state.run_output.clear();
    state.run_output_scroll = 0;
    state.run_status = RunStatus::Compiling;
    state.run_start_time = Some(std::time::Instant::now());
    state.run_elapsed_ms = 0;
    state.run_cancel_flag.store(false, Ordering::Relaxed);
    state.show_run_output = true;
    state.run_target_file = Some(file_path.clone());
    state.last_run_file = Some(file_path.clone());

    let display_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string();
    state.push_log(LogEntry::new(
        LogCategory::Jit,
        format!("building {}...", display_name),
    ));
    state.run_output.push_back(RunOutputLine::new(
        LogCategory::Jit,
        format!("building {}...", display_name),
    ));

    // Switch to full-screen run output view
    state.enter_run_output();

    // Build runner arguments
    let mut runner_args = vec![
        "run-file".to_string(),
        file_path.to_string_lossy().to_string(),
    ];
    if let Some(ref entry_name) = entry {
        runner_args.push("--entry".to_string());
        runner_args.push(entry_name.clone());
    }
    if !passthrough.is_empty() {
        runner_args.push("--".to_string());
        runner_args.extend(passthrough);
    }

    let tx = tx.clone();
    let cancel = Arc::clone(&state.run_cancel_flag);
    let timeout_secs = state.run_config.timeout_secs;
    let runner_env = build_runner_env(daemon);

    std::thread::spawn(move || {
        run_runner_subprocess(runner_args, runner_env, profiles, tx, cancel, timeout_secs);
    });
}

/// Run by logical entry ID through the daemon's in-process runner pipeline.
///
/// Usage: `run-entry <id> [-- <args...>]`
pub fn cmd_run_entry(
    state: &mut TuiState,
    daemon: &Arc<DaemonState>,
    tx: &Sender<DaemonEvent>,
    args: &[&str],
) {
    if !matches!(state.run_status, RunStatus::Idle | RunStatus::Succeeded | RunStatus::Failed | RunStatus::Timeout) {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            "already running — use '/stop' first",
        ));
        return;
    }

    if args.is_empty() {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            "usage: run-entry <path#entry> [-- <args...>]",
        ));
        return;
    }

    let entry_id = args[0].to_string();
    let passthrough: Vec<String> = if let Some(pos) = args.iter().position(|a| *a == "--") {
        args[pos + 1..].iter().map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    };

    let profiles = match daemon_commands::compute_orchestrator_profiles(
        &daemon.runtime.tlsf_stats(),
        &daemon.runtime.stats(),
        &entry_id,
    ) {
        Ok(p) => p,
        Err(msg) => {
            state.push_log(LogEntry::new(LogCategory::Err, msg));
            return;
        }
    };

    // Reset run state
    state.run_output.clear();
    state.run_output_scroll = 0;
    state.run_status = RunStatus::Compiling;
    state.run_start_time = Some(std::time::Instant::now());
    state.run_elapsed_ms = 0;
    state.run_cancel_flag.store(false, Ordering::Relaxed);
    state.show_run_output = true;

    state.push_log(LogEntry::new(
        LogCategory::Jit,
        format!("building entry {}...", entry_id),
    ));
    state.run_output.push_back(RunOutputLine::new(
        LogCategory::Jit,
        format!("building entry {}...", entry_id),
    ));

    // Switch to full-screen run output view
    state.enter_run_output();

    let mut runner_args = vec!["run-entry".to_string(), entry_id];
    if !passthrough.is_empty() {
        runner_args.push("--".to_string());
        runner_args.extend(passthrough);
    }

    let tx = tx.clone();
    let cancel = Arc::clone(&state.run_cancel_flag);
    let timeout_secs = state.run_config.timeout_secs;
    let runner_env = build_runner_env(daemon);

    std::thread::spawn(move || {
        run_runner_subprocess(runner_args, runner_env, profiles, tx, cancel, timeout_secs);
    });
}

/// Discover and list all runnable entries in the workspace.
pub fn cmd_run_list(
    state: &mut TuiState,
    _daemon: &Arc<DaemonState>,
    tx: &Sender<DaemonEvent>,
) {
    state.push_log(LogEntry::new(LogCategory::Jit, "scanning workspace entries..."));

    let tx = tx.clone();

    std::thread::spawn(move || {
        let workspace_root = match daemon_commands::resolve_workspace_root() {
            Ok(root) => root,
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("workspace error: {}", e),
                    is_error: true,
                }).ok();
                return;
            }
        };

        let started = std::time::Instant::now();
        let entries = match discover_run_entries(&workspace_root) {
            Ok(entries) => entries,
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("run-list error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                }).ok();
                return;
            }
        };

        // Each entry becomes a clickable log line — clicking populates the
        // input with `run-entry <id>` so the user just presses Enter.
        tx.send(DaemonEvent::LogAction(
            LogEntry::new(
                LogCategory::Sys,
                format!("── {} entries (click to run) ──", entries.len()),
            ),
        )).ok();
        for entry in entries {
            let default_tag = if entry.is_default { " [default]" } else { "" };
            tx.send(DaemonEvent::LogAction(
                LogEntry::new(
                    LogCategory::Sys,
                    format!(
                        "{} ({}/{}){}",
                        entry.id, entry.crate_name, entry.entry, default_tag
                    ),
                )
                .with_action(format!("run-entry {}", entry.id)),
            ))
            .ok();
        }

        tx.send(DaemonEvent::RunFinished {
            success: true,
            elapsed_ms: started.elapsed().as_millis() as u64,
        }).ok();
    });
}

/// Toggle the run output panel visibility.
pub fn cmd_toggle_output(state: &mut TuiState) {
    state.show_run_output = !state.show_run_output;
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("run output: {}", if state.show_run_output { "visible" } else { "hidden" }),
    ));
}

// ── internal helpers ──────────────────────────────────────────────

fn parse_run_file_tui_args(
    args: &[&str],
    state: &TuiState,
) -> Result<(PathBuf, Option<String>, Vec<String>), String> {
    // Split at -- for passthrough
    let (head, tail) = if let Some(pos) = args.iter().position(|a| *a == "--") {
        (&args[..pos], &args[pos + 1..])
    } else {
        (args, &[] as &[&str])
    };
    let passthrough: Vec<String> = tail.iter().map(|s| s.to_string()).collect();

    // Parse head: [path] [--entry <name>]
    let mut path: Option<PathBuf> = None;
    let mut entry: Option<String> = None;
    let mut idx = 0;

    while idx < head.len() {
        match head[idx] {
            "--entry" => {
                let val = head.get(idx + 1).ok_or("missing value for --entry")?;
                entry = Some(val.to_string());
                idx += 2;
            }
            arg if arg.starts_with("--") => {
                return Err(format!("unknown flag '{}' (expected --entry or --)", arg));
            }
            arg => {
                if path.is_some() {
                    return Err("unexpected extra argument after path".to_string());
                }
                path = Some(PathBuf::from(arg));
                idx += 1;
            }
        }
    }

    // Fall back to currently open file
    let file_path = match path {
        Some(p) => {
            if p.is_absolute() {
                p
            } else {
                state.current_dir.join(p)
            }
        }
        None => {
            state.open_file.clone().ok_or_else(|| {
                "no file specified — open a file or: run-file <path>".to_string()
            })?
        }
    };

    Ok((file_path, entry, passthrough))
}

fn prepare_in_process_run_command(
    workspace_root: &std::path::Path,
    runner_args: &[String],
) -> Result<PreparedRunCommand, String> {
    if runner_args.is_empty() {
        return Err("missing runner mode".to_string());
    }

    let repo_root = workspace_root
        .parent()
        .unwrap_or(workspace_root)
        .to_path_buf();

    match runner_args[0].as_str() {
        "run-file" => {
            let mut req = parse_run_file_request(runner_args.get(1..).unwrap_or(&[]))?;
            if req.path.is_relative() {
                req.path = workspace_root.join(req.path);
            }
            prepare_run_file_command(workspace_root, &repo_root, req)
        }
        "run-entry" => {
            let req = parse_run_entry_request(runner_args.get(1..).unwrap_or(&[]))?;
            prepare_run_entry_command(workspace_root, &repo_root, req)
        }
        mode => Err(format!(
            "unsupported runner mode '{mode}' for in-process execution"
        )),
    }
}

/// Execute run-file/run-entry through the in-process runner orchestration,
/// spawning only the final target process and streaming output as DaemonEvents.
fn run_runner_subprocess(
    runner_args: Vec<String>,
    runner_env: Vec<(String, String)>,
    profiles: Vec<daemon_commands::RunOrchestratorProfile>,
    tx: Sender<DaemonEvent>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    timeout_secs: u64,
) {
    let workspace_root = match daemon_commands::resolve_workspace_root() {
        Ok(root) => root,
        Err(e) => {
            tx.send(DaemonEvent::RunOutput {
                line: format!("workspace error: {}", e),
                is_error: true,
            }).ok();
            tx.send(DaemonEvent::RunFinished {
                success: false,
                elapsed_ms: 0,
            }).ok();
            return;
        }
    };

    let profiles = if profiles.is_empty() {
        vec![daemon_commands::RunOrchestratorProfile::Safe]
    } else {
        profiles
    };
    let total_attempts = profiles.len();
    let mut total_elapsed_ms = 0u64;

    for (idx, profile) in profiles.into_iter().enumerate() {
        let attempt = idx + 1;
        let mut prepared = match prepare_in_process_run_command(&workspace_root, &runner_args) {
            Ok(prepared) => prepared,
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("runner prepare error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: total_elapsed_ms,
                }).ok();
                return;
            }
        };

        let cmd = prepared.command_mut();
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        for (key, value) in &runner_env {
            cmd.env(key, value);
        }
        for (key, value) in daemon_commands::profile_env_pairs(profile) {
            cmd.env(key, value);
        }

        tx.send(DaemonEvent::RunOutput {
            line: format!(
                "[orchestrator] attempt {}/{} profile={} streams={} pool_fraction={:.2}",
                attempt,
                total_attempts,
                profile.as_str(),
                profile.max_streams(),
                profile.pool_fraction(),
            ),
            is_error: false,
        }).ok();

        let attempt_start = std::time::Instant::now();
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("failed to spawn run target: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: total_elapsed_ms,
                }).ok();
                return;
            }
        };

        tx.send(DaemonEvent::RunStarted {
            file: runner_args.get(1).cloned().unwrap_or_default(),
        }).ok();

        // Stream stdout and stderr in separate threads
        let (line_tx, line_rx) = std::sync::mpsc::channel::<(bool, String)>();

        if let Some(stdout) = child.stdout.take() {
            let ltx = line_tx.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    if ltx.send((false, line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let ltx = line_tx.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    if ltx.send((true, line)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(line_tx);

        // Timeout watchdog
        let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let timed_out_flag = Arc::clone(&timed_out);
        let cancel_timeout = Arc::clone(&cancel);
        let watchdog_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog_stop_flag = Arc::clone(&watchdog_stop);
        let tx_timeout = tx.clone();
        let timeout_handle = std::thread::spawn(move || {
            let deadline = std::time::Duration::from_secs(timeout_secs);
            let start = std::time::Instant::now();
            while start.elapsed() < deadline {
                if watchdog_stop_flag.load(Ordering::Relaxed)
                    || cancel_timeout.load(Ordering::Relaxed)
                {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            timed_out_flag.store(true, Ordering::Relaxed);
            cancel_timeout.store(true, Ordering::Relaxed);
            let elapsed = start.elapsed().as_millis() as u64;
            tx_timeout.send(DaemonEvent::RunOutput {
                line: format!("TIMEOUT after {}s — execution killed", timeout_secs),
                is_error: true,
            }).ok();
            tx_timeout.send(DaemonEvent::RunTimeout {
                elapsed_ms: elapsed,
            }).ok();
        });

        let mut captured_stdout = String::new();
        let mut captured_stderr = String::new();

        // Relay output lines to TUI
        for (is_error, line) in line_rx {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if is_error {
                if !captured_stderr.is_empty() {
                    captured_stderr.push('\n');
                }
                captured_stderr.push_str(&line);
            } else {
                if !captured_stdout.is_empty() {
                    captured_stdout.push('\n');
                }
                captured_stdout.push_str(&line);
            }

            let is_build_error = is_error
                && (line.contains("error[E")
                    || line.contains("error:")
                    || line.contains("warning:")
                    || line.starts_with("error"));
            tx.send(DaemonEvent::RunOutput {
                line: line.clone(),
                is_error: is_error || is_build_error,
            }).ok();
        }

        // If cancelled, try to kill the child
        if cancel.load(Ordering::Relaxed) && !timed_out.load(Ordering::Relaxed) {
            let _ = child.kill();
        }

        let status = child.wait();
        watchdog_stop.store(true, Ordering::Relaxed);
        let _ = timeout_handle.join();

        let elapsed_ms = attempt_start.elapsed().as_millis() as u64;
        total_elapsed_ms += elapsed_ms;
        let was_timeout = timed_out.load(Ordering::Relaxed);

        if was_timeout {
            // Timeout already emitted its events.
            return;
        }

        let success = status.as_ref().map(|s| s.success()).unwrap_or(false);
        let exit_code = status.as_ref().ok().and_then(|s| s.code());

        if !success {
            if let Some(code) = exit_code {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("process exited with code {}", code),
                    is_error: true,
                }).ok();
            }
        }

        if success {
            tx.send(DaemonEvent::RunFinished {
                success: true,
                elapsed_ms: total_elapsed_ms,
            }).ok();
            return;
        }

        let user_cancelled = cancel.load(Ordering::Relaxed);
        let oom_like =
            daemon_commands::is_oom_like_failure(&captured_stdout, &captured_stderr, exit_code);
        if !user_cancelled && oom_like && attempt < total_attempts {
            tx.send(DaemonEvent::RunOutput {
                line: "[orchestrator] OOM-like failure detected; retrying with safer profile".to_string(),
                is_error: true,
            }).ok();
            cancel.store(false, Ordering::Relaxed);
            continue;
        }

        tx.send(DaemonEvent::RunFinished {
            success: false,
            elapsed_ms: total_elapsed_ms,
        }).ok();
        return;
    }
}

fn build_runner_env(daemon: &Arc<DaemonState>) -> Vec<(String, String)> {
    daemon.runtime.export_context();

    let mut envs = Vec::with_capacity(5);
    envs.push((
        "FERRITE_DAEMON_SOCKET".to_string(),
        daemon.config.socket_path.clone(),
    ));

    // Signal daemon-client mode so the child creates its own bounded runtime
    // instead of importing the daemon's host-side pointer (invalid cross-process).
    envs.push(("PTX_DAEMON_CLIENT".to_string(), "1".to_string()));

    // Forward context/stream handles (these are CUDA driver handles, valid cross-process).
    // Do NOT forward PTX_RUNTIME_PTR — it is a host pointer, invalid in child.
    for key in ["PTX_CONTEXT_PTR", "PTX_STREAM_PTR"] {
        if let Ok(value) = std::env::var(key) {
            envs.push((key.to_string(), value));
        }
    }

    envs
}
