use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::tui::state::TuiState;
use crate::tui::workspace::{self, fs_ops};

use super::checkpoint;
use super::lock;
use super::protocol::{AgentCommand, AgentResponse, AuditEntry};

const MAX_AUDIT: usize = 200;

pub fn execute_agent_command(
    cmd: &AgentCommand,
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) -> AgentResponse {
    let start = Instant::now();
    let cmd_name = cmd_label(cmd);

    state.push_log(LogEntry::new(
        LogCategory::App,
        format!("[agent] {}", cmd_name),
    ));

    let response = dispatch(cmd, state, runner, tx);

    let duration_us = start.elapsed().as_micros() as u64;

    // Audit trail
    if state.audit_trail.len() >= MAX_AUDIT {
        state.audit_trail.pop_front();
    }
    state.audit_trail.push_back(AuditEntry {
        timestamp: start,
        command: cmd_name.to_string(),
        success: response.ok,
        message: response
            .message
            .clone()
            .or_else(|| response.error.clone())
            .unwrap_or_default(),
        duration_us,
    });

    response
}

fn dispatch(
    cmd: &AgentCommand,
    state: &mut TuiState,
    _runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) -> AgentResponse {
    let root = &state.workspace_root.clone();
    let cwd = &state.current_dir.clone();

    match cmd {
        AgentCommand::Read { path } => {
            match workspace::guard_path(root, path, cwd) {
                Ok(p) => match std::fs::read_to_string(&p) {
                    Ok(content) => AgentResponse::success_data(
                        "Read",
                        serde_json::json!({ "path": p.display().to_string(), "content": content }),
                    ),
                    Err(e) => AgentResponse::error("Read", format!("read failed: {}", e)),
                },
                Err(e) => AgentResponse::error("Read", e),
            }
        }
        AgentCommand::Write { path, content } => {
            match workspace::guard_path(root, path, cwd) {
                Ok(p) => {
                    if let Some(parent) = p.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&p, content) {
                        Ok(()) => {
                            state.agent_modified_files.insert(p.clone());
                            state.file_tree.rebuild();
                            // If the written file is currently open, reload it
                            if state.open_file.as_ref() == Some(&p) {
                                let _ = state.open_file_path(p);
                            }
                            AgentResponse::success("Write", "file written")
                        }
                        Err(e) => AgentResponse::error("Write", format!("write failed: {}", e)),
                    }
                }
                Err(e) => AgentResponse::error("Write", e),
            }
        }
        AgentCommand::Edit { path, line, delete_count, insert } => {
            match workspace::guard_path(root, path, cwd) {
                Ok(p) => {
                    let content = match std::fs::read_to_string(&p) {
                        Ok(c) => c,
                        Err(e) => return AgentResponse::error("Edit", format!("read failed: {}", e)),
                    };
                    let mut lines: Vec<String> = content.lines().map(String::from).collect();
                    if lines.is_empty() {
                        lines.push(String::new());
                    }
                    let start_line = (*line).min(lines.len());
                    let end_line = (start_line + delete_count).min(lines.len());
                    lines.splice(start_line..end_line, insert.iter().cloned());
                    let new_content = lines.join("\n");
                    match std::fs::write(&p, &new_content) {
                        Ok(()) => {
                            state.agent_modified_files.insert(p.clone());
                            if state.open_file.as_ref() == Some(&p) {
                                let _ = state.open_file_path(p);
                            }
                            AgentResponse::success("Edit", format!("edited (deleted {}, inserted {})", delete_count, insert.len()))
                        }
                        Err(e) => AgentResponse::error("Edit", format!("write failed: {}", e)),
                    }
                }
                Err(e) => AgentResponse::error("Edit", e),
            }
        }
        AgentCommand::List { path, recursive } => {
            let target = path.as_deref();
            if recursive.unwrap_or(false) {
                // Recursive listing
                match workspace::guard_path(root, target.unwrap_or("."), cwd) {
                    Ok(p) => {
                        let mut files = Vec::new();
                        collect_recursive(&p, &mut files, 2500);
                        let rel: Vec<String> = files
                            .iter()
                            .filter_map(|f| f.strip_prefix(root).ok())
                            .map(|p| p.display().to_string())
                            .collect();
                        AgentResponse::success_data("List", serde_json::json!(rel))
                    }
                    Err(e) => AgentResponse::error("List", e),
                }
            } else {
                match fs_ops::ws_ls(root, cwd, target) {
                    Ok(entries) => AgentResponse::success_data("List", serde_json::json!(entries)),
                    Err(e) => AgentResponse::error("List", e),
                }
            }
        }
        AgentCommand::Mkdir { path } => match fs_ops::ws_mkdir(root, cwd, path) {
            Ok(msg) => {
                state.file_tree.rebuild();
                AgentResponse::success("Mkdir", msg)
            }
            Err(e) => AgentResponse::error("Mkdir", e),
        },
        AgentCommand::Touch { path } => match fs_ops::ws_touch(root, cwd, path) {
            Ok(msg) => {
                state.file_tree.rebuild();
                AgentResponse::success("Touch", msg)
            }
            Err(e) => AgentResponse::error("Touch", e),
        },
        AgentCommand::Mv { src, dst } => match fs_ops::ws_mv(root, cwd, src, dst) {
            Ok(msg) => {
                state.file_tree.rebuild();
                AgentResponse::success("Mv", msg)
            }
            Err(e) => AgentResponse::error("Mv", e),
        },
        AgentCommand::Cp { src, dst } => match fs_ops::ws_cp(root, cwd, src, dst) {
            Ok(msg) => {
                state.file_tree.rebuild();
                AgentResponse::success("Cp", msg)
            }
            Err(e) => AgentResponse::error("Cp", e),
        },
        AgentCommand::Rm { path, confirmed } => {
            match fs_ops::ws_rm(root, cwd, path, confirmed.unwrap_or(false)) {
                Ok(fs_ops::RmResult::Removed(msg)) => {
                    state.file_tree.rebuild();
                    AgentResponse::success("Rm", msg)
                }
                Ok(fs_ops::RmResult::NeedsConfirm(msg, _)) => {
                    AgentResponse::error("Rm", format!("confirmation required: {} — resend with confirmed:true", msg))
                }
                Err(e) => AgentResponse::error("Rm", e),
            }
        }
        AgentCommand::Run { file, profile: _, args: _ } => {
            if state.run_status != crate::tui::state::run_state::RunStatus::Idle {
                return AgentResponse::error("Run", "a run is already in progress — use Stop first");
            }
            if let Some(f) = file {
                match workspace::guard_path(root, f, cwd) {
                    Ok(p) => {
                        if let Err(e) = state.open_file_path(p) {
                            return AgentResponse::error("Run", format!("open failed: {}", e));
                        }
                    }
                    Err(e) => return AgentResponse::error("Run", e),
                }
            }
            // Trigger run via event
            tx.send(DaemonEvent::Log(LogEntry::new(
                LogCategory::Run,
                "[agent] triggering run",
            )))
            .ok();
            AgentResponse::success("Run", "run started")
        }
        AgentCommand::Stop => {
            state
                .run_cancel_flag
                .store(true, std::sync::atomic::Ordering::Relaxed);
            AgentResponse::success("Stop", "stop signal sent")
        }
        AgentCommand::Status => {
            let status = serde_json::json!({
                "run_status": format!("{:?}", state.run_status),
                "open_file": state.open_file.as_ref().map(|p| p.display().to_string()),
                "file_dirty": state.file_dirty,
                "agent_lock": state.agent_lock,
                "checkpoints": state.agent_checkpoints.keys().collect::<Vec<_>>(),
                "workspace_root": state.workspace_root.display().to_string(),
                "current_dir": state.current_dir.display().to_string(),
            });
            AgentResponse::success_data("Status", status)
        }
        AgentCommand::OpenBuffer { path } => {
            match workspace::guard_path(root, path, cwd) {
                Ok(p) => match state.open_file_path(p) {
                    Ok(()) => AgentResponse::success("OpenBuffer", "buffer opened"),
                    Err(e) => AgentResponse::error("OpenBuffer", e),
                },
                Err(e) => AgentResponse::error("OpenBuffer", e),
            }
        }
        AgentCommand::SaveBuffer => match state.file_save() {
            Ok(()) => AgentResponse::success("SaveBuffer", "buffer saved"),
            Err(e) => AgentResponse::error("SaveBuffer", e),
        },
        AgentCommand::BufferInfo => {
            let info = serde_json::json!({
                "path": state.open_file.as_ref().map(|p| p.display().to_string()),
                "dirty": state.file_dirty,
                "lines": state.file_lines.len(),
                "cursor": { "line": state.file_cursor_line, "col": state.file_cursor_col },
            });
            AgentResponse::success_data("BufferInfo", info)
        }
        AgentCommand::Checkpoint { label } => {
            match checkpoint::create_checkpoint(
                label,
                &state.agent_modified_files,
                &mut state.agent_checkpoints,
            ) {
                Ok(msg) => AgentResponse::success("Checkpoint", msg),
                Err(e) => AgentResponse::error("Checkpoint", e),
            }
        }
        AgentCommand::Rollback { label } => {
            let open = state.open_file.clone();
            match checkpoint::rollback_checkpoint(
                label,
                &mut state.agent_checkpoints,
                open.as_deref(),
            ) {
                Ok((msg, reload)) => {
                    if reload {
                        if let Some(p) = open {
                            let _ = state.open_file_path(p);
                        }
                    }
                    state.file_tree.rebuild();
                    AgentResponse::success("Rollback", msg)
                }
                Err(e) => AgentResponse::error("Rollback", e),
            }
        }
        AgentCommand::Lock { owner } => {
            match lock::acquire_lock(&state.agent_lock, owner) {
                Ok(msg) => {
                    state.agent_lock = Some(owner.clone());
                    AgentResponse::success("Lock", msg)
                }
                Err(e) => AgentResponse::error("Lock", e),
            }
        }
        AgentCommand::Unlock => {
            match lock::release_lock(&state.agent_lock, None) {
                Ok(msg) => {
                    state.agent_lock = None;
                    AgentResponse::success("Unlock", msg)
                }
                Err(e) => AgentResponse::error("Unlock", e),
            }
        }
    }
}

fn collect_recursive(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>, limit: usize) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if name.starts_with(".git") || name == "target" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
                if out.len() >= limit {
                    return;
                }
            }
        }
    }
}

fn cmd_label(cmd: &AgentCommand) -> &'static str {
    match cmd {
        AgentCommand::Read { .. } => "Read",
        AgentCommand::Write { .. } => "Write",
        AgentCommand::Edit { .. } => "Edit",
        AgentCommand::List { .. } => "List",
        AgentCommand::Mkdir { .. } => "Mkdir",
        AgentCommand::Touch { .. } => "Touch",
        AgentCommand::Mv { .. } => "Mv",
        AgentCommand::Cp { .. } => "Cp",
        AgentCommand::Rm { .. } => "Rm",
        AgentCommand::Run { .. } => "Run",
        AgentCommand::Stop => "Stop",
        AgentCommand::Status => "Status",
        AgentCommand::OpenBuffer { .. } => "OpenBuffer",
        AgentCommand::SaveBuffer => "SaveBuffer",
        AgentCommand::BufferInfo => "BufferInfo",
        AgentCommand::Checkpoint { .. } => "Checkpoint",
        AgentCommand::Rollback { .. } => "Rollback",
        AgentCommand::Lock { .. } => "Lock",
        AgentCommand::Unlock => "Unlock",
    }
}
