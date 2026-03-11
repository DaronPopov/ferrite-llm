use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::commands;
use crate::events::{LogCategory, LogEntry};
use crate::state::{DaemonState, MANAGED_APPS};
use crate::tui::state::TuiState;

pub(super) fn cmd_ps(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    commands::cleanup_exited_apps(daemon);
    let apps = daemon.apps.lock();
    if apps.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "no running processes"));
    } else {
        for app in apps.values() {
            let uptime = app.started_at.elapsed().as_secs();
            state.push_log(LogEntry::new(
                LogCategory::App,
                format!(
                    "  {} pid={} id={} up={}s",
                    app.name,
                    app.child.id(),
                    app.id,
                    uptime,
                ),
            ));
        }
    }
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("managed: {}", MANAGED_APPS.join(", ")),
    ));
}

pub(super) fn cmd_start(daemon: &Arc<DaemonState>, state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            "usage: start <app> [args...]",
        ));
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!("available: {}", MANAGED_APPS.join(", ")),
        ));
        return;
    }
    // Reuse the existing handle_app_start logic
    match commands::handle_app_start(daemon, args) {
        Ok(json) => {
            // Parse the JSON to give a clean TUI message
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                if v.get("ok").and_then(|v| v.as_bool()) == Some(true) {
                    state.push_log(LogEntry::new(
                        LogCategory::App,
                        format!(
                            "started {} pid={} id={}",
                            v["name"].as_str().unwrap_or("?"),
                            v["pid"].as_u64().unwrap_or(0),
                            v["id"].as_u64().unwrap_or(0),
                        ),
                    ));
                } else {
                    let msg = v["error"].as_str().unwrap_or("failed");
                    state.push_log(LogEntry::new(LogCategory::Err, msg.to_string()));
                }
            }
        }
        Err(e) => {
            state.push_log(LogEntry::new(LogCategory::Err, format!("start: {}", e)));
        }
    }
}

pub(super) fn cmd_stop(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let mut stopped_something = false;

    // 1. Stop stability test if running
    if state.stability_running.load(Ordering::Relaxed) {
        state.stability_running.store(false, Ordering::Relaxed);
        state.push_log(LogEntry::new(LogCategory::Sys, "stopped stability test"));
        stopped_something = true;
    }

    // 2. Cancel running script (same as /stop)
    if state.run_status != crate::tui::state::run_state::RunStatus::Idle {
        state.run_cancel_flag.store(true, Ordering::Relaxed);
        state.push_log(LogEntry::new(LogCategory::Sys, "stopped running script"));
        stopped_something = true;
    }

    // 3. Kill all managed apps
    {
        let mut apps = daemon.apps.lock();
        if !apps.is_empty() {
            let count = apps.len();
            for (_, app) in apps.iter_mut() {
                let _ = app.child.kill();
                let _ = app.child.wait();
            }
            apps.clear();
            state.push_log(LogEntry::new(
                LogCategory::App,
                format!("stopped {} managed app(s)", count),
            ));
            stopped_something = true;
        }
    }

    // 4. Stop all active durable jobs
    {
        let mut supervisor = daemon.supervisor.lock();
        let stopped_jobs = supervisor.stop_all("stopped by operator");
        if stopped_jobs > 0 {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("stopped {} durable job(s)", stopped_jobs),
            ));
            stopped_something = true;
        }
    }

    if !stopped_something {
        state.push_log(LogEntry::new(LogCategory::Sys, "nothing running"));
    }
}

pub(super) fn cmd_kill(daemon: &Arc<DaemonState>, state: &mut TuiState, args: &[&str]) {
    if args.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "usage: kill <id|name>"));
        return;
    }
    match commands::handle_app_stop(daemon, args) {
        Ok(json) => {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                if v.get("ok").and_then(|v| v.as_bool()) == Some(true) {
                    state.push_log(LogEntry::new(
                        LogCategory::App,
                        format!(
                            "stopped {} pid={}",
                            v["name"].as_str().unwrap_or("?"),
                            v["pid"].as_u64().unwrap_or(0),
                        ),
                    ));
                } else {
                    let msg = v["error"].as_str().unwrap_or("not found");
                    state.push_log(LogEntry::new(LogCategory::Err, msg.to_string()));
                }
            }
        }
        Err(e) => {
            state.push_log(LogEntry::new(LogCategory::Err, format!("kill: {}", e)));
        }
    }
}

pub(super) fn cmd_uptime(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let up = daemon.start_time.elapsed().as_secs();
    let total = daemon
        .total_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let failed = daemon
        .failed_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let clients = daemon
        .active_clients
        .load(std::sync::atomic::Ordering::Relaxed);
    let apps_count = daemon.apps.lock().len();

    let h = up / 3600;
    let m = (up % 3600) / 60;
    let s = up % 60;
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "up {}h {}m {}s  clients={} apps={} requests={} failed={}",
            h, m, s, clients, apps_count, total, failed,
        ),
    ));
}
