mod demo_cmds;
mod gpu;
mod help;
pub mod inspect;
mod memory;
mod process;
pub mod run;
mod workspace;

use std::sync::mpsc::Sender;
use std::sync::Arc;

use super::state::TuiState;
use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;

/// Execute a shell command typed at the prompt.
pub(super) fn exec_command(
    line: &str,
    daemon: &Arc<DaemonState>,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    state: &mut TuiState,
    tx: &Sender<DaemonEvent>,
) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return;
    }

    let cmd = parts[0];
    let args = &parts[1..];

    match cmd {
        // ── lifecycle ────────────────────────────────────────────
        "quit" | "exit" | "shutdown" => {
            daemon.shutdown();
        }
        "help" | "?" => help::cmd_help(state),
        "binds" => help::cmd_binds(state),
        "clear" | "cls" => {
            state.log.clear();
        }

        // ── process management ───────────────────────────────────
        "ps" | "procs" | "apps" => process::cmd_ps(daemon, state),
        "start" | "launch" => process::cmd_start(daemon, state, args),
        "stop" => process::cmd_stop(daemon, state),
        "kill" => process::cmd_kill(daemon, state, args),
        "uptime" => process::cmd_uptime(daemon, state),

        // ── memory ───────────────────────────────────────────────
        "pool" | "status" => memory::cmd_pool(daemon, state, args),
        "defrag" | "defragment" => memory::cmd_defrag(daemon, state),

        // ── gpu / runtime ────────────────────────────────────────
        "stats" => gpu::cmd_stats(daemon, state),
        "hwpoll" => gpu::cmd_hwpoll(daemon, state),
        "streams" => gpu::cmd_streams(daemon, state),
        "snapshot" | "snap" => gpu::cmd_snapshot(daemon, state),
        "health" => gpu::cmd_health(daemon, state),

        // ── benchmark ─────────────────────────────────────────────
        "bench" | "benchmark" => demo_cmds::cmd_bench(state, runner, tx),

        // ── demos ─────────────────────────────────────────────────
        "demos" => demo_cmds::cmd_demos(state),
        "demo" => demo_cmds::cmd_demo(daemon, state, runner, tx, args),

        // ── misc ─────────────────────────────────────────────────
        "ping" => {
            state.push_log(LogEntry::new(LogCategory::Sys, "pong"));
        }
        "metrics" => workspace::cmd_metrics(daemon, state),
        "/sysmon" | "sysmon" => workspace::cmd_sysmon(state, args),
        "/detail" | "detail" => workspace::cmd_detail(state),
        "/density" | "density" => workspace::cmd_density(state, args),
        "/fxscript" | "fxscript" => workspace::cmd_fxscript(state, args),
        "/plot3d" | "plot3d" => workspace::cmd_plot3d(state, args),
        "/files" | "files" => workspace::cmd_files(state),
        "/open" | "open" => workspace::cmd_open(state, args),
        "/save" | "save" => workspace::cmd_save(state),

        // ── filesystem (Plan A) ──────────────────────────────────
        "mkdir" => workspace::cmd_mkdir(state, args),
        "touch" => workspace::cmd_touch(state, args),
        "mv" => workspace::cmd_mv(state, args),
        "cp" => workspace::cmd_cp(state, args),
        "rm" => workspace::cmd_rm(state, args),
        "ls" => workspace::cmd_ls(state, args),
        "cd" => workspace::cmd_cd(state, args),
        "pwd" => workspace::cmd_pwd(state),
        "confirm" | "y" => workspace::cmd_confirm(state),
        "cancel" | "n" => workspace::cmd_cancel(state),

        // ── run (Plan B) ─────────────────────────────────────────
        "run" => run::cmd_run(state, runner, tx),
        "rerun" => run::cmd_rerun(state, runner, tx),
        "/stop" => run::cmd_stop_run(state),
        "args" => run::cmd_args(state, args),
        "profile" => run::cmd_profile(state, args),
        "timeout" => run::cmd_timeout(state, args),

        // ── run-file / run-entry / run-list (daemon runner) ──
        "run-file" => run::cmd_run_file(state, daemon, tx, args),
        "run-entry" => run::cmd_run_entry(state, daemon, tx, args),
        "run-list" => run::cmd_run_list(state, daemon, tx),
        "/output" | "output" => run::cmd_toggle_output(state),

        // ── inspect / profiling ──────────────────────────────────
        "ptx" | "inspect" => inspect::cmd_ptx(state, runner),
        "perf" | "profiling" => inspect::cmd_perf(state),
        "jit-clear" => inspect::cmd_jit_clear(state, runner),
        "jit-cache" => inspect::cmd_jit_cache(state, runner),
        "quota" | "quotas" => inspect::cmd_quota(state, args),

        // ── agent (Plan C) ───────────────────────────────────────
        "audit" => workspace::cmd_audit(state),

        // ── scheduler / control plane (Plan-C) ──────────────────
        "scheduler" | "/scheduler" => cmd_scheduler(state, daemon, args),
        "/policy" => cmd_policy(state, daemon),
        "audit-query" | "/audit-query" => cmd_audit_query(state, daemon, args),

        _ => {
            state.push_log(LogEntry::new(
                LogCategory::Err,
                format!("unknown: '{}' — type 'help'", cmd),
            ));
        }
    }
}

// ── scheduler TUI commands ──────────────────────────────────────

fn cmd_scheduler(state: &mut TuiState, daemon: &Arc<DaemonState>, args: &[&str]) {
    use std::sync::atomic::Ordering;

    if args.is_empty() {
        // Switch to the scheduler TUI panel
        state.ui_mode = super::state::UiMode::Scheduler;
        let paused = daemon.scheduler_paused.load(Ordering::Relaxed);
        let engine = daemon.policy_engine.lock();
        state.push_log(LogEntry::new(LogCategory::Jit, "── scheduler dashboard ──"));
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!(
                "  queue: {}  rules: {}  audit entries: {}",
                if paused { "PAUSED" } else { "ACTIVE" },
                engine.rule_count(),
                engine.audit_log().len(),
            ),
        ));
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            "  subcommands: pause, resume, stats, policies, queue-status",
        ));
        return;
    }

    match args[0] {
        "pause" => {
            daemon.scheduler_paused.store(true, Ordering::Relaxed);
            state.scheduler_paused = true;
            state.push_log(LogEntry::new(LogCategory::Sys, "scheduler queue PAUSED"));
        }
        "resume" => {
            daemon.scheduler_paused.store(false, Ordering::Relaxed);
            state.scheduler_paused = false;
            state.push_log(LogEntry::new(LogCategory::Sys, "scheduler queue RESUMED"));
        }
        "stats" => {
            let engine = daemon.policy_engine.lock();
            let paused = daemon.scheduler_paused.load(Ordering::Relaxed);
            state.push_log(LogEntry::new(LogCategory::Jit, "── scheduler stats ──"));
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "  paused: {}  rules: {}  audit entries: {}  policy decisions: {}",
                    paused,
                    engine.rule_count(),
                    engine.audit_log().len(),
                    state.scheduler_policy_decisions.len(),
                ),
            ));
        }
        "policies" | "rules" => {
            let engine = daemon.policy_engine.lock();
            let rules = engine.list_rules();
            if rules.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no policy rules configured"));
            } else {
                state.push_log(LogEntry::new(LogCategory::Jit, "── active policy rules ──"));
                for (name, desc) in rules {
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("  {} — {}", name, desc),
                    ));
                }
            }
        }
        "queue-status" | "queue" | "qs" => {
            let paused = daemon.scheduler_paused.load(Ordering::Relaxed);
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "scheduler queue: {}  jobs: 0  depth: 0",
                    if paused { "PAUSED" } else { "ACTIVE" },
                ),
            ));
        }
        sub => {
            state.push_log(LogEntry::new(
                LogCategory::Err,
                format!("unknown scheduler subcommand '{}' — try: scheduler [pause|resume|stats|policies|queue]", sub),
            ));
        }
    }
}

fn cmd_policy(state: &mut TuiState, daemon: &Arc<DaemonState>) {
    let engine = daemon.policy_engine.lock();
    state.push_log(LogEntry::new(LogCategory::Jit, "── policy status ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("  rules: {}  audit entries: {}", engine.rule_count(), engine.audit_log().len()),
    ));

    let recent = engine.audit_log().query_recent(10);
    if recent.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "  no recent decisions"));
    } else {
        state.push_log(LogEntry::new(LogCategory::Jit, "  ── recent decisions ──"));
        for entry in recent {
            let marker = if entry.decision == "Allow" { "ALLOW" } else { "DENY" };
            let reason = entry.reason.as_deref().unwrap_or("");
            state.push_log(LogEntry::new(
                if entry.decision == "Allow" { LogCategory::Sys } else { LogCategory::Err },
                format!(
                    "  {:.1}s  {} {} t:{} {}",
                    entry.elapsed_secs, marker, entry.action, entry.tenant_id, reason,
                ),
            ));
        }
    }
}

fn cmd_audit_query(state: &mut TuiState, daemon: &Arc<DaemonState>, args: &[&str]) {
    let engine = daemon.policy_engine.lock();
    let mut tenant_filter: Option<u64> = None;
    let mut limit: usize = 20;

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
                    limit = args[i + 1].parse().unwrap_or(20);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                // Try parsing as a limit number
                if let Ok(n) = args[i].parse::<usize>() {
                    limit = n;
                }
                i += 1;
            }
        }
    }

    let entries = if let Some(tid) = tenant_filter {
        engine.audit_log().query_by_tenant(tid, limit)
    } else {
        engine.audit_log().query_recent(limit)
    };

    if entries.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "no audit entries found"));
    } else {
        let filter_str = if let Some(tid) = tenant_filter {
            format!(" (tenant {})", tid)
        } else {
            String::new()
        };
        state.push_log(LogEntry::new(
            LogCategory::Jit,
            format!("── audit log{} ({} entries) ──", filter_str, entries.len()),
        ));
        for entry in entries {
            let marker = if entry.decision == "Allow" { "ALLOW" } else { "DENY" };
            let reason = entry.reason.as_deref().unwrap_or("");
            state.push_log(LogEntry::new(
                if entry.decision == "Allow" { LogCategory::Sys } else { LogCategory::Err },
                format!(
                    "  {:.1}s  {} {} -> {} t:{} {}",
                    entry.elapsed_secs, marker, entry.action, entry.resource, entry.tenant_id, reason,
                ),
            ));
        }
    }
}

pub(crate) fn push_plot3d_tensor(state: &mut TuiState) {
    workspace::plot3d_push_latest_tensor_if_running(state);
}
