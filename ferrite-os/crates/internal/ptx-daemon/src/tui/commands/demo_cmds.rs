use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;
use crate::tui::demo::{
    find_demo, list_demos, run_bench, run_dataflow_proof, run_logwatch, run_program,
    run_stress_loop, run_stability_test, DEMO_PROGRAMS,
};
use crate::tui::state::{TuiState, UiMode};

pub(super) fn cmd_bench(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("bench: {} programs queued", DEMO_PROGRAMS.len()),
    ));

    let runner = Arc::clone(runner);
    let tx = tx.clone();
    std::thread::spawn(move || {
        run_bench(&runner, &tx);
    });
}

pub(super) fn cmd_demos(state: &mut TuiState) {
    list_demos(state);
}

pub(super) fn cmd_demo(
    daemon: &Arc<DaemonState>,
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
    args: &[&str],
) {
    if args.is_empty() {
        list_demos(state);
        return;
    }
    match args[0] {
        "stop" => {
            if state.stability_running.load(Ordering::Relaxed) {
                state.stability_running.store(false, Ordering::Relaxed);
                state.push_log(LogEntry::new(LogCategory::Sys, "stress demo stopping..."));
            } else {
                state.push_log(LogEntry::new(LogCategory::Sys, "no stress demo running"));
            }
        }
        "inspect" => {
            if args.len() < 2 {
                state.push_log(LogEntry::new(LogCategory::Sys, "usage: demo inspect <name>"));
                return;
            }
            match find_demo(args[1]) {
                Some((name, _desc, src)) => {
                    let demos_dir = state.workspace_root.join("demos");
                    if let Err(e) = std::fs::create_dir_all(&demos_dir) {
                        state.push_log(LogEntry::new(LogCategory::Err, format!("mkdir demos: {}", e)));
                        return;
                    }
                    let file_path = demos_dir.join(format!("{}.fgl", name));
                    if let Err(e) = std::fs::write(&file_path, src) {
                        state.push_log(LogEntry::new(LogCategory::Err, format!("write {}: {}", file_path.display(), e)));
                        return;
                    }
                    match state.open_file_path(file_path.clone()) {
                        Ok(()) => {
                            state.ui_mode = UiMode::Files;
                            state.push_log(LogEntry::new(LogCategory::Sys, format!("opened {}", file_path.display())));
                        }
                        Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
                    }
                }
                None => {
                    state.push_log(LogEntry::new(LogCategory::Err, format!("unknown demo '{}' — type 'demos' to list", args[1])));
                }
            }
        }
        stress @ ("vram-flood" | "mem-churn" | "pipeline-stress" | "extreme-load") => {
            if state.stability_running.load(Ordering::Relaxed) {
                state.push_log(LogEntry::new(LogCategory::Sys,
                    format!("{}: a stress demo is already running — 'demo stop' to halt", stress)));
                return;
            }
            let (name, _desc, src) = find_demo(stress).unwrap();
            state.stability_running.store(true, Ordering::Relaxed);
            state.push_log(LogEntry::new(LogCategory::Sys,
                format!("{}: launching continuous stress loop — 'demo stop' to halt", name)));
            let flag = Arc::clone(&state.stability_running);
            let runtime = Arc::clone(&daemon.runtime);
            let tx = tx.clone();
            let name = name.to_string();
            let src = src.to_string();
            std::thread::spawn(move || {
                run_stress_loop(&name, &src, &flag, &runtime, &tx);
            });
        }
        "stability" => {
            if state.stability_running.load(Ordering::Relaxed) {
                state.push_log(LogEntry::new(LogCategory::Sys, "stability test already running — 'demo stop' to halt"));
                return;
            }
            let n_streams = daemon.runtime.num_streams();
            state.stability_running.store(true, Ordering::Relaxed);
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("stability: launching — {} streams available, 'demo stop' to halt", n_streams),
            ));
            let flag = Arc::clone(&state.stability_running);
            let runtime = Arc::clone(&daemon.runtime);
            let tx = tx.clone();
            std::thread::spawn(move || {
                run_stability_test(&flag, &runtime, &tx);
            });
        }
        "gpu-logwatch" | "logwatch" => {
            if state.stability_running.load(Ordering::Relaxed) {
                state.push_log(LogEntry::new(LogCategory::Sys,
                    "gpu-logwatch: a long-running demo is already active — 'demo stop' to halt"));
                return;
            }
            state.stability_running.store(true, Ordering::Relaxed);
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "gpu-logwatch: launching GPU log stream monitor — 'demo stop' to halt",
            ));
            let flag = Arc::clone(&state.stability_running);
            let runtime = Arc::clone(&daemon.runtime);
            let tx = tx.clone();
            std::thread::spawn(move || {
                run_logwatch(&flag, &runtime, &tx);
            });
        }
        "dataflow-proof" | "dataflow" => {
            if state.stability_running.load(Ordering::Relaxed) {
                state.push_log(LogEntry::new(LogCategory::Sys,
                    "dataflow-proof: a long-running demo is already active — 'demo stop' to halt"));
                return;
            }
            state.stability_running.store(true, Ordering::Relaxed);
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "dataflow-proof: proving TPU-like deterministic memory on GPU — 'demo stop' to halt",
            ));
            let flag = Arc::clone(&state.stability_running);
            let runtime = Arc::clone(&daemon.runtime);
            let tx = tx.clone();
            std::thread::spawn(move || {
                run_dataflow_proof(&flag, &runtime, &tx);
            });
        }
        target => {
            match find_demo(target) {
                Some((name, _desc, src)) => {
                    state.push_log(LogEntry::new(LogCategory::Sys, format!("demo: running '{}'", name)));
                    let name = name.to_string();
                    let src = src.to_string();
                    let runner = Arc::clone(runner);
                    let tx = tx.clone();
                    std::thread::spawn(move || {
                        let mut r = runner.lock();
                        run_program(&name, &src, &mut r, &tx);
                    });
                }
                None => {
                    state.push_log(LogEntry::new(LogCategory::Err, format!("unknown demo '{}' — type 'demos' to list", target)));
                }
            }
        }
    }
}
