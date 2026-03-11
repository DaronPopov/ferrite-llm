use std::io;
use std::sync::{mpsc, Arc};

use ferrite_platform::ipc;
use tracing::info;

use crate::config::DaemonConfig;
use crate::event_stream::SchedulerEvent;
use crate::events::{DaemonEvent, LogCategory, LogEntry};

mod bootstrap;
mod client;
mod command_pipeline;
mod lifecycle;
mod transport;

pub use client::{connect_and_send, run_watch_client};

fn emit_diag(
    component: &'static str,
    status: &'static str,
    code: &'static str,
    summary: impl AsRef<str>,
    remediation: &'static str,
) {
    info!(
        component,
        status,
        code,
        summary = summary.as_ref(),
        remediation,
        "diagnostic"
    );
}

pub fn run_server(config: DaemonConfig) -> io::Result<()> {
    let use_tui = bootstrap::should_use_tui(&config);
    bootstrap::init_logging(&config, use_tui);

    info!("Starting Ferrite-OS daemon");
    emit_diag(
        "daemon.server",
        "PASS",
        "DMN-SRV-0001",
        "starting Ferrite-OS daemon",
        "none",
    );
    info!("Configuration: {:?}", config);

    let _pid_file = bootstrap::create_pid_file(&config)?;
    info!("PID file created: {}", config.pid_file);
    emit_diag(
        "daemon.server",
        "PASS",
        "DMN-SRV-0002",
        format!("pid file created at {}", config.pid_file),
        "none",
    );

    let runtime = bootstrap::init_runtime(&config)?;
    bootstrap::boot_kernel_if_requested(&config, &runtime);

    let listener = bootstrap::prepare_listener(&config)?;
    info!("Daemon listening on {}", config.socket_path);
    emit_diag(
        "daemon.server",
        "PASS",
        "DMN-SRV-0003",
        format!("daemon listening on {}", config.socket_path),
        "none",
    );

    let (state, runner) = match bootstrap::build_state(runtime, config.clone()) {
        Ok(state) => state,
        Err(e) => {
            ipc::remove_endpoint(&ipc::Endpoint::new(&config.socket_path));
            return Err(e);
        }
    };

    lifecycle::start_signal_handler(Arc::clone(&state));
    lifecycle::start_keepalive_thread(Arc::clone(&state));

    if use_tui {
        let (tx, rx) = mpsc::channel::<DaemonEvent>();

        // Bridge: forward SchedulerEvent → DaemonEvent so the TUI can render them.
        let scheduler_rx = state.event_stream.lock().subscribe();
        let tx_bridge = tx.clone();
        std::thread::spawn(move || {
            while let Ok(entry) = scheduler_rx.recv() {
                let daemon_evt = match entry.event {
                    SchedulerEvent::AppEvent { app_name, event_name, payload, .. } => {
                        DaemonEvent::AppEvent {
                            app_name,
                            message: format!("{}: {}", event_name, payload),
                        }
                    }
                    _ => continue,
                };
                if tx_bridge.send(daemon_evt).is_err() {
                    break;
                }
            }
        });

        let state_bg = Arc::clone(&state);
        let runner_bg = Arc::clone(&runner);
        let config_bg = config.clone();
        let tx_socket = tx.clone();
        std::thread::spawn(move || {
            transport::run_socket_listener(listener, state_bg, runner_bg, config_bg, Some(tx_socket));
        });

        tx.send(DaemonEvent::Log(LogEntry::new(
            LogCategory::Sys,
            "daemon ready",
        )))
        .ok();

        info!("Daemon ready (TUI mode)");
        emit_diag("daemon.server", "PASS", "DMN-SRV-0004", "daemon ready (TUI mode)", "none");
        crate::tui::run_tui(state.clone(), runner.clone(), rx, tx)?;
    } else {
        lifecycle::start_watch_thread(Arc::clone(&state));

        info!("Daemon ready (headless mode)");
        emit_diag(
            "daemon.server",
            "PASS",
            "DMN-SRV-0005",
            "daemon ready (headless mode)",
            "none",
        );
        transport::run_socket_listener(listener, state.clone(), runner, config.clone(), None);
    }

    let health = state.health_diagnostic();
    emit_diag(
        health.component,
        health.status,
        health.code,
        health.summary,
        health.remediation,
    );

    info!("Shutting down daemon");
    emit_diag("daemon.server", "PASS", "DMN-SRV-0006", "shutting down daemon", "none");
    lifecycle::shutdown_managed_apps(&state);

    ipc::remove_endpoint(&ipc::Endpoint::new(&config.socket_path));

    info!("Daemon stopped");
    emit_diag("daemon.server", "PASS", "DMN-SRV-0007", "daemon stopped", "none");
    Ok(())
}
