// Ferrite-OS Daemon - Production-Grade GPU Runtime Daemon
//
// Features:
// - Signal handling (SIGTERM, SIGINT, SIGHUP)
// - Structured logging with tracing
// - Graceful shutdown
// - Concurrent client handling
// - PID file management
// - Health monitoring
// - Configuration file support
// - TUI dashboard (ratatui + crossterm)

mod bootstrap;
mod commands;
mod config;
mod event_stream;
mod events;
mod job_store;
mod lifecycle;
mod pid;
mod policy;
mod scheduler_commands;
mod script_runner;
mod server;
mod state;
mod supervisor;
mod tui;

pub fn run_daemon() {
    ferrite_platform::signals::ignore_pipe_signal();

    let invocation = match bootstrap::parse_cli() {
        Ok(v) => v,
        Err(code) => {
            if code == 2 {
                bootstrap::print_usage();
            }
            std::process::exit(code);
        }
    };

    if let Err(code) = lifecycle::dispatch(invocation) {
        if code == 2 {
            bootstrap::print_usage();
        }
        std::process::exit(code);
    }
}
