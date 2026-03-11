use std::io::{self, Read, Write};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use ferrite_platform::ipc::{IpcListener, IpcStream};
use tracing::{debug, error, trace, warn};

use crate::config::DaemonConfig;
use crate::events::DaemonEvent;
use crate::script_runner::ScriptRunner;
use crate::state::{ClientGuard, DaemonState};

use super::command_pipeline;

fn response_has_error(payload: &str) -> bool {
    for line in payload.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with('{') {
            return false;
        }
        return serde_json::from_str::<serde_json::Value>(trimmed)
            .ok()
            .and_then(|value| value.as_object().map(|obj| obj.contains_key("error")))
            .unwrap_or(false);
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestAccounting {
    failed_delta: u64,
    success: bool,
}

impl RequestAccounting {
    fn parse_error() -> Self {
        Self {
            failed_delta: 1,
            success: false,
        }
    }

    fn from_payload(payload: &str) -> Self {
        let success = !response_has_error(payload);
        Self {
            failed_delta: if success { 0 } else { 1 },
            success,
        }
    }

    fn io_error() -> Self {
        Self {
            failed_delta: 1,
            success: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestAccounting, response_has_error};

    #[test]
    fn response_has_error_true_when_json_contains_error_field() {
        assert!(response_has_error("{\"error\":\"boom\"}\n"));
    }

    #[test]
    fn response_has_error_false_when_json_without_error_field() {
        assert!(!response_has_error("{\"ok\":true}\n"));
    }

    #[test]
    fn response_has_error_false_for_non_json_payload() {
        assert!(!response_has_error("plain text response\n"));
    }

    #[test]
    fn response_has_error_ignores_leading_empty_lines() {
        assert!(response_has_error("\n\n{\"error\":\"bad\"}\n"));
    }

    #[test]
    fn accounting_parse_error_is_failed() {
        let acc = RequestAccounting::parse_error();
        assert_eq!(acc.failed_delta, 1);
        assert!(!acc.success);
    }

    #[test]
    fn accounting_success_payload_is_not_failed() {
        let acc = RequestAccounting::from_payload("{\"ok\":true}\n");
        assert_eq!(acc.failed_delta, 0);
        assert!(acc.success);
    }

    #[test]
    fn accounting_error_payload_is_failed() {
        let acc = RequestAccounting::from_payload("{\"error\":\"bad\"}\n");
        assert_eq!(acc.failed_delta, 1);
        assert!(!acc.success);
    }

    #[test]
    fn accounting_io_error_is_failed() {
        let acc = RequestAccounting::io_error();
        assert_eq!(acc.failed_delta, 1);
        assert!(!acc.success);
    }
}

pub(super) fn run_socket_listener(
    listener: IpcListener,
    state: Arc<DaemonState>,
    runner: Arc<parking_lot::Mutex<ScriptRunner>>,
    config: DaemonConfig,
    event_tx: Option<mpsc::Sender<DaemonEvent>>,
) {
    if let Err(e) = listener.set_nonblocking(true) {
        error!(error = %e, "Cannot set listener non-blocking");
        return;
    }

    while state.is_running() {
        match listener.accept() {
            Ok(stream) => {
                let active = state.active_clients.load(Ordering::Relaxed);
                if active >= config.max_clients as u64 {
                    warn!(
                        "Max clients reached ({}), rejecting connection",
                        config.max_clients
                    );
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    continue;
                }

                let state_clone = Arc::clone(&state);
                let runner_clone = Arc::clone(&runner);
                let tx_clone = event_tx.clone();
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, state_clone, runner_clone, tx_clone) {
                        debug!("Client error: {}", e);
                    }
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }
}

fn handle_client(
    mut stream: IpcStream,
    state: Arc<DaemonState>,
    _runner: Arc<parking_lot::Mutex<ScriptRunner>>,
    event_tx: Option<mpsc::Sender<DaemonEvent>>,
) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(state.config.client_timeout_secs)))?;
    stream.set_nonblocking(false)?;

    state.active_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard::new(&state);

    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    let cmdline = buf.trim();
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let parsed = match command_pipeline::parse_command_line(cmdline) {
        Ok(parsed) => parsed,
        Err(message) => {
            let response = serde_json::json!({ "error": message });
            stream.write_all(format!("{}\n", response).as_bytes())?;
            let accounting = RequestAccounting::parse_error();
            state
                .failed_requests
                .fetch_add(accounting.failed_delta, Ordering::Relaxed);
            if let Some(ref tx) = event_tx {
                tx.send(DaemonEvent::ClientHandled {
                    command: "invalid-command".to_string(),
                    success: accounting.success,
                })
                .ok();
            }
            return Ok(());
        }
    };

    trace!(command = %parsed.command, args = ?parsed.args, "Handling client request");

    // Agent commands are routed through the TUI event channel for
    // synchronous execution on the main thread.
    if parsed.command == "agent" {
        if let Some(ref tx) = event_tx {
            let json_str = parsed.args.join(" ");
            let response_payload = match serde_json::from_str::<crate::tui::agent::protocol::AgentCommand>(&json_str) {
                Ok(cmd) => {
                    let (reply_tx, reply_rx) = mpsc::channel();
                    tx.send(DaemonEvent::AgentCommandSync(cmd, reply_tx)).ok();
                    match reply_rx.recv_timeout(Duration::from_secs(30)) {
                        Ok(resp) => {
                            let json = serde_json::to_string(&resp).unwrap_or_else(|_| {
                                "{\"error\":\"serialize failed\"}".to_string()
                            });
                            format!("{json}\n")
                        }
                        Err(_) => "{\"error\":\"timeout waiting for TUI\"}\n".to_string(),
                    }
                }
                Err(e) => format!("{{\"error\":\"invalid agent JSON: {}\"}}\n", e),
            };

            stream.write_all(response_payload.as_bytes())?;
            let accounting = RequestAccounting::from_payload(&response_payload);
            if accounting.failed_delta > 0 {
                state
                    .failed_requests
                    .fetch_add(accounting.failed_delta, Ordering::Relaxed);
            }

            if let Some(ref tx2) = event_tx {
                tx2.send(DaemonEvent::ClientHandled {
                    command: parsed.command,
                    success: accounting.success,
                })
                .ok();
            }
            return Ok(());
        }
    }

    let success = match command_pipeline::execute_command(&state, &parsed) {
        Ok(msg) => {
            stream.write_all(msg.as_bytes())?;
            let accounting = RequestAccounting::from_payload(&msg);
            if accounting.failed_delta > 0 {
                state
                    .failed_requests
                    .fetch_add(accounting.failed_delta, Ordering::Relaxed);
            }
            accounting.success
        }
        Err(e) => {
            let err_msg = format!("{{\"error\":\"{}\"}}\n", e);
            stream.write_all(err_msg.as_bytes())?;
            let accounting = RequestAccounting::io_error();
            state
                .failed_requests
                .fetch_add(accounting.failed_delta, Ordering::Relaxed);
            accounting.success
        }
    };

    if let Some(tx) = event_tx {
        tx.send(DaemonEvent::ClientHandled {
            command: parsed.command,
            success,
        })
        .ok();
    }

    Ok(())
}
