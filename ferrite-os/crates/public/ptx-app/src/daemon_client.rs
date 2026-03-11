//! IPC client for communicating with the Ferrite daemon.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Duration;

use ferrite_platform::ipc::{Endpoint, IpcStream};

use crate::error::AppError;

fn current_default_socket_path() -> String {
    ferrite_platform::paths::default_socket_addr()
}

/// Client for sending commands to the Ferrite daemon over IPC.
pub(crate) struct DaemonClient {
    endpoint: Endpoint,
}

impl DaemonClient {
    /// Create a client targeting the given socket/pipe path.
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            endpoint: Endpoint::new(socket_path),
        }
    }

    /// Auto-detect the daemon endpoint.
    ///
    /// Checks `FERRITE_SOCKET`, then `FERRITE_DAEMON_SOCKET`, then current defaults.
    /// Keeps legacy path as a fallback when present.
    pub fn auto_detect() -> Self {
        if let Ok(path) = std::env::var("FERRITE_SOCKET") {
            return Self::new(path);
        }
        if let Ok(path) = std::env::var("FERRITE_DAEMON_SOCKET") {
            return Self::new(path);
        }

        let current = current_default_socket_path();
        let legacy = ferrite_platform::paths::legacy_socket_path();
        if Path::new(&current).exists() || !Path::new(legacy).exists() {
            return Self::new(current);
        }

        Self::new(legacy)
    }

    /// Check if the daemon endpoint exists (does not guarantee connectivity).
    pub fn socket_exists(&self) -> bool {
        ferrite_platform::ipc::endpoint_exists(&self.endpoint)
    }

    /// Send a command to the daemon and return the response.
    pub fn send_command(&self, command: &str) -> Result<String, AppError> {
        let mut stream = IpcStream::connect(&self.endpoint).map_err(|e| {
            AppError::DaemonUnavailable {
                message: format!(
                    "cannot connect to {}: {}",
                    self.endpoint, e
                ),
            }
        })?;

        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .ok();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .ok();

        stream
            .write_all(command.as_bytes())
            .map_err(|e| AppError::DaemonUnavailable {
                message: format!("write failed: {}", e),
            })?;
        stream
            .write_all(b"\n")
            .map_err(|e| AppError::DaemonUnavailable {
                message: format!("write newline failed: {}", e),
            })?;
        stream.flush().map_err(|e| AppError::DaemonUnavailable {
            message: format!("flush failed: {}", e),
        })?;

        // Shutdown the write side so the daemon sees EOF.
        stream.shutdown(std::net::Shutdown::Write).ok();

        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .map_err(|e| AppError::DaemonUnavailable {
                message: format!("read failed: {}", e),
            })?;

        Ok(response)
    }

    /// Send a structured command envelope to preserve argument boundaries.
    pub fn send_command_parts(&self, command: &str, args: &[String]) -> Result<String, AppError> {
        let envelope = serde_json::json!({
            "command": command,
            "args": args,
        });
        self.send_command(&envelope.to_string())
    }

    /// Submit a job to the daemon and return the job ID.
    pub fn submit_job(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<u64, AppError> {
        let mut payload_args = vec![command.to_string()];
        payload_args.extend(args.iter().cloned());
        let response = self.send_command_parts("job-submit", &payload_args)?;
        let parsed: serde_json::Value =
            serde_json::from_str(response.trim()).map_err(|e| AppError::DaemonUnavailable {
                message: format!("invalid daemon response: {}", e),
            })?;

        if let Some(id) = parsed.get("job_id").and_then(|v| v.as_u64()) {
            Ok(id)
        } else if let Some(err) = parsed.get("error").and_then(|v| v.as_str()) {
            Err(AppError::DaemonUnavailable {
                message: err.to_string(),
            })
        } else {
            Err(AppError::DaemonUnavailable {
                message: format!("unexpected response: {}", response),
            })
        }
    }
}
