use crate::config::SinkConfig;
use crate::model::Event;
use crate::queue::BoundedQueue;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::Shutdown;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferrite_platform::ipc::{Endpoint, IpcStream};

/// Discover the daemon socket path using the same logic as ferrite-daemon.
pub fn discover_socket() -> String {
    if let Ok(s) = std::env::var("FERRITE_SOCKET") {
        return s;
    }
    if let Ok(s) = std::env::var("FERRITE_DAEMON_SOCKET") {
        // Backward-compatible alias used by some clients.
        return s;
    }
    ferrite_platform::paths::default_socket_addr()
}

/// Send a command to the daemon via IPC and return the response.
pub fn ipc_send(socket_path: &str, command: &str) -> anyhow::Result<String> {
    let endpoint = Endpoint::new(socket_path);
    let mut stream = IpcStream::connect(&endpoint)?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.shutdown(Shutdown::Write)?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp)?;
    Ok(resp)
}

/// Run the sink loop: drain events from the queue, write JSONL, optionally ping daemon.
pub fn run_sink(
    config: &SinkConfig,
    queue: Arc<BoundedQueue<Event>>,
    shutdown: Arc<AtomicBool>,
) {
    let socket_path = config
        .socket_path
        .clone()
        .unwrap_or_else(discover_socket);

    let mut output_file: Option<File> = config.output_file.as_ref().map(|path| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("failed to open output file {}: {}", path, e))
    });

    let batch_size = config.batch_size.max(1);
    let flush_interval = Duration::from_millis(config.flush_interval_ms);
    let mut last_ipc_check = Instant::now();
    let ipc_check_interval = Duration::from_secs(30);

    tracing::info!(
        socket = %socket_path,
        output = ?config.output_file,
        ipc_enabled = config.ipc_enabled,
        "sink thread started"
    );

    // Initial liveness check
    if config.ipc_enabled {
        match ipc_send(&socket_path, &config.command_template) {
            Ok(resp) => tracing::info!(response = %resp.trim(), "daemon liveness OK"),
            Err(e) => tracing::warn!(error = %e, "daemon not reachable (will continue writing JSONL)"),
        }
    }

    while !shutdown.load(Ordering::Relaxed) {
        let batch = queue.drain_batch(batch_size);

        if batch.is_empty() {
            std::thread::sleep(flush_interval.min(Duration::from_millis(100)));
            // Periodic IPC liveness check even when idle
            if config.ipc_enabled && last_ipc_check.elapsed() >= ipc_check_interval {
                do_ipc_check(&socket_path, &config.command_template);
                last_ipc_check = Instant::now();
            }
            continue;
        }

        // Write events to JSONL
        if let Some(ref mut file) = output_file {
            for event in &batch {
                match serde_json::to_string(event) {
                    Ok(line) => {
                        if let Err(e) = writeln!(file, "{}", line) {
                            tracing::error!(error = %e, "failed to write event to JSONL");
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "failed to serialize event");
                    }
                }
            }
            if let Err(e) = file.flush() {
                tracing::error!(error = %e, "failed to flush JSONL output");
            }
        }

        tracing::debug!(count = batch.len(), "wrote event batch");

        // Periodic IPC liveness check
        if config.ipc_enabled && last_ipc_check.elapsed() >= ipc_check_interval {
            do_ipc_check(&socket_path, &config.command_template);
            last_ipc_check = Instant::now();
        }
    }

    // Drain remaining events on shutdown
    tracing::info!("sink draining remaining events");
    let remaining = queue.drain_batch(usize::MAX);
    if let Some(ref mut file) = output_file {
        for event in &remaining {
            if let Ok(line) = serde_json::to_string(event) {
                let _ = writeln!(file, "{}", line);
            }
        }
        let _ = file.flush();
    }
    tracing::info!(drained = remaining.len(), "sink shutdown complete");
}

fn do_ipc_check(socket_path: &str, command: &str) {
    match ipc_send(socket_path, command) {
        Ok(resp) => tracing::debug!(response = %resp.trim(), "daemon liveness OK"),
        Err(e) => tracing::warn!(error = %e, "daemon IPC check failed"),
    }
}
