use crate::model::Event;
use crate::normalize::normalize;
use crate::queue::BoundedQueue;
use crate::retry::retry_with_backoff;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tungstenite::Message;

pub fn spawn(
    name: String,
    url: String,
    reconnect_ms: u64,
    max_retries: u32,
    queue: Arc<BoundedQueue<Event>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("src-ws-{}", name))
        .spawn(move || {
            tracing::info!(source = %name, url = %url, "WS source started");
            while !shutdown.load(Ordering::Relaxed) {
                match connect_and_read(&name, &url, max_retries, &queue, &shutdown) {
                    Ok(()) => {
                        tracing::info!(source = %name, "WS connection closed cleanly");
                    }
                    Err(e) => {
                        tracing::error!(source = %name, error = %e, "WS connection error");
                    }
                }
                if !shutdown.load(Ordering::Relaxed) {
                    tracing::info!(source = %name, reconnect_ms, "reconnecting after delay");
                    let mut remaining = reconnect_ms;
                    while remaining > 0 && !shutdown.load(Ordering::Relaxed) {
                        let sleep = remaining.min(200);
                        thread::sleep(Duration::from_millis(sleep));
                        remaining = remaining.saturating_sub(sleep);
                    }
                }
            }
            tracing::info!(source = %name, "WS source stopped");
        })
        .expect("failed to spawn WS source thread")
}

fn connect_and_read(
    name: &str,
    url: &str,
    max_retries: u32,
    queue: &Arc<BoundedQueue<Event>>,
    shutdown: &AtomicBool,
) -> anyhow::Result<()> {
    let (mut socket, _response) = retry_with_backoff(max_retries, 500, || {
        tungstenite::connect(url).map_err(|e| anyhow::anyhow!("ws connect: {}", e))
    })?;

    tracing::info!(source = %name, "WS connected");

    loop {
        if shutdown.load(Ordering::Relaxed) {
            let _ = socket.close(None);
            break;
        }

        match socket.read() {
            Ok(Message::Text(text)) => {
                let payload: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => serde_json::Value::String(text),
                };
                let event = normalize(name, "ws_message", payload);
                if queue.push(event) {
                    tracing::warn!(source = %name, "queue full, dropped oldest event");
                }
            }
            Ok(Message::Binary(data)) => {
                let payload = match serde_json::from_slice(&data) {
                    Ok(v) => v,
                    Err(_) => serde_json::json!({ "binary_len": data.len() }),
                };
                let event = normalize(name, "ws_binary", payload);
                if queue.push(event) {
                    tracing::warn!(source = %name, "queue full, dropped oldest event");
                }
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) => {
                tracing::info!(source = %name, "WS server sent close");
                break;
            }
            Ok(Message::Frame(_)) => continue,
            Err(tungstenite::Error::ConnectionClosed) => break,
            Err(e) => return Err(anyhow::anyhow!("ws read: {}", e)),
        }
    }

    Ok(())
}
