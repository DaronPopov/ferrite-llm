use crate::model::Event;
use crate::normalize::normalize;
use crate::queue::BoundedQueue;
use crate::retry::retry_with_backoff;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub fn spawn(
    name: String,
    url: String,
    interval_ms: u64,
    headers: HashMap<String, String>,
    jq_path: Option<String>,
    max_retries: u32,
    queue: Arc<BoundedQueue<Event>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("src-rest-{}", name))
        .spawn(move || {
            tracing::info!(source = %name, url = %url, "REST source started");
            while !shutdown.load(Ordering::Relaxed) {
                match poll_once(&name, &url, &headers, &jq_path, max_retries) {
                    Ok(events) => {
                        for event in events {
                            if queue.push(event) {
                                tracing::warn!(source = %name, "queue full, dropped oldest event");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(source = %name, error = %e, "REST poll failed after retries");
                    }
                }
                // Sleep in small increments so we can respond to shutdown
                let mut remaining = interval_ms;
                while remaining > 0 && !shutdown.load(Ordering::Relaxed) {
                    let sleep = remaining.min(200);
                    thread::sleep(Duration::from_millis(sleep));
                    remaining = remaining.saturating_sub(sleep);
                }
            }
            tracing::info!(source = %name, "REST source stopped");
        })
        .expect("failed to spawn REST source thread")
}

fn poll_once(
    name: &str,
    url: &str,
    headers: &HashMap<String, String>,
    jq_path: &Option<String>,
    max_retries: u32,
) -> anyhow::Result<Vec<Event>> {
    let url = url.to_string();
    let headers = headers.clone();

    let body: serde_json::Value = retry_with_backoff(max_retries, 500, || {
        let mut req = ureq::get(&url);
        for (k, v) in &headers {
            req = req.set(k, v);
        }
        let resp = req.call().map_err(|e| anyhow::anyhow!("{}", e))?;
        let body = resp
            .into_string()
            .map_err(|e| anyhow::anyhow!("read body: {}", e))?;
        let val: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| anyhow::anyhow!("json parse: {}", e))?;
        Ok::<_, anyhow::Error>(val)
    })?;

    let payload = if let Some(ref path) = jq_path {
        body.pointer(path).cloned().unwrap_or(body)
    } else {
        body
    };

    // If the extracted payload is an array, emit one event per element
    let events = if let serde_json::Value::Array(items) = payload {
        items
            .into_iter()
            .map(|item| normalize(name, "rest_poll", item))
            .collect()
    } else {
        vec![normalize(name, "rest_poll", payload)]
    };

    Ok(events)
}
