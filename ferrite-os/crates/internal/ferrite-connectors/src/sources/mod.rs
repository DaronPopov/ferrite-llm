pub mod rest;
pub mod ws;

use crate::config::SourceConfig;
use crate::model::Event;
use crate::queue::BoundedQueue;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;

pub fn spawn_source(
    config: &SourceConfig,
    queue: Arc<BoundedQueue<Event>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    match config {
        SourceConfig::Rest {
            name,
            url,
            interval_ms,
            headers,
            jq_path,
            max_retries,
        } => rest::spawn(
            name.clone(),
            url.clone(),
            *interval_ms,
            headers.clone(),
            jq_path.clone(),
            *max_retries,
            queue,
            shutdown,
        ),
        SourceConfig::Ws {
            name,
            url,
            reconnect_ms,
            max_retries,
        } => ws::spawn(
            name.clone(),
            url.clone(),
            *reconnect_ms,
            *max_retries,
            queue,
            shutdown,
        ),
    }
}
