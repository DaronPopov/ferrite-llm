use ferrite_connectors::config::load_config;
use ferrite_connectors::http::router::{Response, Router};
use ferrite_connectors::http::server::HttpServer;
use ferrite_connectors::model::Event;
use ferrite_connectors::queue::BoundedQueue;
use ferrite_connectors::sink::ferrite_ipc;
use ferrite_connectors::sources;
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));

    tracing::info!(config = %config_path.display(), "loading configuration");
    let config = load_config(&config_path)?;

    tracing::info!(
        sources = config.sources.len(),
        queue_capacity = config.queue_capacity,
        "starting ferrite-connectors"
    );

    let queue: Arc<BoundedQueue<Event>> = Arc::new(BoundedQueue::new(config.queue_capacity));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Signal handling
    let shutdown_sig = shutdown.clone();
    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    thread::Builder::new()
        .name("signal-handler".into())
        .spawn(move || {
            for sig in signals.forever() {
                tracing::info!(signal = sig, "received shutdown signal");
                shutdown_sig.store(true, Ordering::SeqCst);
                break;
            }
        })?;

    // Spawn source threads
    let mut handles = Vec::new();
    for source_config in &config.sources {
        tracing::info!(source = source_config.name(), "spawning source");
        let handle = sources::spawn_source(source_config, queue.clone(), shutdown.clone());
        handles.push(handle);
    }

    // Optionally spawn HTTP server
    let mut http_server: Option<HttpServer> = None;
    if let Some(ref server_config) = config.server {
        let bind = format!("{}:{}", server_config.bind, server_config.port);
        let router = Arc::new(
            Router::new()
                .get("/health", |_, _| Response::ok("ok"))
                .get("/status", |_, _| {
                    Response::json(&serde_json::json!({"status": "running"}))
                }),
        );
        let q = if server_config.feed_queue {
            Some(queue.clone())
        } else {
            None
        };
        match HttpServer::start(
            &bind,
            router,
            q,
            server_config.max_connections,
            shutdown.clone(),
        ) {
            Ok(server) => {
                tracing::info!(addr = %server.addr(), "HTTP server listening");
                http_server = Some(server);
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to start HTTP server");
            }
        }
    }

    // Run sink on main thread
    ferrite_ipc::run_sink(&config.sink, queue.clone(), shutdown.clone());

    // Stop HTTP server if running
    if let Some(mut server) = http_server {
        server.stop();
    }

    // Wait for source threads to finish
    tracing::info!("waiting for source threads to finish");
    let drain_deadline =
        std::time::Instant::now() + Duration::from_millis(config.shutdown_drain_timeout_ms);
    for handle in handles {
        let remaining = drain_deadline
            .checked_duration_since(std::time::Instant::now())
            .unwrap_or(Duration::from_millis(100));
        // We can't timeout thread joins in std, so just join
        let _ = handle.join();
        if std::time::Instant::now() > drain_deadline {
            tracing::warn!("shutdown drain timeout exceeded");
            break;
        }
        let _ = remaining; // used for deadline tracking
    }

    tracing::info!("ferrite-connectors shut down cleanly");
    Ok(())
}
