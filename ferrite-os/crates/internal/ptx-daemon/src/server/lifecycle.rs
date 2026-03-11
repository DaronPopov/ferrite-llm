use std::io::{self, Write};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ferrite_platform::signals::LifecycleSignal;
use tracing::{info, warn};

use crate::state::DaemonState;

pub(super) fn start_signal_handler(state: Arc<DaemonState>) {
    thread::spawn(move || {
        let rx = match ferrite_platform::signals::subscribe() {
            Ok(rx) => rx,
            Err(e) => {
                warn!(error = %e, "Failed to register signal handlers");
                return;
            }
        };

        for signal in rx.iter() {
            match signal {
                LifecycleSignal::Shutdown => {
                    info!("Received shutdown signal, initiating shutdown");
                    state.shutdown();
                    break;
                }
                LifecycleSignal::Reload => {
                    info!("Received reload signal, reloading configuration not yet implemented");
                }
            }
        }
    });
}

pub(super) fn start_keepalive_thread(state: Arc<DaemonState>) {
    let keepalive_ms = state.config.keepalive_ms;
    thread::spawn(move || {
        info!("Keepalive thread started (interval: {}ms)", keepalive_ms);
        while state.is_running() {
            state.runtime.keepalive();
            thread::sleep(Duration::from_millis(keepalive_ms));
        }
        info!("Keepalive thread stopped");
    });
}

pub(super) fn start_watch_thread(state: Arc<DaemonState>) {
    if !state.config.watch_enabled {
        return;
    }

    let watch_ms = state.config.watch_ms;
    let is_tty = ferrite_platform::tty::stdout_is_tty();

    thread::spawn(move || {
        info!("Watch thread started (interval: {}ms)", watch_ms);
        while state.is_running() {
            let tlsf = state.runtime.tlsf_stats();
            let stats = state.runtime.stats();
            let active = state.active_clients.load(Ordering::Relaxed);

            let line = format!(
                "util={:.1}% frag={:.4}% vram={}MB streams={} clients={}",
                tlsf.utilization_percent,
                tlsf.fragmentation_ratio * 100.0,
                stats.vram_used / (1024 * 1024),
                stats.active_streams,
                active
            );

            if is_tty {
                print!("\r{}", line);
                let _ = io::stdout().flush();
            } else {
                println!("{}", line);
            }

            thread::sleep(Duration::from_millis(watch_ms));
        }
        if is_tty {
            println!();
        }
        info!("Watch thread stopped");
    });
}

pub(super) fn shutdown_managed_apps(state: &Arc<DaemonState>) {
    let mut apps = state.apps.lock();
    for (_, app) in apps.iter_mut() {
        if let Err(e) = app.child.kill() {
            warn!(
                app_id = app.id,
                app = %app.name,
                error = %e,
                "Failed to kill managed app"
            );
            continue;
        }
        if let Err(e) = app.child.wait() {
            warn!(
                app_id = app.id,
                app = %app.name,
                error = %e,
                "Failed waiting managed app"
            );
        }
    }
    apps.clear();
}
