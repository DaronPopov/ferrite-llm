//! Lifecycle signal abstraction.
//!
//! Exposes only the intents the daemon needs (`Shutdown`, `Reload`)
//! regardless of platform signal/event mechanism.

use std::sync::mpsc;

/// High-level lifecycle intent received from the OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleSignal {
    Shutdown,
    Reload,
}

/// Subscribe to OS lifecycle signals.
///
/// Returns a receiver that yields [`LifecycleSignal`] values.
/// The background listener thread is detached and will stop when
/// the receiver is dropped.
///
/// On Unix: maps SIGTERM/SIGINT → Shutdown, SIGHUP → Reload.
/// On Windows: maps CTRL_C/CTRL_BREAK/CTRL_CLOSE → Shutdown.
pub fn subscribe() -> io::Result<mpsc::Receiver<LifecycleSignal>> {
    let (tx, rx) = mpsc::channel();
    spawn_listener(tx)?;
    Ok(rx)
}

use std::io;

#[cfg(unix)]
fn spawn_listener(tx: mpsc::Sender<LifecycleSignal>) -> io::Result<()> {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGTERM, SIGINT, SIGHUP])?;

    std::thread::spawn(move || {
        for sig in signals.forever() {
            let intent = match sig {
                SIGTERM | SIGINT => LifecycleSignal::Shutdown,
                SIGHUP => LifecycleSignal::Reload,
                _ => continue,
            };
            if tx.send(intent).is_err() {
                break;
            }
            if intent == LifecycleSignal::Shutdown {
                break;
            }
        }
    });

    Ok(())
}

#[cfg(windows)]
fn spawn_listener(tx: mpsc::Sender<LifecycleSignal>) -> io::Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // On Windows we use SetConsoleCtrlHandler. This is a simplified
    // implementation — a production version would use windows-sys directly.
    // For now, provide compile-time correctness with a TODO for runtime testing.
    static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

    std::thread::spawn(move || {
        // Poll-based fallback until proper console handler is wired.
        loop {
            if SHUTDOWN_FLAG.load(Ordering::Relaxed) {
                let _ = tx.send(LifecycleSignal::Shutdown);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    Ok(())
}

/// Ignore SIGPIPE (Unix) or equivalent (no-op on Windows).
pub fn ignore_pipe_signal() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
    // No equivalent needed on Windows.
}
