use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::stream::{SensorStream, Stamped};
use crate::pipeline::SharedRing;

/// Background I/O thread that reads from a `SensorStream` into a shared ring buffer.
///
/// Decouples sensor I/O from pipeline compute by running the blocking read loop
/// in a dedicated thread. The pipeline consumes samples via `try_recv()` or
/// `recv_timeout()`.
pub struct CaptureThread<T: Send + 'static> {
    handle: Option<JoinHandle<()>>,
    ring: Arc<SharedRing<Stamped<T>>>,
    stop: Arc<AtomicBool>,
}

impl<T: Send + 'static> CaptureThread<T> {
    /// Spawn a background thread that reads from `sensor` into a ring buffer.
    pub fn spawn<S: SensorStream<Sample = T> + Send + 'static>(
        mut sensor: S,
        capacity: usize,
    ) -> Self {
        let ring = Arc::new(SharedRing::new(capacity));
        let stop = Arc::new(AtomicBool::new(false));

        let ring_clone = Arc::clone(&ring);
        let stop_clone = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            eprintln!("[sensor] capture thread started: {}", sensor.info().name);
            while !stop_clone.load(Ordering::Relaxed) && sensor.is_open() {
                match sensor.read() {
                    Ok(sample) => {
                        ring_clone.push(sample);
                    }
                    Err(e) => {
                        eprintln!("[sensor] read error: {}", e);
                        // On device lost, exit the loop
                        if matches!(e, super::stream::SensorError::DeviceLost) {
                            break;
                        }
                    }
                }
            }
            eprintln!("[sensor] capture thread stopped");
        });

        Self {
            handle: Some(handle),
            ring,
            stop,
        }
    }

    /// Non-blocking: get the latest sample (or None if ring is empty).
    pub fn try_recv(&self) -> Option<Stamped<T>> {
        self.ring.pop()
    }

    /// Blocking: poll for next sample with timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<Stamped<T>> {
        let start = std::time::Instant::now();
        loop {
            if let Some(sample) = self.ring.pop() {
                return Some(sample);
            }
            if start.elapsed() >= timeout {
                return None;
            }
            thread::sleep(Duration::from_micros(100));
        }
    }

    /// Graceful shutdown: signal the thread and wait for it to finish.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl<T: Send + 'static> Drop for CaptureThread<T> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Don't join on drop — the thread will notice the stop flag
    }
}
