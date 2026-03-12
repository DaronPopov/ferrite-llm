use std::time::Instant;

/// Monotonic clock for sensor timestamps.
///
/// Provides a shared time base for `Stamped<T>` timestamps. Used by
/// `CaptureThread` and any custom sensor implementation.
pub struct SensorClock {
    epoch: Instant,
}

impl SensorClock {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// Microseconds elapsed since this clock was created.
    pub fn now_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// Milliseconds elapsed since this clock was created (floating point).
    pub fn elapsed_ms(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }
}

impl Default for SensorClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_monotonic() {
        let clock = SensorClock::new();
        let t1 = clock.now_us();
        // Small busy spin to ensure time passes
        for _ in 0..1000 { std::hint::black_box(0); }
        let t2 = clock.now_us();
        assert!(t2 >= t1);
    }

    #[test]
    fn clock_elapsed_ms() {
        let clock = SensorClock::new();
        let ms = clock.elapsed_ms();
        assert!(ms >= 0.0);
    }
}
