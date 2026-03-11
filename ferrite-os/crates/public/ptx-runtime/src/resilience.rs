//! Error recovery and resilience for Ferrite-OS
//!
//! This module provides retry logic, circuit breakers, and graceful degradation
//! for production deployments.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use crate::telemetry::{emit_diag, DiagnosticEvent, DiagnosticStatus};

/// Retry policy for transient failures
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Exponential backoff with jitter
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let base_delay = self.initial_delay.as_millis() as f64;
        let exponential = base_delay * self.backoff_multiplier.powi(attempt as i32 - 1);
        let capped = exponential.min(self.max_delay.as_millis() as f64);

        // Add jitter (±20%)
        let jitter = 0.8 + (rand::random::<f64>() * 0.4);
        let final_delay = (capped * jitter) as u64;

        Duration::from_millis(final_delay)
    }

    /// Execute function with retries
    pub fn execute<T, E, F>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Display,
    {
        let mut last_error = None;

        for attempt in 0..self.max_attempts {
            if attempt > 0 {
                let delay = self.delay_for_attempt(attempt);
                tracing::debug!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Retrying after delay"
                );
                std::thread::sleep(delay);
            }

            match f() {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(attempt, "Retry succeeded");
                    }
                    return Ok(result);
                }
                Err(e) => {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_attempts = self.max_attempts,
                        error = %e,
                        "Operation failed"
                    );
                    emit_diag(&DiagnosticEvent::new(
                        "runtime.resilience",
                        DiagnosticStatus::WARN,
                        "RT-RES-0001",
                        format!(
                            "retryable operation failed on attempt {}/{}: {}",
                            attempt + 1,
                            self.max_attempts,
                            e
                        ),
                        "operation will retry with backoff; verify persistent failures",
                    ));
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.expect("RetryPolicy::execute called with max_attempts=0"))
    }
}

/// Circuit breaker to prevent cascading failures
pub struct CircuitBreaker {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,

    failures: AtomicU32,
    successes: AtomicU32,
    state: AtomicU32, // 0=Closed, 1=Open, 2=HalfOpen
    last_failure_time: parking_lot::Mutex<Option<Instant>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject requests
    HalfOpen, // Testing if recovered
}

impl From<u32> for CircuitState {
    fn from(value: u32) -> Self {
        match value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32, timeout: Duration) -> Self {
        Self {
            failure_threshold,
            success_threshold,
            timeout,
            failures: AtomicU32::new(0),
            successes: AtomicU32::new(0),
            state: AtomicU32::new(0), // Closed
            last_failure_time: parking_lot::Mutex::new(None),
        }
    }

    pub fn state(&self) -> CircuitState {
        self.state.load(Ordering::Acquire).into()
    }

    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Release);

        let state = self.state();
        if state == CircuitState::HalfOpen {
            let successes = self.successes.fetch_add(1, Ordering::AcqRel) + 1;
            if successes >= self.success_threshold {
                tracing::info!("Circuit breaker closed after recovery");
                self.state.store(0, Ordering::Release); // Closed
                self.successes.store(0, Ordering::Release);
            }
        }
    }

    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::AcqRel) + 1;

        if failures >= self.failure_threshold {
            let state = self.state();
            if state == CircuitState::Closed || state == CircuitState::HalfOpen {
                tracing::warn!(
                    failures,
                    threshold = self.failure_threshold,
                    "Circuit breaker opened"
                );
                emit_diag(&DiagnosticEvent::new(
                    "runtime.resilience",
                    DiagnosticStatus::WARN,
                    "RT-RES-0002",
                    format!(
                        "circuit breaker opened at failure count {} (threshold {})",
                        failures, self.failure_threshold
                    ),
                    "investigate upstream operation failures before load increases",
                ));
                self.state.store(1, Ordering::Release); // Open
                *self.last_failure_time.lock() = Some(Instant::now());
                self.successes.store(0, Ordering::Release);
            }
        }
    }

    pub fn allow_request(&self) -> bool {
        let state = self.state();

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout expired
                let elapsed = self.last_failure_time.lock()
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::MAX);

                if elapsed >= self.timeout {
                    tracing::info!("Circuit breaker entering half-open state");
                    emit_diag(&DiagnosticEvent::new(
                        "runtime.resilience",
                        DiagnosticStatus::PASS,
                        "RT-RES-0003",
                        "circuit breaker entering half-open state",
                        "none",
                    ));
                    self.state.store(2, Ordering::Release); // HalfOpen
                    self.failures.store(0, Ordering::Release);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Execute function with circuit breaker protection
    pub fn execute<T, E, F>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if !self.allow_request() {
            return Err(CircuitBreakerError::CircuitOpen);
        }

        match f() {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitBreakerError::OperationFailed(e))
            }
        }
    }
}

#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    CircuitOpen,
    OperationFailed(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CircuitOpen => write!(f, "Circuit breaker is open"),
            Self::OperationFailed(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::error::Error> std::error::Error for CircuitBreakerError<E> {}

/// Rate limiter to prevent resource exhaustion
pub struct RateLimiter {
    capacity: u64,
    tokens: AtomicU64,
    last_refill: parking_lot::Mutex<Instant>,
    refill_rate: u64, // tokens per second
}

impl RateLimiter {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            tokens: AtomicU64::new(capacity),
            last_refill: parking_lot::Mutex::new(Instant::now()),
            refill_rate,
        }
    }

    pub fn try_acquire(&self, tokens: u64) -> bool {
        self.refill();

        let current = self.tokens.load(Ordering::Acquire);
        if current >= tokens {
            self.tokens.fetch_sub(tokens, Ordering::AcqRel);
            true
        } else {
            tracing::debug!(
                requested = tokens,
                available = current,
                "Rate limit exceeded"
            );
            false
        }
    }

    fn refill(&self) {
        let mut last_refill = self.last_refill.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);

        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64) as u64;
        if tokens_to_add > 0 {
            let current = self.tokens.load(Ordering::Acquire);
            let new_tokens = (current + tokens_to_add).min(self.capacity);
            self.tokens.store(new_tokens, Ordering::Release);
            *last_refill = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_delays() {
        let policy = RetryPolicy::default();

        let delay0 = policy.delay_for_attempt(0);
        assert_eq!(delay0, Duration::ZERO);

        let delay1 = policy.delay_for_attempt(1);
        assert!(delay1 >= Duration::from_millis(8)); // 10ms with -20% jitter
        assert!(delay1 <= Duration::from_millis(12)); // 10ms with +20% jitter
    }

    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::new(3, 2, Duration::from_millis(100));

        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());

        // Record failures
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record successes to close
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, 10);

        // Should allow 10 tokens
        for _ in 0..10 {
            assert!(limiter.try_acquire(1));
        }

        // Should deny 11th
        assert!(!limiter.try_acquire(1));

        // Wait and refill
        std::thread::sleep(Duration::from_millis(1100));
        assert!(limiter.try_acquire(1));
    }
}
