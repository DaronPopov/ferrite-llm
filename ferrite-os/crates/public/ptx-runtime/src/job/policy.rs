//! Restart policies for durable jobs.
//!
//! Determines whether a failed job should be restarted and, if so, the delay
//! before the next attempt.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::job::failure::{FailureClass, FailureClassifier};

/// Configuration for exponential backoff between restart attempts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    /// Initial delay in milliseconds before the first retry.
    pub initial_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Multiplicative factor applied to the delay on each subsequent attempt.
    pub multiplier: f64,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 1000,
            max_delay_ms: 30_000,
            multiplier: 2.0,
        }
    }
}

impl BackoffConfig {
    /// Compute the delay for a given attempt number (1-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(self.initial_delay_ms);
        }
        let base = self.initial_delay_ms as f64;
        let exponential = base * self.multiplier.powi(attempt as i32 - 1);
        let capped = exponential.min(self.max_delay_ms as f64);
        Duration::from_millis(capped as u64)
    }
}

/// Policy that governs whether and how a job is restarted after failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartPolicy {
    /// Never restart the job regardless of exit status.
    Never,
    /// Restart only when the job fails (non-zero exit).
    OnFailure {
        max_retries: u32,
        backoff: BackoffConfig,
    },
    /// Restart on any termination (success or failure) up to a limit.
    Always {
        max_retries: u32,
        backoff: BackoffConfig,
    },
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self::Never
    }
}

/// The outcome of evaluating a restart policy against a failure.
#[derive(Debug, Clone)]
pub enum RestartDecision {
    /// The job should be restarted after the specified delay.
    Restart { delay: Duration },
    /// The job should not be restarted; give up with the stated reason.
    GiveUp { reason: String },
}

impl RestartPolicy {
    /// Evaluate whether a restart should occur given the failure count and the
    /// error that caused the failure.
    ///
    /// `failure_count` is the number of failures that have already occurred
    /// (i.e., the count *before* deciding whether to retry again).
    pub fn evaluate(&self, failure_count: u32, error: &Error) -> RestartDecision {
        match self {
            RestartPolicy::Never => RestartDecision::GiveUp {
                reason: "restart policy is Never".into(),
            },

            RestartPolicy::OnFailure {
                max_retries,
                backoff,
            } => {
                // Classify the failure -- permanent errors are never retried.
                let class = FailureClassifier::classify(error);
                if class == FailureClass::Permanent {
                    return RestartDecision::GiveUp {
                        reason: format!(
                            "permanent failure classified for error: {}",
                            error
                        ),
                    };
                }

                if failure_count >= *max_retries {
                    return RestartDecision::GiveUp {
                        reason: format!(
                            "reached max retries ({}/{})",
                            failure_count, max_retries
                        ),
                    };
                }

                let delay = backoff.delay_for_attempt(failure_count);
                tracing::info!(
                    failure_count,
                    max_retries,
                    delay_ms = delay.as_millis(),
                    "restart decision: retry"
                );
                RestartDecision::Restart { delay }
            }

            RestartPolicy::Always {
                max_retries,
                backoff,
            } => {
                if failure_count >= *max_retries {
                    return RestartDecision::GiveUp {
                        reason: format!(
                            "reached max retries ({}/{})",
                            failure_count, max_retries
                        ),
                    };
                }

                let delay = backoff.delay_for_attempt(failure_count);
                tracing::info!(
                    failure_count,
                    max_retries,
                    delay_ms = delay.as_millis(),
                    "restart decision: retry (always policy)"
                );
                RestartDecision::Restart { delay }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transient_error() -> Error {
        Error::CudaError {
            code: 2,
            message: "out of memory".into(),
        }
    }

    fn permanent_error() -> Error {
        Error::InitFailed { device_id: 0 }
    }

    #[test]
    fn never_policy_always_gives_up() {
        let policy = RestartPolicy::Never;
        match policy.evaluate(0, &transient_error()) {
            RestartDecision::GiveUp { .. } => {}
            RestartDecision::Restart { .. } => panic!("Never policy should not restart"),
        }
    }

    #[test]
    fn on_failure_retries_transient() {
        let policy = RestartPolicy::OnFailure {
            max_retries: 3,
            backoff: BackoffConfig::default(),
        };
        match policy.evaluate(0, &transient_error()) {
            RestartDecision::Restart { delay } => {
                assert!(delay.as_millis() > 0);
            }
            RestartDecision::GiveUp { reason } => {
                panic!("Should retry transient: {}", reason)
            }
        }
    }

    #[test]
    fn on_failure_gives_up_on_permanent() {
        let policy = RestartPolicy::OnFailure {
            max_retries: 3,
            backoff: BackoffConfig::default(),
        };
        match policy.evaluate(0, &permanent_error()) {
            RestartDecision::GiveUp { .. } => {}
            RestartDecision::Restart { .. } => {
                panic!("Should not retry permanent error")
            }
        }
    }

    #[test]
    fn on_failure_gives_up_at_max() {
        let policy = RestartPolicy::OnFailure {
            max_retries: 3,
            backoff: BackoffConfig::default(),
        };
        match policy.evaluate(3, &transient_error()) {
            RestartDecision::GiveUp { .. } => {}
            RestartDecision::Restart { .. } => {
                panic!("Should give up at max retries")
            }
        }
    }

    #[test]
    fn always_policy_retries_even_permanent() {
        let policy = RestartPolicy::Always {
            max_retries: 5,
            backoff: BackoffConfig::default(),
        };
        match policy.evaluate(0, &permanent_error()) {
            RestartDecision::Restart { .. } => {}
            RestartDecision::GiveUp { reason } => {
                panic!("Always should retry: {}", reason)
            }
        }
    }

    #[test]
    fn backoff_increases() {
        let cfg = BackoffConfig {
            initial_delay_ms: 100,
            max_delay_ms: 10_000,
            multiplier: 2.0,
        };
        let d1 = cfg.delay_for_attempt(1);
        let d2 = cfg.delay_for_attempt(2);
        let d3 = cfg.delay_for_attempt(3);
        assert!(d2 > d1);
        assert!(d3 > d2);
    }

    #[test]
    fn backoff_caps_at_max() {
        let cfg = BackoffConfig {
            initial_delay_ms: 1000,
            max_delay_ms: 5000,
            multiplier: 10.0,
        };
        let d5 = cfg.delay_for_attempt(5);
        assert!(d5.as_millis() <= 5000);
    }
}
