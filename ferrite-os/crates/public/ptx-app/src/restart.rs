//! Restart policy wrapper for FerApp.

use ptx_runtime::job::{BackoffConfig, RestartPolicy};

/// Restart policy for a FerApp application.
///
/// Wraps the runtime's `RestartPolicy` with ergonomic constructors.
#[derive(Debug, Clone)]
pub struct Restart {
    /// The underlying runtime restart policy.
    ///
    /// Read by `FerApp::run()` when serializing job config to the daemon.
    #[allow(dead_code)]
    pub(crate) policy: RestartPolicy,
}

impl Restart {
    /// Never restart on failure (default).
    pub fn never() -> Self {
        Self {
            policy: RestartPolicy::Never,
        }
    }

    /// Restart on failure up to `max_retries` times with default backoff.
    pub fn on_failure(max_retries: u32) -> Self {
        Self {
            policy: RestartPolicy::OnFailure {
                max_retries,
                backoff: BackoffConfig::default(),
            },
        }
    }

    /// Restart on failure with custom backoff parameters.
    pub fn on_failure_with_backoff(max_retries: u32, initial_ms: u64, max_ms: u64) -> Self {
        Self {
            policy: RestartPolicy::OnFailure {
                max_retries,
                backoff: BackoffConfig {
                    initial_delay_ms: initial_ms,
                    max_delay_ms: max_ms,
                    multiplier: 2.0,
                },
            },
        }
    }

    /// Always restart (even on success) up to `max_retries` times.
    pub fn always(max_retries: u32) -> Self {
        Self {
            policy: RestartPolicy::Always {
                max_retries,
                backoff: BackoffConfig::default(),
            },
        }
    }
}

impl Default for Restart {
    fn default() -> Self {
        Self::never()
    }
}
