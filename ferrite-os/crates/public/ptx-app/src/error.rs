//! Error types for ptx-app.

use std::fmt;

/// Errors that can occur within a FerApp application.
#[derive(Debug)]
pub enum AppError {
    /// An error from the underlying PTX runtime.
    Runtime(ptx_runtime::Error),
    /// The daemon socket is unavailable or unreachable.
    DaemonUnavailable { message: String },
    /// Builder parameter validation failed.
    ValidationError { message: String },
    /// A policy rule denied the requested action.
    PolicyDenied { action: String, reason: String },
    /// Checkpoint save or restore failed.
    CheckpointError { detail: String },
    /// Generic application error (user-facing).
    App { message: String },
    /// The user closure panicked.
    Panic { message: String },
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Runtime(e) => write!(f, "runtime error: {}", e),
            AppError::DaemonUnavailable { message } => {
                write!(f, "daemon unavailable: {}", message)
            }
            AppError::ValidationError { message } => {
                write!(f, "validation error: {}", message)
            }
            AppError::PolicyDenied { action, reason } => {
                write!(f, "policy denied '{}': {}", action, reason)
            }
            AppError::CheckpointError { detail } => {
                write!(f, "checkpoint error: {}", detail)
            }
            AppError::App { message } => write!(f, "app error: {}", message),
            AppError::Panic { message } => write!(f, "panic: {}", message),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::Runtime(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ptx_runtime::Error> for AppError {
    fn from(e: ptx_runtime::Error) -> Self {
        AppError::Runtime(e)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::CheckpointError {
            detail: e.to_string(),
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::App {
            message: e.to_string(),
        }
    }
}
