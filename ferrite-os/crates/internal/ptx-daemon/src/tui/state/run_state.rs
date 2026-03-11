use std::time::Instant;

use crate::events::LogCategory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Idle,
    Compiling,
    Running,
    Succeeded,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunProfile {
    Debug,
    Release,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub profile: RunProfile,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    #[allow(dead_code)]
    pub features: Vec<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            profile: RunProfile::Debug,
            args: Vec::new(),
            timeout_secs: 60,
            features: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunOutputLine {
    pub timestamp: Instant,
    pub severity: LogCategory,
    pub text: String,
}

impl RunOutputLine {
    pub fn new(severity: LogCategory, text: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            severity,
            text: text.into(),
        }
    }
}
