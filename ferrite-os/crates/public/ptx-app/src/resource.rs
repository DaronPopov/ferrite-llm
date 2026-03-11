//! Resource priority levels for FerApp jobs.

use serde::{Deserialize, Serialize};

/// Priority level for a FerApp job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Realtime,
}

impl Priority {
    /// Convert to the integer priority used by the daemon scheduler.
    pub fn as_i32(self) -> i32 {
        match self {
            Priority::Low => 0,
            Priority::Normal => 1,
            Priority::High => 2,
            Priority::Realtime => 3,
        }
    }
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}
