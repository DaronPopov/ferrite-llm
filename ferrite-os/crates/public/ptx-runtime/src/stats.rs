//! Runtime statistics and operation counting.

use std::sync::atomic::{AtomicU64, Ordering};
use crate::telemetry::{DiagnosticEvent, DiagnosticStatus};

/// Global operation counter for tracking tensor operations.
static OPS_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Increment the operation counter.
#[inline]
pub fn increment_ops() {
    OPS_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Increment the operation counter by a specific amount.
#[inline]
pub fn increment_ops_by(n: u64) {
    OPS_COUNTER.fetch_add(n, Ordering::Relaxed);
}

/// Get the current operation count.
#[inline]
pub fn get_ops_count() -> u64 {
    OPS_COUNTER.load(Ordering::Relaxed)
}

/// Reset the operation counter to zero.
#[inline]
pub fn reset_ops_count() {
    OPS_COUNTER.store(0, Ordering::Relaxed);
}

/// Emit a health diagnostic based on whether operations are flowing.
pub fn ops_health_diagnostic() -> DiagnosticEvent {
    let ops = get_ops_count();
    if ops == 0 {
        DiagnosticEvent::new(
            "runtime.stats",
            DiagnosticStatus::WARN,
            "RT-STATS-0001",
            "operation counter is zero",
            "verify workload dispatch path if this is unexpected",
        )
    } else {
        DiagnosticEvent::new(
            "runtime.stats",
            DiagnosticStatus::PASS,
            "RT-STATS-0002",
            format!("operation counter active: {}", ops),
            "none",
        )
    }
}
