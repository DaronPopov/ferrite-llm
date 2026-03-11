//! Telemetry and observability for Ferrite-OS
//!
//! This module provides structured logging, metrics, and tracing for production deployments.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticStatus {
    PASS,
    WARN,
    FAIL,
}

impl DiagnosticStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PASS => "PASS",
            Self::WARN => "WARN",
            Self::FAIL => "FAIL",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticEvent {
    pub component: &'static str,
    pub status: DiagnosticStatus,
    pub code: &'static str,
    pub summary: String,
    pub remediation: String,
}

impl DiagnosticEvent {
    pub fn new(
        component: &'static str,
        status: DiagnosticStatus,
        code: &'static str,
        summary: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            component,
            status,
            code,
            summary: summary.into(),
            remediation: remediation.into(),
        }
    }
}

pub fn emit_diag(diag: &DiagnosticEvent) {
    match diag.status {
        DiagnosticStatus::PASS => tracing::info!(
            component = diag.component,
            status = diag.status.as_str(),
            code = diag.code,
            summary = %diag.summary,
            remediation = %diag.remediation,
            "diagnostic"
        ),
        DiagnosticStatus::WARN => tracing::warn!(
            component = diag.component,
            status = diag.status.as_str(),
            code = diag.code,
            summary = %diag.summary,
            remediation = %diag.remediation,
            "diagnostic"
        ),
        DiagnosticStatus::FAIL => tracing::error!(
            component = diag.component,
            status = diag.status.as_str(),
            code = diag.code,
            summary = %diag.summary,
            remediation = %diag.remediation,
            "diagnostic"
        ),
    }
}

/// Global metrics counters
pub struct Metrics {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub allocation_failures: AtomicU64,
    pub total_bytes_allocated: AtomicU64,
    pub total_bytes_freed: AtomicU64,
    pub kernel_launches: AtomicU64,
    pub stream_synchronizations: AtomicU64,
}

impl Metrics {
    pub const fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
            total_bytes_allocated: AtomicU64::new(0),
            total_bytes_freed: AtomicU64::new(0),
            kernel_launches: AtomicU64::new(0),
            stream_synchronizations: AtomicU64::new(0),
        }
    }

    pub fn record_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated.fetch_add(size as u64, Ordering::Relaxed);
    }

    pub fn record_deallocation(&self, size: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_freed.fetch_add(size as u64, Ordering::Relaxed);
    }

    pub fn record_allocation_failure(&self) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.allocation_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_kernel_launch(&self) {
        self.kernel_launches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_stream_sync(&self) {
        self.stream_synchronizations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes_allocated.load(Ordering::Relaxed),
            total_bytes_freed: self.total_bytes_freed.load(Ordering::Relaxed),
            kernel_launches: self.kernel_launches.load(Ordering::Relaxed),
            stream_synchronizations: self.stream_synchronizations.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub allocations: u64,
    pub deallocations: u64,
    pub allocation_failures: u64,
    pub total_bytes_allocated: u64,
    pub total_bytes_freed: u64,
    pub kernel_launches: u64,
    pub stream_synchronizations: u64,
}

impl MetricsSnapshot {
    pub fn net_allocated_bytes(&self) -> i64 {
        self.total_bytes_allocated as i64 - self.total_bytes_freed as i64
    }

    pub fn allocation_success_rate(&self) -> f64 {
        if self.allocations == 0 {
            1.0
        } else {
            1.0 - (self.allocation_failures as f64 / self.allocations as f64)
        }
    }
}

/// Global metrics instance
static GLOBAL_METRICS: Metrics = Metrics::new();

pub fn metrics() -> &'static Metrics {
    &GLOBAL_METRICS
}

/// Performance timer for operations
pub struct OpTimer {
    operation: &'static str,
    start: Instant,
}

impl OpTimer {
    pub fn new(operation: &'static str) -> Self {
        tracing::trace!(operation, "started");
        Self {
            operation,
            start: Instant::now(),
        }
    }
}

impl Drop for OpTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        tracing::trace!(
            operation = self.operation,
            duration_us = elapsed.as_micros(),
            "completed"
        );
    }
}

/// Initialize telemetry subsystem
pub fn init() {
    use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    // Check if already initialized
    if tracing::dispatcher::has_been_set() {
        return;
    }

    // Default to INFO level, can be overridden with RUST_LOG
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .compact();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    emit_diag(&DiagnosticEvent::new(
        "runtime.telemetry",
        DiagnosticStatus::PASS,
        "RT-TELEM-0001",
        "Ferrite-OS telemetry initialized",
        "none",
    ));
}

/// Initialize telemetry with file logging
pub fn init_with_file(log_dir: &str) -> std::io::Result<()> {
    use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
    use tracing_appender::rolling::{RollingFileAppender, Rotation};

    // Check if already initialized
    if tracing::dispatcher::has_been_set() {
        return Ok(());
    }

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Console output
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_level(true)
        .compact();

    // File output (daily rotation)
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, "ferrite-os.log");
    let file_layer = fmt::layer()
        .with_writer(file_appender)
        .with_ansi(false)
        .json();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(file_layer)
        .init();

    emit_diag(&DiagnosticEvent::new(
        "runtime.telemetry",
        DiagnosticStatus::PASS,
        "RT-TELEM-0002",
        format!("Ferrite-OS telemetry initialized with file logging at {}", log_dir),
        "none",
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics() {
        let m = Metrics::new();
        m.record_allocation(1024);
        m.record_allocation(2048);
        m.record_deallocation(1024);

        let snapshot = m.snapshot();
        assert_eq!(snapshot.allocations, 2);
        assert_eq!(snapshot.deallocations, 1);
        assert_eq!(snapshot.total_bytes_allocated, 3072);
        assert_eq!(snapshot.total_bytes_freed, 1024);
        assert_eq!(snapshot.net_allocated_bytes(), 2048);
    }

    #[test]
    fn test_success_rate() {
        let m = Metrics::new();
        m.record_allocation(100);
        m.record_allocation(100);
        m.record_allocation(100);
        m.record_allocation_failure();

        let snapshot = m.snapshot();
        assert_eq!(snapshot.allocation_success_rate(), 0.75);
    }
}
