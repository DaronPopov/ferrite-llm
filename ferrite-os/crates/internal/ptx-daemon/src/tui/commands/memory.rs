use std::sync::Arc;

use crate::events::{LogCategory, LogEntry};
use crate::state::DaemonState;
use crate::tui::state::TuiState;

pub(super) fn cmd_pool(daemon: &Arc<DaemonState>, state: &mut TuiState, args: &[&str]) {
    let tlsf = daemon.runtime.tlsf_stats();

    if args.first() == Some(&"check") {
        // Deep validation — use severity-appropriate categories so
        // problems are immediately visible in the log.
        let report = daemon.runtime.validate_pool();
        let summary_cat = if report.is_valid {
            LogCategory::Sys
        } else {
            LogCategory::Err
        };
        state.push_log(LogEntry::new(
            summary_cat,
            format!(
                "pool {}  leaks {}  corrupted {}  broken_chains {}  hash_err {}",
                if report.is_valid { "valid" } else { "INVALID" },
                if report.has_memory_leaks { "YES" } else { "no" },
                if report.has_corrupted_blocks { "YES" } else { "no" },
                if report.has_broken_chains { "YES" } else { "no" },
                if report.has_hash_errors { "YES" } else { "no" },
            ),
        ));
        if report.error_count > 0 {
            for i in 0..report.error_count.min(16) as usize {
                let msg =
                    unsafe { std::ffi::CStr::from_ptr(report.error_messages[i].as_ptr()) };
                if let Ok(s) = msg.to_str() {
                    if !s.is_empty() {
                        state.push_log(LogEntry::new(LogCategory::Err, format!("  {}", s)));
                    }
                }
            }
        }
    } else {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!(
                "total: {:.1}MB  used: {:.1}MB  free: {:.1}MB",
                tlsf.total_pool_size as f64 / (1024.0 * 1024.0),
                tlsf.allocated_bytes as f64 / (1024.0 * 1024.0),
                tlsf.free_bytes as f64 / (1024.0 * 1024.0),
            ),
        ));
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!(
                "util: {:.1}%  frag: {:.1}%  peak: {:.1}MB  largest_free: {:.1}MB  healthy={}",
                tlsf.utilization_percent,
                tlsf.fragmentation_ratio * 100.0,
                tlsf.peak_allocated as f64 / (1024.0 * 1024.0),
                tlsf.largest_free_block as f64 / (1024.0 * 1024.0),
                tlsf.is_healthy,
            ),
        ));
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!(
                "blocks: {} (alloc={} free={})  allocs: {} frees: {}  healthy={}",
                tlsf.total_blocks,
                tlsf.allocated_blocks,
                tlsf.free_blocks,
                tlsf.total_allocations,
                tlsf.total_frees,
                tlsf.is_healthy,
            ),
        ));
    }
}

pub(super) fn cmd_defrag(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let tlsf_before = daemon.runtime.tlsf_stats();
    daemon.runtime.defragment();
    let tlsf_after = daemon.runtime.tlsf_stats();
    let freed = if tlsf_after.fragmentation_ratio < tlsf_before.fragmentation_ratio {
        format!(
            "frag {:.4}% -> {:.4}%",
            tlsf_before.fragmentation_ratio * 100.0,
            tlsf_after.fragmentation_ratio * 100.0,
        )
    } else {
        "no change".to_string()
    };
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("defragment complete  {}", freed),
    ));
}
