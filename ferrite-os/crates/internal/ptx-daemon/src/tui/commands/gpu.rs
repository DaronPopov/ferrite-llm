use std::sync::Arc;

use crate::events::{LogCategory, LogEntry};
use crate::state::DaemonState;
use crate::tui::state::TuiState;

pub(super) fn cmd_stats(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let s = daemon.runtime.stats();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "vram {}MB  sm {}%  mem {}%  streams {}  kernels {}",
            s.vram_used / (1024 * 1024),
            s.gpu_utilization as i32,
            s.mem_utilization as i32,
            s.active_streams,
            s.registered_kernels,
        ),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "ops {}  latency {:.1}us  hw {}  flops {} ({:.2} GFLOPS)",
            s.total_ops,
            s.avg_latency_us,
            if s.nvml_valid { "nvml" } else { "fallback" },
            if s.cupti_valid { "cupti" } else { "na" },
            s.gflops_total,
        ),
    ));
    if s.nvml_valid {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!(
                "clock sm {}MHz  mem {}MHz  power {:.1}W  temp {}C",
                s.sm_clock_mhz, s.mem_clock_mhz, s.power_w, s.temperature_c
            ),
        ));
    }
}

pub(super) fn cmd_hwpoll(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let s = daemon.runtime.stats();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "nvml {}  cupti {}  sm {}%  mem {}%",
            if s.nvml_valid { "active" } else { "off" },
            if s.cupti_valid { "active" } else { "off" },
            s.gpu_utilization as i32,
            s.mem_utilization as i32,
        ),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "hw_ops/s {}  gflops {}  clock {}MHz/{}MHz  power {:.1}W  temp {}C",
            if s.cupti_valid {
                format!("{:.0}", s.hw_ops_per_sec)
            } else {
                "N/A".to_string()
            },
            if s.cupti_valid {
                format!("{:.2}", s.gflops_total)
            } else {
                "N/A".to_string()
            },
            s.sm_clock_mhz,
            s.mem_clock_mhz,
            s.power_w,
            s.temperature_c,
        ),
    ));
}

pub(super) fn cmd_streams(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let s = daemon.runtime.stats();
    let num = daemon.runtime.num_streams();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "pool: {} streams configured  active: {}",
            num, s.active_streams,
        ),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "priorities: realtime / high / normal / low",
    ));
}

pub(super) fn cmd_snapshot(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let snap = daemon.runtime.system_snapshot();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "ops {}  procs {}  tasks {}  vram {}MB",
            snap.total_ops,
            snap.active_processes,
            snap.active_tasks,
            snap.total_vram_used / (1024 * 1024),
        ),
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!(
            "watchdog {}  kernel {}  queue {}/{}  interrupts {}",
            if snap.watchdog_alert { "ALERT" } else { "ok" },
            if snap.kernel_running {
                "running"
            } else {
                "idle"
            },
            snap.queue_head,
            snap.queue_tail,
            snap.interrupt_cnt,
        ),
    ));
}

pub(super) fn cmd_health(daemon: &Arc<DaemonState>, state: &mut TuiState) {
    let tlsf = daemon.runtime.tlsf_stats();
    let snap = daemon.runtime.system_snapshot();
    let active = daemon
        .active_clients
        .load(std::sync::atomic::Ordering::Relaxed);
    let crashed = snap.watchdog_alert || !daemon.is_running();
    let healthy = active < daemon.config.max_clients as u64
        && tlsf.utilization_percent < 95.0
        && !crashed;

    // Use appropriate severity: healthy is good, degraded is a warning,
    // unhealthy is an error — makes failures immediately visible.
    let (cat, label) = if healthy {
        (LogCategory::Sys, "OK")
    } else if crashed {
        (LogCategory::Err, "FAIL")
    } else {
        (LogCategory::App, "WARN")
    };

    state.push_log(LogEntry::new(
        cat,
        format!(
            "health {}  pool {}  util {:.1}%  watchdog {}  clients {}/{}",
            label,
            if crashed { "fail" } else { "ok" },
            tlsf.utilization_percent,
            if snap.watchdog_alert { "ALERT" } else { "ok" },
            active,
            daemon.config.max_clients,
        ),
    ));
}
