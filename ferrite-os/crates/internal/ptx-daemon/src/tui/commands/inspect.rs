use std::sync::Arc;

use crate::events::{LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::tui::state::TuiState;

/// Print last compiled program info to the log.
pub fn cmd_ptx(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
) {
    let r = runner.lock();
    let Some(info) = r.inspect_last() else {
        state.push_log(LogEntry::new(LogCategory::Sys, "no compiled program — run a script first"));
        return;
    };

    state.push_log(LogEntry::new(LogCategory::Jit, "── compiled program ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("  nodes: {}  inputs: {}  total elements: {}",
            info.node_count, info.input_count, info.total_elements),
    ));

    for (i, shape) in info.input_shapes.iter().enumerate() {
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!("  input[{}]: {:?}", i, shape),
        ));
    }
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("  output: {:?}", info.output_shape),
    ));

    if !info.op_summary.is_empty() {
        let ops: Vec<String> = info.op_summary.iter()
            .map(|(name, count)| format!("{}x{}", name, count))
            .collect();
        state.push_log(LogEntry::new(
            LogCategory::Sys,
            format!("  ops: {}", ops.join("  ")),
        ));
    }
}

/// Print profiling history to the log.
pub fn cmd_perf(state: &mut TuiState) {
    let ring = &state.kernel_profiles;
    if ring.entries.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Sys, "no kernel profiles recorded"));
        return;
    }

    // Collect all data before pushing to log (avoid borrow conflict)
    let summary = format!(
        "  total runs: {}  success rate: {:.0}%  avg latency: {:.1}ms  peak: {}ms",
        ring.total_runs,
        ring.success_rate(),
        ring.smoothed_latency,
        ring.peak_ms(),
    );
    let avgs = format!(
        "  avg compile: {:.1}ms  avg exec: {:.1}ms",
        ring.avg_compile_ms(),
        ring.avg_exec_ms(),
    );

    let vram_delta = ring.last_vram_delta();
    let vram_line = format!(
        "  last vram delta: {:+} bytes",
        vram_delta,
    );

    let recent: Vec<(LogCategory, String)> = ring.entries.iter().rev().take(10).map(|e| {
        let elapsed = e.timestamp.elapsed().as_secs();
        let marker = if e.success { "ok" } else { "FAIL" };
        let tag = if e.source_tag.len() > 30 {
            format!("{}...", &e.source_tag[..27])
        } else {
            e.source_tag.clone()
        };
        let delta = e.vram_after as i64 - e.vram_before as i64;
        let vram_str = if e.vram_before > 0 || e.vram_after > 0 {
            format!("  vram {:+}", delta)
        } else {
            String::new()
        };
        let cat = if e.success { LogCategory::Sys } else { LogCategory::Err };
        (cat, format!(
            "  {:>4}s ago  {}ms (jit {}ms + gpu {}ms)  [{}]  {}{}",
            elapsed, e.total_ms, e.compile_ms, e.exec_ms, marker, tag, vram_str,
        ))
    }).collect();

    state.push_log(LogEntry::new(LogCategory::Jit, "── kernel profiling ──"));
    state.push_log(LogEntry::new(LogCategory::Sys, summary));
    state.push_log(LogEntry::new(LogCategory::Sys, avgs));
    state.push_log(LogEntry::new(LogCategory::Sys, vram_line));

    if !recent.is_empty() {
        state.push_log(LogEntry::new(LogCategory::Jit, "  ── recent (newest first) ──"));
        for (cat, msg) in recent {
            state.push_log(LogEntry::new(cat, msg));
        }
    }
}

/// Open PTX inspection as a read-only buffer in the editor.
pub fn cmd_ptx_buffer(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
) {
    let r = runner.lock();
    let Some(info) = r.inspect_last() else {
        state.push_log(LogEntry::new(LogCategory::Sys, "no compiled program"));
        return;
    };

    let mut lines = Vec::new();
    lines.push(format!("# Compiled Program Inspection"));
    lines.push(format!("#"));
    lines.push(format!("# Nodes:          {}", info.node_count));
    lines.push(format!("# Inputs:         {}", info.input_count));
    lines.push(format!("# Total elements: {}", info.total_elements));
    lines.push(String::new());

    lines.push("# Input shapes:".to_string());
    for (i, shape) in info.input_shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        lines.push(format!("#   input[{}]: {:?}  ({} elements)", i, shape, numel));
    }
    lines.push(String::new());

    let out_numel: usize = info.output_shape.iter().product();
    lines.push(format!("# Output shape: {:?}  ({} elements)", info.output_shape, out_numel));
    lines.push(String::new());

    lines.push("# Operation graph:".to_string());
    for (name, count) in &info.op_summary {
        lines.push(format!("#   {:<12} x{}", name, count));
    }
    lines.push(String::new());

    // Include profiling data if available
    let ring = &state.kernel_profiles;
    if !ring.entries.is_empty() {
        lines.push("# Profiling summary:".to_string());
        lines.push(format!("#   Total runs:    {}", ring.total_runs));
        lines.push(format!("#   Success rate:  {:.0}%", ring.success_rate()));
        lines.push(format!("#   Avg latency:   {:.1}ms", ring.smoothed_latency));
        lines.push(format!("#   Avg compile:   {:.1}ms", ring.avg_compile_ms()));
        lines.push(format!("#   Avg exec:      {:.1}ms", ring.avg_exec_ms()));
        lines.push(format!("#   Peak latency:  {}ms", ring.peak_ms()));
        let delta = ring.last_vram_delta();
        lines.push(format!("#   Last VRAM Δ:   {:+} bytes", delta));
    }

    // Load into editor as a virtual read-only buffer
    state.file_lines = lines;
    state.file_cursor_line = 0;
    state.file_cursor_col = 0;
    state.file_scroll = 0;
    state.file_hscroll = 0;
    state.file_dirty = false;
    state.open_file = None; // No backing file — virtual buffer
    state.push_log(LogEntry::new(LogCategory::Jit, "opened PTX inspection buffer (read-only)"));
}

/// Show JIT disk cache status.
pub fn cmd_jit_cache(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
) {
    let r = runner.lock();
    let mem = r.cache_len();
    let disk = r.disk_cache_len();
    let hits = r.disk_hits();
    state.push_log(LogEntry::new(
        LogCategory::Jit,
        format!("jit cache — memory: {} programs  disk: {} programs  disk hits: {}", mem, disk, hits),
    ));
}

/// Show per-owner VRAM usage and quotas.
pub fn cmd_quota(state: &mut TuiState, args: &[&str]) {
    use crate::tui::state::OwnerQuotaConfig;

    match args.first().copied() {
        None | Some("show") => {
            // Display current per-owner usage
            if state.vram_owners.is_empty() {
                state.push_log(LogEntry::new(LogCategory::Sys, "no active VRAM owners"));
                return;
            }
            state.push_log(LogEntry::new(LogCategory::Jit, "── vram owners ──"));
            // Collect data before mutating state
            let lines: Vec<(LogCategory, String)> = state
                .vram_owners
                .iter()
                .map(|o| {
                    let limit_str = if o.soft_limit > 0 {
                        let pct = o.allocated_bytes as f64 * 100.0 / o.soft_limit as f64;
                        format!(
                            "  limit {:.1}MB ({:.0}%)",
                            o.soft_limit as f64 / (1024.0 * 1024.0),
                            pct,
                        )
                    } else {
                        "  (no limit)".to_string()
                    };
                    let cat = if o.over_quota {
                        LogCategory::Err
                    } else {
                        LogCategory::Sys
                    };
                    let marker = if o.over_quota { " OVER-QUOTA" } else { "" };
                    (
                        cat,
                        format!(
                            "  [{}] {}  {:.1}MB  {} blocks{}{}",
                            o.owner_id,
                            o.label,
                            o.allocated_bytes as f64 / (1024.0 * 1024.0),
                            o.block_count,
                            limit_str,
                            marker,
                        ),
                    )
                })
                .collect();
            for (cat, msg) in lines {
                state.push_log(LogEntry::new(cat, msg));
            }
        }
        Some("set") => {
            // quota set <owner_id> <limit_mb>
            if args.len() < 3 {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: quota set <owner_id> <limit_mb>",
                ));
                return;
            }
            let owner_id: u32 = match args[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    state.push_log(LogEntry::new(LogCategory::Err, "invalid owner_id"));
                    return;
                }
            };
            let limit_mb: f64 = match args[2].parse() {
                Ok(v) => v,
                Err(_) => {
                    state.push_log(LogEntry::new(LogCategory::Err, "invalid limit_mb"));
                    return;
                }
            };
            let limit_bytes = (limit_mb * 1024.0 * 1024.0) as u64;
            let entry = state
                .vram_quota_configs
                .entry(owner_id)
                .or_insert_with(|| OwnerQuotaConfig {
                    label: format!("owner-{}", owner_id),
                    soft_limit: 0,
                });
            entry.soft_limit = limit_bytes;
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "quota set: owner {} → soft limit {:.1}MB",
                    owner_id, limit_mb
                ),
            ));
        }
        Some("clear") => {
            if args.len() < 2 {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: quota clear <owner_id>",
                ));
                return;
            }
            let owner_id: u32 = match args[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    state.push_log(LogEntry::new(LogCategory::Err, "invalid owner_id"));
                    return;
                }
            };
            if let Some(cfg) = state.vram_quota_configs.get_mut(&owner_id) {
                cfg.soft_limit = 0;
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("quota cleared for owner {}", owner_id),
                ));
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("no quota configured for owner {}", owner_id),
                ));
            }
        }
        Some("label") => {
            // quota label <owner_id> <name>
            if args.len() < 3 {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    "usage: quota label <owner_id> <name>",
                ));
                return;
            }
            let owner_id: u32 = match args[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    state.push_log(LogEntry::new(LogCategory::Err, "invalid owner_id"));
                    return;
                }
            };
            let label = args[2..].join(" ");
            let entry = state
                .vram_quota_configs
                .entry(owner_id)
                .or_insert_with(|| OwnerQuotaConfig {
                    label: String::new(),
                    soft_limit: 0,
                });
            entry.label = label.clone();
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("owner {} labelled '{}'", owner_id, label),
            ));
        }
        Some(sub) => {
            state.push_log(LogEntry::new(
                LogCategory::Err,
                format!("unknown subcommand '{}' — try: quota [show|set|clear|label]", sub),
            ));
        }
    }
}

/// Clear the JIT cache.
pub fn cmd_jit_clear(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
) {
    let mut r = runner.lock();
    let before = r.cache_len();
    r.clear_cache();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("cleared JIT cache ({} programs dropped)", before),
    ));
}
