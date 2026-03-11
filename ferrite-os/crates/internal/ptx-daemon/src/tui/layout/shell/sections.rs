use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::super::{draw_metric_card, fmt_bytes, fmt_count, fmt_latency, pool_crashed};
use super::detail::draw_pipeline;
use crate::tui::state::TuiState;
use crate::tui::style;
use crate::tui::widgets::{FragMap, Gauge, LiveWave};

pub(in crate::tui::layout) fn draw_system_status(
    frame: &mut Frame,
    area: Rect,
    state: &TuiState,
) {
    if area.height == 0 {
        return;
    }

    let latency_ms = state
        .last_pipeline
        .as_ref()
        .map(|p| p.total_ms as f32)
        .unwrap_or(0.0);

    let vc = style::vram_color(state.display_vram_pct);
    let health_color = if pool_crashed(state) {
        style::bad()
    } else {
        style::good()
    };
    let lat_color = style::latency_color(latency_ms);

    let line1 = Line::from(vec![
        Span::styled("GPU", style::label()),
        Span::styled(
            format!(" {:>6.1}%  ", state.display_gpu_util),
            style::value(),
        ),
        Span::styled("MEM", style::label()),
        Span::styled(
            format!(" {:>6.1}%  ", state.display_vram_pct),
            style::semantic(vc),
        ),
        Span::styled("POOL", style::label()),
        Span::styled(
            format!(" {:>8}  ", if pool_crashed(state) { "FAIL" } else { "OK" }),
            style::semantic_bold(health_color),
        ),
        Span::styled("LAT", style::label()),
        Span::styled(
            format!(" {:>7}", fmt_latency(latency_ms)),
            style::semantic_bold(lat_color),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line1),
        Rect::new(area.x, area.y, area.width, 1),
    );

    if area.height > 1 {
        let health = style::pool_health_token(pool_crashed(state), state.pool_needs_defrag);

        let line2 = Line::from(vec![
            Span::styled("CLIENTS", style::label()),
            Span::styled(format!(" {:>3}  ", state.active_clients), style::value()),
            Span::styled("TOTAL", style::label()),
            Span::styled(
                format!(" {:>8}  ", fmt_count(state.total_ops)),
                style::value(),
            ),
            Span::styled("STATE", style::label()),
            Span::styled(
                format!(" {:>8}", health.0),
                style::semantic_bold(health.1),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(line2),
            Rect::new(area.x, area.y + 1, area.width, 1),
        );
    }

    // Line 3: NVML hardware telemetry (temp, power, clocks)
    if area.height > 2 && state.hardware_poll_valid {
        let temp_color = if state.temperature_c > 85 {
            style::bad()
        } else if state.temperature_c > 70 {
            style::warn()
        } else {
            style::good()
        };
        let line3 = Line::from(vec![
            Span::styled("TEMP", style::label()),
            Span::styled(
                format!(" {:>3}°C  ", state.temperature_c),
                style::semantic(temp_color),
            ),
            Span::styled("PWR", style::label()),
            Span::styled(
                format!(" {:>5.1}W  ", state.power_w),
                style::value(),
            ),
            Span::styled("CLK", style::label()),
            Span::styled(
                format!(" {}MHz  ", state.sm_clock_mhz),
                style::value(),
            ),
            Span::styled("MEM", style::label()),
            Span::styled(
                format!(" {}MHz", state.mem_clock_mhz),
                style::value(),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(line3),
            Rect::new(area.x, area.y + 2, area.width, 1),
        );
    }
}

pub(in crate::tui::layout) fn draw_memory(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let frag_h = if area.height > 2 { 1u16 } else { 0 };
    let owner_h = if area.height > 3 && !state.vram_owners.is_empty() {
        (area.height.saturating_sub(3)).min(state.vram_owners.len() as u16).min(4)
    } else {
        0
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),       // 0: vram gauge
            Constraint::Length(1),       // 1: stats
            Constraint::Length(frag_h),  // 2: frag map
            Constraint::Length(owner_h), // 3: owner bars
            Constraint::Min(0),         // 4: remainder
        ])
        .split(area);

    draw_vram_gauge(frame, rows[0], state);

    if area.height > 1 {
        let health_color = if pool_crashed(state) {
            style::bad()
        } else {
            style::good()
        };
        let line = Line::from(vec![
            Span::styled("peak ", style::label()),
            Span::styled(format!("{:>8}", fmt_bytes(state.pool_peak)), style::value()),
            Span::styled("   largest_free ", style::label()),
            Span::styled(
                format!("{:>8}", fmt_bytes(state.pool_largest_free)),
                style::value(),
            ),
            Span::styled("   blocks ", style::label()),
            Span::styled(
                format!(
                    "{:>4}/{:<4}",
                    state.pool_blocks,
                    state.pool_blocks + state.pool_free_blocks
                ),
                style::value(),
            ),
            Span::styled("   pool ", style::label()),
            Span::styled(
                if pool_crashed(state) { "FAIL" } else { "OK" },
                style::semantic(health_color),
            ),
            Span::styled("   allocs ", style::label()),
            Span::styled(
                if state.alloc_rate >= 1000.0 {
                    format!("{:.1}K/s", state.alloc_rate / 1000.0)
                } else {
                    format!("{:.0}/s", state.alloc_rate)
                },
                style::value(),
            ),
        ]);
        frame.render_widget(Paragraph::new(line), rows[1]);
    }

    if area.height > 2 {
        frame.render_widget(
            FragMap {
                label: "map",
                pool_used: state.pool_used,
                pool_total: state.pool_total,
                allocated_blocks: state.pool_blocks,
                free_blocks: state.pool_free_blocks,
                largest_free: state.pool_largest_free,
                alloc_color: style::info(),
                free_color: style::bar_empty(),
                label_color: style::fg_dim(),
            },
            rows[2],
        );
    }

    // VRAM trend wave removed — the standard activity bar covers it.

    // Per-owner VRAM bars (when owners exist).
    if !state.vram_owners.is_empty() && rows[3].height > 0 {
        draw_owner_bars(frame, rows[3], state);
    }
}

fn draw_owner_bars(frame: &mut Frame, area: Rect, state: &TuiState) {
    let max_rows = area.height as usize;
    let owners = &state.vram_owners;
    let count = owners.len().min(max_rows);
    if count == 0 {
        return;
    }

    for (i, owner) in owners.iter().take(count).enumerate() {
        let y = area.y + i as u16;
        if y >= area.y + area.height {
            break;
        }

        let label_w = 12usize.min(area.width as usize);
        let bar_start = label_w as u16 + 1;
        let bar_w = area.width.saturating_sub(bar_start + 12);

        // Label
        let short_label = if owner.label.len() > label_w {
            format!("{}…", &owner.label[..label_w - 1])
        } else {
            format!("{:<w$}", owner.label, w = label_w)
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(&short_label, style::label()))),
            Rect::new(area.x, y, label_w as u16, 1),
        );

        // Bar
        if bar_w > 2 {
            let (pct, bar_color) = if owner.soft_limit > 0 {
                let p = (owner.allocated_bytes as f32 * 100.0 / owner.soft_limit as f32).min(100.0);
                let c = if owner.over_quota { style::bad() } else { style::info() };
                (p, c)
            } else {
                // No quota — show usage relative to pool total
                let p = if state.pool_total > 0 {
                    (owner.allocated_bytes as f32 * 100.0 / state.pool_total as f32).min(100.0)
                } else {
                    0.0
                };
                (p, style::info())
            };

            frame.render_widget(
                Gauge {
                    label: "",
                    used: owner.allocated_bytes,
                    total: if owner.soft_limit > 0 { owner.soft_limit } else { state.pool_total },
                    percent: pct,
                    bar_color,
                    empty_color: style::bar_empty(),
                    label_color: style::fg_dim(),
                    text_color: style::fg(),
                },
                Rect::new(area.x + bar_start, y, bar_w, 1),
            );
        }

        // Usage text after bar
        let usage_str = format!("{:.1}MB", owner.allocated_bytes as f64 / (1024.0 * 1024.0));
        let text_x = area.x + bar_start + bar_w + 1;
        let text_w = area.width.saturating_sub(text_x - area.x);
        if text_w > 0 {
            let usage_color = if owner.over_quota { style::bad() } else { style::fg() };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    usage_str,
                    style::semantic(usage_color),
                ))),
                Rect::new(text_x, y, text_w, 1),
            );
        }
    }
}

pub(in crate::tui::layout) fn draw_execution(
    frame: &mut Frame,
    area: Rect,
    state: &TuiState,
    pipeline_h: u16,
) -> Rect {
    if area.height == 0 {
        return Rect::default();
    }

    let base_h = area.height.saturating_sub(pipeline_h);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(base_h),
            Constraint::Length(pipeline_h),
        ])
        .split(area);

    draw_execution_base(frame, rows[0], state);

    if pipeline_h > 0 {
        draw_pipeline(frame, rows[1], state);
    }

    rows[1]
}

fn draw_execution_base(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(if area.height > 1 { 1 } else { 0 }),
            Constraint::Min(0),
        ])
        .split(area);

    draw_sparklines(frame, chunks[0], state);

    if area.height > 1 {
        let (compile_ms, exec_ms, total_ms) = state
            .last_pipeline
            .as_ref()
            .map(|p| {
                let compile = p.stages.first().map(|s| s.duration_ms).unwrap_or(0);
                let exec = p.stages.get(1).map(|s| s.duration_ms).unwrap_or(0);
                (compile, exec, p.total_ms)
            })
            .unwrap_or((0, 0, 0));

        let pipeline_state = if total_ms > 0 { "active" } else { "idle" };
        let cards = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(chunks[1]);

        draw_metric_card(
            frame,
            cards[0],
            "pipeline",
            pipeline_state,
            if total_ms > 0 { style::info() } else { style::fg_dim() },
        );
        draw_metric_card(
            frame,
            cards[1],
            "latency",
            &fmt_latency(total_ms as f32),
            style::latency_color(total_ms as f32),
        );
        draw_metric_card(
            frame,
            cards[2],
            "jit/gpu",
            &format!("{}ms/{}ms", compile_ms, exec_ms),
            style::fg(),
        );
    }
}

fn draw_sparklines(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.width < 12 || area.height == 0 {
        return;
    }

    // GPU utilization wave only — VRAM wave is in the standard activity bar.
    frame.render_widget(
        LiveWave {
            label: "gpu",
            data: &state.gpu_util_history,
            max_val: 100.0,
            phase: state.wave_phase_gpu,
            intensity: state.display_gpu_util / 100.0,
            suffix: format!("{:>5.1}%", state.display_gpu_util),
            color: style::info(),
            label_color: style::fg_dim(),
            text_color: style::fg(),
        },
        Rect::new(area.x, area.y, area.width, 1),
    );
}

pub(in crate::tui::layout) fn draw_profiling(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let ring = &state.kernel_profiles;

    if ring.entries.is_empty() {
        let hint = Line::from(vec![
            Span::styled("no kernel executions yet — run a script or demo", style::label()),
        ]);
        frame.render_widget(Paragraph::new(hint), Rect::new(area.x, area.y, area.width, 1));
        return;
    }

    // Line 1: latency sparkline
    let latency_wave_w = area.width.saturating_sub(30);
    if latency_wave_w > 10 {
        frame.render_widget(
            LiveWave {
                label: "lat",
                data: &ring.latency_history,
                max_val: ring.peak_ms().max(1) as f32,
                phase: 0.0,
                intensity: (ring.smoothed_latency / ring.peak_ms().max(1) as f32).min(1.0),
                suffix: format!("{:>5.1}ms", ring.smoothed_latency),
                color: style::latency_color(ring.smoothed_latency),
                label_color: style::fg_dim(),
                text_color: style::fg(),
            },
            Rect::new(area.x, area.y, latency_wave_w, 1),
        );
        // Stats after the wave
        let stats_x = area.x + latency_wave_w;
        let stats_w = area.width.saturating_sub(latency_wave_w);
        let stats_line = Line::from(vec![
            Span::styled(
                format!(" runs:{}", ring.total_runs),
                style::label(),
            ),
            Span::styled(
                format!(" ok:{:.0}%", ring.success_rate()),
                style::semantic(if ring.success_rate() >= 90.0 { style::good() } else { style::warn() }),
            ),
        ]);
        frame.render_widget(Paragraph::new(stats_line), Rect::new(stats_x, area.y, stats_w, 1));
    }

    // Line 2: metric cards — avg compile, avg exec, peak, vram delta
    if area.height > 1 {
        let cards = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(Rect::new(area.x, area.y + 1, area.width, 1));

        draw_metric_card(
            frame,
            cards[0],
            "avg jit",
            &format!("{:.1}ms", ring.avg_compile_ms()),
            style::latency_color(ring.avg_compile_ms()),
        );
        draw_metric_card(
            frame,
            cards[1],
            "avg exec",
            &format!("{:.1}ms", ring.avg_exec_ms()),
            style::latency_color(ring.avg_exec_ms()),
        );
        draw_metric_card(
            frame,
            cards[2],
            "peak",
            &fmt_latency(ring.peak_ms() as f32),
            style::latency_color(ring.peak_ms() as f32),
        );
        let delta = ring.last_vram_delta();
        let delta_color = if delta > 0 { style::warn() } else { style::good() };
        draw_metric_card(
            frame,
            cards[3],
            "vram Δ",
            &fmt_bytes(delta.unsigned_abs()),
            delta_color,
        );
    }
}

fn draw_vram_gauge(frame: &mut Frame, area: Rect, state: &TuiState) {
    let bar_color = style::vram_color(state.display_vram_pct);

    frame.render_widget(
        Gauge {
            label: "vram",
            used: state.vram_used,
            total: state.vram_total,
            percent: state.display_vram_pct,
            bar_color,
            empty_color: style::bar_empty(),
            label_color: style::fg_dim(),
            text_color: style::fg(),
        },
        area,
    );
}
