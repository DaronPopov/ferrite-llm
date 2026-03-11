use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph};
use ratatui::Frame;

use super::{
    draw_activity_wave, fmt_bytes, fmt_count, fmt_elapsed, fmt_latency, fmt_uptime, pool_crashed,
};
use crate::tui::state::TuiState;
use crate::tui::style;

pub(super) fn draw_clean(frame: &mut Frame, state: &mut TuiState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(style::screen()), area);

    let nvml_h = if state.hardware_poll_valid { 1 } else { 0 };
    let owner_h = if !state.vram_owners.is_empty() {
        (state.vram_owners.len() as u16 + 1).min(5) // +1 for header
    } else {
        0
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),        // 0: header
            Constraint::Length(1),        // 1: activity wave
            Constraint::Length(1),        // 2: core stats 1
            Constraint::Length(1),        // 3: core stats 2
            Constraint::Length(1),        // 4: group separator
            Constraint::Length(1),        // 5: memory stats
            Constraint::Length(1),        // 6: health
            Constraint::Length(nvml_h),   // 7: NVML hardware (when available)
            Constraint::Length(owner_h),  // 8: VRAM owners (when active)
            Constraint::Length(1),        // 9: profiling summary
            Constraint::Min(0),           // 10: recent log
            Constraint::Length(1),        // 11: prompt
        ])
        .split(area);

    draw_clean_header(frame, rows[0], state);
    draw_activity_wave(frame, rows[1], state);
    draw_clean_stats(frame, rows[2], rows[3], rows[5], rows[6], state);
    super::draw_thin_rule(frame, rows[4]);
    if nvml_h > 0 {
        draw_clean_nvml(frame, rows[7], state);
    }
    if owner_h > 0 {
        draw_clean_owners(frame, rows[8], state);
    }
    draw_clean_profiling(frame, rows[9], state);
    // Store the log body area for mouse click hit-testing.
    let log_area = rows[10];
    if log_area.height > 1 {
        state.log_body_area = Rect::new(log_area.x, log_area.y + 1, log_area.width, log_area.height - 1);
    } else {
        state.log_body_area = Rect::default();
    }
    draw_clean_recent(frame, log_area, state);
    draw_clean_prompt(frame, rows[11], state);
}

fn draw_clean_header(frame: &mut Frame, area: Rect, state: &TuiState) {
    let status = style::status_token(state.running, state.watchdog);

    let line = Line::from(vec![
        Span::styled("ferrite daemon", style::accent_bold()),
        Span::styled("  ", style::spacer()),
        Span::styled(status.0, style::semantic_bold(status.1)),
        Span::styled("  ", style::spacer()),
        Span::styled(
            format!("uptime {}", fmt_uptime(state.uptime_secs)),
            style::label(),
        ),
        Span::styled("  ", style::spacer()),
        Span::styled(
            format!("device {}", state.device_name),
            style::label(),
        ),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}

fn draw_clean_stats(
    frame: &mut Frame,
    line1: Rect,
    line2: Rect,
    line3: Rect,
    line4: Rect,
    state: &TuiState,
) {
    let latency_ms = state
        .last_pipeline
        .as_ref()
        .map(|p| p.total_ms as f32)
        .unwrap_or(0.0);

    let metrics_1 = Line::from(vec![
        Span::styled("GPU ", style::label()),
        Span::styled(format!("{:>6.1}%  ", state.display_gpu_util), style::value()),
        Span::styled("VRAM ", style::label()),
        Span::styled(
            format!("{:>6.1}%  ", state.display_vram_pct),
            style::semantic(style::vram_color(state.display_vram_pct)),
        ),
        Span::styled("LAT ", style::label()),
        Span::styled(
            format!("{:>8}  ", fmt_latency(latency_ms)),
            style::semantic(style::latency_color(latency_ms)),
        ),
        Span::styled("FREE ", style::label()),
        Span::styled(
            format!("{:>8}", fmt_bytes(state.pool_largest_free)),
            style::value(),
        ),
    ]);
    frame.render_widget(Paragraph::new(metrics_1), line1);

    let metrics_2 = Line::from(vec![
        Span::styled("CLIENTS ", style::label()),
        Span::styled(format!("{:>3}  ", state.active_clients), style::value()),
        Span::styled("TOTAL ", style::label()),
        Span::styled(format!("{:>8}", fmt_count(state.total_ops)), style::value()),
    ]);
    frame.render_widget(Paragraph::new(metrics_2), line2);

    let mem = Line::from(vec![
        Span::styled("VRAM ", style::label()),
        Span::styled(format!("{:>8}", fmt_bytes(state.vram_used)), style::value()),
        Span::styled(" / ", style::label()),
        Span::styled(format!("{:>8}", fmt_bytes(state.vram_total)), style::value()),
        Span::styled("  pool ", style::label()),
        Span::styled(format!("{:>8}", fmt_bytes(state.pool_used)), style::label()),
        Span::styled("  peak ", style::label()),
        Span::styled(format!("{:>8}", fmt_bytes(state.pool_peak)), style::value()),
        Span::styled("  largest_free ", style::label()),
        Span::styled(
            format!("{:>8}", fmt_bytes(state.pool_largest_free)),
            style::value(),
        ),
    ]);
    frame.render_widget(Paragraph::new(mem), line3);

    let health = style::pool_health_token(pool_crashed(state), state.pool_needs_defrag);
    let health_span = Span::styled(
        health.0,
        style::semantic_bold(health.1),
    );
    let health_line = Line::from(vec![
        Span::styled("POOL ", style::label()),
        health_span,
        Span::styled("  blocks ", style::label()),
        Span::styled(
            format!(
                "{}/{}",
                state.pool_blocks,
                state.pool_blocks + state.pool_free_blocks
            ),
            style::value(),
        ),
        Span::styled("  allocs ", style::label()),
        Span::styled(format!("{}", fmt_count(state.pool_allocs)), style::value()),
        Span::styled("  frees ", style::label()),
        Span::styled(format!("{}", fmt_count(state.pool_frees)), style::value()),
    ]);
    frame.render_widget(Paragraph::new(health_line), line4);
}

fn draw_clean_recent(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }
    let title = Line::from(vec![
        Span::styled("recent", style::label()),
    ]);
    frame.render_widget(
        Paragraph::new(title),
        Rect::new(area.x, area.y, area.width, 1),
    );

    if area.height <= 1 {
        return;
    }

    let body = Rect::new(area.x, area.y + 1, area.width, area.height - 1);

    let visible = (area.height as usize - 1).min(40);
    let items: Vec<ListItem> = state
        .log
        .iter()
        .rev()
        .skip(state.log_scroll)
        .take(visible)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|entry| {
            let elapsed = fmt_elapsed(entry.timestamp.elapsed().as_secs());
            let (tag_style, _msg_style) = style::log_styles(entry.category);
            let clickable = entry.action.is_some();
            let msg_style = if clickable { style::accent() } else { style::value() };
            let prefix = if clickable { "\u{25b8} " } else { "  " };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:>4} ", elapsed), style::label()),
                Span::styled(format!("{:<4}", entry.category), tag_style),
                Span::styled(prefix, if clickable { style::accent() } else { style::label() }),
                Span::styled(&entry.message, msg_style),
            ]))
        })
        .collect();

    if items.is_empty() {
        let hint = Line::from(vec![Span::styled(
            "  no events yet \u{2014} type help to get started",
            style::label(),
        )]);
        frame.render_widget(Paragraph::new(hint), body);
    } else {
        frame.render_widget(List::new(items), body);
    }
}

fn draw_clean_nvml(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }
    let temp_color = if state.temperature_c > 85 {
        style::bad()
    } else if state.temperature_c > 70 {
        style::warn()
    } else {
        style::good()
    };
    let line = Line::from(vec![
        Span::styled("HW   ", style::label()),
        Span::styled(
            format!("{}°C  ", state.temperature_c),
            style::semantic(temp_color),
        ),
        Span::styled(
            format!("{:.1}W  ", state.power_w),
            style::value(),
        ),
        Span::styled(
            format!("sm:{}MHz  mem:{}MHz", state.sm_clock_mhz, state.mem_clock_mhz),
            style::label(),
        ),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_clean_owners(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 || state.vram_owners.is_empty() {
        return;
    }
    // Header line
    let header = Line::from(vec![
        Span::styled("OWNERS ", style::label()),
        Span::styled(
            format!("{} active", state.vram_owners.len()),
            style::value(),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), Rect::new(area.x, area.y, area.width, 1));

    // Per-owner lines
    for (i, owner) in state.vram_owners.iter().take((area.height as usize).saturating_sub(1)).enumerate() {
        let y = area.y + 1 + i as u16;
        if y >= area.y + area.height {
            break;
        }
        let usage_mb = owner.allocated_bytes as f64 / (1024.0 * 1024.0);
        let limit_str = if owner.soft_limit > 0 {
            let limit_mb = owner.soft_limit as f64 / (1024.0 * 1024.0);
            let pct = owner.allocated_bytes as f64 * 100.0 / owner.soft_limit as f64;
            format!("/{:.0}MB ({:.0}%)", limit_mb, pct)
        } else {
            String::new()
        };
        let color = if owner.over_quota {
            style::bad()
        } else {
            style::fg()
        };
        let marker = if owner.over_quota { " OVER" } else { "" };
        let line = Line::from(vec![
            Span::styled(format!("  [{:>2}] ", owner.owner_id), style::label()),
            Span::styled(
                format!("{:<10} ", if owner.label.len() > 10 { &owner.label[..10] } else { &owner.label }),
                style::label(),
            ),
            Span::styled(
                format!("{:.1}MB{}", usage_mb, limit_str),
                style::semantic(color),
            ),
            Span::styled(
                format!("  {} blk{}", owner.block_count, marker),
                style::semantic(color),
            ),
        ]);
        frame.render_widget(Paragraph::new(line), Rect::new(area.x, y, area.width, 1));
    }
}

fn draw_clean_profiling(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }
    let ring = &state.kernel_profiles;
    if ring.entries.is_empty() {
        let line = Line::from(vec![
            Span::styled("PROF ", style::label()),
            Span::styled("no kernel runs yet", style::label()),
        ]);
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let last = ring.entries.back().unwrap();
    let success_marker = if last.success { "ok" } else { "FAIL" };
    let success_color = if last.success { style::good() } else { style::bad() };
    let line = Line::from(vec![
        Span::styled("PROF ", style::label()),
        Span::styled(
            format!("last {}ms ", last.total_ms),
            style::semantic(style::latency_color(last.total_ms as f32)),
        ),
        Span::styled(
            format!("(jit {}ms + gpu {}ms)  ", last.compile_ms, last.exec_ms),
            style::label(),
        ),
        Span::styled(success_marker, style::semantic(success_color)),
        Span::styled(
            format!("  runs:{}  avg:{:.0}ms  peak:{}ms",
                ring.total_runs,
                ring.smoothed_latency,
                ring.peak_ms(),
            ),
            style::label(),
        ),
        {
            let delta = ring.last_vram_delta();
            if delta != 0 {
                let sign = if delta > 0 { "+" } else { "-" };
                let color = if delta > 0 { style::warn() } else { style::good() };
                Span::styled(
                    format!("  vram:{}{}", sign, fmt_bytes(delta.unsigned_abs())),
                    style::semantic(color),
                )
            } else {
                Span::styled("", style::label())
            }
        },
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_clean_prompt(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }
    let before = &state.input[..state.cursor];
    let after = &state.input[state.cursor..];
    let (cursor_ch, rest) = if after.is_empty() {
        (" ", "")
    } else {
        let n = after.chars().next().map_or(1, |c| c.len_utf8());
        (&after[..n], &after[n..])
    };

    // Hints adapt to input state.
    let hints = if state.input.is_empty() {
        "  [help] [/sysmon on] [ctrl+o:files]"
    } else {
        "  [enter:run] [esc:clear] [\u{2191}\u{2193}:history]"
    };

    let line = Line::from(vec![
        Span::styled("\u{276f} ", style::accent()),
        Span::styled(before, style::value_bright()),
        Span::styled(cursor_ch, style::cursor()),
        Span::styled(rest, style::value_bright()),
        Span::styled(hints, style::label()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}
