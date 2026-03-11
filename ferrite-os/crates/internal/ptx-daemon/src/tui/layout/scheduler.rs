//! TUI scheduler dashboard panel.
//!
//! Renders three sub-views (toggled via Tab):
//! - **Queue**:   table of jobs with priority, state, and wait time
//! - **Tenants**: table of tenants with VRAM / stream usage
//! - **Policy**:  recent policy decisions with Allow / Deny indicators
//!
//! Quick actions:
//! - `p` — pause / resume queue
//! - `k` — kill selected job
//! - `r` — reprioritize selected job
//! - `Tab` — cycle sub-panels

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table};
use ratatui::Frame;

use crate::tui::state::{SchedulerViewMode, TuiState};
use crate::tui::style;

/// Main scheduler dashboard entry point.
pub fn render_scheduler_panel(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height < 3 || area.width < 20 {
        return;
    }

    // Split: header (2 rows) + body + footer (1 row)
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // header + tab bar
            Constraint::Min(0),   // body
            Constraint::Length(1), // key hints
        ])
        .split(area);

    draw_scheduler_header(frame, rows[0], state);

    match state.scheduler_view_mode {
        SchedulerViewMode::Queue => draw_queue_view(frame, rows[1], state),
        SchedulerViewMode::Tenants => draw_tenant_view(frame, rows[1], state),
        SchedulerViewMode::Policy => draw_policy_view(frame, rows[1], state),
    }

    draw_scheduler_footer(frame, rows[2], state);
}

fn draw_scheduler_header(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    // Title line
    let paused_tag = if state.scheduler_paused {
        Span::styled(" PAUSED ", style::semantic_bold(style::bad()))
    } else {
        Span::styled(" ACTIVE ", style::semantic_bold(style::good()))
    };

    let title = Line::from(vec![
        Span::styled("scheduler", style::accent_bold()),
        Span::styled("  ", style::spacer()),
        paused_tag,
        Span::styled(
            format!(
                "  queue:{}  decisions:{}",
                state.scheduler_queue_snapshot.len(),
                state.scheduler_policy_decisions.len(),
            ),
            style::label(),
        ),
    ]);
    frame.render_widget(Paragraph::new(title), Rect::new(area.x, area.y, area.width, 1));

    // Tab bar
    if area.height >= 2 {
        let tab_y = area.y + 1;
        let tabs = [
            (SchedulerViewMode::Queue, "Queue"),
            (SchedulerViewMode::Tenants, "Tenants"),
            (SchedulerViewMode::Policy, "Policy"),
        ];

        let spans: Vec<Span> = tabs
            .iter()
            .flat_map(|(mode, label)| {
                let is_active = *mode == state.scheduler_view_mode;
                let tab_style = if is_active {
                    style::accent_bold()
                } else {
                    style::label()
                };
                let prefix = if is_active { "[ " } else { "  " };
                let suffix = if is_active { " ]" } else { "  " };
                vec![
                    Span::styled(prefix, style::label()),
                    Span::styled(*label, tab_style),
                    Span::styled(suffix, style::label()),
                ]
            })
            .collect();

        let tab_line = Line::from(spans);
        frame.render_widget(Paragraph::new(tab_line), Rect::new(area.x, tab_y, area.width, 1));
    }
}

fn draw_queue_view(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    if state.scheduler_queue_snapshot.is_empty() {
        let msg = Line::from(vec![
            Span::styled("  no jobs in queue", style::label()),
        ]);
        frame.render_widget(Paragraph::new(msg), area);
        return;
    }

    let header = Row::new(vec!["ID", "Tenant", "Priority", "State", "Wait"])
        .style(style::label_bold())
        .bottom_margin(0);

    let rows: Vec<Row> = state
        .scheduler_queue_snapshot
        .iter()
        .enumerate()
        .map(|(i, job)| {
            let is_selected = i == state.scheduler_selected_index;
            let base_style = if is_selected {
                style::selected_row()
            } else {
                style::value()
            };
            let _state_color = match job.state.as_str() {
                "running" => style::good(),
                "queued" => style::info(),
                "failed" => style::bad(),
                _ => style::fg(),
            };
            Row::new(vec![
                format!("{}", job.job_id),
                format!("{}", job.tenant_id),
                format!("{}", job.priority),
                job.state.clone(),
                format!("{}s", job.wait_secs),
            ])
            .style(base_style)
        })
        .collect();

    let widths = [
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Min(6),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::NONE));

    frame.render_widget(table, area);
}

fn draw_tenant_view(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    if state.scheduler_tenant_snapshot.is_empty() {
        let msg = Line::from(vec![
            Span::styled("  no tenants registered", style::label()),
        ]);
        frame.render_widget(Paragraph::new(msg), area);
        return;
    }

    let header = Row::new(vec!["ID", "Label", "VRAM", "Limit", "Streams", "SLimit", "Jobs"])
        .style(style::label_bold())
        .bottom_margin(0);

    let rows: Vec<Row> = state
        .scheduler_tenant_snapshot
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let is_selected = i == state.scheduler_selected_index;
            let base_style = if is_selected {
                style::selected_row()
            } else {
                style::value()
            };
            Row::new(vec![
                format!("{}", t.tenant_id),
                t.label.clone(),
                format_bytes_short(t.vram_used),
                format_bytes_short(t.vram_limit),
                format!("{}", t.streams_used),
                format!("{}", t.streams_limit),
                format!("{}", t.active_jobs),
            ])
            .style(base_style)
        })
        .collect();

    let widths = [
        Constraint::Length(6),
        Constraint::Length(12),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Min(5),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::NONE));

    frame.render_widget(table, area);
}

fn draw_policy_view(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    if state.scheduler_policy_decisions.is_empty() {
        let msg = Line::from(vec![
            Span::styled("  no policy decisions recorded", style::label()),
        ]);
        frame.render_widget(Paragraph::new(msg), area);
        return;
    }

    let visible = area.height as usize;
    let items: Vec<ListItem> = state
        .scheduler_policy_decisions
        .iter()
        .rev()
        .take(visible)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|d| {
            let (indicator, indicator_color) = if d.allowed {
                ("ALLOW", style::good())
            } else {
                ("DENY ", style::bad())
            };

            let reason_text = d
                .reason
                .as_deref()
                .map(|r| format!("  [{}]", r))
                .unwrap_or_default();

            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:.1}s ", d.elapsed_secs),
                    style::label(),
                ),
                Span::styled(
                    indicator,
                    style::semantic_bold(indicator_color),
                ),
                Span::styled(format!(" {} ", d.action), style::value()),
                Span::styled(
                    format!("t:{}", d.tenant_id),
                    style::label(),
                ),
                Span::styled(
                    reason_text,
                    style::semantic(if d.allowed { style::warn() } else { style::bad() }),
                ),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), area);
}

fn draw_scheduler_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }
    let mode_hints = match state.scheduler_view_mode {
        SchedulerViewMode::Queue => "[p:pause/resume] [k:kill] [r:reprioritize] [tab:next view]",
        SchedulerViewMode::Tenants => "[tab:next view]",
        SchedulerViewMode::Policy => "[tab:next view]",
    };
    let line = Line::from(vec![
        Span::styled(mode_hints, style::label()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn format_bytes_short(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1}G", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.0}M", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.0}K", b as f64 / 1024.0)
    } else if b > 0 {
        format!("{}B", b)
    } else {
        "--".to_string()
    }
}
