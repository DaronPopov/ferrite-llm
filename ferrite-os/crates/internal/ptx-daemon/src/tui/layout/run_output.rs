use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph};
use ratatui::Frame;

use crate::events::LogCategory;
use crate::tui::state::run_state::RunStatus;
use crate::tui::state::TuiState;
use crate::tui::style;

pub(super) fn draw_run_output(frame: &mut Frame, state: &mut TuiState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(style::screen()), area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(1),   // output body
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_header(frame, rows[0], state);
    draw_body(frame, rows[1], state);
    draw_footer(frame, rows[2], state);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let (status_tag, status_color) = match state.run_status {
        RunStatus::Idle => ("IDLE", style::fg_dim()),
        RunStatus::Compiling => ("COMPILING", style::warn()),
        RunStatus::Running => ("RUNNING", style::info()),
        RunStatus::Succeeded => ("OK", style::good()),
        RunStatus::Failed => ("FAIL", style::bad()),
        RunStatus::Timeout => ("TIMEOUT", style::bad()),
    };

    let file_label = state
        .run_target_file
        .as_ref()
        .and_then(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "(no file)".to_string());

    let elapsed = if state.run_elapsed_ms > 0 {
        format!("  {}ms", state.run_elapsed_ms)
    } else if let Some(start) = state.run_start_time {
        format!("  {}ms", start.elapsed().as_millis())
    } else {
        String::new()
    };

    let total = state.run_output.len();
    let count_label = format!("  {} lines", total);

    let rule_len = area
        .width
        .saturating_sub(status_tag.len() as u16 + file_label.len() as u16 + elapsed.len() as u16 + count_label.len() as u16 + 10);

    let line = Line::from(vec![
        Span::styled(format!(" {} ", status_tag), style::badge(status_color)),
        Span::styled("  ", style::screen()),
        Span::styled(&file_label, style::value()),
        Span::styled(&elapsed, style::label()),
        Span::styled(&count_label, style::label()),
        Span::styled(
            "─".repeat(rule_len as usize),
            style::rule_line(),
        ),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_body(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let visible = area.height as usize;
    let total = state.run_output.len();

    // Auto-scroll to bottom when scroll is at 0 (default)
    // or when the user hasn't scrolled up.
    if state.run_output_scroll == 0 {
        // Show the latest output — no skip from the end
    } else if state.run_output_scroll + visible > total {
        // Clamp so we don't scroll past the beginning
        state.run_output_scroll = total.saturating_sub(visible);
    }

    // run_output_scroll counts lines from the *bottom* (0 = pinned to bottom).
    // Convert to a skip-from-start offset.
    let skip = total.saturating_sub(visible + state.run_output_scroll);

    let start_time = state.run_start_time.unwrap_or_else(std::time::Instant::now);

    let items: Vec<ListItem> = state
        .run_output
        .iter()
        .skip(skip)
        .take(visible)
        .map(|entry| {
            let elapsed = entry.timestamp.duration_since(start_time);
            let ts = format!("+{:>7.2}s ", elapsed.as_secs_f64());
            let text_color = match entry.severity {
                LogCategory::Err => style::bad(),
                LogCategory::Jit => style::info(),
                _ => style::fg(),
            };
            ListItem::new(Line::from(vec![
                Span::styled(ts, style::fg_dim()),
                Span::styled(&entry.text, style::semantic(text_color)),
            ]))
        })
        .collect();

    if items.is_empty() {
        let hint = Line::from(vec![Span::styled(
            "  waiting for output...",
            style::label(),
        )]);
        frame.render_widget(Paragraph::new(hint), area);
    } else {
        frame.render_widget(List::new(items), area);
    }
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let is_active = matches!(
        state.run_status,
        RunStatus::Compiling | RunStatus::Running
    );

    let mut spans = vec![
        Span::styled(" esc", style::accent()),
        Span::styled(" back  ", style::label()),
        Span::styled("↑↓", style::accent()),
        Span::styled(" scroll  ", style::label()),
        Span::styled("pgup/pgdn", style::accent()),
        Span::styled(" page  ", style::label()),
    ];

    if is_active {
        spans.push(Span::styled("ctrl+6", style::accent()));
        spans.push(Span::styled(" stop  ", style::label()));
    }

    let scroll_indicator = if state.run_output_scroll > 0 {
        format!("  [+{} from bottom]", state.run_output_scroll)
    } else {
        "  [latest]".to_string()
    };
    spans.push(Span::styled(scroll_indicator, style::label()));

    let line = Line::from(spans);
    frame.render_widget(Paragraph::new(line), area);
}
