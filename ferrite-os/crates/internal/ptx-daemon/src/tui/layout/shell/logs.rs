use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, Paragraph};
use ratatui::Frame;

use super::super::{fmt_elapsed, truncate_for_log};
use crate::events::LogCategory;
use crate::tui::state::{Panel, TuiState};
use crate::tui::style;

pub(in crate::tui::layout) fn draw_logs(
    frame: &mut Frame,
    area: Rect,
    state: &TuiState,
    compact: bool,
) {
    if area.height == 0 {
        return;
    }

    let mut run = 0usize;
    let mut app = 0usize;
    let mut err = 0usize;

    for entry in &state.log {
        match entry.category {
            LogCategory::Sys | LogCategory::Jit => {}
            LogCategory::Run => run += 1,
            LogCategory::App => app += 1,
            LogCategory::Err => err += 1,
        }
    }

    let verbose = state.focus == Panel::Processes && !compact;
    let mode = if verbose { "detail" } else { "compact" };

    let summary = Line::from(vec![
        Span::styled("logs ", style::label()),
        Span::styled(mode, style::accent_bold()),
        Span::styled("  err ", style::label()),
        Span::styled(
            format!("{}", err),
            style::semantic(if err > 0 { style::bad() } else { style::good() }),
        ),
        Span::styled("  app ", style::label()),
        Span::styled(format!("{}", app), style::semantic(style::warn())),
        Span::styled("  run ", style::label()),
        Span::styled(format!("{}", run), style::semantic(style::good())),
        Span::styled("  tab:toggle", style::label()),
    ]);
    frame.render_widget(
        Paragraph::new(summary),
        Rect::new(area.x, area.y, area.width, 1),
    );

    if area.height <= 1 {
        return;
    }

    let visible = area.height as usize - 1;

    let items: Vec<ListItem> = state
        .log
        .iter()
        .rev()
        .skip(state.log_scroll)
        .filter(|entry| {
            if verbose {
                true
            } else {
                !matches!(entry.category, LogCategory::Jit | LogCategory::Run)
            }
        })
        .take(visible)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|entry| {
            let elapsed = fmt_elapsed(entry.timestamp.elapsed().as_secs());
            let (tag_style, default_msg_style) = style::log_styles(entry.category);
            let clickable = entry.action.is_some();
            let msg_style = if clickable { style::accent() } else { default_msg_style };
            let prefix = if clickable { "▸ " } else { "  " };

            ListItem::new(Line::from(vec![
                Span::styled(format!("{:>5} ", elapsed), style::label()),
                Span::styled(format!("{:<4}", entry.category), tag_style),
                Span::styled(prefix, if clickable { style::accent() } else { style::label() }),
                Span::styled(
                    if verbose {
                        entry.message.clone()
                    } else {
                        truncate_for_log(&entry.message, area.width.saturating_sub(16) as usize)
                    },
                    msg_style,
                ),
            ]))
        })
        .collect();

    let body = Rect::new(area.x, area.y + 1, area.width, area.height - 1);

    if items.is_empty() {
        // Empty-state guidance when no log entries are visible.
        let hint = Line::from(vec![
            Span::styled(
                "      no events yet — type a command or run a demo to get started",
                style::label(),
            ),
        ]);
        frame.render_widget(Paragraph::new(hint), body);
    } else {
        frame.render_widget(List::new(items), body);
    }
}
