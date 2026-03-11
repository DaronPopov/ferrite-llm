use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::super::fmt_uptime;
use crate::tui::state::TuiState;
use crate::tui::style;
use crate::tui::widgets::{Heartbeat, Spinner};

pub(in crate::tui::layout) fn draw_title(frame: &mut Frame, area: Rect, state: &TuiState) {
    let hb = Heartbeat;
    let status = style::status_token(state.running, state.watchdog);

    let vc = style::vram_color(state.display_vram_pct);
    let pool = style::pool_token(
        !state.running || state.watchdog,
        state.pool_needs_defrag,
    );

    let mut spans = vec![
        Span::styled("ferrite", style::accent_bold()),
        Span::styled(" daemon", style::label()),
        Span::styled("  ", style::spacer()),
        Span::styled(hb.symbol(), style::semantic(status.1)),
        Span::styled(" ", style::spacer()),
        Span::styled(
            status.0,
            style::semantic_bold(status.1),
        ),
        Span::styled("  ", style::spacer()),
        Span::styled("gpu ", style::label()),
        Span::styled(
            format!("{:.1}%", state.display_gpu_util),
            style::value(),
        ),
        Span::styled("  vram ", style::label()),
        Span::styled(
            format!("{:.1}%", state.display_vram_pct),
            style::semantic(vc),
        ),
        Span::styled("  pool ", style::label()),
        Span::styled(pool.0.to_string(), style::semantic(pool.1)),
    ];
    spans.push(Span::styled("  ", style::spacer()));
    spans.push(Span::styled(
        format!("up {}", fmt_uptime(state.uptime_secs)),
        style::label(),
    ));

    frame.render_widget(
        Paragraph::new(Line::from(spans)),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let spinner = Spinner {
        tick: state.tick_count,
    };
    let proc_line = if state.processes.is_empty() {
        Line::from(vec![
            Span::styled("apps  ", style::label()),
            Span::styled("none", style::label()),
            Span::styled(
                format!("  device {}", state.device_name),
                style::label(),
            ),
        ])
    } else {
        let mut line = vec![Span::styled("apps  ", style::label())];
        for (idx, app) in state.processes.iter().enumerate() {
            if idx > 0 {
                line.push(Span::styled("  ", style::spacer()));
            }
            line.push(Span::styled(spinner.frame(), style::semantic(style::good())));
            line.push(Span::styled(
                format!(" {}", app.name),
                style::value(),
            ));
        }
        line.push(Span::styled(
            format!("  device {}", state.device_name),
            style::label(),
        ));
        Line::from(line)
    };

    if area.height > 1 {
        frame.render_widget(
            Paragraph::new(proc_line),
            Rect::new(area.x, area.y + 1, area.width, 1),
        );
    }
}
