use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::tui::state::TuiState;
use crate::tui::style;
use crate::tui::widgets::{BrailleWaveform, HistogramSparkline, PipelineBar, TensorHeatmap};

pub(in crate::tui::layout) fn draw_pipeline(frame: &mut Frame, area: Rect, state: &TuiState) {
    let pl = match &state.last_pipeline {
        Some(p) => p,
        None => return,
    };

    for (i, stage) in pl.stages.iter().enumerate() {
        if i >= area.height as usize {
            break;
        }

        let total = pl.total_ms.max(1) as f32;
        let color = style::pipeline_stage_color(&stage.name);

        frame.render_widget(
            PipelineBar {
                label: stage.name.clone(),
                offset_frac: stage.offset_ms as f32 / total,
                width_frac: stage.duration_ms as f32 / total,
                duration_ms: stage.duration_ms,
                bar_color: color,
                bg_color: style::bar_empty(),
                label_color: style::fg_dim(),
                text_color: style::fg(),
            },
            Rect::new(area.x, area.y + i as u16, area.width, 1),
        );
    }
}

pub(in crate::tui::layout) fn draw_tensor(
    frame: &mut Frame,
    area: Rect,
    state: &TuiState,
) -> Rect {
    let tv = match &state.last_tensor {
        Some(tv) => tv,
        None => return Rect::default(),
    };

    if area.height <= 2 {
        let line = Line::from(vec![
            Span::styled("shape ", style::label()),
            Span::styled(format!("{:?}", tv.shape), style::value()),
            Span::styled("  min ", style::label()),
            Span::styled(format!("{:.4}", tv.min), style::value()),
            Span::styled("  max ", style::label()),
            Span::styled(format!("{:.4}", tv.max), style::value()),
            Span::styled("  mean ", style::label()),
            Span::styled(format!("{:.4}", tv.mean), style::value()),
        ]);
        frame.render_widget(Paragraph::new(line), area);
        return area;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);

    let header = Line::from(vec![
        Span::styled("shape ", style::label()),
        Span::styled(format!("{:?}", tv.shape), style::value()),
        Span::styled("  range ", style::label()),
        Span::styled(
            format!("{:.4}..{:.4}", tv.min, tv.max),
            style::value(),
        ),
        Span::styled("  mean ", style::label()),
        Span::styled(format!("{:.4}", tv.mean), style::value()),
    ]);
    frame.render_widget(Paragraph::new(header), rows[0]);

    frame.render_widget(
        HistogramSparkline {
            label: "dist",
            bins: &tv.histogram,
            label_color: style::fg_dim(),
        },
        rows[1],
    );

    frame.render_widget(
        BrailleWaveform {
            label: "wave",
            data: &tv.samples,
            min_val: tv.min,
            max_val: tv.max,
            color: style::info(),
            label_color: style::fg_dim(),
        },
        rows[2],
    );

    if rows[3].height > 0 {
        frame.render_widget(
            TensorHeatmap {
                label: "heat",
                samples: &tv.samples,
                min: tv.min,
                max: tv.max,
                label_color: style::fg_dim(),
            },
            rows[3],
        );
    }

    area
}

/// Render per-stream busy/idle timeline using block characters.
pub(in crate::tui::layout) fn draw_streams(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 || state.stream_count == 0 {
        return;
    }

    let max_streams = area.height as usize;
    let busy_color = style::info();
    let idle_color = style::bar_empty();

    for (i, activity) in state.stream_activity.iter().enumerate() {
        if i >= max_streams {
            break;
        }

        let label = format!("s{:<2}", i);
        let mut spans = vec![
            Span::styled(label, style::label()),
            Span::raw(" "),
        ];

        // How many columns for the timeline
        let timeline_w = area.width.saturating_sub(4) as usize;
        let skip = if activity.len() > timeline_w {
            activity.len() - timeline_w
        } else {
            0
        };

        for &active in activity.iter().skip(skip) {
            if active {
                spans.push(Span::styled("▓", style::semantic(busy_color)));
            } else {
                spans.push(Span::styled("░", style::semantic(idle_color)));
            }
        }

        // Pad if timeline is shorter than available width
        let drawn = activity.len().saturating_sub(skip);
        if drawn < timeline_w {
            let pad = "░".repeat(timeline_w - drawn);
            spans.push(Span::styled(pad, style::semantic(idle_color)));
        }

        frame.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect::new(area.x, area.y + i as u16, area.width, 1),
        );
    }
}
