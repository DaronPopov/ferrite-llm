mod detail;
mod header;
mod logs;
mod prompt;
mod sections;

use std::time::Instant;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Block;
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use tachyonfx::Shader;

use super::{draw_activity_wave, draw_section_header, section_body};
use crate::tui::state::TuiState;
use crate::tui::style;

#[derive(Clone, Copy)]
enum SysmonSection {
    System,
    Memory,
    Execution,
    Profiling,
    Detail,
    Logs,
}

#[derive(Clone, Copy)]
struct SysmonSectionSpec {
    kind: SysmonSection,
    height: u16,
}

fn section_label(kind: SysmonSection) -> &'static str {
    match kind {
        SysmonSection::System => "System",
        SysmonSection::Memory => "Memory",
        SysmonSection::Execution => "Execution",
        SysmonSection::Profiling => "Profiling",
        SysmonSection::Detail => "Detail",
        SysmonSection::Logs => "Logs",
    }
}

fn draw_section_index(
    frame: &mut Frame,
    area: Rect,
    specs: &[SysmonSectionSpec],
    start: usize,
    end: usize,
) {
    if area.width < 12 || area.height == 0 || specs.is_empty() {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![Span::styled(" index", style::label_bold())]));
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {}-{} / {}", start + 1, end + 1, specs.len()),
            style::label(),
        ),
    ]));

    let rows_for_items = area.height.saturating_sub(lines.len() as u16) as usize;
    if rows_for_items == 0 {
        frame.render_widget(Paragraph::new(lines), area);
        return;
    }

    let win_start = if specs.len() <= rows_for_items {
        0
    } else {
        start.min(specs.len() - rows_for_items)
    };
    let win_end = (win_start + rows_for_items).min(specs.len());

    for (i, spec) in specs.iter().enumerate().take(win_end).skip(win_start) {
        let marker = if i == start {
            "▸"
        } else if i <= end {
            "•"
        } else {
            " "
        };
        let item_style = if i == start {
            style::accent_bold()
        } else if i <= end {
            style::value()
        } else {
            style::label()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", marker), style::label()),
            Span::styled(section_label(spec.kind), item_style),
        ]));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

pub(super) fn draw_sysmon(frame: &mut Frame, state: &mut TuiState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(style::screen()), area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: area.height,
    };

    let (compact, ultra_compact) = match state.ui_density {
        crate::tui::state::UiDensity::Auto => (inner.height < 32, inner.height < 24),
        crate::tui::state::UiDensity::Compact => (true, true),
        crate::tui::state::UiDensity::Balanced => (true, false),
        crate::tui::state::UiDensity::Comfortable => (false, false),
    };

    // Section heights are fixed by section type for stable visual rhythm.
    // On short terminals, we render a viewport (scrollable section window).
    let header_h: u16 = 2;
    let wave_h: u16 = 1;
    let prompt_h: u16 = 1;
    let system_h: u16 = if ultra_compact { 2 } else { 3 };
    let owner_extra = if !ultra_compact && !state.vram_owners.is_empty() {
        state.vram_owners.len().min(4) as u16
    } else {
        0
    };
    let memory_h: u16 = if ultra_compact { 3 } else { 4 + owner_extra };
    let exec_h: u16 = if ultra_compact { 2 } else if compact { 3 } else { 4 };
    let profile_h: u16 = if ultra_compact { 0 } else { 3 };
    // Tensor and stream details only shown in detail mode to reduce default noise
    let detail_h: u16 = if state.detail_mode && !ultra_compact {
        if compact { 3 } else { 5 }
    } else {
        0
    };

    let mut specs = Vec::with_capacity(6);
    specs.push(SysmonSectionSpec {
        kind: SysmonSection::System,
        height: system_h,
    });
    specs.push(SysmonSectionSpec {
        kind: SysmonSection::Memory,
        height: memory_h,
    });
    specs.push(SysmonSectionSpec {
        kind: SysmonSection::Execution,
        height: exec_h,
    });
    if profile_h > 0 {
        specs.push(SysmonSectionSpec {
            kind: SysmonSection::Profiling,
            height: profile_h,
        });
    }
    if detail_h > 0 {
        specs.push(SysmonSectionSpec {
            kind: SysmonSection::Detail,
            height: detail_h,
        });
    }
    specs.push(SysmonSectionSpec {
        kind: SysmonSection::Logs,
        height: if compact { 6 } else { 8 },
    });

    let chrome_h = header_h + wave_h + prompt_h;
    let body_h = inner.height.saturating_sub(chrome_h);
    let total_sections_h: u16 = specs.iter().map(|s| s.height).sum();
    let max_scroll = if body_h == 0 || total_sections_h <= body_h {
        0
    } else {
        specs.len().saturating_sub(1)
    };
    state.set_sysmon_scroll_bounds(max_scroll);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_h),
            Constraint::Length(wave_h),
            Constraint::Min(0),
            Constraint::Length(prompt_h),
        ])
        .split(inner);

    header::draw_title(frame, rows[0], state);
    draw_activity_wave(frame, rows[1], state);
    let body_area = rows[2];
    prompt::draw_prompt(frame, rows[3], state);

    let body_cols = if body_area.width >= 40 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(16)])
            .split(body_area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(0)])
            .split(body_area)
    };
    let content_area = body_cols[0];
    let index_area = body_cols[1];

    let mut pipeline_fx_area = Rect::default();
    let mut tensor_fx_area = Rect::default();

    if content_area.height > 0 && !specs.is_empty() {
        let start = state.sysmon_section_scroll.min(specs.len().saturating_sub(1));
        let mut last_rendered_idx = start;
        let mut y = content_area.y;
        let mut remaining = content_area.height;
        let mut drew_any = false;

        for (offset, spec) in specs.iter().skip(start).enumerate() {
            if remaining == 0 {
                break;
            }
            let h = if spec.height <= remaining {
                spec.height
            } else if drew_any {
                break;
            } else {
                remaining
            };
            if h == 0 {
                break;
            }

            let section_rect = Rect::new(content_area.x, y, content_area.width, h);
            match spec.kind {
                SysmonSection::System => {
                    draw_section_header(frame, section_rect, "System");
                    sections::draw_system_status(frame, section_body(section_rect), state);
                }
                SysmonSection::Memory => {
                    draw_section_header(frame, section_rect, "Memory");
                    sections::draw_memory(frame, section_body(section_rect), state);
                }
                SysmonSection::Execution => {
                    draw_section_header(frame, section_rect, "Execution");
                    let pipeline_h = if state.detail_mode && section_rect.height > 2 {
                        2u16
                    } else {
                        0u16
                    };
                    pipeline_fx_area =
                        sections::draw_execution(frame, section_body(section_rect), state, pipeline_h);
                }
                SysmonSection::Profiling => {
                    draw_section_header(frame, section_rect, "Profiling");
                    sections::draw_profiling(frame, section_body(section_rect), state);
                }
                SysmonSection::Detail => {
                    draw_section_header(frame, section_rect, "Detail");
                    let detail_area = section_body(section_rect);
                    if state.last_tensor.is_some() {
                        tensor_fx_area = detail::draw_tensor(frame, detail_area, state);
                    } else if state.last_pipeline.is_some() {
                        detail::draw_pipeline(frame, detail_area, state);
                    } else {
                        detail::draw_streams(frame, detail_area, state);
                    }
                }
                SysmonSection::Logs => {
                    draw_section_header(frame, section_rect, "Logs");
                    let log_body = section_body(section_rect);
                    // Store log body area for click hit-testing.
                    if log_body.height > 1 {
                        state.log_body_area = Rect::new(log_body.x, log_body.y + 1, log_body.width, log_body.height - 1);
                    } else {
                        state.log_body_area = Rect::default();
                    }
                    logs::draw_logs(frame, log_body, state, compact);
                }
            }

            drew_any = true;
            last_rendered_idx = start + offset;
            y = y.saturating_add(h);
            remaining = remaining.saturating_sub(h);
        }

        draw_section_index(frame, index_area, &specs, start, last_rendered_idx);
    }

    let now = Instant::now();
    let delta = now.duration_since(state.last_frame);
    state.last_frame = now;

    let buf = frame.buffer_mut();
    for (name, effect) in state.effects.iter_mut() {
        let effect_area = match *name {
            "tensor" => tensor_fx_area,
            "pipeline" => pipeline_fx_area,
            _ => continue,
        };
        if effect_area.width > 0 && effect_area.height > 0 {
            effect.process(delta.into(), buf, effect_area);
        }
    }
    state.effects.retain(|(_, effect)| !effect.done());
}
