use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, Paragraph};
use ratatui::Frame;

use super::draw_activity_wave;
use crate::tui::state::run_state::RunStatus;
use crate::tui::state::{EditorMode, FilesFocus, TuiState};
use crate::tui::style;

pub(super) fn draw_files(frame: &mut Frame, state: &mut TuiState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(style::screen()), area);

    let show_cmdline = matches!(state.editor_mode, EditorMode::Command);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header / status line
            Constraint::Length(1), // activity wave
            Constraint::Min(1),   // body (tree + editor)
            Constraint::Length(1), // footer / command line
        ])
        .split(area);

    // ── status line (header) ─────────────────────────────────
    let filename = state
        .open_file
        .as_ref()
        .and_then(|p| p.strip_prefix(&state.workspace_root).ok())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(no file)".to_string());

    let (mode_label, mode_color) = style::editor_mode_token(state.editor_mode);

    let dirty_indicator = if state.file_dirty { " [+]" } else { "" };
    let cursor_pos = format!(
        "{}:{}",
        state.file_cursor_line + 1,
        state.file_cursor_col + 1,
    );
    let line_count = state.file_lines.len();
    let pct = if line_count <= 1 {
        "All".to_string()
    } else if state.file_cursor_line == 0 {
        "Top".to_string()
    } else if state.file_cursor_line >= line_count - 1 {
        "Bot".to_string()
    } else {
        format!("{}%", state.file_cursor_line * 100 / line_count)
    };

    let run_indicator = match state.run_status {
        RunStatus::Idle => String::new(),
        RunStatus::Compiling => "  [COMPILING...]".to_string(),
        RunStatus::Running => format!("  [RUNNING {}ms]", state.run_elapsed_ms),
        RunStatus::Succeeded => format!("  [OK {}ms]", state.run_elapsed_ms),
        RunStatus::Failed => "  [FAIL]".to_string(),
        RunStatus::Timeout => format!("  [TIMEOUT {}ms]", state.run_elapsed_ms),
    };
    let run_color = match state.run_status {
        RunStatus::Idle => style::fg_dim(),
        RunStatus::Compiling => style::warn(),
        RunStatus::Running => style::info(),
        RunStatus::Succeeded => style::good(),
        RunStatus::Failed | RunStatus::Timeout => style::bad(),
    };

    let lock_indicator = if let Some(ref owner) = state.agent_lock {
        format!("  [LOCKED:{}]", owner)
    } else {
        String::new()
    };

    let confirm_indicator = if let Some(ref pc) = state.pending_confirm {
        if !pc.is_expired() {
            "  [CONFIRM? y/n]".to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let mut header_spans = vec![
        Span::styled(
            format!(" {} ", mode_label),
            style::badge(mode_color),
        ),
        Span::styled("  ", style::spacer()),
        Span::styled(&filename, style::value()),
        Span::styled(dirty_indicator, style::semantic(style::warn())),
        Span::styled("  ", style::spacer()),
        Span::styled(&cursor_pos, style::label()),
        Span::styled("  ", style::spacer()),
        Span::styled(&pct, style::label()),
    ];
    if !run_indicator.is_empty() {
        header_spans.push(Span::styled(run_indicator, style::semantic(run_color)));
    }
    if !lock_indicator.is_empty() {
        header_spans.push(Span::styled(lock_indicator, style::semantic(style::warn())));
    }
    if !confirm_indicator.is_empty() {
        header_spans.push(Span::styled(confirm_indicator, style::semantic_bold(style::warn())));
    }
    let header = Line::from(header_spans);
    frame.render_widget(Paragraph::new(header), rows[0]);

    draw_activity_wave(frame, rows[1], state);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(rows[2]);

    draw_files_tree(frame, cols[0], state);

    // If run output is visible, split editor area 60/40
    if state.show_run_output {
        let editor_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(cols[1]);
        draw_files_editor(frame, editor_rows[0], state);
        draw_run_output(frame, editor_rows[1], state);
    } else {
        draw_files_editor(frame, cols[1], state);
    }

    // ── footer / command line ────────────────────────────────
    if show_cmdline {
        let before = &state.editor_cmdline[..state.editor_cmd_cursor];
        let after = &state.editor_cmdline[state.editor_cmd_cursor..];
        let (cursor_ch, rest) = if after.is_empty() {
            (" ", "")
        } else {
            let n = after.chars().next().map_or(1, |c| c.len_utf8());
            (&after[..n], &after[n..])
        };
        let cmdline = Line::from(vec![
            Span::styled(":", style::value_bright()),
            Span::styled(before, style::value_bright()),
            Span::styled(cursor_ch, style::cursor()),
            Span::styled(rest, style::value_bright()),
        ]);
        frame.render_widget(Paragraph::new(cmdline), rows[3]);
    } else {
        let footer_spans = match state.editor_mode {
            EditorMode::Normal => vec![
                Span::styled("i", style::accent()),
                Span::styled(" ins  ", style::label()),
                Span::styled(":", style::accent()),
                Span::styled(" cmd  ", style::label()),
                Span::styled("^5", style::accent()),
                Span::styled(" run  ", style::label()),
                Span::styled("^6", style::accent()),
                Span::styled(" stop  ", style::label()),
                Span::styled("^J", style::accent()),
                Span::styled(" output  ", style::label()),
                Span::styled("^O", style::accent()),
                Span::styled(" exit", style::label()),
            ],
            EditorMode::Insert => vec![
                Span::styled("esc", style::accent()),
                Span::styled(" normal  ", style::label()),
                Span::styled("ctrl+s", style::accent()),
                Span::styled(" save  ", style::label()),
                Span::styled("type to edit", style::label()),
            ],
            EditorMode::Visual => vec![
                Span::styled("esc", style::accent()),
                Span::styled(" cancel  ", style::label()),
                Span::styled("d", style::accent()),
                Span::styled(" delete  ", style::label()),
                Span::styled("y", style::accent()),
                Span::styled(" yank  ", style::label()),
                Span::styled("hjkl", style::accent()),
                Span::styled(" extend", style::label()),
            ],
            EditorMode::Command => vec![], // Should not reach here
        };
        let footer = Line::from(footer_spans);
        frame.render_widget(Paragraph::new(footer), rows[3]);
    }
}

fn draw_files_tree(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    // Breadcrumb: show current_dir relative to workspace
    let breadcrumb = state
        .current_dir
        .strip_prefix(&state.workspace_root)
        .map(|p| {
            let s = p.display().to_string();
            if s.is_empty() { "/".to_string() } else { format!("/{}", s) }
        })
        .unwrap_or_else(|_| "/".to_string());

    let title = Line::from(vec![
        Span::styled(" files ", style::files_tree_title(state.files_focus)),
        Span::styled(&breadcrumb, style::label()),
        Span::styled(
            "─".repeat(area.width.saturating_sub(7 + breadcrumb.len() as u16) as usize),
            style::rule_line(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let body = Rect::new(
        area.x,
        area.y + 1,
        area.width,
        area.height.saturating_sub(1),
    );
    state.files_tree_area = body;
    if body.height == 0 {
        return;
    }

    let visible = body.height as usize;
    if state.file_cursor < state.file_list_offset {
        state.file_list_offset = state.file_cursor;
    } else if state.file_cursor >= state.file_list_offset + visible {
        state.file_list_offset = state.file_cursor + 1 - visible;
    }

    let items: Vec<ListItem> = state
        .file_tree
        .visible
        .iter()
        .enumerate()
        .skip(state.file_list_offset)
        .take(visible)
        .map(|(idx, entry)| {
            let indent = "  ".repeat(entry.depth);
            let icon = if entry.is_dir {
                if state.file_tree.expanded.contains(&entry.path) {
                    "v "
                } else {
                    "> "
                }
            } else {
                "  "
            };
            let name = entry
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            let suffix = if entry.is_dir { "/" } else { "" };
            let label = format!("{}{}{}{}", indent, icon, name, suffix);

            let item_style = if idx == state.file_cursor {
                style::badge(style::info())
            } else if entry.is_dir {
                style::semantic(style::info())
            } else {
                style::value()
            };
            ListItem::new(Line::from(vec![Span::styled(label, item_style)]))
        })
        .collect();
    frame.render_widget(List::new(items), body);
}

fn draw_files_editor(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let (mode_tag, _) = style::editor_mode_tag(state.editor_mode);

    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", mode_tag),
            style::files_editor_badge(state.editor_mode, state.files_focus),
        ),
        Span::styled(
            "─".repeat(area.width.saturating_sub(6) as usize),
            style::rule_line(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let body = Rect::new(
        area.x,
        area.y + 1,
        area.width,
        area.height.saturating_sub(1),
    );
    state.files_editor_area = body;
    if body.height == 0 {
        return;
    }

    // Vertical scroll — keep cursor visible
    if state.file_cursor_line < state.file_scroll {
        state.file_scroll = state.file_cursor_line;
    } else if state.file_cursor_line >= state.file_scroll + body.height as usize {
        state.file_scroll = state.file_cursor_line + 1 - body.height as usize;
    }

    // Horizontal scroll — keep cursor visible
    let gutter_w: usize = 6; // "12345 " = 6 chars
    let text_w = (body.width as usize).saturating_sub(gutter_w);
    if text_w > 0 {
        if state.file_cursor_col < state.file_hscroll {
            state.file_hscroll = state.file_cursor_col;
        } else if state.file_cursor_col >= state.file_hscroll + text_w {
            state.file_hscroll = state.file_cursor_col + 1 - text_w;
        }
    }

    let is_focused = matches!(state.files_focus, FilesFocus::Editor);
    let is_insert = matches!(state.editor_mode, EditorMode::Insert);

    let mut items: Vec<ListItem> = Vec::new();
    for line_idx in state.file_scroll..state.file_lines.len().min(state.file_scroll + body.height as usize)
    {
        let line_no = format!("{:>5} ", line_idx + 1);
        let line = &state.file_lines[line_idx];
        let chars: Vec<char> = line.chars().collect();
        let mut spans = vec![Span::styled(
            line_no,
            style::semantic(if line_idx == state.file_cursor_line {
                style::fg_bright()
            } else {
                style::fg_dim()
            }),
        )];

        // Render visible portion of line with horizontal scroll
        let visible_start = state.file_hscroll;
        let visible_end = (state.file_hscroll + text_w).min(chars.len());

        for col in visible_start..visible_end {
            let ch = chars[col];
            let mut cell_style = style::value();
            if state.is_selected_char(line_idx, col) {
                cell_style = cell_style.bg(style::selection());
            }
            if line_idx == state.file_cursor_line && col == state.file_cursor_col && is_focused {
                // Cursor style: block cursor in Normal/Visual, thin cursor in Insert
                if is_insert {
                    cell_style = style::cursor();
                } else {
                    cell_style = style::badge(style::info());
                }
            }
            spans.push(Span::styled(ch.to_string(), cell_style));
        }

        // Cursor past end of line
        if line_idx == state.file_cursor_line && is_focused {
            if state.file_cursor_col >= chars.len()
                && state.file_cursor_col >= visible_start
                && state.file_cursor_col < visible_start + text_w
            {
                if is_insert {
                    spans.push(Span::styled(" ", style::cursor()));
                } else {
                    spans.push(Span::styled(" ", style::badge(style::info())));
                }
            }
        }

        items.push(ListItem::new(Line::from(spans)));
    }

    // Fill remaining rows with tildes (vim-style)
    let rendered = items.len();
    for _ in rendered..body.height as usize {
        items.push(ListItem::new(Line::from(vec![
            Span::styled("    ~ ", style::label()),
        ])));
    }

    if state.file_lines.len() == 1 && state.file_lines[0].is_empty() && state.open_file.is_none() {
        // No file open — show hint
        if items.len() > 2 {
            items[2] = ListItem::new(Line::from(vec![Span::styled(
                "      (no file open — select from tree or :e <path>)",
                style::label(),
            )]));
        }
    }

    frame.render_widget(List::new(items), body);
}

fn draw_run_output(frame: &mut Frame, area: Rect, state: &mut TuiState) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let (status_tag, status_color) = match state.run_status {
        RunStatus::Idle => ("IDLE", style::fg_dim()),
        RunStatus::Compiling => ("COMPILING...", style::warn()),
        RunStatus::Running => ("RUNNING...", style::info()),
        RunStatus::Succeeded => ("OK", style::good()),
        RunStatus::Failed => ("FAIL", style::bad()),
        RunStatus::Timeout => ("TIMEOUT", style::bad()),
    };

    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", status_tag),
            style::badge(status_color),
        ),
        Span::styled(" output ", style::label()),
        Span::styled(
            "─".repeat(area.width.saturating_sub(status_tag.len() as u16 + 12) as usize),
            style::rule_line(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let body = Rect::new(
        area.x,
        area.y + 1,
        area.width,
        area.height.saturating_sub(1),
    );
    if body.height == 0 {
        return;
    }

    let visible = body.height as usize;
    let total = state.run_output.len();

    // Auto-scroll to bottom unless user scrolled up
    if state.run_output_scroll == 0 || state.run_output_scroll + visible >= total {
        state.run_output_scroll = total.saturating_sub(visible);
    }

    let start_time = state.run_start_time.unwrap_or_else(std::time::Instant::now);

    let items: Vec<ListItem> = state
        .run_output
        .iter()
        .skip(state.run_output_scroll)
        .take(visible)
        .map(|line| {
            let elapsed = line.timestamp.duration_since(start_time);
            let ts = format!("+{:.2}s ", elapsed.as_secs_f64());
            let text_color = match line.severity {
                crate::events::LogCategory::Err => style::bad(),
                crate::events::LogCategory::Jit => style::info(),
                _ => style::fg(),
            };
            ListItem::new(Line::from(vec![
                Span::styled(ts, style::fg_dim()),
                Span::styled(&line.text, style::semantic(text_color)),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), body);
}
