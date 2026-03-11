mod command;
mod insert;
mod normal;
mod visual;

use std::sync::mpsc::Sender;
use std::sync::Arc;

use crossterm::event::KeyCode;
use ratatui::layout::Rect;

use super::commands::run as run_cmds;
use super::state::{EditorMode, FilesFocus, TuiState};
use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;

pub(super) fn handle_files_key(
    state: &mut TuiState,
    code: KeyCode,
    ctrl: bool,
    shift: bool,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
    daemon: &Arc<DaemonState>,
) -> bool {
    // Global shortcuts available in all editor modes
    if ctrl {
        match code {
            KeyCode::Char('5') => {
                // Ctrl+5: Run current file via ptx-runner
                run_cmds::cmd_run_file(state, daemon, tx, &[]);
                return true;
            }
            KeyCode::Char('6') => {
                // Ctrl+6: Stop running process
                run_cmds::cmd_stop_run(state);
                return true;
            }
            KeyCode::Char('s') => {
                match state.file_save() {
                    Ok(()) => state.push_log(LogEntry::new(LogCategory::Sys, "saved file")),
                    Err(e) => state.push_log(LogEntry::new(LogCategory::Err, e)),
                }
                return true;
            }
            KeyCode::Char('j') => {
                // Toggle run output panel
                run_cmds::cmd_toggle_output(state);
                return true;
            }
            _ => {}
        }
    }

    // Tree focus uses same keybinds regardless of editor mode
    if matches!(state.files_focus, FilesFocus::Tree) {
        return handle_files_tree_key(state, code, ctrl);
    }

    // Editor focus: route by editor mode
    match state.editor_mode {
        EditorMode::Normal => normal::handle_editor_normal(state, code, ctrl, runner, tx),
        EditorMode::Insert => insert::handle_editor_insert(state, code, ctrl, shift),
        EditorMode::Visual => visual::handle_editor_visual(state, code, ctrl),
        EditorMode::Command => command::handle_editor_command(state, code, ctrl, runner, tx),
    }
}

fn handle_files_tree_key(state: &mut TuiState, code: KeyCode, ctrl: bool) -> bool {
    match code {
        KeyCode::Up | KeyCode::Char('k') => state.file_move_cursor(1),
        KeyCode::Down | KeyCode::Char('j') => state.file_move_cursor(-1),
        KeyCode::PageUp => state.file_move_cursor(12),
        KeyCode::PageDown => state.file_move_cursor(-12),
        KeyCode::Enter => {
            if let Err(e) = state.open_selected_file() {
                state.push_log(LogEntry::new(LogCategory::Err, e));
            }
        }
        KeyCode::Char('l') => {
            // Expand dir or open file
            let is_dir = state
                .file_tree
                .visible
                .get(state.file_cursor)
                .map_or(false, |e| e.is_dir);
            if is_dir {
                state.file_tree.expand(state.file_cursor);
            } else {
                state.files_toggle_focus();
            }
        }
        KeyCode::Char('h') => {
            // Collapse dir or navigate to parent
            if let Some(parent_idx) = state.file_tree.collapse(state.file_cursor) {
                state.file_cursor = parent_idx;
            }
        }
        KeyCode::Char('r') if !ctrl => {
            state.reload_file_entries();
            state.push_log(LogEntry::new(LogCategory::Sys, "file index refreshed"));
        }
        KeyCode::Tab => state.files_toggle_focus(),
        KeyCode::Esc | KeyCode::Char('q') => state.toggle_ui_mode(),
        _ => return false,
    }
    true
}

pub(super) fn handle_files_mouse_down(state: &mut TuiState, col: u16, row: u16) {
    let c = col;
    let r = row;
    if point_in_rect(c, r, state.files_tree_area) {
        state.files_focus = FilesFocus::Tree;
        let rel_row = r.saturating_sub(state.files_tree_area.y) as usize;
        let idx = state.file_list_offset + rel_row;
        if idx < state.file_tree.len() {
            state.file_cursor = idx;
        }
    } else if point_in_rect(c, r, state.files_editor_area) {
        state.files_focus = FilesFocus::Editor;
        let rel_row = r.saturating_sub(state.files_editor_area.y) as usize;
        let line = (state.file_scroll + rel_row).min(state.file_lines.len().saturating_sub(1));
        let rel_col = c
            .saturating_sub(state.files_editor_area.x)
            .saturating_sub(7) as usize;
        let max_col = state
            .file_lines
            .get(line)
            .map(|s| s.chars().count())
            .unwrap_or(0);
        let col = rel_col.min(max_col);
        state.file_cursor_line = line;
        state.file_cursor_col = col;
        state.set_selection_anchor(line, col);
    }
}

pub(super) fn handle_files_mouse_drag(state: &mut TuiState, col: u16, row: u16) {
    if !state.selecting || !point_in_rect(col, row, state.files_editor_area) {
        return;
    }
    let rel_row = row.saturating_sub(state.files_editor_area.y) as usize;
    let line = (state.file_scroll + rel_row).min(state.file_lines.len().saturating_sub(1));
    let rel_col = col
        .saturating_sub(state.files_editor_area.x)
        .saturating_sub(7) as usize;
    let max_col = state
        .file_lines
        .get(line)
        .map(|s| s.chars().count())
        .unwrap_or(0);
    state.file_cursor_line = line;
    state.file_cursor_col = rel_col.min(max_col);
    state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
}

fn point_in_rect(col: u16, row: u16, area: Rect) -> bool {
    col >= area.x
        && col < area.x.saturating_add(area.width)
        && row >= area.y
        && row < area.y.saturating_add(area.height)
}
