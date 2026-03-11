use crossterm::event::KeyCode;

use crate::events::{LogCategory, LogEntry};
use crate::tui::state::{EditorMode, TuiState};

pub(super) fn handle_editor_visual(state: &mut TuiState, code: KeyCode, ctrl: bool) -> bool {
    if ctrl {
        match code {
            KeyCode::Char('c') => {
                if state.file_copy_selection() {
                    state.push_log(LogEntry::new(LogCategory::Sys, "copied selection"));
                }
                state.editor_mode = EditorMode::Normal;
                state.clear_selection();
                return true;
            }
            _ => {}
        }
    }

    match code {
        KeyCode::Esc | KeyCode::Char('v') => {
            state.editor_mode = EditorMode::Normal;
            state.clear_selection();
        }
        // Motions — extend selection
        KeyCode::Char('h') | KeyCode::Left => {
            if state.file_cursor_col > 0 {
                state.file_cursor_col -= 1;
                state.preferred_col = state.file_cursor_col;
            }
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('l') | KeyCode::Right => {
            let line_len = state
                .file_lines
                .get(state.file_cursor_line)
                .map(|l| l.chars().count())
                .unwrap_or(0);
            if state.file_cursor_col + 1 < line_len {
                state.file_cursor_col += 1;
                state.preferred_col = state.file_cursor_col;
            }
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('j') | KeyCode::Down => {
            state.vim_move_line(1);
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('k') | KeyCode::Up => {
            state.vim_move_line(-1);
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('w') => {
            state.word_forward();
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('b') => {
            state.word_backward();
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('e') => {
            state.word_end();
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('0') | KeyCode::Home => {
            state.file_cursor_col = 0;
            state.preferred_col = 0;
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('$') | KeyCode::End => {
            let line_len = state
                .file_lines
                .get(state.file_cursor_line)
                .map(|l| l.chars().count())
                .unwrap_or(0);
            state.file_cursor_col = line_len.saturating_sub(1);
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('G') => {
            state.file_cursor_line = state.file_lines.len().saturating_sub(1);
            state.clamp_cursor_normal();
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        KeyCode::Char('g') => {
            // gg in visual mode
            state.file_cursor_line = 0;
            state.clamp_cursor_normal();
            state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
        }
        // Actions on selection
        KeyCode::Char('d') | KeyCode::Char('x') => {
            state.push_undo();
            if state.file_copy_selection() {
                state.push_log(LogEntry::new(LogCategory::Sys, "deleted selection"));
            }
            state.delete_selection_if_any();
            state.editor_mode = EditorMode::Normal;
            state.clamp_cursor_normal();
        }
        KeyCode::Char('y') => {
            if state.file_copy_selection() {
                state.push_log(LogEntry::new(LogCategory::Sys, "yanked selection"));
            }
            state.editor_mode = EditorMode::Normal;
            state.clear_selection();
        }
        _ => return false,
    }
    true
}
