use crossterm::event::KeyCode;

use crate::events::{LogCategory, LogEntry};
use crate::tui::state::{EditorMode, TuiState};

pub(super) fn handle_editor_insert(state: &mut TuiState, code: KeyCode, ctrl: bool, shift: bool) -> bool {
    if ctrl {
        match code {
            KeyCode::Char('v') => {
                let clip = state.clipboard.clone();
                if !clip.is_empty() {
                    state.push_undo();
                    state.file_paste(&clip);
                }
                return true;
            }
            KeyCode::Char('c') => {
                if state.file_copy_selection() {
                    state.push_log(LogEntry::new(LogCategory::Sys, "copied selection"));
                }
                return true;
            }
            _ => {}
        }
    }

    match code {
        KeyCode::Esc => {
            state.editor_mode = EditorMode::Normal;
            state.clear_selection();
            // In normal mode cursor can't be past last char
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        KeyCode::Enter => {
            state.push_undo();
            state.file_insert_newline();
        }
        KeyCode::Backspace => {
            state.push_undo();
            state.file_backspace();
        }
        KeyCode::Delete => {
            state.push_undo();
            state.file_delete();
        }
        KeyCode::Up => {
            state.file_move_cursor_line(1);
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::Down => {
            state.file_move_cursor_line(-1);
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::Left => {
            state.file_move_cursor_col(1);
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::Right => {
            state.file_move_cursor_col(-1);
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::Home => {
            state.file_cursor_col = 0;
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::End => {
            state.file_cursor_col = state
                .file_lines
                .get(state.file_cursor_line)
                .map(|s| s.chars().count())
                .unwrap_or(0);
            if shift {
                state.update_selection_head(state.file_cursor_line, state.file_cursor_col);
            } else {
                state.clear_selection();
            }
        }
        KeyCode::PageUp => state.file_scroll_lines(12),
        KeyCode::PageDown => state.file_scroll_lines(-12),
        KeyCode::Tab => {
            // Insert 4 spaces (soft tab) in insert mode
            state.push_undo();
            for _ in 0..4 {
                state.file_insert_char(' ');
            }
        }
        KeyCode::Char(ch) => {
            state.push_undo();
            state.file_insert_char(ch);
        }
        _ => return false,
    }
    true
}
