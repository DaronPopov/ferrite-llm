use std::sync::mpsc::Sender;
use std::sync::Arc;

use crossterm::event::KeyCode;

use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::tui::editor::keymap::{EditorAction, KeyPress};
use crate::tui::editor::search::SearchDirection;
use crate::tui::state::{EditorMode, FindCharState, PendingOperator, TuiState};

pub(super) fn handle_editor_normal(
    state: &mut TuiState,
    code: KeyCode,
    ctrl: bool,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) -> bool {
    // ── macro recording: capture every key ──
    state.macro_engine.record_key(code, ctrl, false);

    // ── count prefix accumulation (e.g. 5j, 12G) ──
    if !ctrl {
        if let KeyCode::Char(c) = code {
            if c.is_ascii_digit() && (state.count_prefix.is_some() || c != '0') {
                let digit = (c as u8 - b'0') as usize;
                let current = state.count_prefix.unwrap_or(0);
                state.count_prefix = Some(current * 10 + digit);
                return true;
            }
        }
    }

    let count = state.count_prefix.take().unwrap_or(1);

    // ── waiting for char argument (f/F/t/T/r/m/'/`/"/q/@) ──
    if let Some(pending) = state.pending_key.take() {
        return handle_char_argument(state, pending, code, ctrl, count, runner, tx);
    }

    // ── paste / copy handled before keymap (Ctrl+v, Ctrl+c) ──
    if ctrl {
        match code {
            KeyCode::Char('v') => {
                let text = state.registers.get_paste();
                if !text.is_empty() {
                    state.push_undo();
                    state.file_paste(&text);
                }
                return true;
            }
            KeyCode::Char('c') => {
                if state.file_copy_selection() {
                    state.push_log(LogEntry::new(LogCategory::Sys, "copied selection"));
                }
                return true;
            }
            KeyCode::Char('s') => {
                // Handled globally in files/mod.rs
                return false;
            }
            _ => {}
        }
    }

    // ── search input mode ──
    if state.search.active {
        return handle_search_input(state, code, ctrl);
    }

    // ── resolve action via keymap or direct dispatch ──
    let kp = KeyPress {
        code,
        ctrl,
        shift: false,
    };

    // Build pending sequence
    state.pending_keys.push(kp.clone());
    let seq = state.pending_keys.clone();

    // Check if this is a prefix of a longer binding
    if state.keymap.is_prefix(EditorMode::Normal, &seq) {
        // Need more keys — wait
        return true;
    }

    // Try to look up the complete sequence
    let action = state.keymap.lookup(EditorMode::Normal, &seq).cloned();
    state.pending_keys.clear();

    if let Some(action) = action {
        // Check operator-pending: if we have a pending operator and this is a motion,
        // execute the operator over the motion range.
        if state.pending_operator.is_some() {
            return execute_operator_with_action(state, &action, count, runner, tx);
        }
        return execute_action(state, &action, count, runner, tx);
    }

    // No binding found — try single-key fallback (clear pending)
    if seq.len() > 1 {
        // Multi-key sequence didn't match — ignore
        return true;
    }

    false
}

fn handle_char_argument(
    state: &mut TuiState,
    pending: char,
    code: KeyCode,
    _ctrl: bool,
    count: usize,
    _runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    _tx: &Sender<DaemonEvent>,
) -> bool {
    let ch = match code {
        KeyCode::Char(c) => c,
        KeyCode::Esc => return true, // Cancel
        _ => return true,
    };

    match pending {
        'f' => {
            // Find char forward
            state.last_find_char = Some(FindCharState {
                ch,
                forward: true,
                till: false,
            });
            for _ in 0..count {
                state.find_char_forward(ch, false);
            }
        }
        'F' => {
            state.last_find_char = Some(FindCharState {
                ch,
                forward: false,
                till: false,
            });
            for _ in 0..count {
                state.find_char_backward(ch, false);
            }
        }
        't' => {
            state.last_find_char = Some(FindCharState {
                ch,
                forward: true,
                till: true,
            });
            for _ in 0..count {
                state.find_char_forward(ch, true);
            }
        }
        'T' => {
            state.last_find_char = Some(FindCharState {
                ch,
                forward: false,
                till: true,
            });
            for _ in 0..count {
                state.find_char_backward(ch, true);
            }
        }
        'r' => {
            // Replace char under cursor
            state.push_undo();
            state.replace_char_at_cursor(ch);
        }
        'm' => {
            // Set mark
            if state.marks.set_mark(ch, state.file_cursor_line, state.file_cursor_col) {
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("mark '{}' set", ch),
                ));
            }
        }
        '\'' => {
            // Go to mark line
            if let Some(pos) = state.marks.get_mark(ch) {
                state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                state.file_cursor_line = pos.line.min(state.file_lines.len().saturating_sub(1));
                state.file_cursor_col = 0;
                state.clamp_cursor_normal();
                // Go to first non-blank
                let line = state.file_lines.get(state.file_cursor_line).cloned().unwrap_or_default();
                let col = line.chars().take_while(|c| c.is_whitespace()).count();
                state.file_cursor_col = col;
                state.preferred_col = col;
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("mark '{}' not set", ch),
                ));
            }
        }
        '`' => {
            // Go to mark exact position
            if let Some(pos) = state.marks.get_mark(ch) {
                state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                state.file_cursor_line = pos.line.min(state.file_lines.len().saturating_sub(1));
                state.file_cursor_col = pos.col;
                state.clamp_cursor_normal();
                state.preferred_col = state.file_cursor_col;
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("mark '{}' not set", ch),
                ));
            }
        }
        '"' => {
            // Select register
            state.registers.select(ch);
        }
        'q' => {
            if state.macro_engine.is_recording() {
                // Stop recording
                if let Some(reg) = state.macro_engine.stop_recording() {
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("recorded macro @{}", reg),
                    ));
                }
            } else {
                // Start recording
                if state.macro_engine.start_recording(ch) {
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("recording @{} ...", ch),
                    ));
                }
            }
        }
        '@' => {
            // Play macro
            if state.macro_engine.start_replay(ch) {
                // Replay keys are pumped by the main event loop
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("playing @{}", ch),
                ));
            } else {
                state.push_log(LogEntry::new(
                    LogCategory::Err,
                    format!("macro @{} empty", ch),
                ));
            }
        }
        'g' => {
            match ch {
                'g' => {
                    state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                    state.file_cursor_line = 0;
                    state.clamp_cursor_normal();
                    state.preferred_col = state.file_cursor_col;
                }
                'd' => {
                    // gd — go to definition placeholder
                    state.push_log(LogEntry::new(LogCategory::Sys, "go-to-definition: not yet available"));
                }
                _ => {}
            }
        }
        'z' => {
            match ch {
                'z' => state.center_cursor_on_screen(),
                't' => state.scroll_cursor_to_top(),
                'b' => state.scroll_cursor_to_bottom(),
                _ => {}
            }
        }
        'd' => {
            // Operator-pending: d + motion
            match ch {
                'd' => {
                    // dd — delete line(s)
                    for _ in 0..count {
                        state.delete_line();
                    }
                }
                'w' => {
                    state.push_undo();
                    for _ in 0..count {
                        state.delete_word_forward();
                    }
                }
                'b' => {
                    state.push_undo();
                    for _ in 0..count {
                        state.delete_word_backward();
                    }
                }
                'e' => {
                    state.push_undo();
                    for _ in 0..count {
                        state.delete_to_word_end();
                    }
                }
                '$' => {
                    state.push_undo();
                    state.delete_to_end_of_line();
                }
                '0' => {
                    state.push_undo();
                    state.delete_to_start_of_line();
                }
                'i' => {
                    // text object: diw, di(, di", etc — wait for next char
                    state.pending_key = Some('\x01'); // sentinel for "inner text object after d"
                    state.pending_operator = Some(PendingOperator::Delete);
                }
                'a' => {
                    state.pending_key = Some('\x02'); // sentinel for "around text object after d"
                    state.pending_operator = Some(PendingOperator::Delete);
                }
                _ => {}
            }
        }
        'c' => {
            match ch {
                'c' => {
                    // cc — change entire line
                    state.push_undo();
                    state.change_line();
                }
                'w' => {
                    state.push_undo();
                    state.delete_word_forward();
                    state.editor_mode = EditorMode::Insert;
                }
                'b' => {
                    state.push_undo();
                    state.delete_word_backward();
                    state.editor_mode = EditorMode::Insert;
                }
                'e' => {
                    state.push_undo();
                    state.delete_to_word_end();
                    state.editor_mode = EditorMode::Insert;
                }
                '$' => {
                    state.push_undo();
                    state.delete_to_end_of_line();
                    state.editor_mode = EditorMode::Insert;
                }
                'i' => {
                    state.pending_key = Some('\x01');
                    state.pending_operator = Some(PendingOperator::Change);
                }
                'a' => {
                    state.pending_key = Some('\x02');
                    state.pending_operator = Some(PendingOperator::Change);
                }
                _ => {}
            }
        }
        'y' => {
            match ch {
                'y' => {
                    state.yank_line();
                    state.push_log(LogEntry::new(LogCategory::Sys, "yanked line"));
                }
                'w' => {
                    state.yank_word_forward();
                }
                'e' => {
                    state.yank_to_word_end();
                }
                '$' => {
                    state.yank_to_end_of_line();
                }
                _ => {}
            }
        }
        '\x01' => {
            // Inner text object: {operator}i{object}
            let op = state.pending_operator.take();
            state.push_undo();
            let deleted = state.select_inner_text_object(ch);
            if deleted {
                if let Some(PendingOperator::Change) = op {
                    state.editor_mode = EditorMode::Insert;
                }
            }
        }
        '\x02' => {
            // Around text object: {operator}a{object}
            let op = state.pending_operator.take();
            state.push_undo();
            let deleted = state.select_around_text_object(ch);
            if deleted {
                if let Some(PendingOperator::Change) = op {
                    state.editor_mode = EditorMode::Insert;
                }
            }
        }
        _ => {}
    }
    true
}

fn handle_search_input(state: &mut TuiState, code: KeyCode, _ctrl: bool) -> bool {
    match code {
        KeyCode::Esc => {
            state.search.cancel();
            state.search.matches.clear();
        }
        KeyCode::Enter => {
            if let Some(pattern) = state.search.submit() {
                state.search.find_all(&state.file_lines, &pattern);
                let dir = state.search.direction;
                if let Some(m) = state.search.find_next(
                    state.file_cursor_line,
                    state.file_cursor_col,
                    dir,
                ) {
                    state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                    state.file_cursor_line = m.line;
                    state.file_cursor_col = m.col_start;
                    state.clamp_cursor_normal();
                    state.preferred_col = state.file_cursor_col;
                }
                let summary = state.search.match_summary();
                state.push_log(LogEntry::new(
                    LogCategory::Sys,
                    format!("search: {} ({})", pattern, summary),
                ));
                state.registers.search = pattern;
            }
        }
        KeyCode::Backspace => {
            state.search.backspace();
            if state.search.incremental && !state.search.input.is_empty() {
                let pat = state.search.input.clone();
                state.search.find_all(&state.file_lines, &pat);
            } else if state.search.input.is_empty() {
                state.search.matches.clear();
            }
        }
        KeyCode::Char(ch) => {
            state.search.insert_char(ch);
            if state.search.incremental {
                let pat = state.search.input.clone();
                state.search.find_all(&state.file_lines, &pat);
            }
        }
        _ => return false,
    }
    true
}

fn execute_action(
    state: &mut TuiState,
    action: &EditorAction,
    count: usize,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) -> bool {
    match action {
        // ── motions ──
        EditorAction::MoveLeft => {
            for _ in 0..count {
                if state.file_cursor_col > 0 {
                    state.file_cursor_col -= 1;
                }
            }
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::MoveRight => {
            let line_len = state.current_line_len();
            for _ in 0..count {
                if state.file_cursor_col + 1 < line_len {
                    state.file_cursor_col += 1;
                }
            }
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::MoveDown => {
            for _ in 0..count {
                state.vim_move_line(1);
            }
        }
        EditorAction::MoveUp => {
            for _ in 0..count {
                state.vim_move_line(-1);
            }
        }
        EditorAction::WordForward => {
            for _ in 0..count {
                state.word_forward();
            }
        }
        EditorAction::WordBackward => {
            for _ in 0..count {
                state.word_backward();
            }
        }
        EditorAction::WordEnd => {
            for _ in 0..count {
                state.word_end();
            }
        }
        EditorAction::BigWordForward => {
            for _ in 0..count {
                state.big_word_forward();
            }
        }
        EditorAction::BigWordBackward => {
            for _ in 0..count {
                state.big_word_backward();
            }
        }
        EditorAction::BigWordEnd => {
            for _ in 0..count {
                state.big_word_end();
            }
        }
        EditorAction::LineStart => {
            state.file_cursor_col = 0;
            state.preferred_col = 0;
        }
        EditorAction::LineEnd => {
            let line_len = state.current_line_len();
            state.file_cursor_col = line_len.saturating_sub(1);
            state.preferred_col = usize::MAX;
        }
        EditorAction::FirstNonBlank => {
            let line = state.file_lines.get(state.file_cursor_line).cloned().unwrap_or_default();
            let col = line.chars().take_while(|c| c.is_whitespace()).count();
            state.file_cursor_col = col;
            state.preferred_col = col;
        }
        EditorAction::GotoFirstLine => {
            state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
            state.file_cursor_line = 0;
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::GotoLastLine => {
            state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
            if count > 1 {
                state.file_cursor_line = (count - 1).min(state.file_lines.len().saturating_sub(1));
            } else {
                state.file_cursor_line = state.file_lines.len().saturating_sub(1);
            }
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::GotoLine(n) => {
            state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
            state.file_cursor_line = n.saturating_sub(1).min(state.file_lines.len().saturating_sub(1));
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::PageDown => {
            let rows = state.editor_visible_rows.max(1);
            state.file_scroll_lines(-(rows as i32));
            state.file_cursor_line = (state.file_cursor_line + rows)
                .min(state.file_lines.len().saturating_sub(1));
            state.clamp_cursor_normal();
        }
        EditorAction::PageUp => {
            let rows = state.editor_visible_rows.max(1);
            state.file_scroll_lines(rows as i32);
            state.file_cursor_line = state.file_cursor_line.saturating_sub(rows);
            state.clamp_cursor_normal();
        }
        EditorAction::HalfPageDown => {
            let half = (state.editor_visible_rows / 2).max(1);
            state.file_scroll_lines(-(half as i32));
            state.file_cursor_line = (state.file_cursor_line + half)
                .min(state.file_lines.len().saturating_sub(1));
            state.clamp_cursor_normal();
        }
        EditorAction::HalfPageUp => {
            let half = (state.editor_visible_rows / 2).max(1);
            state.file_scroll_lines(half as i32);
            state.file_cursor_line = state.file_cursor_line.saturating_sub(half);
            state.clamp_cursor_normal();
        }
        EditorAction::ScreenTop => {
            state.file_cursor_line = state.file_scroll;
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::ScreenMiddle => {
            let mid = state.file_scroll + state.editor_visible_rows / 2;
            state.file_cursor_line = mid.min(state.file_lines.len().saturating_sub(1));
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::ScreenBottom => {
            let bot = state.file_scroll + state.editor_visible_rows.saturating_sub(1);
            state.file_cursor_line = bot.min(state.file_lines.len().saturating_sub(1));
            state.clamp_cursor_normal();
            state.preferred_col = state.file_cursor_col;
        }
        EditorAction::ParagraphForward => {
            for _ in 0..count {
                state.paragraph_forward();
            }
        }
        EditorAction::ParagraphBackward => {
            for _ in 0..count {
                state.paragraph_backward();
            }
        }
        EditorAction::MatchingBracket => {
            state.goto_matching_bracket();
        }
        EditorAction::FindCharForward => {
            state.pending_key = Some('f');
        }
        EditorAction::FindCharBackward => {
            state.pending_key = Some('F');
        }
        EditorAction::TillCharForward => {
            state.pending_key = Some('t');
        }
        EditorAction::TillCharBackward => {
            state.pending_key = Some('T');
        }
        EditorAction::RepeatFindChar => {
            if let Some(fcs) = state.last_find_char {
                for _ in 0..count {
                    if fcs.forward {
                        state.find_char_forward(fcs.ch, fcs.till);
                    } else {
                        state.find_char_backward(fcs.ch, fcs.till);
                    }
                }
            }
        }
        EditorAction::RepeatFindCharReverse => {
            if let Some(fcs) = state.last_find_char {
                for _ in 0..count {
                    if fcs.forward {
                        state.find_char_backward(fcs.ch, fcs.till);
                    } else {
                        state.find_char_forward(fcs.ch, fcs.till);
                    }
                }
            }
        }
        EditorAction::ScrollLineDown => {
            state.file_scroll_lines(-1);
        }
        EditorAction::ScrollLineUp => {
            state.file_scroll_lines(1);
        }
        EditorAction::CenterCursorLine => {
            state.center_cursor_on_screen();
        }

        // ── mode transitions ──
        EditorAction::InsertMode => {
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::InsertAtLineStart => {
            let line = state.file_lines.get(state.file_cursor_line).cloned().unwrap_or_default();
            let col = line.chars().take_while(|c| c.is_whitespace()).count();
            state.file_cursor_col = col;
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::AppendAfterCursor => {
            let line_len = state.current_line_len();
            if line_len > 0 {
                state.file_cursor_col = (state.file_cursor_col + 1).min(line_len);
            }
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::AppendAtLineEnd => {
            let line_len = state.current_line_len();
            state.file_cursor_col = line_len;
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::OpenLineBelow => {
            state.open_line_below();
        }
        EditorAction::OpenLineAbove => {
            state.open_line_above();
        }
        EditorAction::VisualMode => {
            state.editor_mode = EditorMode::Visual;
            state.set_selection_anchor(state.file_cursor_line, state.file_cursor_col);
        }
        EditorAction::VisualLineMode => {
            state.editor_mode = EditorMode::Visual;
            state.set_selection_anchor(state.file_cursor_line, 0);
            let line_len = state.current_line_len();
            state.update_selection_head(state.file_cursor_line, line_len);
        }
        EditorAction::CommandMode => {
            state.editor_mode = EditorMode::Command;
            state.editor_cmdline.clear();
            state.editor_cmd_cursor = 0;
        }
        EditorAction::NormalMode => {
            state.editor_mode = EditorMode::Normal;
            state.clear_selection();
            state.clamp_cursor_normal();
        }
        EditorAction::ReplaceChar => {
            state.pending_key = Some('r');
        }

        // ── editing ──
        EditorAction::DeleteChar => {
            for _ in 0..count {
                state.delete_char_at_cursor();
            }
        }
        EditorAction::DeleteCharBefore => {
            for _ in 0..count {
                if state.file_cursor_col > 0 {
                    state.file_cursor_col -= 1;
                    state.delete_char_at_cursor();
                }
            }
        }
        EditorAction::DeleteLine => {
            for _ in 0..count {
                state.delete_line();
            }
        }
        EditorAction::YankLine => {
            state.yank_line();
            state.push_log(LogEntry::new(LogCategory::Sys, "yanked line"));
        }
        EditorAction::PutAfter => {
            let text = state.registers.get_paste();
            if text.is_empty() {
                state.put_line_below();
            } else {
                state.push_undo();
                // Line-mode paste if text ends with newline
                if text.ends_with('\n') || state.clipboard == text {
                    state.put_line_below();
                } else {
                    // Character-mode paste
                    state.file_cursor_col += 1;
                    state.file_paste(&text);
                }
            }
        }
        EditorAction::PutBefore => {
            let text = state.registers.get_paste();
            if text.is_empty() {
                // Use clipboard
                if !state.clipboard.is_empty() {
                    state.push_undo();
                    state.file_lines.insert(state.file_cursor_line, state.clipboard.clone());
                    state.file_cursor_col = 0;
                    state.clamp_cursor_normal();
                    state.file_dirty = true;
                }
            } else {
                state.push_undo();
                state.file_paste(&text);
            }
        }
        EditorAction::JoinLines => {
            for _ in 0..count {
                state.join_lines();
            }
        }
        EditorAction::Undo => {
            if state.undo() {
                state.push_log(LogEntry::new(LogCategory::Sys, "undo"));
            }
        }
        EditorAction::Redo => {
            if state.redo() {
                state.push_log(LogEntry::new(LogCategory::Sys, "redo"));
            }
        }
        EditorAction::ChangeToEndOfLine => {
            state.push_undo();
            state.delete_to_end_of_line();
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::DeleteToEndOfLine => {
            state.push_undo();
            state.delete_to_end_of_line();
        }
        EditorAction::YankToEndOfLine => {
            state.yank_to_end_of_line();
        }
        EditorAction::SubstituteChar => {
            state.push_undo();
            state.delete_char_at_cursor();
            state.editor_mode = EditorMode::Insert;
        }
        EditorAction::SubstituteLine => {
            state.push_undo();
            state.change_line();
        }
        EditorAction::Indent => {
            state.push_undo();
            for _ in 0..count {
                state.indent_line();
            }
        }
        EditorAction::Dedent => {
            state.push_undo();
            for _ in 0..count {
                state.dedent_line();
            }
        }
        EditorAction::RepeatLastCommand => {
            // Dot repeat — simplified: repeat last edit
            state.push_log(LogEntry::new(LogCategory::Sys, "repeat: coming soon"));
        }

        // ── operators ──
        EditorAction::OperatorDelete => {
            state.pending_key = Some('d');
        }
        EditorAction::OperatorChange => {
            state.pending_key = Some('c');
        }
        EditorAction::OperatorYank => {
            state.pending_key = Some('y');
        }
        EditorAction::OperatorIndent => {
            state.push_undo();
            for _ in 0..count {
                state.indent_line();
            }
        }
        EditorAction::OperatorDedent => {
            state.push_undo();
            for _ in 0..count {
                state.dedent_line();
            }
        }

        // ── search ──
        EditorAction::SearchForward => {
            state.search.start(SearchDirection::Forward);
        }
        EditorAction::SearchBackward => {
            state.search.start(SearchDirection::Backward);
        }
        EditorAction::SearchNext => {
            let dir = state.search.last_direction;
            if let Some(pat) = state.search.last_pattern.clone() {
                if state.search.matches.is_empty() {
                    state.search.find_all(&state.file_lines, &pat);
                }
                if let Some(m) =
                    state.search.find_next(state.file_cursor_line, state.file_cursor_col, dir)
                {
                    state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                    state.file_cursor_line = m.line;
                    state.file_cursor_col = m.col_start;
                    state.clamp_cursor_normal();
                    state.preferred_col = state.file_cursor_col;
                }
                let summary = state.search.match_summary();
                state.push_log(LogEntry::new(LogCategory::Sys, format!("/{} ({})", pat, summary)));
            }
        }
        EditorAction::SearchPrev => {
            let dir = match state.search.last_direction {
                SearchDirection::Forward => SearchDirection::Backward,
                SearchDirection::Backward => SearchDirection::Forward,
            };
            if let Some(pat) = state.search.last_pattern.clone() {
                if state.search.matches.is_empty() {
                    state.search.find_all(&state.file_lines, &pat);
                }
                if let Some(m) =
                    state.search.find_next(state.file_cursor_line, state.file_cursor_col, dir)
                {
                    state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                    state.file_cursor_line = m.line;
                    state.file_cursor_col = m.col_start;
                    state.clamp_cursor_normal();
                    state.preferred_col = state.file_cursor_col;
                }
                let summary = state.search.match_summary();
                state.push_log(LogEntry::new(LogCategory::Sys, format!("?{} ({})", pat, summary)));
            }
        }
        EditorAction::SearchWordUnderCursor => {
            let word = state.word_under_cursor();
            if !word.is_empty() {
                state.search.find_all(&state.file_lines, &word);
                state.search.last_pattern = Some(word.clone());
                state.search.last_direction = SearchDirection::Forward;
                if let Some(m) = state.search.find_next(
                    state.file_cursor_line,
                    state.file_cursor_col,
                    SearchDirection::Forward,
                ) {
                    state.marks.record_jump(state.file_cursor_line, state.file_cursor_col);
                    state.file_cursor_line = m.line;
                    state.file_cursor_col = m.col_start;
                    state.clamp_cursor_normal();
                    state.preferred_col = state.file_cursor_col;
                }
                let summary = state.search.match_summary();
                state.push_log(LogEntry::new(LogCategory::Sys, format!("*{} ({})", word, summary)));
            }
        }

        // ── marks / jumps ──
        EditorAction::SetMark => {
            state.pending_key = Some('m');
        }
        EditorAction::GotoMark => {
            state.pending_key = Some('`');
        }
        EditorAction::GotoMarkLine => {
            state.pending_key = Some('\'');
        }
        EditorAction::JumpBack => {
            if let Some(pos) = state.marks.jump_back() {
                state.file_cursor_line = pos.line.min(state.file_lines.len().saturating_sub(1));
                state.file_cursor_col = pos.col;
                state.clamp_cursor_normal();
                state.preferred_col = state.file_cursor_col;
            }
        }
        EditorAction::JumpForward => {
            if let Some(pos) = state.marks.jump_forward() {
                state.file_cursor_line = pos.line.min(state.file_lines.len().saturating_sub(1));
                state.file_cursor_col = pos.col;
                state.clamp_cursor_normal();
                state.preferred_col = state.file_cursor_col;
            }
        }

        // ── registers ──
        EditorAction::SelectRegister => {
            state.pending_key = Some('"');
        }

        // ── macros ──
        EditorAction::RecordMacro => {
            if state.macro_engine.is_recording() {
                if let Some(reg) = state.macro_engine.stop_recording() {
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("recorded macro @{}", reg),
                    ));
                }
            } else {
                state.pending_key = Some('q');
            }
        }
        EditorAction::PlayMacro => {
            state.pending_key = Some('@');
        }

        // ── buffers ──
        EditorAction::BufferNext => {
            state.buffer_next();
        }
        EditorAction::BufferPrev => {
            state.buffer_prev();
        }
        EditorAction::BufferClose => {
            state.push_log(LogEntry::new(LogCategory::Sys, "use :bclose to close buffer"));
        }
        EditorAction::BufferList => {
            state.push_log(LogEntry::new(LogCategory::Sys, "use :buffers to list buffers"));
        }

        // ── misc ──
        EditorAction::ToggleFocus => {
            state.files_toggle_focus();
        }
        EditorAction::ToggleUiMode => {
            state.toggle_ui_mode();
        }
        EditorAction::Run => {
            crate::tui::commands::run::cmd_run(state, runner, tx);
        }
        EditorAction::Stop => {
            crate::tui::commands::run::cmd_stop_run(state);
        }
        EditorAction::ToggleRunOutput => {
            state.show_run_output = !state.show_run_output;
        }
        EditorAction::NoOp => {}
    }
    true
}

fn execute_operator_with_action(
    state: &mut TuiState,
    action: &EditorAction,
    _count: usize,
    _runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    _tx: &Sender<DaemonEvent>,
) -> bool {
    // Cancel operator-pending if non-motion action
    state.pending_operator = None;
    // For now, just execute the action normally
    // A full implementation would compute the range and apply the operator
    match action {
        _ => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                "operator+motion: use d{motion}/c{motion}/y{motion} keys directly",
            ));
        }
    }
    true
}
