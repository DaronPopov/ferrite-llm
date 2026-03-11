use std::path::PathBuf;

use super::{char_to_byte, EditorMode, FilesFocus, TuiState, UndoEntry};

impl TuiState {
    pub fn reload_file_entries(&mut self) {
        self.file_tree.rebuild();
        let len = self.file_tree.len();
        if len == 0 {
            self.file_cursor = 0;
        } else if self.file_cursor >= len {
            self.file_cursor = len - 1;
        }
    }

    pub fn file_move_cursor(&mut self, delta: i32) {
        let len = self.file_tree.len();
        if len == 0 {
            self.file_cursor = 0;
            return;
        }
        if delta > 0 {
            self.file_cursor = self
                .file_cursor
                .saturating_sub(delta as usize)
                .min(len - 1);
        } else if delta < 0 {
            self.file_cursor = (self.file_cursor + (-delta) as usize).min(len - 1);
        }
    }

    pub fn open_selected_file(&mut self) -> Result<(), String> {
        let entry = self
            .file_tree
            .visible
            .get(self.file_cursor)
            .ok_or_else(|| "no file selected".to_string())?;

        if entry.is_dir {
            self.file_tree.toggle(self.file_cursor);
            return Ok(());
        }

        let path = entry.path.clone();
        self.open_file_path(path)
    }

    pub fn open_file_path(&mut self, path: PathBuf) -> Result<(), String> {
        let bytes = std::fs::read(&path).map_err(|e| format!("open failed: {}", e))?;
        if bytes.contains(&0) {
            return Err("binary file not supported in editor".into());
        }
        let text = String::from_utf8(bytes).map_err(|_| "file is not valid UTF-8".to_string())?;
        self.open_file = Some(path);
        self.file_lines = text.lines().map(ToOwned::to_owned).collect();
        if self.file_lines.is_empty() {
            self.file_lines.push(String::new());
        }
        self.file_scroll = 0;
        self.file_hscroll = 0;
        self.file_cursor_line = 0;
        self.file_cursor_col = 0;
        self.preferred_col = 0;
        self.file_dirty = false;
        self.selection_anchor = None;
        self.selection_head = None;
        self.selecting = false;
        self.editor_mode = EditorMode::Normal;
        self.pending_key = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.files_focus = FilesFocus::Editor;
        Ok(())
    }

    pub fn file_scroll_lines(&mut self, delta: i32) {
        if delta > 0 {
            self.file_scroll = self.file_scroll.saturating_sub(delta as usize);
        } else if delta < 0 {
            self.file_scroll =
                (self.file_scroll + (-delta) as usize).min(self.file_lines.len().saturating_sub(1));
        }
    }

    pub fn file_move_cursor_line(&mut self, delta: i32) {
        if self.file_lines.is_empty() {
            return;
        }
        if delta > 0 {
            self.file_cursor_line = self.file_cursor_line.saturating_sub(delta as usize);
        } else if delta < 0 {
            self.file_cursor_line =
                (self.file_cursor_line + (-delta) as usize).min(self.file_lines.len() - 1);
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        self.file_cursor_col = self.file_cursor_col.min(line_len);
    }

    pub fn file_move_cursor_col(&mut self, delta: i32) {
        let line_len = self
            .file_lines
            .get(self.file_cursor_line)
            .map(|l| l.chars().count())
            .unwrap_or(0);
        if delta > 0 {
            self.file_cursor_col = self.file_cursor_col.saturating_sub(delta as usize);
        } else if delta < 0 {
            self.file_cursor_col = (self.file_cursor_col + (-delta) as usize).min(line_len);
        }
    }

    pub fn file_insert_char(&mut self, ch: char) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line = &mut self.file_lines[self.file_cursor_line];
        let byte_idx = char_to_byte(line, self.file_cursor_col);
        line.insert(byte_idx, ch);
        self.file_cursor_col += 1;
        self.file_dirty = true;
        self.clear_selection();
    }

    pub fn file_insert_newline(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let current = self.file_lines[self.file_cursor_line].clone();
        let split = char_to_byte(&current, self.file_cursor_col);
        let (left, right) = current.split_at(split);
        self.file_lines[self.file_cursor_line] = left.to_string();
        self.file_lines
            .insert(self.file_cursor_line + 1, right.to_string());
        self.file_cursor_line += 1;
        self.file_cursor_col = 0;
        self.file_dirty = true;
        self.clear_selection();
    }

    pub fn file_backspace(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        if self.delete_selection_if_any() {
            return;
        }
        if self.file_cursor_col > 0 {
            let line = &mut self.file_lines[self.file_cursor_line];
            let end = char_to_byte(line, self.file_cursor_col);
            let start = char_to_byte(line, self.file_cursor_col - 1);
            line.drain(start..end);
            self.file_cursor_col -= 1;
            self.file_dirty = true;
        } else if self.file_cursor_line > 0 {
            let current = self.file_lines.remove(self.file_cursor_line);
            self.file_cursor_line -= 1;
            let prev_len = self.file_lines[self.file_cursor_line].chars().count();
            self.file_lines[self.file_cursor_line].push_str(&current);
            self.file_cursor_col = prev_len;
            self.file_dirty = true;
        }
    }

    pub fn file_delete(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        if self.delete_selection_if_any() {
            return;
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        if self.file_cursor_col < line_len {
            let line = &mut self.file_lines[self.file_cursor_line];
            let start = char_to_byte(line, self.file_cursor_col);
            let end = char_to_byte(line, self.file_cursor_col + 1);
            line.drain(start..end);
            self.file_dirty = true;
        } else if self.file_cursor_line + 1 < self.file_lines.len() {
            let next = self.file_lines.remove(self.file_cursor_line + 1);
            self.file_lines[self.file_cursor_line].push_str(&next);
            self.file_dirty = true;
        }
    }

    pub fn file_save(&mut self) -> Result<(), String> {
        let path = self
            .open_file
            .as_ref()
            .ok_or_else(|| "no file open".to_string())?
            .clone();
        let content = self.file_lines.join("\n");
        std::fs::write(&path, content).map_err(|e| format!("save failed: {}", e))?;
        self.file_dirty = false;
        Ok(())
    }

    // ── vim editor methods ──────────────────────────────────

    pub fn push_undo(&mut self) {
        self.undo_stack.push(UndoEntry {
            lines: self.file_lines.clone(),
            cursor_line: self.file_cursor_line,
            cursor_col: self.file_cursor_col,
        });
        self.redo_stack.clear();
        // Cap undo depth
        if self.undo_stack.len() > 200 {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self) -> bool {
        if let Some(entry) = self.undo_stack.pop() {
            self.redo_stack.push(UndoEntry {
                lines: self.file_lines.clone(),
                cursor_line: self.file_cursor_line,
                cursor_col: self.file_cursor_col,
            });
            self.file_lines = entry.lines;
            self.file_cursor_line = entry.cursor_line;
            self.file_cursor_col = entry.cursor_col;
            self.file_dirty = true;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(entry) = self.redo_stack.pop() {
            self.undo_stack.push(UndoEntry {
                lines: self.file_lines.clone(),
                cursor_line: self.file_cursor_line,
                cursor_col: self.file_cursor_col,
            });
            self.file_lines = entry.lines;
            self.file_cursor_line = entry.cursor_line;
            self.file_cursor_col = entry.cursor_col;
            self.file_dirty = true;
            true
        } else {
            false
        }
    }

    /// Move cursor line (vim-style: preserves preferred_col).
    pub fn vim_move_line(&mut self, delta: i32) {
        if self.file_lines.is_empty() {
            return;
        }
        if delta < 0 {
            self.file_cursor_line = self.file_cursor_line.saturating_sub((-delta) as usize);
        } else {
            self.file_cursor_line =
                (self.file_cursor_line + delta as usize).min(self.file_lines.len() - 1);
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        // In normal mode, cursor can't go past last char (len-1), but empty lines allow 0
        let max_col = if matches!(self.editor_mode, EditorMode::Insert) {
            line_len
        } else {
            line_len.saturating_sub(1)
        };
        self.file_cursor_col = self.preferred_col.min(max_col);
    }

    /// Clamp cursor col for normal mode (can't be past last char).
    pub fn clamp_cursor_normal(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        let max_col = line_len.saturating_sub(1);
        if self.file_cursor_col > max_col {
            self.file_cursor_col = max_col;
        }
    }

    /// Move to next word start (vim 'w').
    pub fn word_forward(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        if col < chars.len() {
            // Skip current word
            let start_is_word = chars[col].is_alphanumeric() || chars[col] == '_';
            if start_is_word {
                while col < chars.len() && (chars[col].is_alphanumeric() || chars[col] == '_') {
                    col += 1;
                }
            } else if !chars[col].is_whitespace() {
                while col < chars.len()
                    && !chars[col].is_whitespace()
                    && !chars[col].is_alphanumeric()
                    && chars[col] != '_'
                {
                    col += 1;
                }
            }
            // Skip whitespace
            while col < chars.len() && chars[col].is_whitespace() {
                col += 1;
            }
        }

        if col >= chars.len() && self.file_cursor_line + 1 < self.file_lines.len() {
            // Move to next line
            self.file_cursor_line += 1;
            self.file_cursor_col = 0;
            // Skip leading whitespace on new line
            let next_chars: Vec<char> = self.file_lines[self.file_cursor_line].chars().collect();
            let mut nc = 0;
            while nc < next_chars.len() && next_chars[nc].is_whitespace() {
                nc += 1;
            }
            self.file_cursor_col = nc.min(next_chars.len().saturating_sub(1));
        } else {
            self.file_cursor_col = col.min(chars.len().saturating_sub(1));
        }
        self.preferred_col = self.file_cursor_col;
    }

    /// Move to previous word start (vim 'b').
    pub fn word_backward(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        if col == 0 {
            if self.file_cursor_line > 0 {
                self.file_cursor_line -= 1;
                let prev_len = self.file_lines[self.file_cursor_line].chars().count();
                self.file_cursor_col = prev_len.saturating_sub(1);
            }
            self.preferred_col = self.file_cursor_col;
            return;
        }

        col = col.saturating_sub(1);
        // Skip whitespace backward
        while col > 0 && chars[col].is_whitespace() {
            col -= 1;
        }
        // Skip word backward
        let is_word = chars[col].is_alphanumeric() || chars[col] == '_';
        if is_word {
            while col > 0 && (chars[col - 1].is_alphanumeric() || chars[col - 1] == '_') {
                col -= 1;
            }
        } else {
            while col > 0
                && !chars[col - 1].is_whitespace()
                && !chars[col - 1].is_alphanumeric()
                && chars[col - 1] != '_'
            {
                col -= 1;
            }
        }

        self.file_cursor_col = col;
        self.preferred_col = col;
    }

    /// Move to end of current word (vim 'e').
    pub fn word_end(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        if col + 1 < chars.len() {
            col += 1;
        } else if self.file_cursor_line + 1 < self.file_lines.len() {
            self.file_cursor_line += 1;
            let next_chars: Vec<char> = self.file_lines[self.file_cursor_line].chars().collect();
            col = 0;
            while col < next_chars.len() && next_chars[col].is_whitespace() {
                col += 1;
            }
            self.file_cursor_col = col;
            self.preferred_col = col;
            // Find end of word on new line
            let new_chars: Vec<char> = self.file_lines[self.file_cursor_line].chars().collect();
            let is_word = col < new_chars.len()
                && (new_chars[col].is_alphanumeric() || new_chars[col] == '_');
            if is_word {
                while col + 1 < new_chars.len()
                    && (new_chars[col + 1].is_alphanumeric() || new_chars[col + 1] == '_')
                {
                    col += 1;
                }
            }
            self.file_cursor_col = col.min(new_chars.len().saturating_sub(1));
            self.preferred_col = self.file_cursor_col;
            return;
        }

        // Skip whitespace forward
        while col < chars.len() && chars[col].is_whitespace() {
            col += 1;
        }
        // Skip to end of word
        let is_word =
            col < chars.len() && (chars[col].is_alphanumeric() || chars[col] == '_');
        if is_word {
            while col + 1 < chars.len()
                && (chars[col + 1].is_alphanumeric() || chars[col + 1] == '_')
            {
                col += 1;
            }
        } else {
            while col + 1 < chars.len()
                && !chars[col + 1].is_whitespace()
                && !chars[col + 1].is_alphanumeric()
                && chars[col + 1] != '_'
            {
                col += 1;
            }
        }

        self.file_cursor_col = col.min(chars.len().saturating_sub(1));
        self.preferred_col = self.file_cursor_col;
    }

    /// Delete current line (vim 'dd').
    pub fn delete_line(&mut self) {
        if self.file_lines.is_empty() || self.open_file.is_none() {
            return;
        }
        self.push_undo();
        let removed = self.file_lines.remove(self.file_cursor_line);
        self.clipboard = removed;
        if self.file_lines.is_empty() {
            self.file_lines.push(String::new());
        }
        if self.file_cursor_line >= self.file_lines.len() {
            self.file_cursor_line = self.file_lines.len() - 1;
        }
        self.clamp_cursor_normal();
        self.preferred_col = self.file_cursor_col;
        self.file_dirty = true;
    }

    /// Delete char at cursor (vim 'x').
    pub fn delete_char_at_cursor(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        if self.file_cursor_col < line_len {
            self.push_undo();
            let line = &mut self.file_lines[self.file_cursor_line];
            let start = char_to_byte(line, self.file_cursor_col);
            let end = char_to_byte(line, self.file_cursor_col + 1);
            line.drain(start..end);
            self.clamp_cursor_normal();
            self.file_dirty = true;
        }
    }

    /// Open new line below cursor and enter insert mode (vim 'o').
    pub fn open_line_below(&mut self) {
        if self.open_file.is_none() {
            return;
        }
        self.push_undo();
        self.file_lines
            .insert(self.file_cursor_line + 1, String::new());
        self.file_cursor_line += 1;
        self.file_cursor_col = 0;
        self.preferred_col = 0;
        self.file_dirty = true;
        self.editor_mode = EditorMode::Insert;
    }

    /// Open new line above cursor and enter insert mode (vim 'O').
    pub fn open_line_above(&mut self) {
        if self.open_file.is_none() {
            return;
        }
        self.push_undo();
        self.file_lines
            .insert(self.file_cursor_line, String::new());
        self.file_cursor_col = 0;
        self.preferred_col = 0;
        self.file_dirty = true;
        self.editor_mode = EditorMode::Insert;
    }

    /// Join current line with next line (vim 'J').
    pub fn join_lines(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        if self.file_cursor_line + 1 >= self.file_lines.len() {
            return;
        }
        self.push_undo();
        let current_len = self.file_lines[self.file_cursor_line].chars().count();
        let next = self.file_lines.remove(self.file_cursor_line + 1);
        let trimmed = next.trim_start();
        if !self.file_lines[self.file_cursor_line].is_empty() && !trimmed.is_empty() {
            self.file_lines[self.file_cursor_line].push(' ');
            self.file_cursor_col = current_len;
        } else {
            self.file_cursor_col = current_len;
        }
        self.file_lines[self.file_cursor_line].push_str(trimmed);
        self.preferred_col = self.file_cursor_col;
        self.file_dirty = true;
    }

    /// Put (paste) line below current (vim 'p' after dd).
    pub fn put_line_below(&mut self) {
        if self.open_file.is_none() || self.clipboard.is_empty() {
            return;
        }
        self.push_undo();
        if self.clipboard.contains('\n') {
            // Multi-line or line-mode paste
            for (i, line) in self.clipboard.lines().enumerate() {
                self.file_lines
                    .insert(self.file_cursor_line + 1 + i, line.to_string());
            }
            self.file_cursor_line += 1;
        } else {
            self.file_lines
                .insert(self.file_cursor_line + 1, self.clipboard.clone());
            self.file_cursor_line += 1;
        }
        self.file_cursor_col = 0;
        self.clamp_cursor_normal();
        self.preferred_col = self.file_cursor_col;
        self.file_dirty = true;
    }

    /// Yank (copy) current line (vim 'yy').
    pub fn yank_line(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        self.clipboard = self.file_lines[self.file_cursor_line].clone();
    }

    // ── editor command-line helpers ──────────────────────────

    pub fn cmdline_insert(&mut self, ch: char) {
        self.editor_cmdline.insert(self.editor_cmd_cursor, ch);
        self.editor_cmd_cursor += ch.len_utf8();
    }

    pub fn cmdline_backspace(&mut self) {
        if self.editor_cmd_cursor > 0 {
            let prev = self.editor_cmdline[..self.editor_cmd_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.editor_cmdline.drain(prev..self.editor_cmd_cursor);
            self.editor_cmd_cursor = prev;
        }
    }

    pub fn cmdline_clear(&mut self) {
        self.editor_cmdline.clear();
        self.editor_cmd_cursor = 0;
    }

    pub fn cmdline_submit(&mut self) -> Option<String> {
        let line = self.editor_cmdline.trim().to_string();
        self.editor_cmdline.clear();
        self.editor_cmd_cursor = 0;
        self.editor_mode = EditorMode::Normal;
        if line.is_empty() {
            None
        } else {
            Some(line)
        }
    }

    // ── selection helpers ────────────────────────────────────

    pub fn set_selection_anchor(&mut self, line: usize, col: usize) {
        self.selection_anchor = Some((line, col));
        self.selection_head = Some((line, col));
        self.selecting = true;
    }

    pub fn update_selection_head(&mut self, line: usize, col: usize) {
        if self.selection_anchor.is_some() {
            self.selection_head = Some((line, col));
        }
    }

    pub fn finish_selection(&mut self) {
        self.selecting = false;
    }

    pub fn clear_selection(&mut self) {
        self.selection_anchor = None;
        self.selection_head = None;
        self.selecting = false;
    }

    pub fn file_copy_selection(&mut self) -> bool {
        if let Some(text) = self.selected_text() {
            self.clipboard = text;
            true
        } else {
            false
        }
    }

    pub fn file_paste(&mut self, text: &str) {
        if self.open_file.is_none() {
            return;
        }
        if self.delete_selection_if_any() {
            // continue inserting into collapsed cursor position
        }
        for ch in text.chars() {
            if ch == '\n' {
                self.file_insert_newline();
            } else if ch != '\r' {
                self.file_insert_char(ch);
            }
        }
    }

    pub fn selected_text(&self) -> Option<String> {
        let ((sl, sc), (el, ec)) = self.selection_bounds()?;
        if sl >= self.file_lines.len() || el >= self.file_lines.len() {
            return None;
        }
        if sl == el {
            let line = &self.file_lines[sl];
            let sb = char_to_byte(line, sc);
            let eb = char_to_byte(line, ec);
            return Some(line[sb..eb].to_string());
        }
        let mut out = String::new();
        for li in sl..=el {
            let line = &self.file_lines[li];
            if li == sl {
                let sb = char_to_byte(line, sc);
                out.push_str(&line[sb..]);
                out.push('\n');
            } else if li == el {
                let eb = char_to_byte(line, ec);
                out.push_str(&line[..eb]);
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Some(out)
    }

    pub fn is_selected_char(&self, line: usize, col: usize) -> bool {
        if let Some(((sl, sc), (el, ec))) = self.selection_bounds() {
            if line < sl || line > el {
                return false;
            }
            if sl == el {
                return line == sl && col >= sc && col < ec;
            }
            if line == sl {
                return col >= sc;
            }
            if line == el {
                return col < ec;
            }
            return true;
        }
        false
    }

    fn selection_bounds(&self) -> Option<((usize, usize), (usize, usize))> {
        let a = self.selection_anchor?;
        let b = self.selection_head?;
        if a <= b {
            Some((a, b))
        } else {
            Some((b, a))
        }
    }

    // ── rich vim motions (Plan C) ──────────────────────────────

    /// Get current line length in chars.
    pub fn current_line_len(&self) -> usize {
        self.file_lines
            .get(self.file_cursor_line)
            .map(|l| l.chars().count())
            .unwrap_or(0)
    }

    /// Move to next WORD start (whitespace-delimited, vim 'W').
    pub fn big_word_forward(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        // Skip non-whitespace
        while col < chars.len() && !chars[col].is_whitespace() {
            col += 1;
        }
        // Skip whitespace
        while col < chars.len() && chars[col].is_whitespace() {
            col += 1;
        }

        if col >= chars.len() && self.file_cursor_line + 1 < self.file_lines.len() {
            self.file_cursor_line += 1;
            let next_chars: Vec<char> = self.file_lines[self.file_cursor_line].chars().collect();
            col = 0;
            while col < next_chars.len() && next_chars[col].is_whitespace() {
                col += 1;
            }
            self.file_cursor_col = col.min(next_chars.len().saturating_sub(1));
        } else {
            self.file_cursor_col = col.min(chars.len().saturating_sub(1));
        }
        self.preferred_col = self.file_cursor_col;
    }

    /// Move to previous WORD start (vim 'B').
    pub fn big_word_backward(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        if col == 0 {
            if self.file_cursor_line > 0 {
                self.file_cursor_line -= 1;
                let prev_len = self.file_lines[self.file_cursor_line].chars().count();
                self.file_cursor_col = prev_len.saturating_sub(1);
            }
            self.preferred_col = self.file_cursor_col;
            return;
        }

        col = col.saturating_sub(1);
        while col > 0 && chars[col].is_whitespace() {
            col -= 1;
        }
        while col > 0 && !chars[col - 1].is_whitespace() {
            col -= 1;
        }

        self.file_cursor_col = col;
        self.preferred_col = col;
    }

    /// Move to end of WORD (vim 'E').
    pub fn big_word_end(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let mut col = self.file_cursor_col;

        if col + 1 < chars.len() {
            col += 1;
        } else if self.file_cursor_line + 1 < self.file_lines.len() {
            self.file_cursor_line += 1;
            col = 0;
        }

        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();

        while col < chars.len() && chars[col].is_whitespace() {
            col += 1;
        }
        while col + 1 < chars.len() && !chars[col + 1].is_whitespace() {
            col += 1;
        }

        self.file_cursor_col = col.min(chars.len().saturating_sub(1));
        self.preferred_col = self.file_cursor_col;
    }

    /// Find character forward on current line (vim 'f'/'t').
    pub fn find_char_forward(&mut self, ch: char, till: bool) {
        let line = self.file_lines.get(self.file_cursor_line).cloned().unwrap_or_default();
        let chars: Vec<char> = line.chars().collect();
        for i in (self.file_cursor_col + 1)..chars.len() {
            if chars[i] == ch {
                self.file_cursor_col = if till { i.saturating_sub(1) } else { i };
                self.preferred_col = self.file_cursor_col;
                return;
            }
        }
    }

    /// Find character backward on current line (vim 'F'/'T').
    pub fn find_char_backward(&mut self, ch: char, till: bool) {
        let line = self.file_lines.get(self.file_cursor_line).cloned().unwrap_or_default();
        let chars: Vec<char> = line.chars().collect();
        if self.file_cursor_col == 0 {
            return;
        }
        for i in (0..self.file_cursor_col).rev() {
            if chars[i] == ch {
                self.file_cursor_col = if till { i + 1 } else { i };
                self.preferred_col = self.file_cursor_col;
                return;
            }
        }
    }

    /// Move to next empty line (vim '}').
    pub fn paragraph_forward(&mut self) {
        let mut line = self.file_cursor_line + 1;
        while line < self.file_lines.len() {
            if self.file_lines[line].trim().is_empty() {
                self.file_cursor_line = line;
                self.file_cursor_col = 0;
                self.preferred_col = 0;
                return;
            }
            line += 1;
        }
        self.file_cursor_line = self.file_lines.len().saturating_sub(1);
        self.file_cursor_col = 0;
        self.preferred_col = 0;
    }

    /// Move to previous empty line (vim '{').
    pub fn paragraph_backward(&mut self) {
        if self.file_cursor_line == 0 {
            return;
        }
        let mut line = self.file_cursor_line - 1;
        loop {
            if self.file_lines[line].trim().is_empty() {
                self.file_cursor_line = line;
                self.file_cursor_col = 0;
                self.preferred_col = 0;
                return;
            }
            if line == 0 {
                break;
            }
            line -= 1;
        }
        self.file_cursor_line = 0;
        self.file_cursor_col = 0;
        self.preferred_col = 0;
    }

    /// Jump to matching bracket (vim '%').
    pub fn goto_matching_bracket(&mut self) {
        let line = self.file_lines.get(self.file_cursor_line).cloned().unwrap_or_default();
        let chars: Vec<char> = line.chars().collect();
        if self.file_cursor_col >= chars.len() {
            return;
        }

        let ch = chars[self.file_cursor_col];
        let (target, forward) = match ch {
            '(' => (')', true),
            ')' => ('(', false),
            '[' => (']', true),
            ']' => ('[', false),
            '{' => ('}', true),
            '}' => ('{', false),
            '<' => ('>', true),
            '>' => ('<', false),
            _ => return,
        };

        let mut depth = 1i32;
        let mut l = self.file_cursor_line;
        let mut c = self.file_cursor_col;

        if forward {
            loop {
                c += 1;
                let line_chars: Vec<char> = self.file_lines.get(l).map(|s| s.chars().collect()).unwrap_or_default();
                while c < line_chars.len() {
                    if line_chars[c] == ch {
                        depth += 1;
                    } else if line_chars[c] == target {
                        depth -= 1;
                        if depth == 0 {
                            self.file_cursor_line = l;
                            self.file_cursor_col = c;
                            self.preferred_col = c;
                            return;
                        }
                    }
                    c += 1;
                }
                l += 1;
                c = 0;
                if l >= self.file_lines.len() {
                    return;
                }
            }
        } else {
            loop {
                if c == 0 {
                    if l == 0 {
                        return;
                    }
                    l -= 1;
                    let line_chars: Vec<char> = self.file_lines.get(l).map(|s| s.chars().collect()).unwrap_or_default();
                    c = line_chars.len().saturating_sub(1);
                } else {
                    c -= 1;
                }
                let line_chars: Vec<char> = self.file_lines.get(l).map(|s| s.chars().collect()).unwrap_or_default();
                if c < line_chars.len() {
                    if line_chars[c] == ch {
                        depth += 1;
                    } else if line_chars[c] == target {
                        depth -= 1;
                        if depth == 0 {
                            self.file_cursor_line = l;
                            self.file_cursor_col = c;
                            self.preferred_col = c;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Replace char at cursor without changing mode (vim 'r').
    pub fn replace_char_at_cursor(&mut self, ch: char) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line_len = self.file_lines[self.file_cursor_line].chars().count();
        if self.file_cursor_col < line_len {
            let line = &mut self.file_lines[self.file_cursor_line];
            let start = char_to_byte(line, self.file_cursor_col);
            let end = char_to_byte(line, self.file_cursor_col + 1);
            line.drain(start..end);
            line.insert(start, ch);
            self.file_dirty = true;
        }
    }

    /// Delete from cursor to end of line (vim 'D').
    pub fn delete_to_end_of_line(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line = &mut self.file_lines[self.file_cursor_line];
        let byte_idx = char_to_byte(line, self.file_cursor_col);
        let deleted = line[byte_idx..].to_string();
        line.truncate(byte_idx);
        if !deleted.is_empty() {
            self.registers.delete(&deleted);
            self.clipboard = deleted;
        }
        self.clamp_cursor_normal();
        self.file_dirty = true;
    }

    /// Delete from cursor to start of line (vim 'd0').
    pub fn delete_to_start_of_line(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line = &mut self.file_lines[self.file_cursor_line];
        let byte_idx = char_to_byte(line, self.file_cursor_col);
        let deleted = line[..byte_idx].to_string();
        let rest = line[byte_idx..].to_string();
        *line = rest;
        if !deleted.is_empty() {
            self.registers.delete(&deleted);
            self.clipboard = deleted;
        }
        self.file_cursor_col = 0;
        self.preferred_col = 0;
        self.file_dirty = true;
    }

    /// Delete a word forward (vim 'dw').
    pub fn delete_word_forward(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let start_col = self.file_cursor_col;
        self.word_forward();
        let end_col = self.file_cursor_col;

        if start_col < end_col {
            let line = &mut self.file_lines[self.file_cursor_line];
            let sb = char_to_byte(line, start_col);
            let eb = char_to_byte(line, end_col);
            let deleted: String = line.drain(sb..eb).collect();
            self.registers.delete(&deleted);
            self.clipboard = deleted;
            self.file_cursor_col = start_col;
            self.clamp_cursor_normal();
            self.file_dirty = true;
        }
    }

    /// Delete a word backward (vim 'db').
    pub fn delete_word_backward(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let end_col = self.file_cursor_col;
        self.word_backward();
        let start_col = self.file_cursor_col;

        if start_col < end_col {
            let line = &mut self.file_lines[self.file_cursor_line];
            let sb = char_to_byte(line, start_col);
            let eb = char_to_byte(line, end_col);
            let deleted: String = line.drain(sb..eb).collect();
            self.registers.delete(&deleted);
            self.clipboard = deleted;
            self.file_cursor_col = start_col;
            self.file_dirty = true;
        }
    }

    /// Delete to end of word (vim 'de').
    pub fn delete_to_word_end(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let start_col = self.file_cursor_col;
        self.word_end();
        let end_col = self.file_cursor_col + 1; // Include the end char

        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, start_col);
        let eb = char_to_byte(line, end_col.min(line.chars().count()));
        if sb < eb {
            let deleted: String = line.drain(sb..eb).collect();
            self.registers.delete(&deleted);
            self.clipboard = deleted;
            self.file_cursor_col = start_col;
            self.clamp_cursor_normal();
            self.file_dirty = true;
        }
    }

    /// Change entire line content (vim 'cc'/'S').
    pub fn change_line(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let indent_len = line.chars().take_while(|c| c.is_whitespace()).count();
        let indent: String = line.chars().take(indent_len).collect();
        let deleted = line.clone();
        self.registers.delete(&deleted);
        self.clipboard = deleted;
        self.file_lines[self.file_cursor_line] = indent;
        self.file_cursor_col = indent_len;
        self.editor_mode = EditorMode::Insert;
        self.file_dirty = true;
    }

    /// Yank word forward (vim 'yw').
    pub fn yank_word_forward(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let start_col = self.file_cursor_col;
        let start_line = self.file_cursor_line;
        self.word_forward();
        let end_col = self.file_cursor_col;

        // Restore position
        self.file_cursor_line = start_line;
        self.file_cursor_col = start_col;

        if start_col < end_col {
            let line = &self.file_lines[start_line];
            let sb = char_to_byte(line, start_col);
            let eb = char_to_byte(line, end_col);
            let text = line[sb..eb].to_string();
            self.registers.yank(&text);
            self.clipboard = text;
        }
    }

    /// Yank to end of word (vim 'ye').
    pub fn yank_to_word_end(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let start_col = self.file_cursor_col;
        let start_line = self.file_cursor_line;
        self.word_end();
        let end_col = self.file_cursor_col + 1;

        self.file_cursor_line = start_line;
        self.file_cursor_col = start_col;

        let line = &self.file_lines[start_line];
        let sb = char_to_byte(line, start_col);
        let eb = char_to_byte(line, end_col.min(line.chars().count()));
        if sb < eb {
            let text = line[sb..eb].to_string();
            self.registers.yank(&text);
            self.clipboard = text;
        }
    }

    /// Yank to end of line (vim 'Y'/'y$').
    pub fn yank_to_end_of_line(&mut self) {
        if self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, self.file_cursor_col);
        let text = line[sb..].to_string();
        self.registers.yank(&text);
        self.clipboard = text;
    }

    /// Indent current line by 4 spaces.
    pub fn indent_line(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        self.file_lines[self.file_cursor_line].insert_str(0, "    ");
        self.file_cursor_col += 4;
        self.preferred_col = self.file_cursor_col;
        self.file_dirty = true;
    }

    /// Dedent current line by up to 4 spaces.
    pub fn dedent_line(&mut self) {
        if self.open_file.is_none() || self.file_lines.is_empty() {
            return;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let remove = line.chars().take(4).take_while(|c| *c == ' ').count();
        if remove > 0 {
            self.file_lines[self.file_cursor_line] = self.file_lines[self.file_cursor_line][remove..].to_string();
            self.file_cursor_col = self.file_cursor_col.saturating_sub(remove);
            self.preferred_col = self.file_cursor_col;
            self.file_dirty = true;
        }
    }

    /// Get word under cursor.
    pub fn word_under_cursor(&self) -> String {
        if self.file_lines.is_empty() {
            return String::new();
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        if self.file_cursor_col >= chars.len() {
            return String::new();
        }

        let is_word_char = |c: char| c.is_alphanumeric() || c == '_';
        if !is_word_char(chars[self.file_cursor_col]) {
            return String::new();
        }

        let mut start = self.file_cursor_col;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        let mut end = self.file_cursor_col;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }

        chars[start..end].iter().collect()
    }

    /// Select and delete inner text object (e.g., diw, di(, di").
    /// Returns true if something was deleted.
    pub fn select_inner_text_object(&mut self, ch: char) -> bool {
        match ch {
            'w' => self.delete_inner_word(),
            '(' | ')' | 'b' => self.delete_inner_pair('(', ')'),
            '[' | ']' => self.delete_inner_pair('[', ']'),
            '{' | '}' | 'B' => self.delete_inner_pair('{', '}'),
            '<' | '>' => self.delete_inner_pair('<', '>'),
            '"' => self.delete_inner_quote('"'),
            '\'' => self.delete_inner_quote('\''),
            '`' => self.delete_inner_quote('`'),
            _ => false,
        }
    }

    /// Select and delete around text object (e.g., daw, da().
    /// Returns true if something was deleted.
    pub fn select_around_text_object(&mut self, ch: char) -> bool {
        match ch {
            'w' => self.delete_around_word(),
            '(' | ')' | 'b' => self.delete_around_pair('(', ')'),
            '[' | ']' => self.delete_around_pair('[', ']'),
            '{' | '}' | 'B' => self.delete_around_pair('{', '}'),
            '<' | '>' => self.delete_around_pair('<', '>'),
            '"' => self.delete_around_quote('"'),
            '\'' => self.delete_around_quote('\''),
            '`' => self.delete_around_quote('`'),
            _ => false,
        }
    }

    fn delete_inner_word(&mut self) -> bool {
        let word = self.word_under_cursor();
        if word.is_empty() {
            return false;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let is_word_char = |c: char| c.is_alphanumeric() || c == '_';

        let mut start = self.file_cursor_col;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        let mut end = self.file_cursor_col;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }

        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, start);
        let eb = char_to_byte(line, end);
        let deleted: String = line.drain(sb..eb).collect();
        self.registers.delete(&deleted);
        self.clipboard = deleted;
        self.file_cursor_col = start;
        self.clamp_cursor_normal();
        self.file_dirty = true;
        true
    }

    fn delete_around_word(&mut self) -> bool {
        let word = self.word_under_cursor();
        if word.is_empty() {
            return false;
        }
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();
        let is_word_char = |c: char| c.is_alphanumeric() || c == '_';

        let mut start = self.file_cursor_col;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        let mut end = self.file_cursor_col;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }
        // Include trailing whitespace
        while end < chars.len() && chars[end].is_whitespace() {
            end += 1;
        }

        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, start);
        let eb = char_to_byte(line, end);
        let deleted: String = line.drain(sb..eb).collect();
        self.registers.delete(&deleted);
        self.clipboard = deleted;
        self.file_cursor_col = start;
        self.clamp_cursor_normal();
        self.file_dirty = true;
        true
    }

    fn delete_inner_pair(&mut self, open: char, close: char) -> bool {
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();

        // Find surrounding pair on current line
        let mut open_pos = None;
        let mut depth = 0i32;

        // Search backward for opening bracket
        for i in (0..=self.file_cursor_col.min(chars.len().saturating_sub(1))).rev() {
            if chars[i] == close && i != self.file_cursor_col {
                depth += 1;
            } else if chars[i] == open {
                if depth == 0 {
                    open_pos = Some(i);
                    break;
                }
                depth -= 1;
            }
        }

        let open_pos = match open_pos {
            Some(p) => p,
            None => return false,
        };

        // Search forward for closing bracket
        depth = 0;
        let mut close_pos = None;
        for i in (open_pos + 1)..chars.len() {
            if chars[i] == open {
                depth += 1;
            } else if chars[i] == close {
                if depth == 0 {
                    close_pos = Some(i);
                    break;
                }
                depth -= 1;
            }
        }

        let close_pos = match close_pos {
            Some(p) => p,
            None => return false,
        };

        // Delete contents between brackets (exclusive)
        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, open_pos + 1);
        let eb = char_to_byte(line, close_pos);
        if sb < eb {
            let deleted: String = line.drain(sb..eb).collect();
            self.registers.delete(&deleted);
            self.clipboard = deleted;
            self.file_cursor_col = open_pos + 1;
            self.file_dirty = true;
            return true;
        }
        false
    }

    fn delete_around_pair(&mut self, open: char, close: char) -> bool {
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();

        let mut open_pos = None;
        let mut depth = 0i32;

        for i in (0..=self.file_cursor_col.min(chars.len().saturating_sub(1))).rev() {
            if chars[i] == close && i != self.file_cursor_col {
                depth += 1;
            } else if chars[i] == open {
                if depth == 0 {
                    open_pos = Some(i);
                    break;
                }
                depth -= 1;
            }
        }

        let open_pos = match open_pos {
            Some(p) => p,
            None => return false,
        };

        depth = 0;
        let mut close_pos = None;
        for i in (open_pos + 1)..chars.len() {
            if chars[i] == open {
                depth += 1;
            } else if chars[i] == close {
                if depth == 0 {
                    close_pos = Some(i);
                    break;
                }
                depth -= 1;
            }
        }

        let close_pos = match close_pos {
            Some(p) => p,
            None => return false,
        };

        // Delete including brackets
        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, open_pos);
        let eb = char_to_byte(line, close_pos + 1);
        let deleted: String = line.drain(sb..eb).collect();
        self.registers.delete(&deleted);
        self.clipboard = deleted;
        self.file_cursor_col = open_pos;
        self.clamp_cursor_normal();
        self.file_dirty = true;
        true
    }

    fn delete_inner_quote(&mut self, quote: char) -> bool {
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();

        // Find quote boundaries around cursor
        let mut open_pos = None;
        for i in (0..=self.file_cursor_col.min(chars.len().saturating_sub(1))).rev() {
            if chars[i] == quote {
                open_pos = Some(i);
                break;
            }
        }

        let open_pos = match open_pos {
            Some(p) => p,
            None => return false,
        };

        let mut close_pos = None;
        for i in (open_pos + 1)..chars.len() {
            if chars[i] == quote {
                close_pos = Some(i);
                break;
            }
        }

        let close_pos = match close_pos {
            Some(p) => p,
            None => return false,
        };

        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, open_pos + 1);
        let eb = char_to_byte(line, close_pos);
        if sb < eb {
            let deleted: String = line.drain(sb..eb).collect();
            self.registers.delete(&deleted);
            self.clipboard = deleted;
            self.file_cursor_col = open_pos + 1;
            self.file_dirty = true;
            return true;
        }
        false
    }

    fn delete_around_quote(&mut self, quote: char) -> bool {
        let line = &self.file_lines[self.file_cursor_line];
        let chars: Vec<char> = line.chars().collect();

        let mut open_pos = None;
        for i in (0..=self.file_cursor_col.min(chars.len().saturating_sub(1))).rev() {
            if chars[i] == quote {
                open_pos = Some(i);
                break;
            }
        }

        let open_pos = match open_pos {
            Some(p) => p,
            None => return false,
        };

        let mut close_pos = None;
        for i in (open_pos + 1)..chars.len() {
            if chars[i] == quote {
                close_pos = Some(i);
                break;
            }
        }

        let close_pos = match close_pos {
            Some(p) => p,
            None => return false,
        };

        let line = &mut self.file_lines[self.file_cursor_line];
        let sb = char_to_byte(line, open_pos);
        let eb = char_to_byte(line, close_pos + 1);
        let deleted: String = line.drain(sb..eb).collect();
        self.registers.delete(&deleted);
        self.clipboard = deleted;
        self.file_cursor_col = open_pos;
        self.clamp_cursor_normal();
        self.file_dirty = true;
        true
    }

    /// Scroll so cursor is in center of screen.
    pub fn center_cursor_on_screen(&mut self) {
        let half = self.editor_visible_rows / 2;
        self.file_scroll = self.file_cursor_line.saturating_sub(half);
    }

    /// Scroll so cursor is at top of screen.
    pub fn scroll_cursor_to_top(&mut self) {
        self.file_scroll = self.file_cursor_line;
    }

    /// Scroll so cursor is at bottom of screen.
    pub fn scroll_cursor_to_bottom(&mut self) {
        self.file_scroll = self.file_cursor_line
            .saturating_sub(self.editor_visible_rows.saturating_sub(1));
    }

    // ── multi-buffer ─────────────────────────────────────────

    /// Save current file state to buffer list before switching.
    pub fn save_current_to_buffer(&mut self) {
        if self.open_file.is_none() {
            return;
        }
        // Update existing buffer or add new
        let path = self.open_file.clone();
        if let Some(buf) = self.buffers.iter_mut().find(|b| b.path == path) {
            buf.lines = self.file_lines.clone();
            buf.cursor_line = self.file_cursor_line;
            buf.cursor_col = self.file_cursor_col;
            buf.scroll = self.file_scroll;
            buf.dirty = self.file_dirty;
        } else {
            self.buffers.push(super::BufferEntry {
                path: self.open_file.clone(),
                lines: self.file_lines.clone(),
                cursor_line: self.file_cursor_line,
                cursor_col: self.file_cursor_col,
                scroll: self.file_scroll,
                dirty: self.file_dirty,
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
            });
            self.active_buffer = self.buffers.len() - 1;
        }
    }

    /// Switch to next buffer.
    pub fn buffer_next(&mut self) {
        if self.buffers.len() <= 1 {
            return;
        }
        self.save_current_to_buffer();
        self.active_buffer = (self.active_buffer + 1) % self.buffers.len();
        self.load_buffer(self.active_buffer);
    }

    /// Switch to previous buffer.
    pub fn buffer_prev(&mut self) {
        if self.buffers.len() <= 1 {
            return;
        }
        self.save_current_to_buffer();
        self.active_buffer = if self.active_buffer == 0 {
            self.buffers.len() - 1
        } else {
            self.active_buffer - 1
        };
        self.load_buffer(self.active_buffer);
    }

    /// Load a buffer by index into the editor state.
    fn load_buffer(&mut self, idx: usize) {
        if let Some(buf) = self.buffers.get(idx) {
            self.open_file = buf.path.clone();
            self.file_lines = buf.lines.clone();
            self.file_cursor_line = buf.cursor_line;
            self.file_cursor_col = buf.cursor_col;
            self.file_scroll = buf.scroll;
            self.file_dirty = buf.dirty;
            self.editor_mode = EditorMode::Normal;
            self.clamp_cursor_normal();
        }
    }

    /// Close the current buffer.
    pub fn buffer_close(&mut self) -> Result<(), String> {
        if self.file_dirty {
            return Err("unsaved changes — use :bclose! to discard".to_string());
        }
        if self.buffers.is_empty() {
            return Ok(());
        }
        self.buffers.remove(self.active_buffer);
        if self.buffers.is_empty() {
            self.open_file = None;
            self.file_lines = vec![String::new()];
            self.file_cursor_line = 0;
            self.file_cursor_col = 0;
            self.file_dirty = false;
        } else {
            self.active_buffer = self.active_buffer.min(self.buffers.len() - 1);
            self.load_buffer(self.active_buffer);
        }
        Ok(())
    }

    pub fn delete_selection_if_any(&mut self) -> bool {
        let ((sl, sc), (el, ec)) = match self.selection_bounds() {
            Some(b) => b,
            None => return false,
        };
        if sl == el {
            let line = &mut self.file_lines[sl];
            let sb = char_to_byte(line, sc);
            let eb = char_to_byte(line, ec);
            line.drain(sb..eb);
            self.file_cursor_line = sl;
            self.file_cursor_col = sc;
        } else {
            let start_prefix = {
                let line = &self.file_lines[sl];
                let sb = char_to_byte(line, sc);
                line[..sb].to_string()
            };
            let end_suffix = {
                let line = &self.file_lines[el];
                let eb = char_to_byte(line, ec);
                line[eb..].to_string()
            };
            self.file_lines[sl] = format!("{}{}", start_prefix, end_suffix);
            for _ in sl + 1..=el {
                self.file_lines.remove(sl + 1);
            }
            self.file_cursor_line = sl;
            self.file_cursor_col = sc;
        }
        self.file_dirty = true;
        self.clear_selection();
        true
    }
}
