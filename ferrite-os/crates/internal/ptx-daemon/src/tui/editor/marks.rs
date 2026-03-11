// Marks and jump list for the vim-like editor.
//
// Marks:
//   m{a-z} — set mark at current position
//   '{a-z} — jump to mark line
//   `{a-z} — jump to mark exact position
//   '' — jump to last position before jump
//
// Jump list:
//   Ctrl-o — go back in jump history
//   Ctrl-i — go forward in jump history

use std::collections::HashMap;

const MAX_JUMP_LIST: usize = 100;

/// A position in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: usize,
    pub col: usize,
}

/// Stores named marks and the jump list.
pub struct MarkStore {
    /// Named marks a-z.
    marks: HashMap<char, Position>,
    /// The position before the last jump ('' mark).
    pub last_jump_from: Option<Position>,
    /// Jump list for Ctrl-o / Ctrl-i navigation.
    jump_list: Vec<Position>,
    /// Current position in the jump list.
    jump_index: usize,
}

impl MarkStore {
    pub fn new() -> Self {
        Self {
            marks: HashMap::new(),
            last_jump_from: None,
            jump_list: Vec::with_capacity(MAX_JUMP_LIST),
            jump_index: 0,
        }
    }

    /// Set a named mark.
    pub fn set_mark(&mut self, name: char, line: usize, col: usize) -> bool {
        if name.is_ascii_lowercase() {
            self.marks.insert(name, Position { line, col });
            true
        } else {
            false
        }
    }

    /// Get a named mark position.
    pub fn get_mark(&self, name: char) -> Option<Position> {
        if name == '\'' || name == '`' {
            return self.last_jump_from;
        }
        self.marks.get(&name).copied()
    }

    /// Record a jump (for Ctrl-o/Ctrl-i).
    pub fn record_jump(&mut self, from_line: usize, from_col: usize) {
        let pos = Position {
            line: from_line,
            col: from_col,
        };
        self.last_jump_from = Some(pos);

        // Truncate any forward history
        if self.jump_index < self.jump_list.len() {
            self.jump_list.truncate(self.jump_index);
        }

        // Don't duplicate the same position
        if self.jump_list.last() == Some(&pos) {
            return;
        }

        self.jump_list.push(pos);
        if self.jump_list.len() > MAX_JUMP_LIST {
            self.jump_list.remove(0);
        }
        self.jump_index = self.jump_list.len();
    }

    /// Jump backward in jump list (Ctrl-o).
    pub fn jump_back(&mut self) -> Option<Position> {
        if self.jump_index == 0 || self.jump_list.is_empty() {
            return None;
        }
        self.jump_index = self.jump_index.saturating_sub(1);
        self.jump_list.get(self.jump_index).copied()
    }

    /// Jump forward in jump list (Ctrl-i).
    pub fn jump_forward(&mut self) -> Option<Position> {
        if self.jump_index + 1 >= self.jump_list.len() {
            return None;
        }
        self.jump_index += 1;
        self.jump_list.get(self.jump_index).copied()
    }

    /// List all set marks for :marks display.
    pub fn list(&self) -> Vec<(char, Position)> {
        let mut result: Vec<_> = self.marks.iter().map(|(&k, &v)| (k, v)).collect();
        result.sort_by_key(|(k, _)| *k);
        if let Some(pos) = self.last_jump_from {
            result.push(('\'', pos));
        }
        result
    }

    /// Clear all marks.
    pub fn clear(&mut self) {
        self.marks.clear();
        self.last_jump_from = None;
    }
}

impl Default for MarkStore {
    fn default() -> Self {
        Self::new()
    }
}
