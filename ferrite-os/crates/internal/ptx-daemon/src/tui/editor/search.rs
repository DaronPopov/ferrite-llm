// Search engine for the vim-like editor.
//
// Supports:
//   - / forward search with regex
//   - ? backward search
//   - n/N next/prev match
//   - * search word under cursor
//   - Highlight positions for rendering

/// Direction of search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDirection {
    Forward,
    Backward,
}

/// A match position in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchMatch {
    pub line: usize,
    pub col_start: usize,
    pub col_end: usize,
}

/// Search state for the editor.
#[allow(dead_code)]
pub struct SearchState {
    /// Current search pattern.
    pub pattern: String,
    /// Search direction.
    pub direction: SearchDirection,
    /// Whether search input is active (/ or ? prompt visible).
    pub active: bool,
    /// Current input buffer while typing search.
    pub input: String,
    /// Cursor position in the input buffer.
    pub input_cursor: usize,
    /// All match positions (recomputed on pattern change).
    pub matches: Vec<SearchMatch>,
    /// Index into `matches` of the current match (the one we jumped to).
    pub current_match: Option<usize>,
    /// Whether search wraps around.
    pub wrap: bool,
    /// Incremental search: highlight as you type.
    pub incremental: bool,
    /// Last search pattern (for n/N when no active search).
    pub last_pattern: Option<String>,
    /// Last direction.
    pub last_direction: SearchDirection,
}

#[allow(dead_code)]
impl SearchState {
    pub fn new() -> Self {
        Self {
            pattern: String::new(),
            direction: SearchDirection::Forward,
            active: false,
            input: String::new(),
            input_cursor: 0,
            matches: Vec::new(),
            current_match: None,
            wrap: true,
            incremental: true,
            last_pattern: None,
            last_direction: SearchDirection::Forward,
        }
    }

    /// Start a new search in the given direction.
    pub fn start(&mut self, direction: SearchDirection) {
        self.direction = direction;
        self.active = true;
        self.input.clear();
        self.input_cursor = 0;
    }

    /// Insert a character into the search input.
    pub fn insert_char(&mut self, ch: char) {
        self.input.insert(self.input_cursor, ch);
        self.input_cursor += ch.len_utf8();
    }

    /// Delete the character before the cursor.
    pub fn backspace(&mut self) {
        if self.input_cursor > 0 {
            let prev = self.input[..self.input_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.input_cursor);
            self.input_cursor = prev;
        }
    }

    /// Submit the search (Enter pressed).
    pub fn submit(&mut self) -> Option<String> {
        self.active = false;
        let pat = self.input.trim().to_string();
        if pat.is_empty() {
            // Use last pattern
            return self.last_pattern.clone();
        }
        self.pattern = pat.clone();
        self.last_pattern = Some(pat.clone());
        self.last_direction = self.direction;
        Some(pat)
    }

    /// Cancel the search (Esc pressed).
    pub fn cancel(&mut self) {
        self.active = false;
        self.input.clear();
        self.input_cursor = 0;
    }

    /// Compute all matches for a pattern in the given buffer lines.
    pub fn find_all(&mut self, lines: &[String], pattern: &str) {
        self.matches.clear();
        self.current_match = None;

        if pattern.is_empty() {
            return;
        }

        let lower_pattern = pattern.to_lowercase();

        for (line_idx, line) in lines.iter().enumerate() {
            let lower_line = line.to_lowercase();
            let mut search_from = 0;
            while let Some(byte_pos) = lower_line[search_from..].find(&lower_pattern) {
                let abs_byte = search_from + byte_pos;
                // Convert byte offset to char offset
                let col_start = line[..abs_byte].chars().count();
                let col_end = col_start + pattern.chars().count();
                self.matches.push(SearchMatch {
                    line: line_idx,
                    col_start,
                    col_end,
                });
                search_from = abs_byte + lower_pattern.len().max(1);
            }
        }
    }

    /// Find the next match from the given cursor position.
    pub fn find_next(
        &mut self,
        cursor_line: usize,
        cursor_col: usize,
        direction: SearchDirection,
    ) -> Option<SearchMatch> {
        if self.matches.is_empty() {
            return None;
        }

        match direction {
            SearchDirection::Forward => {
                // Find first match after cursor
                let idx = self.matches.iter().position(|m| {
                    m.line > cursor_line
                        || (m.line == cursor_line && m.col_start > cursor_col)
                });
                let idx = idx.or_else(|| {
                    if self.wrap {
                        Some(0)
                    } else {
                        None
                    }
                })?;
                self.current_match = Some(idx);
                Some(self.matches[idx])
            }
            SearchDirection::Backward => {
                // Find last match before cursor
                let idx = self.matches.iter().rposition(|m| {
                    m.line < cursor_line
                        || (m.line == cursor_line && m.col_start < cursor_col)
                });
                let idx = idx.or_else(|| {
                    if self.wrap {
                        Some(self.matches.len() - 1)
                    } else {
                        None
                    }
                })?;
                self.current_match = Some(idx);
                Some(self.matches[idx])
            }
        }
    }

    /// Jump to the match at the current index + delta.
    pub fn jump_match(&mut self, delta: i32) -> Option<SearchMatch> {
        if self.matches.is_empty() {
            return None;
        }
        let current = self.current_match.unwrap_or(0);
        let new_idx = if delta >= 0 {
            (current + delta as usize) % self.matches.len()
        } else {
            let back = (-delta) as usize;
            if back > current {
                self.matches.len() - (back - current) % self.matches.len()
            } else {
                current - back
            }
        };
        self.current_match = Some(new_idx);
        Some(self.matches[new_idx])
    }

    /// Check if a character at (line, col) is inside a match (for highlighting).
    pub fn is_match_char(&self, line: usize, col: usize) -> bool {
        self.matches
            .iter()
            .any(|m| m.line == line && col >= m.col_start && col < m.col_end)
    }

    /// Check if a character is at the current match position.
    pub fn is_current_match_char(&self, line: usize, col: usize) -> bool {
        if let Some(idx) = self.current_match {
            if let Some(m) = self.matches.get(idx) {
                return m.line == line && col >= m.col_start && col < m.col_end;
            }
        }
        false
    }

    /// Get match count summary string like "3/15".
    pub fn match_summary(&self) -> String {
        if self.matches.is_empty() {
            return "no matches".to_string();
        }
        match self.current_match {
            Some(idx) => format!("{}/{}", idx + 1, self.matches.len()),
            None => format!("{} matches", self.matches.len()),
        }
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}
