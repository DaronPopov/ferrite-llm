use super::{TuiState, MAX_HISTORY};

impl TuiState {
    pub fn input_insert(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.history_idx = None;
    }

    pub fn input_backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.cursor);
            self.cursor = prev;
        }
    }

    pub fn input_delete(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor..next);
        }
    }

    pub fn input_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.input[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn input_right(&mut self) {
        if self.cursor < self.input.len() {
            self.cursor = self.input[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.input.len());
        }
    }

    pub fn input_home(&mut self) {
        self.cursor = 0;
    }
    pub fn input_end(&mut self) {
        self.cursor = self.input.len();
    }

    pub fn input_clear(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
    }

    pub fn input_kill_line(&mut self) {
        self.input.truncate(self.cursor);
    }

    pub fn input_kill_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.input[..self.cursor];
        let trimmed = before.trim_end();
        let new_end = trimmed
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        self.input.drain(new_end..self.cursor);
        self.cursor = new_end;
    }

    pub fn input_submit(&mut self) -> Option<String> {
        let line = self.input.trim().to_string();
        if line.is_empty() {
            return None;
        }
        if self.history.last().map_or(true, |h| h != &line) {
            self.history.push(line.clone());
            if self.history.len() > MAX_HISTORY {
                self.history.remove(0);
            }
        }
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        Some(line)
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_idx {
            None => self.history.len().saturating_sub(1),
            Some(i) => i.saturating_sub(1),
        };
        self.history_idx = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor = self.input.len();
    }

    pub fn history_down(&mut self) {
        match self.history_idx {
            None => {}
            Some(i) => {
                if i + 1 < self.history.len() {
                    let idx = i + 1;
                    self.history_idx = Some(idx);
                    self.input = self.history[idx].clone();
                    self.cursor = self.input.len();
                } else {
                    self.history_idx = None;
                    self.input.clear();
                    self.cursor = 0;
                }
            }
        }
    }

    pub fn input_is_empty(&self) -> bool {
        self.input.is_empty()
    }
}
