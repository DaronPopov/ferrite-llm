// Named registers for the vim-like editor.
//
// Registers a-z store yanked/deleted text.
// Special registers:
//   "" (unnamed) — default for d/y/p
//   "0 — last yank
//   "1-9 — delete history (newest first)
//   "/ — last search pattern
//   ". — last inserted text
//   "% — current filename

use std::collections::HashMap;

const MAX_DELETE_HISTORY: usize = 9;

pub struct RegisterFile {
    /// Named registers a-z.
    named: HashMap<char, String>,
    /// Unnamed register (default for d/y/p).
    pub unnamed: String,
    /// Last yank register ("0).
    pub last_yank: String,
    /// Delete history registers ("1-"9, index 0 = most recent).
    delete_history: Vec<String>,
    /// Currently selected register for next operation (set by " prefix).
    pub pending_register: Option<char>,
    /// Last search pattern register.
    pub search: String,
    /// Last inserted text register.
    pub insert: String,
}

#[allow(dead_code)]
impl RegisterFile {
    pub fn new() -> Self {
        Self {
            named: HashMap::new(),
            unnamed: String::new(),
            last_yank: String::new(),
            delete_history: Vec::with_capacity(MAX_DELETE_HISTORY),
            pending_register: None,
            search: String::new(),
            insert: String::new(),
        }
    }

    /// Set the register to use for the next operation.
    pub fn select(&mut self, reg: char) {
        self.pending_register = Some(reg);
    }

    /// Store text from a yank operation.
    pub fn yank(&mut self, text: &str) {
        if let Some(reg) = self.pending_register.take() {
            if reg.is_ascii_lowercase() {
                self.named.insert(reg, text.to_string());
            } else if reg.is_ascii_uppercase() {
                // Uppercase = append to register
                let lower = reg.to_ascii_lowercase();
                self.named
                    .entry(lower)
                    .or_default()
                    .push_str(text);
            }
        }
        self.unnamed = text.to_string();
        self.last_yank = text.to_string();
    }

    /// Store text from a delete operation.
    pub fn delete(&mut self, text: &str) {
        if let Some(reg) = self.pending_register.take() {
            if reg.is_ascii_lowercase() {
                self.named.insert(reg, text.to_string());
            } else if reg.is_ascii_uppercase() {
                let lower = reg.to_ascii_lowercase();
                self.named
                    .entry(lower)
                    .or_default()
                    .push_str(text);
            }
        }
        self.unnamed = text.to_string();

        // Push to delete history
        self.delete_history.insert(0, text.to_string());
        if self.delete_history.len() > MAX_DELETE_HISTORY {
            self.delete_history.truncate(MAX_DELETE_HISTORY);
        }
    }

    /// Get the text to paste.
    pub fn get_paste(&mut self) -> String {
        if let Some(reg) = self.pending_register.take() {
            return self.get(reg);
        }
        self.unnamed.clone()
    }

    /// Get the contents of a specific register.
    pub fn get(&self, reg: char) -> String {
        match reg {
            'a'..='z' => self.named.get(&reg).cloned().unwrap_or_default(),
            'A'..='Z' => self
                .named
                .get(&reg.to_ascii_lowercase())
                .cloned()
                .unwrap_or_default(),
            '"' => self.unnamed.clone(),
            '0' => self.last_yank.clone(),
            '1'..='9' => {
                let idx = (reg as u8 - b'1') as usize;
                self.delete_history
                    .get(idx)
                    .cloned()
                    .unwrap_or_default()
            }
            '/' => self.search.clone(),
            '.' => self.insert.clone(),
            _ => String::new(),
        }
    }

    /// List all non-empty registers for :registers display.
    pub fn list(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();

        // Unnamed
        if !self.unnamed.is_empty() {
            result.push(("\"\"".to_string(), truncate_display(&self.unnamed)));
        }

        // Last yank
        if !self.last_yank.is_empty() {
            result.push(("\"0".to_string(), truncate_display(&self.last_yank)));
        }

        // Delete history
        for (i, text) in self.delete_history.iter().enumerate() {
            if !text.is_empty() {
                result.push((format!("\"{}",  i + 1), truncate_display(text)));
            }
        }

        // Named a-z
        let mut names: Vec<_> = self.named.keys().copied().collect();
        names.sort();
        for ch in names {
            if let Some(text) = self.named.get(&ch) {
                if !text.is_empty() {
                    result.push((format!("\"{}",  ch), truncate_display(text)));
                }
            }
        }

        // Search
        if !self.search.is_empty() {
            result.push(("\"/".to_string(), truncate_display(&self.search)));
        }

        result
    }

    /// Clear pending register selection.
    pub fn clear_pending(&mut self) {
        self.pending_register = None;
    }
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

fn truncate_display(s: &str) -> String {
    let first_line = s.lines().next().unwrap_or("");
    if first_line.len() > 60 {
        format!("{}...", &first_line[..57])
    } else if s.lines().count() > 1 {
        format!("{} [+{} lines]", first_line, s.lines().count() - 1)
    } else {
        first_line.to_string()
    }
}
