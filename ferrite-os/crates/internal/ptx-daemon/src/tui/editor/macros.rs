// Macro recording and playback for the vim-like editor.
//
// q{a-z} — start recording into register
// q      — stop recording (when already recording)
// @{a-z} — play macro from register
// @@     — repeat last played macro

use std::collections::HashMap;

use crossterm::event::KeyCode;

/// A recorded key event.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RecordedKey {
    pub code: KeyCode,
    pub ctrl: bool,
    pub shift: bool,
}

/// Macro engine: records key sequences and replays them.
#[allow(dead_code)]
pub struct MacroEngine {
    /// Stored macros (register -> key sequence).
    macros: HashMap<char, Vec<RecordedKey>>,
    /// Currently recording into this register (None = not recording).
    pub recording: Option<char>,
    /// Buffer for the macro being recorded.
    record_buffer: Vec<RecordedKey>,
    /// Last played macro register (for @@ repeat).
    pub last_played: Option<char>,
    /// Replay queue: keys waiting to be dispatched.
    replay_queue: Vec<RecordedKey>,
    /// Current position in the replay queue.
    replay_pos: usize,
    /// Whether we're currently replaying (to prevent recursive recording).
    pub replaying: bool,
}

#[allow(dead_code)]
impl MacroEngine {
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
            recording: None,
            record_buffer: Vec::new(),
            last_played: None,
            replay_queue: Vec::new(),
            replay_pos: 0,
            replaying: false,
        }
    }

    /// Start recording a macro into the given register.
    /// Returns false if the register is invalid.
    pub fn start_recording(&mut self, register: char) -> bool {
        if !register.is_ascii_lowercase() {
            return false;
        }
        self.recording = Some(register);
        self.record_buffer.clear();
        true
    }

    /// Stop recording and store the macro.
    /// Returns the register name if recording was active.
    pub fn stop_recording(&mut self) -> Option<char> {
        if let Some(reg) = self.recording.take() {
            // Don't store the final 'q' that stopped recording
            self.macros.insert(reg, self.record_buffer.clone());
            self.record_buffer.clear();
            Some(reg)
        } else {
            None
        }
    }

    /// Record a key event (called for every key while recording).
    pub fn record_key(&mut self, code: KeyCode, ctrl: bool, shift: bool) {
        if self.recording.is_some() && !self.replaying {
            self.record_buffer.push(RecordedKey { code, ctrl, shift });
        }
    }

    /// Start replaying a macro.
    /// Returns true if the macro exists and playback started.
    pub fn start_replay(&mut self, register: char) -> bool {
        let reg = if register == '@' {
            // @@ = repeat last
            match self.last_played {
                Some(r) => r,
                None => return false,
            }
        } else {
            register
        };

        if let Some(keys) = self.macros.get(&reg) {
            self.replay_queue = keys.clone();
            self.replay_pos = 0;
            self.replaying = true;
            self.last_played = Some(reg);
            true
        } else {
            false
        }
    }

    /// Get the next key from the replay queue.
    /// Returns None when replay is complete.
    pub fn next_replay_key(&mut self) -> Option<RecordedKey> {
        if !self.replaying {
            return None;
        }
        if self.replay_pos >= self.replay_queue.len() {
            self.replaying = false;
            self.replay_queue.clear();
            self.replay_pos = 0;
            return None;
        }
        let key = self.replay_queue[self.replay_pos].clone();
        self.replay_pos += 1;
        Some(key)
    }

    /// Check if we're currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// Check if we're currently replaying.
    pub fn is_replaying(&self) -> bool {
        self.replaying
    }

    /// List stored macros for display.
    pub fn list(&self) -> Vec<(char, usize)> {
        let mut result: Vec<_> = self
            .macros
            .iter()
            .map(|(&k, v)| (k, v.len()))
            .collect();
        result.sort_by_key(|(k, _)| *k);
        result
    }

    /// Get the recording status string for the status bar.
    pub fn status(&self) -> Option<String> {
        self.recording.map(|r| format!("recording @{}", r))
    }
}

impl Default for MacroEngine {
    fn default() -> Self {
        Self::new()
    }
}
