// Reprogrammable keymap engine for the vim-like editor.
//
// Design:
//   - A `KeyMap` maps (EditorMode, KeySequence) -> EditorAction.
//   - Default bindings replicate standard vim.
//   - Users can remap at runtime via :map/:unmap.
//   - Keymaps can be serialized/deserialized for config persistence.

use std::collections::HashMap;
use std::fmt;

use crossterm::event::KeyCode;

use crate::tui::state::EditorMode;

// ── key sequence ──────────────────────────────────────────────────

/// A single key press with modifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyPress {
    pub code: KeyCode,
    pub ctrl: bool,
    pub shift: bool,
}

impl KeyPress {
    pub fn new(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: false,
            shift: false,
        }
    }

    pub fn ctrl(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: true,
            shift: false,
        }
    }

    pub fn ch(c: char) -> Self {
        Self::new(KeyCode::Char(c))
    }

    pub fn ctrl_ch(c: char) -> Self {
        Self::ctrl(KeyCode::Char(c))
    }
}

impl fmt::Display for KeyPress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ctrl {
            write!(f, "C-")?;
        }
        match self.code {
            KeyCode::Char(c) => write!(f, "{}", c),
            KeyCode::Enter => write!(f, "<CR>"),
            KeyCode::Esc => write!(f, "<Esc>"),
            KeyCode::Tab => write!(f, "<Tab>"),
            KeyCode::Backspace => write!(f, "<BS>"),
            KeyCode::Delete => write!(f, "<Del>"),
            KeyCode::Up => write!(f, "<Up>"),
            KeyCode::Down => write!(f, "<Down>"),
            KeyCode::Left => write!(f, "<Left>"),
            KeyCode::Right => write!(f, "<Right>"),
            KeyCode::Home => write!(f, "<Home>"),
            KeyCode::End => write!(f, "<End>"),
            KeyCode::PageUp => write!(f, "<PageUp>"),
            KeyCode::PageDown => write!(f, "<PageDown>"),
            KeyCode::F(n) => write!(f, "<F{}>", n),
            _ => write!(f, "<?>"),
        }
    }
}

/// A sequence of key presses (supports multi-key bindings like `gg`, `dd`, `ci(`).
pub type KeySeq = Vec<KeyPress>;

/// Parse a vim-style key notation string into a KeySeq.
/// Examples: "gg", "dd", "<C-f>", "<C-r>", "ciw", "<F5>"
pub fn parse_key_notation(s: &str) -> Option<KeySeq> {
    let mut result = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch == '<' {
            // Parse <...> notation
            chars.next();
            let mut token = String::new();
            while let Some(&c) = chars.peek() {
                chars.next();
                if c == '>' {
                    break;
                }
                token.push(c);
            }
            let lower = token.to_lowercase();
            if lower.starts_with("c-") {
                let key_char = token[2..].chars().next()?;
                result.push(KeyPress::ctrl_ch(key_char.to_ascii_lowercase()));
            } else {
                match lower.as_str() {
                    "cr" | "enter" | "return" => result.push(KeyPress::new(KeyCode::Enter)),
                    "esc" | "escape" => result.push(KeyPress::new(KeyCode::Esc)),
                    "tab" => result.push(KeyPress::new(KeyCode::Tab)),
                    "bs" | "backspace" => result.push(KeyPress::new(KeyCode::Backspace)),
                    "del" | "delete" => result.push(KeyPress::new(KeyCode::Delete)),
                    "up" => result.push(KeyPress::new(KeyCode::Up)),
                    "down" => result.push(KeyPress::new(KeyCode::Down)),
                    "left" => result.push(KeyPress::new(KeyCode::Left)),
                    "right" => result.push(KeyPress::new(KeyCode::Right)),
                    "home" => result.push(KeyPress::new(KeyCode::Home)),
                    "end" => result.push(KeyPress::new(KeyCode::End)),
                    "pageup" => result.push(KeyPress::new(KeyCode::PageUp)),
                    "pagedown" => result.push(KeyPress::new(KeyCode::PageDown)),
                    "space" => result.push(KeyPress::ch(' ')),
                    s if s.starts_with('f') => {
                        let n: u8 = s[1..].parse().ok()?;
                        result.push(KeyPress::new(KeyCode::F(n)));
                    }
                    _ => return None,
                }
            }
        } else {
            chars.next();
            result.push(KeyPress::ch(ch));
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Format a KeySeq back to vim notation string.
pub fn format_key_notation(seq: &KeySeq) -> String {
    seq.iter()
        .map(|kp| {
            if kp.ctrl {
                match kp.code {
                    KeyCode::Char(c) => format!("<C-{}>", c),
                    _ => format!("<C-{}>", kp),
                }
            } else {
                match kp.code {
                    KeyCode::Char(c) => c.to_string(),
                    _ => format!("{}", kp),
                }
            }
        })
        .collect()
}

// ── editor actions ────────────────────────────────────────────────

/// Every action the editor can perform. The keymap maps key sequences to these.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum EditorAction {
    // ── cursor motions ──
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    WordForward,
    WordBackward,
    WordEnd,
    BigWordForward,
    BigWordBackward,
    BigWordEnd,
    LineStart,
    LineEnd,
    FirstNonBlank,
    GotoFirstLine,
    GotoLastLine,
    GotoLine(usize),
    PageDown,
    PageUp,
    HalfPageDown,
    HalfPageUp,
    ScreenTop,
    ScreenMiddle,
    ScreenBottom,
    ParagraphForward,
    ParagraphBackward,
    MatchingBracket,
    FindCharForward,
    FindCharBackward,
    TillCharForward,
    TillCharBackward,
    RepeatFindChar,
    RepeatFindCharReverse,

    // ── mode transitions ──
    InsertMode,
    InsertAtLineStart,
    AppendAfterCursor,
    AppendAtLineEnd,
    OpenLineBelow,
    OpenLineAbove,
    VisualMode,
    VisualLineMode,
    CommandMode,
    NormalMode,
    ReplaceChar,

    // ── editing ──
    DeleteChar,
    DeleteCharBefore,
    DeleteLine,
    YankLine,
    PutAfter,
    PutBefore,
    JoinLines,
    Undo,
    Redo,
    Indent,
    Dedent,
    ChangeToEndOfLine,
    DeleteToEndOfLine,
    YankToEndOfLine,
    SubstituteChar,
    SubstituteLine,

    // ── operator-pending (compose with a motion) ──
    OperatorDelete,
    OperatorChange,
    OperatorYank,
    OperatorIndent,
    OperatorDedent,

    // ── search ──
    SearchForward,
    SearchBackward,
    SearchNext,
    SearchPrev,
    SearchWordUnderCursor,

    // ── marks / jumps ──
    SetMark,
    GotoMark,
    GotoMarkLine,
    JumpBack,
    JumpForward,

    // ── registers ──
    SelectRegister,

    // ── macros ──
    RecordMacro,
    PlayMacro,

    // ── buffers ──
    BufferNext,
    BufferPrev,
    BufferClose,
    BufferList,

    // ── misc ──
    ToggleFocus,
    ToggleUiMode,
    Run,
    Stop,
    ToggleRunOutput,
    RepeatLastCommand,
    NoOp,
    ScrollLineUp,
    ScrollLineDown,
    CenterCursorLine,
}

// ── keymap ────────────────────────────────────────────────────────

/// Maps (mode, key sequence) to an action. Supports multi-key sequences and
/// user remapping at runtime.
#[allow(dead_code)]
pub struct KeyMap {
    /// Primary binding table: mode -> (key sequence -> action).
    bindings: HashMap<EditorMode, HashMap<KeySeq, EditorAction>>,
    /// User overrides (persisted separately from defaults).
    user_overrides: HashMap<EditorMode, HashMap<KeySeq, EditorAction>>,
    /// Disabled default bindings (user explicitly unmapped).
    disabled: HashMap<EditorMode, Vec<KeySeq>>,
}

#[allow(dead_code)]
impl KeyMap {
    pub fn new() -> Self {
        let mut km = Self {
            bindings: HashMap::new(),
            user_overrides: HashMap::new(),
            disabled: HashMap::new(),
        };
        km.load_defaults();
        km
    }

    /// Look up an action for a complete key sequence in a given mode.
    pub fn lookup(&self, mode: EditorMode, seq: &KeySeq) -> Option<&EditorAction> {
        // User overrides take priority
        if let Some(mode_map) = self.user_overrides.get(&mode) {
            if let Some(action) = mode_map.get(seq) {
                return Some(action);
            }
        }
        // Check if explicitly disabled
        if let Some(disabled_list) = self.disabled.get(&mode) {
            if disabled_list.contains(seq) {
                return None;
            }
        }
        // Default bindings
        self.bindings.get(&mode).and_then(|m| m.get(seq))
    }

    /// Check if a key sequence is a prefix of any binding (for multi-key handling).
    pub fn is_prefix(&self, mode: EditorMode, seq: &KeySeq) -> bool {
        let check_prefix = |map: &HashMap<KeySeq, EditorAction>| -> bool {
            map.keys().any(|k| k.len() > seq.len() && k.starts_with(seq))
        };

        if let Some(mode_map) = self.user_overrides.get(&mode) {
            if check_prefix(mode_map) {
                return true;
            }
        }
        if let Some(mode_map) = self.bindings.get(&mode) {
            if check_prefix(mode_map) {
                return true;
            }
        }
        false
    }

    /// Add a user override mapping.
    pub fn map(&mut self, mode: EditorMode, seq: KeySeq, action: EditorAction) {
        self.user_overrides
            .entry(mode)
            .or_default()
            .insert(seq, action);
    }

    /// Remove a user override (restore default if any).
    pub fn unmap(&mut self, mode: EditorMode, seq: &KeySeq) {
        if let Some(mode_map) = self.user_overrides.get_mut(&mode) {
            mode_map.remove(seq);
        }
        // Also add to disabled so default doesn't fire
        self.disabled.entry(mode).or_default().push(seq.clone());
    }

    /// Restore a binding (remove from disabled list).
    pub fn restore(&mut self, mode: EditorMode, seq: &KeySeq) {
        if let Some(disabled_list) = self.disabled.get_mut(&mode) {
            disabled_list.retain(|k| k != seq);
        }
        if let Some(overrides) = self.user_overrides.get_mut(&mode) {
            overrides.remove(seq);
        }
    }

    /// List all active bindings for a mode (merged defaults + overrides - disabled).
    pub fn list_bindings(&self, mode: EditorMode) -> Vec<(String, &EditorAction)> {
        let mut result = Vec::new();
        let disabled = self.disabled.get(&mode);

        // Defaults
        if let Some(mode_map) = self.bindings.get(&mode) {
            for (seq, action) in mode_map {
                let is_disabled = disabled
                    .map(|d| d.contains(seq))
                    .unwrap_or(false);
                if !is_disabled {
                    // Check if overridden
                    let overridden = self
                        .user_overrides
                        .get(&mode)
                        .and_then(|m| m.get(seq))
                        .is_some();
                    if !overridden {
                        result.push((format_key_notation(seq), action));
                    }
                }
            }
        }

        // User overrides
        if let Some(mode_map) = self.user_overrides.get(&mode) {
            for (seq, action) in mode_map {
                result.push((format_key_notation(seq), action));
            }
        }

        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Serialize all user overrides to a simple text format for persistence.
    pub fn export_user_config(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (mode, map) in &self.user_overrides {
            let mode_str = match mode {
                EditorMode::Normal => "nmap",
                EditorMode::Insert => "imap",
                EditorMode::Visual => "vmap",
                EditorMode::Command => "cmap",
            };
            for (seq, action) in map {
                lines.push(format!("{} {} {:?}", mode_str, format_key_notation(seq), action));
            }
        }
        lines
    }

    // ── default bindings ──────────────────────────────────────────

    fn bind(&mut self, mode: EditorMode, seq: KeySeq, action: EditorAction) {
        self.bindings.entry(mode).or_default().insert(seq, action);
    }

    fn load_defaults(&mut self) {
        use EditorAction::*;
        use EditorMode::*;

        // ── Normal mode — motions ──
        self.bind(Normal, vec![KeyPress::ch('h')], MoveLeft);
        self.bind(Normal, vec![KeyPress::ch('l')], MoveRight);
        self.bind(Normal, vec![KeyPress::ch('j')], MoveDown);
        self.bind(Normal, vec![KeyPress::ch('k')], MoveUp);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Left)], MoveLeft);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Right)], MoveRight);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Down)], MoveDown);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Up)], MoveUp);

        self.bind(Normal, vec![KeyPress::ch('w')], WordForward);
        self.bind(Normal, vec![KeyPress::ch('b')], WordBackward);
        self.bind(Normal, vec![KeyPress::ch('e')], WordEnd);
        self.bind(Normal, vec![KeyPress::ch('W')], BigWordForward);
        self.bind(Normal, vec![KeyPress::ch('B')], BigWordBackward);
        self.bind(Normal, vec![KeyPress::ch('E')], BigWordEnd);

        self.bind(Normal, vec![KeyPress::ch('0')], LineStart);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Home)], LineStart);
        self.bind(Normal, vec![KeyPress::ch('^')], FirstNonBlank);
        self.bind(Normal, vec![KeyPress::ch('$')], LineEnd);
        self.bind(Normal, vec![KeyPress::new(KeyCode::End)], LineEnd);

        self.bind(Normal, vec![KeyPress::ch('G')], GotoLastLine);
        self.bind(Normal, vec![KeyPress::ch('g'), KeyPress::ch('g')], GotoFirstLine);

        self.bind(Normal, vec![KeyPress::ctrl_ch('f')], PageDown);
        self.bind(Normal, vec![KeyPress::ctrl_ch('b')], PageUp);
        self.bind(Normal, vec![KeyPress::ctrl_ch('d')], HalfPageDown);
        self.bind(Normal, vec![KeyPress::ctrl_ch('u')], HalfPageUp);
        self.bind(Normal, vec![KeyPress::new(KeyCode::PageDown)], PageDown);
        self.bind(Normal, vec![KeyPress::new(KeyCode::PageUp)], PageUp);

        self.bind(Normal, vec![KeyPress::ch('H')], ScreenTop);
        self.bind(Normal, vec![KeyPress::ch('M')], ScreenMiddle);
        self.bind(Normal, vec![KeyPress::ch('L')], ScreenBottom);

        self.bind(Normal, vec![KeyPress::ch('{')], ParagraphBackward);
        self.bind(Normal, vec![KeyPress::ch('}')], ParagraphForward);
        self.bind(Normal, vec![KeyPress::ch('%')], MatchingBracket);

        self.bind(Normal, vec![KeyPress::ch('f')], FindCharForward);
        self.bind(Normal, vec![KeyPress::ch('F')], FindCharBackward);
        self.bind(Normal, vec![KeyPress::ch('t')], TillCharForward);
        self.bind(Normal, vec![KeyPress::ch('T')], TillCharBackward);
        self.bind(Normal, vec![KeyPress::ch(';')], RepeatFindChar);
        self.bind(Normal, vec![KeyPress::ch(',')], RepeatFindCharReverse);

        // ── Normal mode — mode transitions ──
        self.bind(Normal, vec![KeyPress::ch('i')], InsertMode);
        self.bind(Normal, vec![KeyPress::ch('I')], InsertAtLineStart);
        self.bind(Normal, vec![KeyPress::ch('a')], AppendAfterCursor);
        self.bind(Normal, vec![KeyPress::ch('A')], AppendAtLineEnd);
        self.bind(Normal, vec![KeyPress::ch('o')], OpenLineBelow);
        self.bind(Normal, vec![KeyPress::ch('O')], OpenLineAbove);
        self.bind(Normal, vec![KeyPress::ch('v')], VisualMode);
        self.bind(Normal, vec![KeyPress::ch('V')], VisualLineMode);
        self.bind(Normal, vec![KeyPress::ch(':')], CommandMode);
        self.bind(Normal, vec![KeyPress::ch('r')], ReplaceChar);

        // ── Normal mode — editing ──
        self.bind(Normal, vec![KeyPress::ch('x')], DeleteChar);
        self.bind(Normal, vec![KeyPress::ch('X')], DeleteCharBefore);
        self.bind(Normal, vec![KeyPress::ch('d'), KeyPress::ch('d')], DeleteLine);
        self.bind(Normal, vec![KeyPress::ch('y'), KeyPress::ch('y')], YankLine);
        self.bind(Normal, vec![KeyPress::ch('p')], PutAfter);
        self.bind(Normal, vec![KeyPress::ch('P')], PutBefore);
        self.bind(Normal, vec![KeyPress::ch('J')], JoinLines);
        self.bind(Normal, vec![KeyPress::ch('u')], Undo);
        self.bind(Normal, vec![KeyPress::ctrl_ch('r')], Redo);
        self.bind(Normal, vec![KeyPress::ch('>')], OperatorIndent);
        self.bind(Normal, vec![KeyPress::ch('<')], OperatorDedent);
        self.bind(Normal, vec![KeyPress::ch('C')], ChangeToEndOfLine);
        self.bind(Normal, vec![KeyPress::ch('D')], DeleteToEndOfLine);
        self.bind(Normal, vec![KeyPress::ch('Y')], YankToEndOfLine);
        self.bind(Normal, vec![KeyPress::ch('s')], SubstituteChar);
        self.bind(Normal, vec![KeyPress::ch('S')], SubstituteLine);
        self.bind(Normal, vec![KeyPress::ch('.')], RepeatLastCommand);

        // ── Normal mode — operators (compose with motion) ──
        self.bind(Normal, vec![KeyPress::ch('d')], OperatorDelete);
        self.bind(Normal, vec![KeyPress::ch('c')], OperatorChange);
        self.bind(Normal, vec![KeyPress::ch('y')], OperatorYank);

        // ── Normal mode — search ──
        self.bind(Normal, vec![KeyPress::ch('/')], SearchForward);
        self.bind(Normal, vec![KeyPress::ch('?')], SearchBackward);
        self.bind(Normal, vec![KeyPress::ch('n')], SearchNext);
        self.bind(Normal, vec![KeyPress::ch('N')], SearchPrev);
        self.bind(Normal, vec![KeyPress::ch('*')], SearchWordUnderCursor);

        // ── Normal mode — marks / jumps ──
        self.bind(Normal, vec![KeyPress::ch('m')], SetMark);
        self.bind(Normal, vec![KeyPress::ch('\'')], GotoMarkLine);
        self.bind(Normal, vec![KeyPress::ch('`')], GotoMark);
        self.bind(Normal, vec![KeyPress::ctrl_ch('o')], JumpBack);
        self.bind(Normal, vec![KeyPress::ctrl_ch('i')], JumpForward);

        // ── Normal mode — registers ──
        self.bind(Normal, vec![KeyPress::ch('"')], SelectRegister);

        // ── Normal mode — macros ──
        self.bind(Normal, vec![KeyPress::ch('q')], RecordMacro);
        self.bind(Normal, vec![KeyPress::ch('@')], PlayMacro);

        // ── Normal mode — buffers / misc ──
        self.bind(Normal, vec![KeyPress::new(KeyCode::Tab)], ToggleFocus);
        self.bind(Normal, vec![KeyPress::new(KeyCode::Esc)], ToggleUiMode);
        self.bind(Normal, vec![KeyPress::new(KeyCode::F(5))], Run);
        self.bind(Normal, vec![KeyPress::new(KeyCode::F(6))], Stop);
        self.bind(Normal, vec![KeyPress::ctrl_ch('j')], ToggleRunOutput);
        self.bind(Normal, vec![KeyPress::ctrl_ch('v')], NoOp); // Paste handled separately
        self.bind(Normal, vec![KeyPress::ctrl_ch('c')], NoOp); // Copy handled separately
        self.bind(Normal, vec![KeyPress::ctrl_ch('e')], ScrollLineDown);
        self.bind(Normal, vec![KeyPress::ctrl_ch('y')], ScrollLineUp);
        self.bind(Normal, vec![KeyPress::ch('z'), KeyPress::ch('z')], CenterCursorLine);

        // ── Visual mode — motions (same as normal) ──
        self.bind(Visual, vec![KeyPress::ch('h')], MoveLeft);
        self.bind(Visual, vec![KeyPress::ch('l')], MoveRight);
        self.bind(Visual, vec![KeyPress::ch('j')], MoveDown);
        self.bind(Visual, vec![KeyPress::ch('k')], MoveUp);
        self.bind(Visual, vec![KeyPress::new(KeyCode::Left)], MoveLeft);
        self.bind(Visual, vec![KeyPress::new(KeyCode::Right)], MoveRight);
        self.bind(Visual, vec![KeyPress::new(KeyCode::Down)], MoveDown);
        self.bind(Visual, vec![KeyPress::new(KeyCode::Up)], MoveUp);
        self.bind(Visual, vec![KeyPress::ch('w')], WordForward);
        self.bind(Visual, vec![KeyPress::ch('b')], WordBackward);
        self.bind(Visual, vec![KeyPress::ch('e')], WordEnd);
        self.bind(Visual, vec![KeyPress::ch('0')], LineStart);
        self.bind(Visual, vec![KeyPress::new(KeyCode::Home)], LineStart);
        self.bind(Visual, vec![KeyPress::ch('$')], LineEnd);
        self.bind(Visual, vec![KeyPress::new(KeyCode::End)], LineEnd);
        self.bind(Visual, vec![KeyPress::ch('G')], GotoLastLine);
        self.bind(Visual, vec![KeyPress::ch('g')], GotoFirstLine);
        self.bind(Visual, vec![KeyPress::ch('{')], ParagraphBackward);
        self.bind(Visual, vec![KeyPress::ch('}')], ParagraphForward);
        self.bind(Visual, vec![KeyPress::ch('%')], MatchingBracket);

        // ── Visual mode — actions ──
        self.bind(Visual, vec![KeyPress::new(KeyCode::Esc)], NormalMode);
        self.bind(Visual, vec![KeyPress::ch('v')], NormalMode);
        self.bind(Visual, vec![KeyPress::ch('d')], OperatorDelete);
        self.bind(Visual, vec![KeyPress::ch('x')], OperatorDelete);
        self.bind(Visual, vec![KeyPress::ch('y')], OperatorYank);
        self.bind(Visual, vec![KeyPress::ch('c')], OperatorChange);
    }
}

impl Default for KeyMap {
    fn default() -> Self {
        Self::new()
    }
}
