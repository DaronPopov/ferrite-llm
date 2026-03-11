//! Visual token system for the ferrite TUI.
//!
//! Every color, semantic style, and style-related helper lives here.
//! Layout and widget modules import from this module instead of defining
//! ad-hoc inline styles.
//!
//! Colors are resolved at runtime via [`init`] so the daemon can switch
//! between palettes.  Press Ctrl+4 to cycle themes live.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};

use crate::events::LogCategory;
use crate::tui::state::{EditorMode, FilesFocus};

// ── theme variant ────────────────────────────────────────────────

/// Available theme variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeVariant {
    Default,
    HighContrast,
    Sandstone,
}

const THEME_COUNT: usize = 3;

impl ThemeVariant {
    /// Parse from a string (config value / env var).  Unknown values
    /// fall back to [`ThemeVariant::Default`].
    pub fn from_str_loose(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "high-contrast" | "high_contrast" | "highcontrast" | "hc" => Self::HighContrast,
            "sandstone" | "sand" | "beige" | "warm" => Self::Sandstone,
            _ => Self::Default,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Default => 0,
            Self::HighContrast => 1,
            Self::Sandstone => 2,
        }
    }

    fn from_index(i: usize) -> Self {
        match i % THEME_COUNT {
            0 => Self::Default,
            1 => Self::HighContrast,
            _ => Self::Sandstone,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::HighContrast => "high-contrast",
            Self::Sandstone => "sandstone",
        }
    }
}

// ── color palette struct ─────────────────────────────────────────

/// All palette colors for one theme variant.
#[derive(Debug, Clone)]
pub struct ThemeColors {
    pub bg: Color,
    pub fg: Color,
    pub fg_dim: Color,
    pub fg_bright: Color,
    pub rule: Color,
    pub info: Color,
    pub good: Color,
    pub warn: Color,
    pub bad: Color,
    pub bar_empty: Color,
    pub card_bg: Color,
    pub selection: Color,
}

impl ThemeColors {
    fn default_palette() -> Self {
        Self {
            bg: Color::Rgb(9, 14, 20),
            fg: Color::Rgb(200, 208, 218),
            fg_dim: Color::Rgb(112, 126, 142),
            fg_bright: Color::Rgb(230, 236, 244),
            rule: Color::Rgb(34, 66, 88),
            info: Color::Rgb(78, 154, 186),
            good: Color::Rgb(90, 170, 120),
            warn: Color::Rgb(200, 168, 78),
            bad: Color::Rgb(214, 98, 98),
            bar_empty: Color::Rgb(30, 40, 52),
            card_bg: Color::Rgb(16, 24, 35),
            selection: Color::Rgb(42, 68, 108),
        }
    }

    fn high_contrast_palette() -> Self {
        Self {
            bg: Color::Rgb(9, 14, 20),
            fg: Color::Rgb(230, 236, 244),
            fg_dim: Color::Rgb(150, 165, 180),
            fg_bright: Color::Rgb(250, 252, 255),
            rule: Color::Rgb(60, 100, 130),
            info: Color::Rgb(110, 185, 220),
            good: Color::Rgb(120, 210, 150),
            warn: Color::Rgb(230, 200, 100),
            bad: Color::Rgb(230, 110, 110),
            bar_empty: Color::Rgb(40, 52, 65),
            card_bg: Color::Rgb(22, 32, 46),
            selection: Color::Rgb(55, 85, 135),
        }
    }

    /// Warm sandstone palette — beige accents on dark warm brown.
    ///
    /// Inspired by desert clay terminals: earthy tones, warm neutrals,
    /// low-fatigue amber/sand highlights instead of cool blues.
    fn sandstone_palette() -> Self {
        Self {
            bg: Color::Rgb(22, 18, 14),           // dark roasted umber
            fg: Color::Rgb(210, 198, 180),         // warm parchment
            fg_dim: Color::Rgb(138, 126, 108),     // weathered clay
            fg_bright: Color::Rgb(238, 228, 210),  // bright sand
            rule: Color::Rgb(62, 52, 40),          // dark leather
            info: Color::Rgb(196, 170, 118),       // desert gold / beige accent
            good: Color::Rgb(138, 164, 108),       // sage brush
            warn: Color::Rgb(210, 148, 72),        // burnt sienna
            bad: Color::Rgb(204, 98, 82),          // terracotta
            bar_empty: Color::Rgb(38, 32, 26),     // shadow loam
            card_bg: Color::Rgb(30, 26, 20),       // dark warm card
            selection: Color::Rgb(72, 60, 42),     // warm sepia highlight
        }
    }
}

// ── global singleton (runtime-switchable) ────────────────────────

/// All palettes, indexed by ThemeVariant.
static PALETTES: OnceLock<[ThemeColors; THEME_COUNT]> = OnceLock::new();

/// Current active palette index.  Switched atomically.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Initialise the theme system.  Must be called once before any
/// drawing happens (typically at the top of [`super::app::run_tui`]).
/// Sets the starting theme.  Subsequent calls are no-ops.
pub fn init(variant: ThemeVariant) {
    PALETTES.get_or_init(|| [
        ThemeColors::default_palette(),
        ThemeColors::high_contrast_palette(),
        ThemeColors::sandstone_palette(),
    ]);
    ACTIVE.store(variant.index(), Ordering::Relaxed);
}

/// Cycle to the next theme.  Returns the newly active variant label.
pub fn cycle_theme() -> &'static str {
    let prev = ACTIVE.load(Ordering::Relaxed);
    let next = (prev + 1) % THEME_COUNT;
    ACTIVE.store(next, Ordering::Relaxed);
    ThemeVariant::from_index(next).label()
}

/// Get the current theme variant.
#[allow(dead_code)]
pub fn current_variant() -> ThemeVariant {
    ThemeVariant::from_index(ACTIVE.load(Ordering::Relaxed))
}

/// Access the active palette.  Panics if [`init`] was never called.
#[inline]
fn colors() -> &'static ThemeColors {
    let palettes = PALETTES.get().expect("style::init() must be called before drawing");
    let idx = ACTIVE.load(Ordering::Relaxed);
    &palettes[idx % THEME_COUNT]
}

// ── color accessors ──────────────────────────────────────────────

#[inline] pub fn bg() -> Color { colors().bg }
#[inline] pub fn fg() -> Color { colors().fg }
#[inline] pub fn fg_dim() -> Color { colors().fg_dim }
#[inline] pub fn fg_bright() -> Color { colors().fg_bright }
#[inline] pub fn rule() -> Color { colors().rule }
#[inline] pub fn info() -> Color { colors().info }
#[inline] pub fn good() -> Color { colors().good }
#[inline] pub fn warn() -> Color { colors().warn }
#[inline] pub fn bad() -> Color { colors().bad }
#[inline] pub fn bar_empty() -> Color { colors().bar_empty }
#[inline] pub fn card_bg() -> Color { colors().card_bg }
#[inline] pub fn selection() -> Color { colors().selection }

// ── semantic style accessors ─────────────────────────────────────

/// Dimmed label text (e.g. "GPU", "VRAM", "peak").
#[inline] pub fn label() -> Style { Style::new().fg(fg_dim()) }

/// Bold label for section headers and metric card keys.
#[inline] pub fn label_bold() -> Style { Style::new().fg(fg_dim()).add_modifier(Modifier::BOLD) }

/// Standard data value.
#[inline] pub fn value() -> Style { Style::new().fg(fg()) }

/// Bright text for user input and prominent values.
#[inline] pub fn value_bright() -> Style { Style::new().fg(fg_bright()) }

/// Section heading: bright, bold.
#[inline] pub fn heading() -> Style { Style::new().fg(fg_bright()).add_modifier(Modifier::BOLD) }

/// Rule/divider line.
#[inline] pub fn rule_line() -> Style { Style::new().fg(rule()) }

/// Left-side section indicator.
#[inline] pub fn indicator() -> Style { Style::new().fg(info()) }

/// Neutral accent (mode names, active links).
#[inline] pub fn accent() -> Style { Style::new().fg(info()) }

/// Bold accent for titles and primary emphasis.
#[inline] pub fn accent_bold() -> Style { Style::new().fg(info()).add_modifier(Modifier::BOLD) }

/// Metric card base style (background only — compose with `.fg()`).
#[inline] pub fn card() -> Style { Style::new().bg(card_bg()) }

/// Block cursor for input fields.
#[inline] pub fn cursor() -> Style { Style::new().fg(bg()).bg(fg_dim()) }

/// Error text: red + bold.
#[inline] pub fn error_bold() -> Style { Style::new().fg(bad()).add_modifier(Modifier::BOLD) }

/// Full-screen base (applied to the root block).
#[inline] pub fn screen() -> Style { Style::new().bg(bg()).fg(fg()) }

/// Foreground-only semantic color.
#[inline] pub fn semantic(color: Color) -> Style { Style::new().fg(color) }

/// Foreground semantic color with bold emphasis.
#[inline] pub fn semantic_bold(color: Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

/// Inverse badge: dark text over semantic background.
#[inline] pub fn badge(color: Color) -> Style {
    Style::new().fg(bg()).bg(color).add_modifier(Modifier::BOLD)
}

/// Selected row style for tables/lists.
#[inline] pub fn selected_row() -> Style { Style::new().fg(fg_bright()).bg(selection()) }

/// Inert spacer — carries no visual weight, just inherits surroundings.
#[inline] pub fn spacer() -> Style { Style::new() }

// ── panel / focus tokens ──────────────────────────────────────────

/// Panel title when focused (bright accent, bold).
#[inline] pub fn panel_title_focused() -> Style {
    Style::new().fg(info()).add_modifier(Modifier::BOLD)
}

/// Panel title when unfocused (dim, bold).
#[inline] pub fn panel_title_unfocused() -> Style {
    Style::new().fg(fg_dim()).add_modifier(Modifier::BOLD)
}

/// Editor mode badge when the editor pane is focused: inverse badge
/// using the mode's semantic color.
#[inline] pub fn editor_badge_focused(mode: EditorMode) -> Style {
    let (_, color) = editor_mode_tag(mode);
    badge(color)
}

/// Editor mode badge when the editor pane is unfocused: dim text, no
/// background — the badge fades to a quiet label.
#[inline] pub fn editor_badge_unfocused() -> Style {
    Style::new().fg(fg_dim()).add_modifier(Modifier::BOLD)
}

/// Panel title style dispatched by focus state.
#[inline] pub fn panel_title(focused: bool) -> Style {
    if focused { panel_title_focused() } else { panel_title_unfocused() }
}

/// Files tree title style dispatched by focus state.
#[inline] pub fn files_tree_title(focus: FilesFocus) -> Style {
    panel_title(matches!(focus, FilesFocus::Tree))
}

/// Files editor mode badge dispatched by focus state.
#[inline] pub fn files_editor_badge(mode: EditorMode, focus: FilesFocus) -> Style {
    if matches!(focus, FilesFocus::Editor) {
        editor_badge_focused(mode)
    } else {
        editor_badge_unfocused()
    }
}

// ── semantic helpers ───────────────────────────────────────────────

/// Daemon status: returns (label, color) for the running state.
pub fn status_token(running: bool, watchdog: bool) -> (&'static str, Color) {
    if !running {
        ("OFFLINE", bad())
    } else if watchdog {
        ("WATCHDOG", warn())
    } else {
        ("ONLINE", good())
    }
}

/// Pool health: returns (label, color).
pub fn pool_token(crashed: bool, needs_defrag: bool) -> (&'static str, Color) {
    if crashed {
        ("FAIL", bad())
    } else if needs_defrag {
        ("DEFRAG", warn())
    } else {
        ("OK", good())
    }
}

/// Pool health with verbose labels: returns (label, color).
pub fn pool_health_token(crashed: bool, needs_defrag: bool) -> (&'static str, Color) {
    if crashed {
        ("FAIL", bad())
    } else if needs_defrag {
        ("ADVISORY", warn())
    } else {
        ("HEALTHY", good())
    }
}

/// VRAM color: red only at true exhaustion, amber only at pressure.
pub fn vram_color(percent: f32) -> Color {
    if percent >= 95.0 {
        bad()
    } else if percent >= 90.0 {
        warn()
    } else {
        info()
    }
}

/// Latency color: red at 100ms+, amber at 50ms+, good below.
pub fn latency_color(ms: f32) -> Color {
    if ms >= 100.0 {
        bad()
    } else if ms >= 50.0 {
        warn()
    } else {
        good()
    }
}

/// Log entry styles: returns (tag_style, message_style) for a category.
pub fn log_styles(cat: LogCategory) -> (Style, Style) {
    match cat {
        LogCategory::Sys => (label(), value()),
        LogCategory::Jit => (semantic(info()), value()),
        LogCategory::Run => (semantic(good()), value()),
        LogCategory::App => (semantic(warn()), value()),
        LogCategory::Err => (error_bold(), semantic(bad())),
    }
}

/// Editor mode badge: returns (label, color).
pub fn editor_mode_token(mode: EditorMode) -> (&'static str, Color) {
    match mode {
        EditorMode::Normal => ("NORMAL", info()),
        EditorMode::Insert => ("INSERT", good()),
        EditorMode::Visual => ("VISUAL", warn()),
        EditorMode::Command => ("COMMAND", warn()),
    }
}

/// Short editor mode tag: returns (tag, color).
pub fn editor_mode_tag(mode: EditorMode) -> (&'static str, Color) {
    match mode {
        EditorMode::Normal => ("NOR", info()),
        EditorMode::Insert => ("INS", good()),
        EditorMode::Visual => ("VIS", warn()),
        EditorMode::Command => ("CMD", warn()),
    }
}

/// Pipeline stage color based on stage name.
pub fn pipeline_stage_color(name: &str) -> Color {
    match name {
        "jit" | "compile" => warn(),
        "gpu" | "exec" | "run" => good(),
        _ => info(),
    }
}
