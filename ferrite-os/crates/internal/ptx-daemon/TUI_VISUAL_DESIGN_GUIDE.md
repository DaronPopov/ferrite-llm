# Ferrite TUI Visual Design Guide

Status: Active  
Scope: `crates/internal/ptx-daemon/src/tui/**`  
Non-goal: No operational behavior or keybinding changes

## 1. Purpose

This guide defines a cohesive visual system for the Ferrite daemon TUI:

- high readability in common terminal environments
- consistent semantic status language (PASS/WARN/FAIL/info/debug)
- clear component hierarchy across shell/files/scheduler views
- maintainable token-first styling rules for contributors

## 2. No-Break Invariants

1. No runtime/scheduler/input behavior changes.
2. Existing status vocabulary stays intact (`OK`, `FAIL`, `TIMEOUT`, `ONLINE`, `WATCHDOG`, etc.).
3. Visual semantics must remain understandable in monochrome fallback (color is never the only signal).

## 3. Baseline Inventory (Current State)

### 3.1 Token Source of Truth

- Primary token module: `crates/internal/ptx-daemon/src/tui/style.rs`
- Theme variants currently present:
1. `Default`
2. `HighContrast`
3. `Sandstone`

### 3.2 Rendering Usage by Surface

- Shell header + status: `crates/internal/ptx-daemon/src/tui/layout/shell/header.rs`
- Shell logs panel: `crates/internal/ptx-daemon/src/tui/layout/shell/logs.rs`
- System/detail sections: `crates/internal/ptx-daemon/src/tui/layout/shell/sections.rs`
- Files tree/editor/output: `crates/internal/ptx-daemon/src/tui/layout/files.rs`
- Scheduler panel (queue/tenants/policy): `crates/internal/ptx-daemon/src/tui/layout/scheduler.rs`
- Clean dashboard mode: `crates/internal/ptx-daemon/src/tui/layout/clean.rs`

### 3.3 Readability Debt List

1. Critical `bad` contrast is borderline/low in some themes.
Default `bad` on `bg` is ~4.32:1; Sandstone `bad` on `bg` is ~3.80:1.
2. `fg_dim` is below strong body readability thresholds in Default (~3.99:1), so it should stay metadata-only.
3. ~~Resolved~~ Inline `Style::default()` composition has been eliminated from all layout files. All visual styling now routes through `style.rs` semantic tokens (`spacer()`, `panel_title()`, `files_tree_title()`, `files_editor_badge()`, etc.).
4. Some status values rely heavily on color; text tags exist, but icon/tag pairing should be enforced as a rule.

## 4. Color and Contrast Specification

## 4.1 Canonical Palette (Design Tokens)

### Default (Balanced)

| Token | RGB | Hex |
|---|---|---|
| `bg` | (9,14,20) | `#090E14` |
| `fg` | (200,208,218) | `#C8D0DA` |
| `fg_dim` | (112,126,142) | `#707E8E` |
| `fg_bright` | (230,236,244) | `#E6ECF4` |
| `rule` | (34,66,88) | `#224258` |
| `info` | (78,154,186) | `#4E9ABA` |
| `good` | (90,170,120) | `#5AAA78` |
| `warn` | (200,168,78) | `#C8A84E` |
| `bad` | (214,98,98) | `#D66262` |
| `bar_empty` | (30,40,52) | `#1E2834` |
| `card_bg` | (16,24,35) | `#101823` |
| `selection` | (42,68,108) | `#2A446C` |

### High-Contrast (Accessibility)

| Token | RGB | Hex |
|---|---|---|
| `bg` | (9,14,20) | `#090E14` |
| `fg` | (230,236,244) | `#E6ECF4` |
| `fg_dim` | (150,165,180) | `#96A5B4` |
| `fg_bright` | (250,252,255) | `#FAFCFF` |
| `rule` | (60,100,130) | `#3C6482` |
| `info` | (110,185,220) | `#6EB9DC` |
| `good` | (120,210,150) | `#78D296` |
| `warn` | (230,200,100) | `#E6C864` |
| `bad` | (230,110,110) | `#E66E6E` |
| `bar_empty` | (40,52,65) | `#283441` |
| `card_bg` | (22,32,46) | `#16202E` |
| `selection` | (55,85,135) | `#375587` |

### Sandstone (Optional Alternative)

| Token | RGB | Hex |
|---|---|---|
| `bg` | (22,18,14) | `#16120E` |
| `fg` | (210,198,180) | `#D2C6B4` |
| `fg_dim` | (138,126,108) | `#8A7E6C` |
| `fg_bright` | (238,228,210) | `#EEE4D2` |
| `rule` | (62,52,40) | `#3E3428` |
| `info` | (196,170,118) | `#C4AA76` |
| `good` | (138,164,108) | `#8AA46C` |
| `warn` | (210,148,72) | `#D29448` |
| `bad` | (204,98,82) | `#CC6252` |
| `bar_empty` | (38,32,26) | `#26201A` |
| `card_bg` | (30,26,20) | `#1E1A14` |
| `selection` | (72,60,42) | `#483C2A` |

## 4.2 Contrast Thresholds (Policy)

Use these thresholds for all themes:

1. Body text (`value`, `fg`) on background: target `>= 7:1`, minimum `>= 4.5:1`.
2. Metadata (`label`, `fg_dim`): minimum `>= 3:1`, never used for critical values.
3. Status color text on background (`good/warn/bad/info`): minimum `>= 4.5:1` when used as standalone text.
4. Inverse badges (`bg` text over status color background): minimum `>= 4.5:1`.
5. Rules/dividers can be lower contrast (`>= 1.5:1`) because they are structural, not textual content.

## 4.3 Semantic Status Mapping

| Semantic | Color Token | Required text cue |
|---|---|---|
| PASS / OK / ONLINE | `good` | `OK`/`ONLINE` |
| WARN / WATCHDOG / ADVISORY | `warn` | explicit status word |
| FAIL / ERROR / TIMEOUT | `bad` | `FAIL`/`ERROR`/`TIMEOUT` |
| INFO / ACTIVE | `info` | label text and context |
| DEBUG / secondary system detail | `fg_dim` + label text | clear key name |

## 5. Typography and Emphasis Rules

## 5.1 Hierarchy Levels

1. Title: `heading()` or `accent_bold()`
2. Section label/key: `label()` or `label_bold()`
3. Body/value text: `value()`
4. High-attention live value/input: `value_bright()`
5. Meta/hints/rules: `label()` + `rule_line()`

## 5.2 Emphasis Usage

1. `BOLD`: only for status chips, active tab/title, and critical state words.
2. `DIM` equivalent (`fg_dim`): only for non-critical metadata and helper hints.
3. Reverse/inverse (`fg(bg).bg(status)`): only for compact badges/tags.
4. Avoid stacked emphasis (bold + bright + saturated color) unless alerting.

## 6. Component Pattern Guide

## 6.1 Header and Global Status

- Product identity: `accent_bold()` + `label()`
- Runtime status word: semantic token + `BOLD`
- Keep one primary metric group and one secondary context group per row

Example:

```text
ferrite daemon  ♥ ONLINE  gpu 42.1%  vram 68.2%  pool OK  up 00:05:31
```

## 6.2 Panels (Shell / Scheduler / Files)

1. Panel title row always starts with a semantic tag or mode badge.
2. Horizontal separators use `rule_line()` only.
3. Active tab/view is indicated by both:
- stronger style (accent/bold)
- explicit brackets/marker text (`[ Queue ]`)

## 6.3 Files Tree and Editor

1. Focused pane: color + explicit mode/focus tag (`NOR`, `INS`, etc.).
2. Cursor and selection:
- cursor has dedicated `cursor()` style
- selection uses `selection` background, never color-only signal
3. Footer help remains label-heavy, with key names in `accent()`.

## 6.4 Logs and Output Streams

1. Timestamp: `fg_dim`
2. Category tag: semantic color by category
3. Message body: `value()` unless error/severity overrides
4. Empty-state hints: `label()` only

## 6.5 PASS/WARN/FAIL Grammar Examples

```text
POOL HEALTHY     (PASS)  -> text: HEALTHY + color: good
POOL ADVISORY    (WARN)  -> text: ADVISORY + color: warn
POOL FAIL        (FAIL)  -> text: FAIL + color: bad
```

```text
[ OK ] run complete
[ WARN ] watchdog tripped
[ FAIL ] kernel launch error
```

## 7. Theme Policy and Accessibility

## 7.1 Supported Curated Themes

1. `default`: balanced daily driver
2. `high-contrast`: accessibility-first, stronger separation

`sandstone` remains optional stylistic theme and must still meet contrast policy before being considered default-safe for critical status text.

## 7.2 Accessibility Requirements

1. Never use color as the only status channel.
2. Pair every semantic color with explicit status words (`OK`, `WARN`, `FAIL`, etc.).
3. Keep selected/focused state visible even in monochrome terminals (prefix markers, brackets, tags).
4. Validate on common backgrounds:
- dark ANSI terminals
- truecolor dark terminals
- low-brightness laptop displays

## 8. Token/Rendering Separation Rules

1. `style.rs` is the only place for theme palette values.
2. New semantic intent should first be added to `style.rs` as accessor/helper.
3. Layout modules should consume semantic helpers (`style::label()`, `style::status_token()`) rather than raw ad-hoc RGB definitions.
4. Inline `Style::default()` composition is allowed only in `widgets.rs` (parameterized widget rendering) and `style.rs` itself (token definitions). Layout files must use semantic token helpers exclusively.

## 9. Adoption Protocol

## 9.1 For New UI Work

1. Define semantic intent first (what does this text mean?).
2. Reuse existing token/helper; add new helper if needed.
3. Check contrast against thresholds in section 4.2.
4. Ensure status can be understood without color.

## 9.2 Review Gates

1. No new raw RGB literals in layout files.
2. No critical status rendered with `fg_dim`.
3. Active/focus states include non-color marker.
4. Status vocabulary unchanged unless explicitly approved.

## 10. Validation Commands

```bash
cd /home/daron/fdfd/ferriterc/ferrite-os
cargo check -p ferrite-daemon

cd /home/daron/fdfd/ferriterc/ferrite-os
cargo run -p ferrite-daemon -- serve --config crates/internal/ptx-daemon/dev-config.toml
```

## 11. Visual Reference Captures

Capture these when style changes are made:

1. Shell view with healthy system.
2. Shell view with warning/failure status.
3. Files view (tree focus, editor focus, run output visible).
4. Scheduler view (Queue, Tenants, Policy tabs).
5. High-contrast theme equivalents for 1-4.

Store captures in PR artifacts or linked issue comments to keep regressions reviewable.
