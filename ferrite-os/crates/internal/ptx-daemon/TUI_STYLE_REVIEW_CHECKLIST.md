# TUI Style Review Checklist

Use this checklist for any PR that touches `crates/internal/ptx-daemon/src/tui/**`.

## Required Checks

1. Token usage:
- no new raw RGB literals in layout/render modules
- palette edits are confined to `src/tui/style.rs`

2. Readability:
- body text is readable against background (`>= 4.5:1`, target `>= 7:1`)
- metadata (`fg_dim`) is not used for critical status/value text

3. Semantic consistency:
- PASS/WARN/FAIL meanings map to `good`/`warn`/`bad`
- existing status vocabulary is preserved (`OK`, `FAIL`, `TIMEOUT`, `ONLINE`, etc.)

4. Non-color signaling:
- status always has explicit text cue
- active/focus/selected state has a shape/text marker (tag, bracket, prefix), not only color

5. Emphasis discipline:
- bold is limited to headings, active tabs, badges, critical labels
- no excessive combined emphasis (bold + bright + saturated) without clear intent

6. Component coherence:
- headers, panel titles, log rows, file editor, and scheduler tabs follow the same token language
- divider lines and spacing remain visually calm and scannable

## Validation

```bash
cd /home/daron/fdfd/ferriterc/ferrite-os
cargo check -p ferrite-daemon
```

