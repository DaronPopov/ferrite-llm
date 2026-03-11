use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::tui::state::{Panel, TuiState};
use crate::tui::style;

pub(in crate::tui::layout) fn draw_prompt(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 {
        return;
    }

    let before = &state.input[..state.cursor];
    let after = &state.input[state.cursor..];

    let (cursor_ch, rest) = if after.is_empty() {
        (" ", "")
    } else {
        let n = after.chars().next().map_or(1, |c| c.len_utf8());
        (&after[..n], &after[n..])
    };

    let focus_hint = match state.focus {
        Panel::Log => "log",
        Panel::Processes => "summary",
    };

    let detail_hint = if state.detail_mode {
        "detail:on"
    } else {
        "detail:off"
    };
    let density_hint = state.ui_density.label();

    // Hints adapt to input state: empty input shows navigation hints,
    // active input shows editing hints.
    let hints = if state.input.is_empty() {
        let scroll_hint = if state.sysmon_section_max_scroll > 0 {
            format!(
                " [pg↑↓/wheel:sections {}/{}]",
                state.sysmon_section_scroll + 1,
                state.sysmon_section_max_scroll + 1
            )
        } else {
            String::new()
        };
        format!(
            "  [help] [tab:{}] [{}] [dens:{}]{} [ctrl+o:files]",
            focus_hint, detail_hint, density_hint, scroll_hint
        )
    } else {
        format!(
            "  [enter:run] [esc:clear] [↑↓:history]",
        )
    };

    let line = Line::from(vec![
        Span::styled(">", style::accent_bold()),
        Span::styled(" ", style::label()),
        Span::styled(before, style::value_bright()),
        Span::styled(cursor_ch, style::cursor()),
        Span::styled(rest, style::value_bright()),
        Span::styled(hints, style::label()),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}
