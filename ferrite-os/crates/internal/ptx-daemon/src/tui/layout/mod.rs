mod clean;
mod files;
mod run_output;
pub mod scheduler;
mod shell;

use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::state::{TuiState, UiMode};
use super::style;
use super::widgets::LiveWave;

pub fn draw(frame: &mut Frame, state: &mut TuiState) {
    if matches!(state.ui_mode, UiMode::RunOutput) {
        run_output::draw_run_output(frame, state);
    } else if matches!(state.ui_mode, UiMode::Files) {
        files::draw_files(frame, state);
    } else if matches!(state.ui_mode, UiMode::Scheduler) {
        scheduler::render_scheduler_panel(frame, frame.area(), state);
    } else if !state.sysmon_enabled {
        clean::draw_clean(frame, state);
    } else {
        shell::draw_sysmon(frame, state);
    }
    draw_system_spirit(frame, state);
}

fn draw_section_header(frame: &mut Frame, area: Rect, title: &str) {
    if area.height == 0 {
        return;
    }
    let label = format!(" {} ", title);
    let tail_len = area.width.saturating_sub(label.len() as u16 + 2) as usize;
    let line = Line::from(vec![
        Span::styled("▌", style::indicator()),
        Span::styled(" ", style::spacer()),
        Span::styled(label, style::heading()),
        Span::styled("─".repeat(tail_len), style::rule_line()),
    ]);
    frame.render_widget(
        Paragraph::new(line),
        Rect::new(area.x, area.y, area.width, 1),
    );
}

fn draw_thin_rule(frame: &mut Frame, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let line = Line::from(Span::styled(
        "─".repeat(area.width as usize),
        style::rule_line(),
    ));
    frame.render_widget(Paragraph::new(line), Rect::new(area.x, area.y, area.width, 1));
}

fn section_body(area: Rect) -> Rect {
    Rect::new(
        area.x,
        area.y.saturating_add(1),
        area.width,
        area.height.saturating_sub(1),
    )
}

/// Standard activity wave — rendered on every screen mode.
/// Single row showing live VRAM pressure as an animated wave with
/// GPU utilization, memory stats, and pool health alongside.
fn draw_activity_wave(frame: &mut Frame, area: Rect, state: &TuiState) {
    if area.height == 0 || area.width < 20 {
        return;
    }

    // Right-side stats label:  "4.8/7.7GB  pool OK"
    let pool_tag = style::pool_token(pool_crashed(state), state.pool_needs_defrag);
    let stats = format!(
        " {}/{} {}",
        fmt_bytes(state.vram_used),
        fmt_bytes(state.vram_total),
        pool_tag.0,
    );
    let stats_len = stats.len() as u16;

    // Wave fills remaining space
    let wave_w = area.width.saturating_sub(stats_len);
    let wave_area = Rect::new(area.x, area.y, wave_w, 1);

    frame.render_widget(
        LiveWave {
            label: "vram",
            data: &state.vram_history,
            max_val: 100.0,
            phase: state.wave_phase_vram,
            intensity: state.display_vram_pct / 100.0,
            suffix: format!("{:>5.1}%", state.display_vram_pct),
            color: style::vram_color(state.display_vram_pct),
            label_color: style::fg_dim(),
            text_color: style::fg(),
        },
        wave_area,
    );

    // Stats after the wave
    let stats_x = area.x + wave_w;
    let stats_line = Line::from(vec![
        Span::styled(
            format!(
                " {}/{}  ",
                fmt_bytes(state.vram_used),
                fmt_bytes(state.vram_total),
            ),
            style::label(),
        ),
        Span::styled(pool_tag.0, style::semantic(pool_tag.1)),
    ]);
    frame.render_widget(
        Paragraph::new(stats_line),
        Rect::new(stats_x, area.y, area.width.saturating_sub(wave_w), 1),
    );
}

fn draw_metric_card(frame: &mut Frame, area: Rect, label: &str, value: &str, value_color: Color) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let line = Line::from(vec![
        Span::styled("▎ ", style::indicator()),
        Span::styled(format!("{:<8}", label), style::label_bold()),
        Span::styled(
            truncate_for_log(value, area.width.saturating_sub(11) as usize),
            style::semantic(value_color),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(style::card()),
        area,
    );
}

fn draw_system_spirit(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();
    if area.width < 22 || area.height == 0 {
        return;
    }

    // Top-right waveform spirit: compact multi-harmonic braille wave.
    let gpu = (state.display_gpu_util / 100.0).clamp(0.0, 1.0);
    let vram = (state.display_vram_pct / 100.0).clamp(0.0, 1.0);
    let t = state.tick_count as f32;

    let (color, amp, freq, speed, baseline, harmonic_mix) = if !state.running {
        (style::bad(), 0.00, 0.80, 0.00, 0.10, 0.00)
    } else if state.watchdog {
        (style::bad(), 0.95, 1.90, 0.80, 0.05, 0.10)
    } else if state.pool_needs_defrag {
        (style::warn(), 0.56, 1.35, 0.46, 0.22, 0.52)
    } else {
        (
            style::good(),
            0.28 + gpu * 0.48,
            0.90 + vram * 0.90,
            0.20 + gpu * 0.42,
            0.26,
            0.68,
        )
    };
    let wave = build_spirit_wave(t, amp, freq, speed, baseline, harmonic_mix, state.watchdog);
    let aura = if state.watchdog {
        "!"
    } else if state.pool_needs_defrag {
        "~"
    } else {
        " "
    };

    let spirit_width: u16 = 20;
    let spirit_area = Rect::new(
        area.x + area.width.saturating_sub(spirit_width),
        area.y,
        spirit_width,
        1,
    );
    // Clear first to avoid stale corner artifacts on narrow/resize frames.
    frame.render_widget(
        Paragraph::new(" ".repeat(spirit_area.width as usize)).style(style::screen()),
        spirit_area,
    );

    let rendered = format!("[{}]{}", wave, aura);
    let pad = spirit_area
        .width
        .saturating_sub(rendered.chars().count() as u16) as usize;
    let line = Line::from(vec![
        Span::styled(rendered, style::semantic_bold(color)),
        Span::styled(" ".repeat(pad), style::screen()),
    ]);
    frame.render_widget(Paragraph::new(line), spirit_area);
}

fn build_spirit_wave(
    t: f32,
    amp: f32,
    freq: f32,
    speed: f32,
    baseline: f32,
    harmonic_mix: f32,
    watchdog: bool,
) -> String {
    // Braille: 2 columns x 4 rows per cell.
    const DOTS: [[u8; 2]; 4] = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ];
    const BRAILLE_BASE: u32 = 0x2800;
    let cells = 9usize;
    let samples = cells * 2;
    let mut bits = vec![0u8; cells];

    for i in 0..samples {
        let x = i as f32 / (samples.saturating_sub(1)) as f32;
        let carrier = (x * std::f32::consts::TAU * freq + t * speed).sin();
        let overtone = (x * std::f32::consts::TAU * (freq * 2.1) + t * (speed * 1.7)).sin();
        let ripple = (x * std::f32::consts::TAU * (freq * 4.6) - t * (speed * 0.6)).sin();
        let raw = if watchdog {
            if carrier > 0.0 { 1.0 } else { 0.02 }
        } else {
            let layered = carrier + overtone * harmonic_mix * 0.42 + ripple * harmonic_mix * 0.18;
            (baseline + layered * amp).clamp(0.0, 1.0)
        };

        let row = 3 - (raw * 3.0).round().clamp(0.0, 3.0) as usize;
        let cell = i / 2;
        let col = i % 2;
        bits[cell] |= DOTS[row][col];
    }

    let mut out = String::with_capacity(cells);
    for b in bits {
        let ch = char::from_u32(BRAILLE_BASE + b as u32).unwrap_or('⣀');
        out.push(ch);
    }
    out
}

fn truncate_for_log(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    if max_chars <= 1 {
        return "…".to_string();
    }
    let keep = max_chars - 1;
    let mut out: String = chars[..keep].iter().collect();
    out.push('…');
    out
}

fn pool_crashed(state: &TuiState) -> bool {
    !state.running || state.watchdog
}

fn fmt_uptime(s: u64) -> String {
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    if h > 0 {
        format!("{}h{:02}m", h, m)
    } else if m > 0 {
        format!("{}m{:02}s", m, sec)
    } else {
        format!("{}s", sec)
    }
}

fn fmt_elapsed(s: u64) -> String {
    if s == 0 {
        "now".into()
    } else if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else {
        format!("{}h", s / 3600)
    }
}

fn fmt_latency(ms: f32) -> String {
    if ms <= 0.0 {
        "--".into()
    } else if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1}GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.0}MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.0}K", b as f64 / 1024.0)
    } else {
        format!("{}B", b)
    }
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
