use std::collections::VecDeque;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

const BRAILLE_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct Gauge<'a> {
    pub label: &'a str,
    pub used: u64,
    pub total: u64,
    pub percent: f32,
    pub bar_color: Color,
    pub empty_color: Color,
    pub label_color: Color,
    pub text_color: Color,
}

impl<'a> Widget for Gauge<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 30 || area.height < 1 {
            return;
        }

        // Format stats: "4.8 / 7.7 GB  62%"
        let (used_s, total_s, unit) = format_bytes_pair(self.used, self.total);
        let pct = format!("{:.0}%", self.percent);
        let stats = format!("  {} / {} {}  {}", used_s, total_s, unit, pct);
        let stats_len = stats.len() as u16;

        // Label: fixed-width column for numeric alignment across widgets.
        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        // Bar fills the space between label and stats
        let bar_w = area.width.saturating_sub(label_w + stats_len + 1).max(8) as usize;
        let bar_x = area.x + label_w;

        // Smooth sub-character fill
        let fill_f = (self.percent / 100.0) * bar_w as f32;
        let full = fill_f as usize;
        let frac = ((fill_f - full as f32) * 8.0) as usize;

        // Fractional block characters for smooth edge
        const EIGHTHS: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

        for i in 0..bar_w {
            let (ch, fg, bg) = if i < full {
                ('▆', self.bar_color, Color::Reset)
            } else if i == full && frac > 0 {
                // Partial fill: fractional char in bar_color, rest is empty
                (EIGHTHS[frac], self.bar_color, Color::Reset)
            } else {
                ('·', self.empty_color, Color::Reset)
            };
            buf.set_string(
                bar_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(fg).bg(bg),
            );
        }

        // Stats after bar
        let stats_x = bar_x + bar_w as u16;
        buf.set_string(
            stats_x,
            area.y,
            &stats,
            Style::default().fg(self.text_color),
        );
    }
}

pub struct Spinner {
    pub tick: u64,
}

impl Spinner {
    pub fn frame(&self) -> &'static str {
        BRAILLE_FRAMES[(self.tick as usize) % BRAILLE_FRAMES.len()]
    }
}

pub struct Heartbeat;

impl Heartbeat {
    pub fn symbol(&self) -> &'static str {
        "●"
    }
}

// ── sparkline blocks ─────────────────────────────────────────────
const SPARK_BLOCKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

#[allow(dead_code)]
pub struct MiniSparkline<'a> {
    pub label: &'a str,
    pub data: &'a VecDeque<f32>,
    pub max_val: f32,
    pub suffix: String,
    pub bar_color: Color,
    pub label_color: Color,
    pub text_color: Color,
}

impl<'a> Widget for MiniSparkline<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 1 {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        // Suffix right-aligned
        let suffix_len = self.suffix.len() as u16;
        let suffix_x = area.x + area.width - suffix_len;
        buf.set_string(
            suffix_x,
            area.y,
            &self.suffix,
            Style::default().fg(self.text_color),
        );

        // Sparkline bars fill space between label and suffix
        let bar_w = area.width.saturating_sub(label_w + suffix_len + 1) as usize;
        let bar_x = area.x + label_w;

        let max = if self.max_val > 0.0 {
            self.max_val
        } else {
            1.0
        };
        let data_len = self.data.len();
        let start = data_len.saturating_sub(bar_w);

        for i in 0..bar_w {
            let idx = start + i;
            let ch = if idx < data_len {
                let val = self.data[idx];
                let norm = (val / max).clamp(0.0, 1.0);
                let level = (norm * 7.0) as usize;
                SPARK_BLOCKS[level.min(7)]
            } else {
                ' '
            };
            buf.set_string(
                bar_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(self.bar_color),
            );
        }
    }
}

// ── live wave ───────────────────────────────────────────────────
// Animated single-row wave whose motion speed and amplitude scale
// with load intensity.  At idle the wave is a calm, nearly flat
// ripple.  Under stress it moves faster with higher peaks —
// giving an immediate visceral read on system pressure without
// needing to decode numbers.
//
// The wave is overlaid on the historical data so you still see the
// recent trend underneath the live modulation.

pub struct LiveWave<'a> {
    pub label: &'a str,
    pub data: &'a VecDeque<f32>,
    pub max_val: f32,
    /// Wave phase — advanced each tick, speed proportional to intensity.
    pub phase: f64,
    /// 0.0–1.0 normalized load fraction (drives amplitude + visual energy).
    pub intensity: f32,
    pub suffix: String,
    pub color: Color,
    pub label_color: Color,
    pub text_color: Color,
}

impl<'a> Widget for LiveWave<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 1 {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        // Suffix right-aligned (percentage readout)
        let suffix_len = self.suffix.len() as u16;
        let suffix_x = area.x + area.width - suffix_len;
        buf.set_string(
            suffix_x,
            area.y,
            &self.suffix,
            Style::default().fg(self.text_color),
        );

        let bar_w = area.width.saturating_sub(label_w + suffix_len + 1) as usize;
        let bar_x = area.x + label_w;
        if bar_w == 0 {
            return;
        }

        let max = if self.max_val > 0.0 {
            self.max_val
        } else {
            1.0
        };
        let data_len = self.data.len();
        let start = data_len.saturating_sub(bar_w);

        // Wave amplitude scales with intensity:
        //   idle  (0.0) → amp ≈ 0.04  (barely visible ripple)
        //   full  (1.0) → amp ≈ 0.28  (strong undulation)
        let amp = 0.04 + self.intensity * 0.24;

        let ph = self.phase;

        for i in 0..bar_w {
            let idx = start + i;

            // Historical base level
            let base = if idx < data_len {
                (self.data[idx] / max).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Two-harmonic wave for organic feel:
            //   primary   — long wavelength, full amplitude
            //   secondary — short wavelength, half amplitude, faster drift
            let x = i as f64;
            let w1 = (x * 0.30 + ph).sin() as f32;
            let w2 = (x * 0.73 + ph * 1.4).sin() as f32 * 0.5;
            let wave = (w1 + w2) * amp;

            let height = (base + wave).clamp(0.0, 1.0);
            let level = (height * 7.0) as usize;
            let ch = SPARK_BLOCKS[level.min(7)];

            // Brightness modulation: crests slightly brighter than troughs
            let bright = 0.65 + 0.35 * height;
            let fg = dim_color(self.color, bright);

            buf.set_string(
                bar_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(fg),
            );
        }
    }
}

// ── tensor heatmap ──────────────────────────────────────────────

pub const HEAT_GRADIENT: &[(u8, u8, u8)] = &[
    (30, 35, 60),
    (55, 60, 120),
    (80, 120, 180),
    (90, 160, 150),
    (90, 185, 120),
    (180, 175, 90),
    (200, 100, 85),
];

fn lerp_color(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> Color {
    let r = (a.0 as f32 + (b.0 as f32 - a.0 as f32) * t) as u8;
    let g = (a.1 as f32 + (b.1 as f32 - a.1 as f32) * t) as u8;
    let b_val = (a.2 as f32 + (b.2 as f32 - a.2 as f32) * t) as u8;
    Color::Rgb(r, g, b_val)
}

pub fn heat_color(norm: f32) -> Color {
    let t = norm.clamp(0.0, 1.0) * (HEAT_GRADIENT.len() - 1) as f32;
    let idx = (t as usize).min(HEAT_GRADIENT.len() - 2);
    let frac = t - idx as f32;
    lerp_color(HEAT_GRADIENT[idx], HEAT_GRADIENT[idx + 1], frac)
}

pub struct TensorHeatmap<'a> {
    pub label: &'a str,
    pub samples: &'a [f32],
    pub min: f32,
    pub max: f32,
    pub label_color: Color,
}

impl<'a> Widget for TensorHeatmap<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 1 {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        let bar_x = area.x + label_w;
        let bar_w = (area.width - label_w) as usize;
        let range = self.max - self.min;

        for i in 0..bar_w.min(self.samples.len()) {
            let norm = if range > 0.0 {
                ((self.samples[i] - self.min) / range).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let color = heat_color(norm);
            buf.set_string(bar_x + i as u16, area.y, "▄", Style::default().fg(color));
        }
    }
}

/// Render histogram bins as sparkline bars with gradient coloring.
pub struct HistogramSparkline<'a> {
    pub label: &'a str,
    pub bins: &'a [u32],
    pub label_color: Color,
}

impl<'a> Widget for HistogramSparkline<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 1 || self.bins.is_empty() {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        let bar_x = area.x + label_w;
        let bar_w = (area.width - label_w) as usize;
        let max_bin = *self.bins.iter().max().unwrap_or(&1).max(&1);

        for i in 0..bar_w.min(self.bins.len()) {
            let norm = self.bins[i] as f32 / max_bin as f32;
            let level = (norm * 7.0) as usize;
            let ch = SPARK_BLOCKS[level.min(7)];
            let grad_t = i as f32 / (self.bins.len().max(1) - 1).max(1) as f32;
            let color = heat_color(grad_t);
            buf.set_string(
                bar_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(color),
            );
        }
    }
}

// ── braille waveform ─────────────────────────────────────────────
// 2×4 dot grid per cell = 2x horizontal resolution over block chars.
// Renders a high-fidelity line plot using Unicode braille U+2800..U+28FF.

const BRAILLE_BASE: u32 = 0x2800;

// Bit positions: [row][col] where row 0=top, col 0=left
const BRAILLE_DOTS: [[u8; 2]; 4] = [
    [0x01, 0x08], // row 0 (top)
    [0x02, 0x10], // row 1
    [0x04, 0x20], // row 2
    [0x40, 0x80], // row 3 (bottom)
];

pub struct BrailleWaveform<'a> {
    pub label: &'a str,
    pub data: &'a [f32],
    pub min_val: f32,
    pub max_val: f32,
    pub color: Color,
    pub label_color: Color,
}

impl<'a> Widget for BrailleWaveform<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 1 {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        let plot_w = (area.width - label_w) as usize;
        let plot_x = area.x + label_w;

        if self.data.is_empty() || plot_w == 0 {
            return;
        }

        let data_points = plot_w * 2; // 2 x-positions per braille cell
        let range = (self.max_val - self.min_val).max(f32::EPSILON);

        // Take the last data_points values
        let start = self.data.len().saturating_sub(data_points);
        let visible = &self.data[start..];
        let offset = data_points - visible.len();

        let mut cells = vec![0u8; plot_w];

        for (di, &val) in visible.iter().enumerate() {
            let x_pos = offset + di;
            let cell_idx = x_pos / 2;
            let col = x_pos % 2;

            let norm = ((val - self.min_val) / range).clamp(0.0, 1.0);
            // Row 0 = top (high value), row 3 = bottom (low value)
            let row = 3 - (norm * 3.0).min(3.0) as usize;

            if cell_idx < plot_w {
                cells[cell_idx] |= BRAILLE_DOTS[row][col];
            }
        }

        for (i, &bits) in cells.iter().enumerate() {
            let ch = char::from_u32(BRAILLE_BASE + bits as u32).unwrap_or('⠀');
            buf.set_string(
                plot_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(self.color),
            );
        }
    }
}

fn dim_color(color: Color, brightness: f32) -> Color {
    match color {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f32 * brightness) as u8,
            (g as f32 * brightness) as u8,
            (b as f32 * brightness) as u8,
        ),
        other => other,
    }
}

// ── fragmentation map ───────────────────────────────────────────
// Spatial view of TLSF pool layout — allocated blocks interleaved
// with free regions, largest_free shown as a contiguous gap.

pub struct FragMap {
    pub label: &'static str,
    pub pool_used: u64,
    pub pool_total: u64,
    pub allocated_blocks: u32,
    pub free_blocks: u32,
    pub largest_free: u64,
    pub alloc_color: Color,
    pub free_color: Color,
    pub label_color: Color,
}

impl Widget for FragMap {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 15 || area.height < 1 {
            return;
        }

        let label_w: u16 = 7;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(area.x, area.y, label, Style::default().fg(self.label_color));

        let map_w = (area.width - label_w) as usize;
        let map_x = area.x + label_w;

        if self.pool_total == 0 || map_w == 0 {
            return;
        }

        let alloc_frac = self.pool_used as f64 / self.pool_total as f64;
        let total_blocks = (self.allocated_blocks + self.free_blocks).max(1) as f64;
        let period = (map_w as f64 / total_blocks).max(0.5);

        // Reserve cells for the largest free block, placed at the end
        let largest_cells =
            ((self.largest_free as f64 / self.pool_total as f64) * map_w as f64) as usize;
        let largest_start = map_w.saturating_sub(largest_cells);

        for i in 0..map_w {
            let (ch, color) = if i >= largest_start {
                // Largest free region
                ('▁', dim_color(self.free_color, 0.75))
            } else {
                // Alternating pattern: period based on block count,
                // alloc fraction determines fill within each period
                let phase = (i as f64 / period) % 1.0;
                if phase < alloc_frac {
                    ('▆', self.alloc_color)
                } else {
                    ('▁', dim_color(self.free_color, 0.75))
                }
            };
            buf.set_string(
                map_x + i as u16,
                area.y,
                &ch.to_string(),
                Style::default().fg(color),
            );
        }
    }
}

// ── pipeline bar ────────────────────────────────────────────────
// Single-row waterfall stage: shows a timing bar at a proportional
// offset within the total pipeline duration.

pub struct PipelineBar {
    pub label: String,
    pub offset_frac: f32,
    pub width_frac: f32,
    pub duration_ms: u128,
    pub bar_color: Color,
    pub bg_color: Color,
    pub label_color: Color,
    pub text_color: Color,
}

impl Widget for PipelineBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 15 || area.height < 1 {
            return;
        }

        let label_w: u16 = 9;
        let label = format!("{:<w$}", self.label, w = label_w as usize);
        buf.set_string(
            area.x,
            area.y,
            &label,
            Style::default().fg(self.label_color),
        );

        let suffix = format!(" {}ms", self.duration_ms);
        let suffix_len = suffix.len() as u16;
        let suffix_x = area.x + area.width - suffix_len;
        buf.set_string(
            suffix_x,
            area.y,
            &suffix,
            Style::default().fg(self.text_color),
        );

        let bar_w = area.width.saturating_sub(label_w + suffix_len) as usize;
        let bar_x = area.x + label_w;

        let start = (self.offset_frac * bar_w as f32) as usize;
        let end = ((self.offset_frac + self.width_frac) * bar_w as f32) as usize;

        for i in 0..bar_w {
            if i >= start && i < end {
                buf.set_string(
                    bar_x + i as u16,
                    area.y,
                    "▆",
                    Style::default().fg(self.bar_color),
                );
            } else {
                buf.set_string(
                    bar_x + i as u16,
                    area.y,
                    "·",
                    Style::default().fg(dim_color(self.bg_color, 0.55)),
                );
            }
        }
    }
}

fn format_bytes_pair(used: u64, total: u64) -> (String, String, &'static str) {
    let max = used.max(total);
    if max >= 1024 * 1024 * 1024 {
        (
            format!("{:.1}", used as f64 / (1024.0 * 1024.0 * 1024.0)),
            format!("{:.1}", total as f64 / (1024.0 * 1024.0 * 1024.0)),
            "GB",
        )
    } else if max >= 1024 * 1024 {
        (
            format!("{:.0}", used as f64 / (1024.0 * 1024.0)),
            format!("{:.0}", total as f64 / (1024.0 * 1024.0)),
            "MB",
        )
    } else if max >= 1024 {
        (
            format!("{:.0}", used as f64 / 1024.0),
            format!("{:.0}", total as f64 / 1024.0),
            "K",
        )
    } else {
        (format!("{}", used), format!("{}", total), "B")
    }
}
