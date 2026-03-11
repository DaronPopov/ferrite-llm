use crate::events::{LogCategory, LogEntry};

use super::{Panel, TuiState, UiMode, SPARKLINE_LEN};

impl TuiState {
    pub fn tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        // Static heartbeat — no continuous animation. Phase unused but kept for compat.
        self.heartbeat_phase = 1.0;

        // Smooth display values toward actuals (damped approach, no jitter)
        const LERP: f32 = 0.15;
        self.display_gpu_util += (self.gpu_util - self.display_gpu_util) * LERP;
        self.display_vram_pct += (self.vram_percent - self.display_vram_pct) * LERP;
        // Snap when close to avoid asymptotic drift
        if (self.gpu_util - self.display_gpu_util).abs() < 0.05 {
            self.display_gpu_util = self.gpu_util;
        }
        if (self.vram_percent - self.display_vram_pct).abs() < 0.05 {
            self.display_vram_pct = self.vram_percent;
        }

        // Advance wave phases — speed proportional to load intensity.
        // Idle:  slow drift (0.06 rad/tick ≈ 1 rad/s)  → calm ripple
        // Full:  fast flow  (0.36 rad/tick ≈ 6 rad/s)  → energetic wave
        let gpu_frac = (self.display_gpu_util / 100.0).clamp(0.0, 1.0) as f64;
        let vram_frac = (self.display_vram_pct / 100.0).clamp(0.0, 1.0) as f64;
        self.wave_phase_gpu += 0.06 + gpu_frac * 0.30;
        self.wave_phase_vram += 0.04 + vram_frac * 0.22;

        // Throttle sparkline updates to ~5 Hz (every 3rd tick at 16 FPS).
        // 48 entries × 3 ticks × 60ms = ~8.6 seconds of visible history.
        let push_sparkline = self.tick_count % 3 == 0;

        if push_sparkline {
            if self.ops_rate_history.len() >= SPARKLINE_LEN {
                self.ops_rate_history.pop_front();
            }
            self.ops_rate_history.push_back(if self.ops_rate_valid {
                self.ops_rate_hw
            } else {
                0.0
            });

            if self.gflops_history.len() >= SPARKLINE_LEN {
                self.gflops_history.pop_front();
            }
            self.gflops_history.push_back(if self.gflops_valid {
                self.gflops_hw
            } else {
                0.0
            });
        }

        // Auto-fade tensor viz after 60s
        if let Some(ref tv) = self.last_tensor {
            if tv.timestamp.elapsed().as_secs() > 60 {
                self.last_tensor = None;
            }
        }

        // Auto-fade pipeline after 30s
        if let Some(ref pl) = self.last_pipeline {
            if pl.timestamp.elapsed().as_secs() > 30 {
                self.last_pipeline = None;
            }
        }
    }

    pub fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            Panel::Processes => Panel::Log,
            Panel::Log => Panel::Processes,
        };
    }

    pub fn toggle_ui_mode(&mut self) {
        self.ui_mode = match self.ui_mode {
            UiMode::Shell => {
                self.reload_file_entries();
                UiMode::Files
            }
            UiMode::Files => UiMode::Scheduler,
            UiMode::Scheduler => {
                self.push_log(LogEntry::new(LogCategory::Sys, "shell mode"));
                UiMode::Shell
            }
            UiMode::RunOutput => {
                // Return to wherever we came from
                self.run_output_return_mode
            }
        };
        self.selection_anchor = None;
        self.selection_head = None;
        self.selecting = false;
    }

    /// Switch to the full-screen run output view, remembering where to return.
    pub fn enter_run_output(&mut self) {
        if !matches!(self.ui_mode, UiMode::RunOutput) {
            self.run_output_return_mode = self.ui_mode;
        }
        self.ui_mode = UiMode::RunOutput;
    }

    /// Leave run output view and return to the previous mode.
    pub fn exit_run_output(&mut self) {
        self.ui_mode = self.run_output_return_mode;
    }

    /// Scroll the run output panel.
    pub fn scroll_run_output(&mut self, delta: i32) {
        let total = self.run_output.len();
        if delta < 0 {
            // scroll up (toward older output)
            self.run_output_scroll = self.run_output_scroll
                .saturating_add((-delta) as usize)
                .min(total.saturating_sub(1));
        } else if delta > 0 {
            // scroll down (toward newer output)
            self.run_output_scroll = self.run_output_scroll.saturating_sub(delta as usize);
        }
    }

    pub fn files_toggle_focus(&mut self) {
        self.files_focus = match self.files_focus {
            super::FilesFocus::Tree => super::FilesFocus::Editor,
            super::FilesFocus::Editor => super::FilesFocus::Tree,
        };
    }

    pub fn scroll_log(&mut self, delta: i32) {
        if delta > 0 {
            self.log_scroll = self.log_scroll.saturating_sub(delta as usize);
        } else if delta < 0 {
            self.log_scroll =
                (self.log_scroll + (-delta) as usize).min(self.log.len().saturating_sub(1));
        }
    }

    pub fn set_sysmon_scroll_bounds(&mut self, max: usize) {
        self.sysmon_section_max_scroll = max;
        if self.sysmon_section_scroll > max {
            self.sysmon_section_scroll = max;
        }
    }

    pub fn scroll_sysmon_sections(&mut self, delta: i32) {
        if self.sysmon_section_max_scroll == 0 || delta == 0 {
            return;
        }
        if delta < 0 {
            self.sysmon_section_scroll = self.sysmon_section_scroll.saturating_sub((-delta) as usize);
        } else {
            self.sysmon_section_scroll =
                (self.sysmon_section_scroll + delta as usize).min(self.sysmon_section_max_scroll);
        }
    }
}
