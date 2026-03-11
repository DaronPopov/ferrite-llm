use std::sync::atomic::Ordering;
use std::time::Instant;

use tachyonfx::{CellFilter, Interpolation, fx};

use crate::events::LogEntry;
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;
use crate::tui::style;

use super::{
    AppInfo, OwnerUsage, PipelineStage, PipelineState, TensorViz, TuiState, MAX_LOG_ENTRIES,
    SPARKLINE_LEN,
};

impl TuiState {
    fn set_named_effect(&mut self, name: &'static str, effect: tachyonfx::Effect) {
        if let Some((_, slot)) = self.effects.iter_mut().find(|(n, _)| *n == name) {
            *slot = effect;
        } else {
            self.effects.push((name, effect));
        }
    }

    /// Pull live data from the runtime. Called on every tick (~16 FPS).
    pub fn refresh(&mut self, daemon: &DaemonState, runner: &ScriptRunner) {
        self.running = daemon.is_running();
        self.uptime_secs = daemon.start_time.elapsed().as_secs();
        self.active_clients = daemon.active_clients.load(Ordering::Relaxed);

        // TLSF allocator tracks managed pool state.
        let tlsf = daemon.runtime.tlsf_stats();
        self.pool_used = tlsf.allocated_bytes as u64;
        self.pool_total = tlsf.total_pool_size as u64;
        self.pool_percent = tlsf.utilization_percent;
        self.pool_peak = tlsf.peak_allocated as u64;
        self.pool_frag = tlsf.fragmentation_ratio;
        self.pool_blocks = tlsf.allocated_blocks;
        self.pool_free_blocks = tlsf.free_blocks;
        self.pool_largest_free = tlsf.largest_free_block as u64;
        // Hysteresis on boolean health states — hold for 16 ticks (~1 s)
        // before flipping back, preventing per-frame flicker.
        const HOLD_TICKS: u8 = 16;

        if tlsf.needs_defrag != self.pool_needs_defrag {
            if self.pool_defrag_hold == 0 {
                self.pool_needs_defrag = tlsf.needs_defrag;
                self.pool_defrag_hold = HOLD_TICKS;
            } else {
                self.pool_defrag_hold -= 1;
            }
        } else {
            self.pool_defrag_hold = 0;
        }

        if tlsf.is_healthy != self.pool_healthy {
            if self.pool_healthy_hold == 0 {
                self.pool_healthy = tlsf.is_healthy;
                self.pool_healthy_hold = HOLD_TICKS;
            } else {
                self.pool_healthy_hold -= 1;
            }
        } else {
            self.pool_healthy_hold = 0;
        }

        self.pool_allocs = tlsf.total_allocations;
        self.pool_frees = tlsf.total_frees;

        // Allocation event rate (from ring buffer)
        let ring = daemon.runtime.alloc_events();
        let delta = ring.count.wrapping_sub(self.last_event_count);
        self.last_event_count = ring.count;
        // ~16 FPS → multiply by 16 for events/sec, then smooth
        let raw_rate = delta as f64 * 16.0;
        self.alloc_rate = self.alloc_rate * 0.8 + raw_rate * 0.2;

        if self.tick_count % 3 == 0 {
            if self.alloc_rate_history.len() >= SPARKLINE_LEN {
                self.alloc_rate_history.pop_front();
            }
            self.alloc_rate_history.push_back(self.alloc_rate as f32);
        }

        // Runtime stats — includes global VRAM usage from stable runtime.
        let stats = daemon.runtime.stats();
        self.total_ops = stats.total_ops;
        self.gpu_util = stats.gpu_utilization;
        self.mem_util = stats.mem_utilization;

        if stats.watchdog_tripped != self.watchdog {
            if self.watchdog_hold == 0 {
                self.watchdog = stats.watchdog_tripped;
                self.watchdog_hold = HOLD_TICKS;
            } else {
                self.watchdog_hold -= 1;
            }
        } else {
            self.watchdog_hold = 0;
        }
        self.avg_latency_us = stats.avg_latency_us;
        self.hardware_poll_valid = stats.nvml_valid;
        self.temperature_c = stats.temperature_c;
        self.power_w = stats.power_w;
        self.sm_clock_mhz = stats.sm_clock_mhz;
        self.mem_clock_mhz = stats.mem_clock_mhz;
        self.ops_rate_hw = stats.hw_ops_per_sec as f64;
        self.ops_rate_valid = stats.cupti_valid;
        self.gflops_hw = stats.gflops_total;
        self.gflops_valid = stats.cupti_valid;

        let runtime_used = stats.vram_used as u64;
        let runtime_total = (stats.vram_used + stats.vram_free) as u64;
        self.vram_used = runtime_used.max(self.pool_used);
        self.vram_total = if runtime_total > 0 {
            if self.pool_total > 0 {
                runtime_total.max(self.pool_total)
            } else {
                runtime_total
            }
        } else {
            self.pool_total
        };

        // Use TLSF pool utilization for the live percent — this is what
        // actually changes when demos allocate/free.  cudaMemGetInfo sees
        // the pre-allocated pool as static "used" VRAM and never moves.
        self.vram_percent = if self.pool_total > 0 {
            self.pool_percent
        } else if self.vram_total > 0 {
            self.vram_used as f32 * 100.0 / self.vram_total as f32
        } else {
            0.0
        };

        // Process list
        self.processes.clear();
        {
            let apps = daemon.apps.lock();
            for app in apps.values() {
                self.processes.push(AppInfo {
                    name: app.name.clone(),
                });
            }
        }

        self.script_cache_len = runner.cache_len();

        // Sparkline history pushed at throttled rate (every 3rd tick in tick()).
        // Store latest values; tick() reads them when pushing.
        // gpu_util and vram_percent are already set above.

        // Push sparkline samples at reduced cadence (~5 Hz) to show
        // meaningful trends instead of per-frame noise.
        if self.tick_count % 3 == 0 {
            if self.gpu_util_history.len() >= SPARKLINE_LEN {
                self.gpu_util_history.pop_front();
            }
            self.gpu_util_history.push_back(self.gpu_util);

            if self.vram_history.len() >= SPARKLINE_LEN {
                self.vram_history.pop_front();
            }
            self.vram_history.push_back(self.vram_percent);
        }

        // Update per-stream activity lanes from real cudaStreamQuery results.
        // Each bit in stream_busy[] indicates a stream with pending GPU work.
        let polled = stats.stream_poll_count as usize;
        for i in 0..self.stream_count {
            if i >= self.stream_activity.len() {
                break;
            }
            let is_active = if i < polled {
                (stats.stream_busy[i / 8] >> (i % 8)) & 1 != 0
            } else {
                false
            };
            if self.stream_activity[i].len() >= SPARKLINE_LEN {
                self.stream_activity[i].pop_front();
            }
            self.stream_activity[i].push_back(is_active);
        }

        // Per-owner VRAM usage (from TLSF allocator's owner tracking).
        let owner_stats = daemon.runtime.owner_stats();
        self.vram_owners.clear();
        for i in 0..owner_stats.num_owners as usize {
            let ou = &owner_stats.owners[i];
            if ou.allocated_bytes == 0 && ou.block_count == 0 {
                continue;
            }
            let cfg = self.vram_quota_configs.get(&ou.owner_id);
            let label = cfg
                .map(|c| c.label.clone())
                .unwrap_or_else(|| format!("owner-{}", ou.owner_id));
            let soft_limit = cfg.map(|c| c.soft_limit).unwrap_or(0);
            let over = soft_limit > 0 && (ou.allocated_bytes as u64) > soft_limit;
            self.vram_owners.push(OwnerUsage {
                owner_id: ou.owner_id,
                label,
                allocated_bytes: ou.allocated_bytes as u64,
                block_count: ou.block_count,
                soft_limit,
                over_quota: over,
            });
        }

        // ── scheduler / control plane state refresh ─────────────
        // Mirror the paused state from DaemonState (atomic, cheap).
        self.scheduler_paused = daemon.scheduler_paused.load(Ordering::Relaxed);

        // Pull tenant and queue snapshots from the scheduler (every 8th tick
        // to avoid lock contention — refreshes ~2 Hz at 16 FPS).
        if self.tick_count % 8 == 0 {
            let sched = daemon.scheduler.lock();

            // Tenant snapshot
            self.scheduler_tenant_snapshot.clear();
            for tid in sched.registry().list() {
                if let Some(tenant) = sched.registry().get(tid) {
                    let snap = tenant.usage.snapshot();
                    self.scheduler_tenant_snapshot.push(super::TenantSummary {
                        tenant_id: tid.0,
                        label: tenant.label.clone(),
                        vram_used: snap.current_vram_bytes,
                        vram_limit: tenant.quotas.max_vram_bytes,
                        streams_used: snap.active_streams as u32,
                        streams_limit: if tenant.quotas.max_streams == u64::MAX {
                            0
                        } else {
                            tenant.quotas.max_streams as u32
                        },
                        active_jobs: snap.active_jobs as u32,
                    });
                }
            }

            // Queue snapshot — pull from dispatcher
            self.scheduler_queue_snapshot.clear();
            // The dispatcher exposes queue_len() and active_count() but not
            // individual queued jobs directly.  For the TUI we show active
            // dispatched jobs since those are what the operator cares about.
            // (Queued-but-not-yet-dispatched jobs are transient and flushed
            // within one dispatch cycle.)
            // In a future iteration the dispatcher could expose an iterator
            // over its queue; for now the queue length is shown in the header
            // and active jobs are shown per-tenant.
        }
    }

    pub fn push_log(&mut self, entry: LogEntry) {
        if self.log.len() >= MAX_LOG_ENTRIES {
            self.log.pop_front();
        }
        self.log.push_back(entry);
        self.log_scroll = 0;
    }

    pub fn set_pipeline(&mut self, name: String, compile_ms: u128, exec_ms: u128) {
        let total = compile_ms + exec_ms;
        self.last_pipeline = Some(PipelineState {
            name,
            stages: vec![
                PipelineStage {
                    name: "jit".into(),
                    offset_ms: 0,
                    duration_ms: compile_ms,
                },
                PipelineStage {
                    name: "gpu".into(),
                    offset_ms: compile_ms,
                    duration_ms: exec_ms,
                },
            ],
            total_ms: total,
            timestamp: Instant::now(),
        });

        // Lightweight pulse tuned for frequent pipeline updates.
        let pulse_color = if total >= 100 {
            style::bad()
        } else if total >= 50 {
            style::warn()
        } else {
            style::info()
        };

        let pipeline_fx = fx::sequence(&[
            fx::parallel(&[
                fx::fade_from_fg(pulse_color, (110, Interpolation::SineOut))
                    .with_filter(CellFilter::Text),
                fx::hsl_shift_fg([10.0, 8.0, 4.0], (140, Interpolation::QuadOut))
                    .with_filter(CellFilter::Text),
            ]),
            fx::fade_to_fg(style::fg(), (170, Interpolation::SineInOut))
                .with_filter(CellFilter::Text),
        ]);
        if let Some(effect) = self
            .fx_script
            .as_ref()
            .and_then(|cfg| cfg.build_pipeline_effect(pulse_color))
        {
            self.set_named_effect("pipeline", effect);
        } else {
            self.set_named_effect("pipeline", pipeline_fx);
        }
    }

    pub fn set_tensor_viz(&mut self, shape: Vec<usize>, data: Vec<f32>) {
        if data.is_empty() {
            return;
        }
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len() as f32;

        let mut histogram = vec![0u32; 32];
        let range = max - min;
        if range > 0.0 {
            for &v in &data {
                let bin = (((v - min) / range) * 31.0) as usize;
                histogram[bin.min(31)] += 1;
            }
        } else {
            histogram[0] = data.len() as u32;
        }

        let sample_count = data.len().min(80);
        let samples = data[..sample_count].to_vec();

        self.last_tensor = Some(TensorViz {
            shape,
            min,
            max,
            mean,
            histogram,
            samples,
            timestamp: Instant::now(),
        });

        // Slightly richer color-wave for tensor snapshots (still short and cheap).
        let tensor_base = if mean.abs() > 1.0 { style::warn() } else { style::info() };
        let tensor_fx = fx::sequence(&[
            fx::parallel(&[
                fx::fade_from_fg(tensor_base, (150, Interpolation::SineOut))
                    .with_filter(CellFilter::Text),
                fx::hsl_shift_fg([24.0, 12.0, 6.0], (210, Interpolation::SineInOut))
                    .with_filter(CellFilter::Text),
            ]),
            fx::fade_to_fg(style::fg_bright(), (130, Interpolation::QuadOut))
                .with_filter(CellFilter::Text),
            fx::fade_to_fg(style::fg(), (120, Interpolation::SineIn))
                .with_filter(CellFilter::Text),
        ]);
        if let Some(effect) = self
            .fx_script
            .as_ref()
            .and_then(|cfg| cfg.build_tensor_effect(tensor_base))
        {
            self.set_named_effect("tensor", effect);
        } else {
            self.set_named_effect("tensor", tensor_fx);
        }
    }
}
