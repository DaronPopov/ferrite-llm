use std::collections::VecDeque;
use std::time::Instant;

const MAX_PROFILES: usize = 64;
const SPARKLINE_LEN: usize = 48;

/// Profiling snapshot for a single kernel execution.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct KernelProfile {
    pub timestamp: Instant,
    pub compile_ms: u64,
    pub exec_ms: u64,
    pub total_ms: u64,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub node_count: usize,
    pub total_elements: usize,
    pub success: bool,
    /// Script source identifier (first 40 chars of script or filename).
    pub source_tag: String,
    /// VRAM pool usage before execution (bytes).
    pub vram_before: u64,
    /// VRAM pool usage after execution (bytes).
    pub vram_after: u64,
}

/// Ring buffer of recent kernel profiles with derived statistics.
pub struct ProfileRing {
    pub entries: VecDeque<KernelProfile>,
    pub latency_history: VecDeque<f32>,
    pub compile_history: VecDeque<f32>,
    /// VRAM delta history (bytes, signed via i64).
    pub vram_delta_history: VecDeque<i64>,
    /// Lifetime execution count.
    pub total_runs: u64,
    /// Lifetime successful executions.
    pub total_success: u64,
    /// Exponentially smoothed total latency (ms).
    pub smoothed_latency: f32,
}

impl ProfileRing {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(MAX_PROFILES),
            latency_history: VecDeque::with_capacity(SPARKLINE_LEN),
            compile_history: VecDeque::with_capacity(SPARKLINE_LEN),
            vram_delta_history: VecDeque::with_capacity(SPARKLINE_LEN),
            total_runs: 0,
            total_success: 0,
            smoothed_latency: 0.0,
        }
    }

    pub fn push(&mut self, profile: KernelProfile) {
        self.total_runs += 1;
        if profile.success {
            self.total_success += 1;
        }

        // Update smoothed latency (EMA, alpha=0.3)
        let lat = profile.total_ms as f32;
        if self.smoothed_latency == 0.0 {
            self.smoothed_latency = lat;
        } else {
            self.smoothed_latency = self.smoothed_latency * 0.7 + lat * 0.3;
        }

        // Push to sparkline histories
        if self.latency_history.len() >= SPARKLINE_LEN {
            self.latency_history.pop_front();
        }
        self.latency_history.push_back(lat);

        if self.compile_history.len() >= SPARKLINE_LEN {
            self.compile_history.pop_front();
        }
        self.compile_history.push_back(profile.compile_ms as f32);

        let delta = profile.vram_after as i64 - profile.vram_before as i64;
        if self.vram_delta_history.len() >= SPARKLINE_LEN {
            self.vram_delta_history.pop_front();
        }
        self.vram_delta_history.push_back(delta);

        // Push to ring
        if self.entries.len() >= MAX_PROFILES {
            self.entries.pop_front();
        }
        self.entries.push_back(profile);
    }

    /// Last N profiles (most recent first).
    #[allow(dead_code)]
    pub fn recent(&self, n: usize) -> Vec<&KernelProfile> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Average compile time over the ring.
    pub fn avg_compile_ms(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.compile_ms).sum();
        sum as f32 / self.entries.len() as f32
    }

    /// Average execution time over the ring.
    pub fn avg_exec_ms(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.exec_ms).sum();
        sum as f32 / self.entries.len() as f32
    }

    /// Peak total latency in the ring.
    pub fn peak_ms(&self) -> u64 {
        self.entries.iter().map(|e| e.total_ms).max().unwrap_or(0)
    }

    /// Net VRAM change of the last execution (bytes, signed).
    pub fn last_vram_delta(&self) -> i64 {
        self.entries
            .back()
            .map(|e| e.vram_after as i64 - e.vram_before as i64)
            .unwrap_or(0)
    }

    /// Success rate as percentage.
    pub fn success_rate(&self) -> f32 {
        if self.total_runs == 0 {
            return 100.0;
        }
        self.total_success as f32 * 100.0 / self.total_runs as f32
    }
}
