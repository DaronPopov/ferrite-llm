use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;

#[derive(Clone, Debug)]
pub struct StepMetric {
    pub step: usize,
    pub loss: f64,
    pub lr: f64,
    pub step_time_s: f64,
    pub vram_allocated_mb: f64,
    pub vram_peak_mb: f64,
    pub fragmentation: f64,
    pub tlsf_allocs: u64,
    pub tlsf_frees: u64,
    pub grad_norm: f64,
    pub non_finite: bool,
}

#[derive(Clone, Debug)]
pub struct TlsfHealthSnapshot {
    pub timestamp_s: f64,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub pool_bytes: usize,
    pub fragmentation: f64,
    pub total_allocs: u64,
    pub total_frees: u64,
    pub pool_valid: bool,
}

pub struct MetricsCollector {
    run_id: String,
    start_time: Instant,
    steps: Vec<StepMetric>,
    health_snapshots: Vec<TlsfHealthSnapshot>,
    loss_window: Vec<f64>,
    window_size: usize,
}

impl MetricsCollector {
    pub fn new(run_id: &str, window_size: usize) -> Self {
        Self {
            run_id: run_id.to_string(),
            start_time: Instant::now(),
            steps: Vec::new(),
            health_snapshots: Vec::new(),
            loss_window: Vec::new(),
            window_size,
        }
    }

    pub fn record_step(&mut self, metric: StepMetric) {
        if metric.loss.is_finite() {
            self.loss_window.push(metric.loss);
            if self.loss_window.len() > self.window_size {
                self.loss_window.remove(0);
            }
        }
        self.steps.push(metric);
    }

    pub fn record_health(&mut self, snap: TlsfHealthSnapshot) {
        self.health_snapshots.push(snap);
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn latest_loss(&self) -> f64 {
        self.steps.last().map(|s| s.loss).unwrap_or(f64::NAN)
    }

    pub fn smoothed_loss(&self) -> f64 {
        if self.loss_window.is_empty() { return f64::NAN; }
        self.loss_window.iter().sum::<f64>() / self.loss_window.len() as f64
    }

    pub fn avg_loss(&self) -> f64 {
        let finite: Vec<f64> = self.steps.iter()
            .filter(|s| s.loss.is_finite())
            .map(|s| s.loss)
            .collect();
        if finite.is_empty() { f64::NAN } else { finite.iter().sum::<f64>() / finite.len() as f64 }
    }

    pub fn min_loss(&self) -> f64 {
        self.steps.iter()
            .filter(|s| s.loss.is_finite())
            .map(|s| s.loss)
            .fold(f64::MAX, f64::min)
    }

    pub fn avg_step_time(&self) -> f64 {
        if self.steps.is_empty() { return 0.0; }
        self.steps.iter().map(|s| s.step_time_s).sum::<f64>() / self.steps.len() as f64
    }

    pub fn peak_vram_mb(&self) -> f64 {
        self.steps.iter().map(|s| s.vram_peak_mb).fold(0.0f64, f64::max)
    }

    pub fn max_fragmentation(&self) -> f64 {
        self.health_snapshots.iter()
            .map(|s| s.fragmentation)
            .fold(0.0f64, f64::max)
    }

    pub fn non_finite_count(&self) -> usize {
        self.steps.iter().filter(|s| s.non_finite).count()
    }

    pub fn total_wall_s(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Detect if training is diverging: smoothed loss increasing over last N steps
    pub fn is_diverging(&self, lookback: usize) -> bool {
        if self.steps.len() < lookback * 2 { return false; }
        let recent: Vec<f64> = self.steps[self.steps.len() - lookback..]
            .iter()
            .filter(|s| s.loss.is_finite())
            .map(|s| s.loss)
            .collect();
        let earlier: Vec<f64> = self.steps[self.steps.len() - lookback * 2..self.steps.len() - lookback]
            .iter()
            .filter(|s| s.loss.is_finite())
            .map(|s| s.loss)
            .collect();

        if recent.is_empty() || earlier.is_empty() { return false; }
        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let earlier_avg = earlier.iter().sum::<f64>() / earlier.len() as f64;
        recent_avg > earlier_avg * 1.5
    }

    /// Write metrics to a CSV file for analysis/plotting
    pub fn write_csv(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "step,loss,lr,step_time_s,vram_allocated_mb,vram_peak_mb,fragmentation,tlsf_allocs,tlsf_frees,grad_norm,non_finite")?;
        for s in &self.steps {
            writeln!(w, "{},{:.9},{:.9},{:.6},{:.1},{:.1},{:.9},{},{},{:.9},{}",
                s.step, s.loss, s.lr, s.step_time_s,
                s.vram_allocated_mb, s.vram_peak_mb, s.fragmentation,
                s.tlsf_allocs, s.tlsf_frees, s.grad_norm,
                if s.non_finite { 1 } else { 0 })?;
        }
        w.flush()?;
        Ok(())
    }

    /// Write TLSF health log
    pub fn write_health_csv(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "timestamp_s,allocated_bytes,peak_bytes,pool_bytes,fragmentation,total_allocs,total_frees,pool_valid")?;
        for s in &self.health_snapshots {
            writeln!(w, "{:.6},{},{},{},{:.9},{},{},{}",
                s.timestamp_s, s.allocated_bytes, s.peak_bytes, s.pool_bytes,
                s.fragmentation, s.total_allocs, s.total_frees,
                if s.pool_valid { 1 } else { 0 })?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn print_summary(&self) {
        println!("=== Training Metrics Summary ===\n");
        println!("  Run ID:           {}", self.run_id);
        println!("  Steps:            {}", self.step_count());
        println!("  Wall time:        {:.2}s", self.total_wall_s());
        println!("  Avg step time:    {:.3}s", self.avg_step_time());
        println!("  Avg loss:         {:.6}", self.avg_loss());
        println!("  Min loss:         {:.6}", self.min_loss());
        println!("  Smoothed loss:    {:.6}", self.smoothed_loss());
        println!("  Peak VRAM:        {:.1} MB", self.peak_vram_mb());
        println!("  Max fragmentation:{:.6}", self.max_fragmentation());
        println!("  Non-finite:       {}", self.non_finite_count());
        println!("  Diverging:        {}", self.is_diverging(20));
    }

    pub fn emit_results(&self) {
        println!("\nRESULT run_id={}", self.run_id);
        println!("RESULT total_steps={}", self.step_count());
        println!("RESULT wall_s={:.6}", self.total_wall_s());
        println!("RESULT avg_step_s={:.6}", self.avg_step_time());
        println!("RESULT avg_loss={:.9}", self.avg_loss());
        println!("RESULT min_loss={:.9}", self.min_loss());
        println!("RESULT smoothed_loss={:.9}", self.smoothed_loss());
        println!("RESULT peak_vram_mb={:.6}", self.peak_vram_mb());
        println!("RESULT max_fragmentation={:.9}", self.max_fragmentation());
        println!("RESULT non_finite_count={}", self.non_finite_count());
        println!("RESULT is_diverging={}", if self.is_diverging(20) { 1 } else { 0 });
    }
}

fn parse_args() -> Result<(Option<PathBuf>, Option<PathBuf>, usize)> {
    let mut metrics_csv = None;
    let mut health_csv = None;
    let mut demo_steps = 100usize;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--metrics-csv" => metrics_csv = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--health-csv" => health_csv = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--demo-steps" => demo_steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: training_metrics.rs [--metrics-csv PATH] [--health-csv PATH] [--demo-steps N]");
                println!("  Demonstrates the metrics collection system with synthetic training data.");
                println!("  In production, MetricsCollector is imported and used from the training loop.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    Ok((metrics_csv, health_csv, demo_steps))
}

fn main() -> Result<()> {
    let (metrics_csv, health_csv, demo_steps) = parse_args()?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let run_id = format!("ferrite_{}", timestamp);

    let mut collector = MetricsCollector::new(&run_id, 20);

    println!("[telemetry] running {} demo steps...\n", demo_steps);

    // Simulate a training run with decreasing loss and realistic TLSF stats
    let mut loss = 2.5f64;
    let mut rng_state = 0xDEADBEEFu64;

    for step in 0..demo_steps {
        let t0 = Instant::now();

        // Simulated loss decay + noise
        rng_state ^= step as u64;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state & 0xFFFF) as f64 / 65536.0 - 0.5) * 0.1;
        loss = (loss * 0.995 + noise).max(0.01);

        let lr = 1e-3 * (1.0 - step as f64 / demo_steps as f64).max(0.01);

        // Simulated grad norm
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let grad_norm = 0.5 + ((rng_state & 0xFFFF) as f64 / 65536.0) * 2.0;

        let elapsed = t0.elapsed().as_secs_f64() + 0.05; // simulated step time

        collector.record_step(StepMetric {
            step,
            loss,
            lr,
            step_time_s: elapsed,
            vram_allocated_mb: 2048.0 + (step as f64 * 0.1),
            vram_peak_mb: 2200.0 + (step as f64 * 0.05),
            fragmentation: 0.0001 + (step as f64 * 0.000001),
            tlsf_allocs: (step as u64 + 1) * 1600,
            tlsf_frees: (step as u64 + 1) * 1598,
            grad_norm,
            non_finite: false,
        });

        // Record health every 10 steps
        if step % 10 == 0 {
            collector.record_health(TlsfHealthSnapshot {
                timestamp_s: collector.total_wall_s(),
                allocated_bytes: (2048.0 * 1e6) as usize + step * 100000,
                peak_bytes: (2200.0 * 1e6) as usize + step * 50000,
                pool_bytes: (4000.0 * 1e6) as usize,
                fragmentation: 0.0001 + (step as f64 * 0.000001),
                total_allocs: (step as u64 + 1) * 1600,
                total_frees: (step as u64 + 1) * 1598,
                pool_valid: true,
            });
        }

        if (step + 1) % 20 == 0 || step == 0 {
            println!("  step {:>4} | loss={:.6} | smooth={:.6} | lr={:.6} | grad={:.4}",
                step + 1, loss, collector.smoothed_loss(), lr, grad_norm);
        }
    }

    collector.print_summary();
    collector.emit_results();

    if let Some(path) = metrics_csv {
        collector.write_csv(&path)?;
        println!("\n[telemetry] metrics CSV written to: {}", path.display());
    }

    if let Some(path) = health_csv {
        collector.write_health_csv(&path)?;
        println!("[telemetry] health CSV written to: {}", path.display());
    }

    Ok(())
}
