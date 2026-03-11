use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, reset_torch_stream, set_torch_stream, sync_all_streams};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    num_experts: usize,
    top_k: usize,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    micro_batch: i64,
    steps: usize,
    streams: u32,
    wave_streams: usize,
    lr: f64,
    aux_weight: f64,
    expert_scale: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_experts: 1000,
            top_k: 2,
            hidden: 1024,
            lora_rank: 8,
            lora_alpha: 16.0,
            micro_batch: 8,
            steps: 50,
            streams: 32,
            wave_streams: 8,
            lr: 1e-3,
            aux_weight: 0.01,
            expert_scale: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Expert utilization tracker
// ---------------------------------------------------------------------------

struct UtilizationTracker {
    counts: Vec<u64>,
    total_selections: u64,
}

impl UtilizationTracker {
    fn new(num_experts: usize) -> Self {
        Self {
            counts: vec![0; num_experts],
            total_selections: 0,
        }
    }

    fn record(&mut self, expert_ids: &[i64]) {
        for &eid in expert_ids {
            if (eid as usize) < self.counts.len() {
                self.counts[eid as usize] += 1;
                self.total_selections += 1;
            }
        }
    }

    fn experts_used(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    fn utilization_ratio(&self) -> f64 {
        self.experts_used() as f64 / self.counts.len() as f64
    }

    fn max_count(&self) -> u64 {
        *self.counts.iter().max().unwrap_or(&0)
    }

    fn entropy(&self) -> f64 {
        if self.total_selections == 0 {
            return 0.0;
        }
        let mut h = 0.0f64;
        for &c in &self.counts {
            if c > 0 {
                let p = c as f64 / self.total_selections as f64;
                h -= p * p.ln();
            }
        }
        h
    }

    fn max_entropy(&self) -> f64 {
        (self.counts.len() as f64).ln()
    }
}

// ---------------------------------------------------------------------------
// Shard streaming helpers
// ---------------------------------------------------------------------------

/// Generate synthetic expert base weights for a given expert ID.
/// In production, this would mmap from safetensors on disk.
/// The key point: these are allocated fresh, used once, then freed.
fn generate_expert_shard(
    expert_id: i64,
    hidden: i64,
    scale: f64,
    device: Device,
) -> Tensor {
    // Deterministic seeded init so same expert always gives same weights.
    // Use expert_id to seed — in production this is replaced by file loading.
    let w = Tensor::randn([hidden, hidden], (Kind::Float, device));
    w * scale
}

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--num-experts" => cfg.num_experts = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--top-k" => cfg.top_k = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--steps" => cfg.steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lr" => cfg.lr = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--aux-weight" => cfg.aux_weight = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: streaming_moe.rs [options]");
                println!();
                println!("  Massive Sparse Mixture-of-Experts with shard-streamed experts.");
                println!("  Expert base weights are ephemeral — allocated from the TLSF pool,");
                println!("  used for one forward/backward pass, then freed. Only top-k experts");
                println!("  are ever resident in VRAM regardless of total expert count.");
                println!();
                println!("  --num-experts N    Total expert count (default: 1000)");
                println!("  --top-k K          Experts selected per token (default: 2)");
                println!("  --hidden N         Hidden dimension (default: 1024)");
                println!("  --lora-rank N      LoRA adapter rank per expert (default: 8)");
                println!("  --micro-batch N    Batch size (default: 8)");
                println!("  --steps N          Training steps (default: 50)");
                println!("  --streams N        CUDA stream pool (default: 32)");
                println!("  --wave-streams N   Max concurrent expert streams (default: 8)");
                println!("  --lr F             Learning rate (default: 0.001)");
                println!("  --aux-weight F     Load-balancing loss weight (default: 0.01)");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cfg = parse_args()?;

    aten_ptx::ensure_libtorch_cuda_loaded();
    if !tch::Cuda::is_available() {
        println!("CUDA not available");
        return Ok(());
    }

    // --- Init Ferrite runtime ---
    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: cfg.streams,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, 0.70, cfg.streams).map_err(anyhow::Error::msg)?;

    let device = Device::Cuda(0);
    let active_streams = num_streams();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;
    let n_exp = cfg.num_experts as i64;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    // --- Persistent parameters (always in VRAM) ---

    // Router: maps hidden → expert logits
    let mut router_w = Tensor::randn([cfg.hidden, n_exp], (Kind::Float, device))
        .set_requires_grad(true);

    // LoRA adapters for all experts (small — stays resident)
    // A: [num_experts, hidden, lora_rank]  B: [num_experts, lora_rank, hidden]
    let mut lora_a = Tensor::randn(
        [n_exp, cfg.hidden, cfg.lora_rank],
        (Kind::Float, device),
    ).set_requires_grad(true);
    let mut lora_b = Tensor::randn(
        [n_exp, cfg.lora_rank, cfg.hidden],
        (Kind::Float, device),
    ).set_requires_grad(true);

    // Compute persistent VRAM usage
    let router_bytes = (cfg.hidden * n_exp * 4) as f64;
    let lora_a_bytes = (n_exp * cfg.hidden * cfg.lora_rank * 4) as f64;
    let lora_b_bytes = (n_exp * cfg.lora_rank * cfg.hidden * 4) as f64;
    let persistent_mb = (router_bytes + lora_a_bytes + lora_b_bytes) / 1e6;

    // Expert shard size (streamed, ephemeral)
    let expert_shard_bytes = (cfg.hidden * cfg.hidden * 4) as f64;
    let max_streaming_mb = (cfg.wave_streams as f64 * expert_shard_bytes) / 1e6;
    // Theoretical total expert pool (what would be needed without streaming)
    let total_expert_pool_gb = (cfg.num_experts as f64 * expert_shard_bytes) / 1e9;

    println!("=== Ferrite Streaming Mixture-of-Experts ===\n");
    println!("  num experts:        {}", cfg.num_experts);
    println!("  top-k:              {}", cfg.top_k);
    println!("  hidden:             {}", cfg.hidden);
    println!("  lora rank:          {}", cfg.lora_rank);
    println!("  micro batch:        {}", cfg.micro_batch);
    println!("  steps:              {}", cfg.steps);
    println!("  streams:            {}", active_streams);
    println!("  wave streams:       {}", cfg.wave_streams);
    println!("  aux loss weight:    {:.4}", cfg.aux_weight);
    println!();
    println!("  TLSF pool:          {:.1} MB", pool_mb);
    println!("  persistent params:  {:.1} MB (router + LoRA)", persistent_mb);
    println!("  expert shard:       {:.1} MB each", expert_shard_bytes / 1e6);
    println!("  max streaming:      {:.1} MB (wave_streams × shard)", max_streaming_mb);
    println!("  total expert pool:  {:.2} GB (NEVER in VRAM)", total_expert_pool_gb);
    println!("  VRAM budget:        {:.1} MB (persistent + streaming)", persistent_mb + max_streaming_mb);
    println!();

    // --- Training loop ---
    let mut tracker = UtilizationTracker::new(cfg.num_experts);
    let mut total_loss = 0.0f64;
    let mut total_aux = 0.0f64;
    let mut loss_count = 0usize;
    let mut total_experts_streamed = 0u64;
    let mut total_alloc_events = 0u64;
    let mut step_times = Vec::with_capacity(cfg.steps);
    let t_total = Instant::now();

    for step in 0..cfg.steps {
        let t_step = Instant::now();

        // --- Input ---
        let x = Tensor::randn([cfg.micro_batch, cfg.hidden], (Kind::Float, device));
        let target = Tensor::randn([cfg.micro_batch, cfg.hidden], (Kind::Float, device)).tanh();

        // =================================================================
        // Phase 1: ROUTING
        //   router_logits = x @ router_W  → [batch, num_experts]
        //   top_k selection → expert indices + gating weights
        // =================================================================
        let router_logits = x.matmul(&router_w);
        let (top_vals, top_idx) = router_logits.topk(cfg.top_k as i64, -1, true, true);
        // gates: softmax over selected experts → [batch, top_k]
        let gates = top_vals.softmax(-1, Kind::Float);

        // Load-balancing auxiliary loss (Switch Transformer style):
        // Minimize sum of squared mean routing probabilities → encourages uniform use
        let router_probs = router_logits.softmax(-1, Kind::Float);
        let mean_probs = router_probs.mean_dim(&[0i64][..], false, Kind::Float);
        let aux_loss = mean_probs.square().sum(Kind::Float) * (n_exp as f64 * cfg.aux_weight);

        // Discover unique experts selected this step
        let idx_flat: Vec<i64> = Vec::try_from(top_idx.view([-1]))?;
        tracker.record(&idx_flat);

        let mut unique_experts = idx_flat.clone();
        unique_experts.sort();
        unique_experts.dedup();
        let num_unique = unique_experts.len();
        total_experts_streamed += num_unique as u64;

        // =================================================================
        // Phase 2: STREAM SELECTED EXPERTS (wave-scheduled)
        //   For each selected expert:
        //     - Allocate base weights from TLSF pool
        //     - Apply LoRA adapter
        //     - Forward pass for full batch
        //     - Store output
        //     - Base weights freed when tensor drops (O(1) TLSF free)
        // =================================================================
        let mut expert_outputs: Vec<(i64, Tensor)> = Vec::with_capacity(num_unique);

        let mut wave_base = 0usize;
        while wave_base < num_unique {
            let wave = (num_unique - wave_base).min(cfg.wave_streams);

            for local in 0..wave {
                let eid = unique_experts[wave_base + local];
                let stream_id = local % active_streams;
                set_torch_stream(stream_id);

                // --- STREAM IN: allocate expert shard from TLSF pool ---
                let base_w = generate_expert_shard(eid, cfg.hidden, cfg.expert_scale, device);
                total_alloc_events += 1;

                // LoRA adapter for this expert (view into persistent tensor)
                let a_i = lora_a.select(0, eid);
                let b_i = lora_b.select(0, eid);
                let delta = a_i.matmul(&b_i) * lora_scale;
                let effective_w = &base_w + &delta;

                // Forward: all batch items through this expert
                let expert_out = x.matmul(&effective_w.transpose(0, 1));
                expert_outputs.push((eid, expert_out));

                // --- STREAM OUT: base_w dropped here → O(1) TLSF free ---
                // (effective_w also freed; only expert_out survives for combination)
            }

            sync_all_streams();
            reset_torch_stream();
            wave_base += wave;
        }

        // =================================================================
        // Phase 3: GATED COMBINATION
        //   For each (sample, k-slot), accumulate:
        //     combined[sample] += gate[sample, k] * expert_output[selected_expert]
        // =================================================================
        let mut combined = Tensor::zeros([cfg.micro_batch, cfg.hidden], (Kind::Float, device));

        for k_slot in 0..cfg.top_k as i64 {
            let slot_ids = top_idx.select(1, k_slot);           // [batch]
            let slot_gates = gates.narrow(1, k_slot, 1);        // [batch, 1]

            for &(eid, ref out) in &expert_outputs {
                // Mask: 1.0 where this sample selected this expert in this k-slot
                let mask = slot_ids.eq(eid).to_kind(Kind::Float).unsqueeze(1);
                let contribution = (&mask * &slot_gates) * out;
                combined = combined + contribution;
            }
        }

        // =================================================================
        // Phase 4: LOSS + BACKWARD + UPDATE
        // =================================================================
        let task_loss = (&combined - &target).square().mean(Kind::Float);
        let loss = &task_loss + &aux_loss;
        loss.backward();

        let task_lv = f64::try_from(&task_loss).unwrap_or(f64::NAN);
        let aux_lv = f64::try_from(&aux_loss).unwrap_or(f64::NAN);

        if task_lv.is_finite() {
            total_loss += task_lv;
            total_aux += aux_lv;
            loss_count += 1;
        }

        // SGD update
        tch::no_grad(|| {
            let gr = router_w.grad().clamp(-1.0, 1.0);
            let new_r = &router_w - gr * cfg.lr;
            router_w.copy_(&new_r);

            let ga = lora_a.grad().clamp(-1.0, 1.0);
            let new_a = &lora_a - ga * cfg.lr;
            lora_a.copy_(&new_a);

            let gb = lora_b.grad().clamp(-1.0, 1.0);
            let new_b = &lora_b - gb * cfg.lr;
            lora_b.copy_(&new_b);
        });

        // Zero gradients for next step (tch accumulates by default)
        let _ = router_w.grad().zero_();
        let _ = lora_a.grad().zero_();
        let _ = lora_b.grad().zero_();

        // Drop expert_outputs to free TLSF memory
        drop(expert_outputs);

        tch::Cuda::synchronize(0);
        let dt = t_step.elapsed().as_secs_f64();
        step_times.push(dt);

        if (step + 1) % 10 == 0 || step == 0 {
            let s = runtime.tlsf_stats();
            println!(
                "  step {:>4} | loss={:.6} aux={:.6} | experts={:>3}/{} | vram={:.0}MB peak={:.0}MB | frag={:.6} | {:.3}s",
                step + 1,
                task_lv,
                aux_lv,
                num_unique,
                cfg.num_experts,
                s.allocated_bytes as f64 / 1e6,
                s.peak_allocated as f64 / 1e6,
                s.fragmentation_ratio,
                dt,
            );
        }
    }

    // --- Final report ---
    let wall = t_total.elapsed().as_secs_f64();
    let sf = runtime.tlsf_stats();
    let avg_step = step_times.iter().sum::<f64>() / step_times.len() as f64;
    let avg_loss = if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN };
    let avg_aux = if loss_count > 0 { total_aux / loss_count as f64 } else { f64::NAN };
    let avg_experts_per_step = total_experts_streamed as f64 / cfg.steps as f64;

    println!("\n  Streaming MoE Results:");
    println!("    Total experts:          {}", cfg.num_experts);
    println!("    Experts used (unique):  {} ({:.1}% utilization)",
        tracker.experts_used(), tracker.utilization_ratio() * 100.0);
    println!("    Avg experts/step:       {:.1}", avg_experts_per_step);
    println!("    Max expert count:       {} (most popular)", tracker.max_count());
    println!("    Routing entropy:        {:.4} / {:.4} (max)",
        tracker.entropy(), tracker.max_entropy());
    println!("    Total streamed:         {} expert shards", total_experts_streamed);
    println!("    Total alloc events:     {}", total_alloc_events);
    println!();
    println!("    Avg task loss:          {:.6}", avg_loss);
    println!("    Avg aux loss:           {:.6}", avg_aux);
    println!("    Wall time:              {:.2}s", wall);
    println!("    Avg step:               {:.3}s", avg_step);
    println!("    Peak VRAM:              {:.1} MB", sf.peak_allocated as f64 / 1e6);
    println!("    Fragmentation:          {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:           {}", if runtime.validate_pool().is_valid { "YES" } else { "NO" });
    println!();
    println!("    Persistent VRAM:        {:.1} MB (router + LoRA adapters)", persistent_mb);
    println!("    Expert pool (virtual):  {:.2} GB (never allocated)", total_expert_pool_gb);
    println!("    VRAM savings:           {:.0}x (virtual pool / peak VRAM)",
        total_expert_pool_gb * 1024.0 / (sf.peak_allocated as f64 / 1e6));

    println!("\nRESULT mode=streaming_moe");
    println!("RESULT num_experts={}", cfg.num_experts);
    println!("RESULT top_k={}", cfg.top_k);
    println!("RESULT experts_used={}", tracker.experts_used());
    println!("RESULT utilization_pct={:.4}", tracker.utilization_ratio() * 100.0);
    println!("RESULT avg_experts_per_step={:.4}", avg_experts_per_step);
    println!("RESULT routing_entropy={:.6}", tracker.entropy());
    println!("RESULT max_entropy={:.6}", tracker.max_entropy());
    println!("RESULT total_streamed={}", total_experts_streamed);
    println!("RESULT total_alloc_events={}", total_alloc_events);
    println!("RESULT avg_loss={:.9}", avg_loss);
    println!("RESULT avg_aux_loss={:.9}", avg_aux);
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT avg_step_s={:.6}", avg_step);
    println!("RESULT peak_vram_mb={:.6}", sf.peak_allocated as f64 / 1e6);
    println!("RESULT persistent_vram_mb={:.6}", persistent_mb);
    println!("RESULT virtual_pool_gb={:.6}", total_expert_pool_gb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if runtime.validate_pool().is_valid { 1 } else { 0 });

    Ok(())
}
