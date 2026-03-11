use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, set_torch_stream, sync_all_streams, reset_torch_stream};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

#[derive(Clone)]
struct Config {
    gpu_count: usize,
    model_gb: usize,
    shard_mb: usize,
    steps: usize,
    streams_per_gpu: u32,
    wave_streams: usize,
    micro_batch: i64,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    lr: f64,
    grad_accum: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpu_count: 1,
            model_gb: 100,
            shard_mb: 64,
            steps: 100,
            streams_per_gpu: 64,
            wave_streams: 16,
            micro_batch: 8,
            hidden: 2048,
            lora_rank: 16,
            lora_alpha: 32.0,
            lr: 1e-3,
            grad_accum: 1,
        }
    }
}

#[derive(Clone, Debug)]
struct GpuState {
    device_id: i32,
    device: Device,
    shards_assigned: Vec<usize>,
    shard_loss: Vec<f64>,
}

#[derive(Clone, Debug)]
struct WavePlan {
    gpu_assignments: Vec<Vec<usize>>, // gpu_id -> list of shard indices
    total_shards: usize,
    shards_per_gpu: usize,
}

fn plan_distribution(total_shards: usize, gpu_count: usize) -> WavePlan {
    let shards_per_gpu = total_shards.div_ceil(gpu_count);
    let mut assignments = vec![Vec::new(); gpu_count];

    for shard_idx in 0..total_shards {
        let gpu_id = shard_idx % gpu_count;
        assignments[gpu_id].push(shard_idx);
    }

    WavePlan {
        gpu_assignments: assignments,
        total_shards,
        shards_per_gpu,
    }
}

/// All-reduce simulation: average gradients across GPUs.
/// In production with NCCL this would be a true ring all-reduce.
fn allreduce_average(tensors: &[Tensor]) -> Tensor {
    if tensors.is_empty() {
        return Tensor::zeros([], (Kind::Float, Device::Cpu));
    }
    let sum = tensors.iter().skip(1).fold(tensors[0].shallow_clone(), |acc, t| {
        &acc + &t.to_device(tensors[0].device())
    });
    sum / tensors.len() as f64
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gpus" => cfg.gpu_count = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--model-gb" => cfg.model_gb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--shard-mb" => cfg.shard_mb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--steps" => cfg.steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams-per-gpu" => cfg.streams_per_gpu = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lr" => cfg.lr = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--grad-accum" => cfg.grad_accum = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: multi_gpu_wave.rs --gpus N [options]");
                println!("  Distributes shard-streamed LoRA fine-tuning across multiple GPUs.");
                println!("  Each GPU processes a subset of shards per step, with gradient");
                println!("  all-reduce across devices.");
                println!();
                println!("  --gpus N              Number of GPUs (default: 1, max: available)");
                println!("  --grad-accum N        Gradient accumulation steps before sync");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }
    Ok(cfg)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    let available_gpus = tch::Cuda::device_count() as usize;
    if available_gpus == 0 {
        println!("No CUDA devices available");
        return Ok(());
    }

    let gpu_count = cfg.gpu_count.min(available_gpus);

    // Initialize PTX runtime on primary GPU
    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: cfg.streams_per_gpu,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, 0.70, cfg.streams_per_gpu).map_err(anyhow::Error::msg)?;

    let active_streams = num_streams();
    let inv_sqrt_d = 1.0 / (cfg.hidden as f64).sqrt();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;

    let model_bytes = cfg.model_gb * 1024 * 1024 * 1024;
    let shard_bytes = cfg.shard_mb * 1024 * 1024;
    let total_shards = model_bytes.div_ceil(shard_bytes);

    let plan = plan_distribution(total_shards, gpu_count);
    let pool_bytes = runtime.tlsf_stats().total_pool_size;

    println!("=== Ferrite Multi-GPU Wave Scheduler ===\n");
    println!("  available GPUs:    {}", available_gpus);
    println!("  using GPUs:        {}", gpu_count);
    println!("  model:             {} GB", cfg.model_gb);
    println!("  total shards:      {}", total_shards);
    println!("  shards/GPU:        ~{}", plan.shards_per_gpu);
    println!("  streams/GPU:       {}", active_streams);
    println!("  wave streams:      {}", cfg.wave_streams);
    println!("  grad accumulation: {}", cfg.grad_accum);
    println!("  steps:             {}", cfg.steps);
    println!("  TLSF pool (GPU 0): {:.1} MB\n", pool_bytes as f64 / 1e6);

    for (gpu_id, shards) in plan.gpu_assignments.iter().enumerate() {
        println!("  GPU {}: {} shards assigned", gpu_id, shards.len());
    }
    println!();

    // For single-GPU or when only one is available, run the full shard set on device 0
    // For multi-GPU, each device would get its shard partition
    let primary_device = Device::Cuda(0);

    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut non_finite = 0usize;
    let mut step_times = Vec::with_capacity(cfg.steps);
    let t_total = Instant::now();

    // Per-GPU gradient accumulators (simulated for all GPUs on device 0 for now)
    let mut accum_steps = 0usize;

    for step in 0..cfg.steps {
        let t_step = Instant::now();
        let mut step_loss = 0.0f64;
        let mut gpu_grad_a_accum: Vec<Vec<Tensor>> = (0..gpu_count).map(|_| Vec::new()).collect();
        let mut gpu_grad_b_accum: Vec<Vec<Tensor>> = (0..gpu_count).map(|_| Vec::new()).collect();

        for gpu_id in 0..gpu_count {
            let device = if gpu_count > 1 && gpu_id < available_gpus {
                Device::Cuda(gpu_id)
            } else {
                primary_device
            };

            let my_shards = &plan.gpu_assignments[gpu_id];
            let mut shard_base = 0usize;

            while shard_base < my_shards.len() {
                let wave = (my_shards.len() - shard_base).min(cfg.wave_streams);

                for local_idx in 0..wave {
                    let shard_idx = my_shards[shard_base + local_idx];
                    let stream_id = (shard_base + local_idx) % active_streams;
                    set_torch_stream(stream_id);

                    let elems = shard_bytes.div_ceil(std::mem::size_of::<f32>()) as i64;
                    let rows = (elems / cfg.hidden).max(1);
                    let used = rows * cfg.hidden;

                    let base_w = (Tensor::randn([used], (Kind::Float, device)) * inv_sqrt_d)
                        .view([rows, cfg.hidden]);

                    let mut lora_a = Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device))
                        .set_requires_grad(true);
                    let mut lora_b = Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device))
                        .set_requires_grad(true);

                    let x = Tensor::randn([cfg.micro_batch, cfg.hidden], (Kind::Float, device));
                    let delta = lora_a.matmul(&lora_b) * lora_scale;
                    let effective_w = &base_w + &delta;
                    let y = (x.matmul(&effective_w.transpose(0, 1)) * inv_sqrt_d).tanh();
                    let target = Tensor::randn([cfg.micro_batch, rows], (Kind::Float, device)).tanh();

                    let loss = (&y - &target).clamp(-10.0, 10.0).square().mean(Kind::Float);
                    loss.backward();

                    gpu_grad_a_accum[gpu_id].push(lora_a.grad().clamp(-1.0, 1.0));
                    gpu_grad_b_accum[gpu_id].push(lora_b.grad().clamp(-1.0, 1.0));

                    let lv = f64::try_from(&loss).unwrap_or(f64::NAN);
                    if lv.is_finite() {
                        total_loss += lv;
                        step_loss += lv;
                        loss_count += 1;
                    } else {
                        non_finite += 1;
                    }
                }

                sync_all_streams();
                reset_torch_stream();
                shard_base += wave;
            }
        }

        accum_steps += 1;

        // Gradient synchronization across GPUs every grad_accum steps
        if accum_steps >= cfg.grad_accum {
            // All-reduce: average the accumulated gradients from each GPU
            // In production, this uses NCCL ring all-reduce
            for shard_local in 0..plan.shards_per_gpu.min(
                gpu_grad_a_accum.iter().map(|v| v.len()).min().unwrap_or(0)
            ) {
                let ga_per_gpu: Vec<Tensor> = gpu_grad_a_accum.iter()
                    .filter(|v| shard_local < v.len())
                    .map(|v| v[shard_local].shallow_clone())
                    .collect();
                let gb_per_gpu: Vec<Tensor> = gpu_grad_b_accum.iter()
                    .filter(|v| shard_local < v.len())
                    .map(|v| v[shard_local].shallow_clone())
                    .collect();

                if !ga_per_gpu.is_empty() {
                    let _synced_a = allreduce_average(&ga_per_gpu);
                    let _synced_b = allreduce_average(&gb_per_gpu);
                    // Apply synced gradients to adapters (in production)
                }
            }
            accum_steps = 0;
            gpu_grad_a_accum.iter_mut().for_each(|v| v.clear());
            gpu_grad_b_accum.iter_mut().for_each(|v| v.clear());
        }

        tch::Cuda::synchronize(0);
        let dt = t_step.elapsed().as_secs_f64();
        step_times.push(dt);

        if (step + 1) % 20 == 0 || step == 0 {
            let s = runtime.tlsf_stats();
            println!(
                "  step {:>4} | loss={:.6} | gpus={} | time={:.3}s | vram_0={:.0}MB | frag={:.6}",
                step + 1,
                step_loss / total_shards as f64,
                gpu_count,
                dt,
                s.allocated_bytes as f64 / 1e6,
                s.fragmentation_ratio
            );
        }
    }

    let wall = t_total.elapsed().as_secs_f64();
    let sf = runtime.tlsf_stats();
    let avg_step = step_times.iter().sum::<f64>() / step_times.len() as f64;
    let logical_gb = (total_shards * shard_bytes * cfg.steps) as f64 / 1e9;
    let throughput_gb = logical_gb / wall;

    println!("\n  Multi-GPU Results:");
    println!("    GPUs used:          {}", gpu_count);
    println!("    Total shards/step:  {}", total_shards);
    println!("    Shards per GPU:     ~{}", plan.shards_per_gpu);
    println!("    Logical streamed:   {:.2} GB", logical_gb);
    println!("    Throughput:         {:.2} GB/s", throughput_gb);
    println!("    Wall time:          {:.2}s", wall);
    println!("    Avg step:           {:.3}s", avg_step);
    println!("    Avg loss:           {:.6}", if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN });
    println!("    Non-finite:         {}", non_finite);
    println!("    Peak VRAM (GPU 0):  {:.1} MB", sf.peak_allocated as f64 / 1e6);
    println!("    Fragmentation:      {:.6}", sf.fragmentation_ratio);

    println!("\nRESULT mode=multi_gpu_wave");
    println!("RESULT gpu_count={}", gpu_count);
    println!("RESULT total_shards={}", total_shards);
    println!("RESULT grad_accum={}", cfg.grad_accum);
    println!("RESULT logical_gb={:.6}", logical_gb);
    println!("RESULT throughput_gbs={:.6}", throughput_gb);
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT avg_step_s={:.6}", avg_step);
    println!("RESULT avg_loss={:.9}", if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN });
    println!("RESULT non_finite={}", non_finite);
    println!("RESULT peak_vram_mb={:.6}", sf.peak_allocated as f64 / 1e6);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);

    Ok(())
}
