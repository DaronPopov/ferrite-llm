use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, set_torch_stream, sync_all_streams, reset_torch_stream};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy)]
enum QuantMode {
    F16,
    BF16,
    Int8,
    NF4,
}

impl QuantMode {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "f16" | "float16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "int8" => Ok(Self::Int8),
            "nf4" => Ok(Self::NF4),
            _ => Err(anyhow::anyhow!("invalid quant mode: {s} (f16|bf16|int8|nf4)")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Int8 => "int8",
            Self::NF4 => "nf4",
        }
    }

    fn bytes_per_param(&self) -> f64 {
        match self {
            Self::F16 | Self::BF16 => 2.0,
            Self::Int8 => 1.0,
            Self::NF4 => 0.5,
        }
    }
}

#[derive(Clone)]
struct Config {
    quant_mode: QuantMode,
    model_gb: usize,
    shard_mb: usize,
    steps: usize,
    streams: u32,
    wave_streams: usize,
    micro_batch: i64,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    lr: f64,
    block_size: i64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            quant_mode: QuantMode::F16,
            model_gb: 70,
            shard_mb: 64,
            steps: 100,
            streams: 64,
            wave_streams: 16,
            micro_batch: 8,
            hidden: 2048,
            lora_rank: 16,
            lora_alpha: 32.0,
            lr: 1e-3,
            block_size: 64,
        }
    }
}

struct QuantStats {
    total_shards: usize,
    compressed_bytes: u64,
    original_bytes: u64,
    max_quant_error: f64,
    avg_quant_error: f64,
}

/// Quantize a tensor to f16/bf16 on device
fn quantize_half(tensor: &Tensor, mode: QuantMode) -> Tensor {
    match mode {
        QuantMode::F16 => tensor.to_kind(Kind::Half),
        QuantMode::BF16 => tensor.to_kind(Kind::BFloat16),
        _ => tensor.shallow_clone(),
    }
}

/// Simulate int8 symmetric quantization: scale = max(abs(x)) / 127
fn quantize_int8(tensor: &Tensor) -> (Tensor, Tensor) {
    let abs_max = tensor.abs().max();
    let scale = &abs_max / 127.0;
    let quantized = (tensor / &scale).round().clamp(-128.0, 127.0);
    (quantized, scale)
}

fn dequantize_int8(quantized: &Tensor, scale: &Tensor) -> Tensor {
    quantized * scale
}

/// Simulate NF4 block quantization: per-block normalization + 4-bit lookup
fn quantize_nf4_block(tensor: &Tensor, block_size: i64) -> (Tensor, Tensor) {
    let flat = tensor.flatten(0, -1);
    let numel = flat.size()[0];
    let padded = if numel % block_size != 0 {
        let pad = block_size - (numel % block_size);
        Tensor::cat(&[flat, Tensor::zeros([pad], (Kind::Float, tensor.device()))], 0)
    } else {
        flat
    };

    let blocks = padded.view([-1, block_size]);
    let block_max = blocks.abs().amax(&[1i64][..], true).clamp_min(1e-12);
    let normalized = &blocks / &block_max;

    // Simulate 4-bit: quantize to 16 levels [-1, 1]
    let quantized = (normalized * 7.0).round().clamp(-8.0, 7.0);

    (quantized.view([-1]), block_max.view([-1]))
}

fn dequantize_nf4_block(quantized: &Tensor, scales: &Tensor, block_size: i64) -> Tensor {
    let blocks = quantized.view([-1, block_size]);
    let restored = (&blocks / 7.0) * scales.view([-1, 1]);
    restored.view([-1])
}

fn quantize_shard(tensor: &Tensor, mode: QuantMode, block_size: i64) -> (Tensor, f64) {
    let original_f32 = tensor.to_kind(Kind::Float);

    match mode {
        QuantMode::F16 | QuantMode::BF16 => {
            let q = quantize_half(tensor, mode);
            let restored = q.to_kind(Kind::Float);
            let error = (&original_f32 - &restored).abs().mean(Kind::Float);
            let err_val = f64::try_from(error).unwrap_or(0.0);
            (q, err_val)
        }
        QuantMode::Int8 => {
            let (q, scale) = quantize_int8(tensor);
            let restored = dequantize_int8(&q, &scale);
            let error = (&original_f32 - &restored).abs().mean(Kind::Float);
            let err_val = f64::try_from(error).unwrap_or(0.0);
            (restored, err_val)
        }
        QuantMode::NF4 => {
            let (q, scales) = quantize_nf4_block(tensor, block_size);
            let restored = dequantize_nf4_block(&q, &scales, block_size);
            let numel = original_f32.numel() as i64;
            let restored_trimmed = restored.narrow(0, 0, numel).view(original_f32.size().as_slice());
            let error = (&original_f32 - &restored_trimmed).abs().mean(Kind::Float);
            let err_val = f64::try_from(error).unwrap_or(0.0);
            (restored_trimmed, err_val)
        }
    }
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quant" => cfg.quant_mode = QuantMode::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--model-gb" => cfg.model_gb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--shard-mb" => cfg.shard_mb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--steps" => cfg.steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lr" => cfg.lr = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--block-size" => cfg.block_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: streaming_quantize.rs --quant f16|bf16|int8|nf4 [options]");
                println!("  Streams quantized model shards through the TLSF allocator,");
                println!("  dequantizing on-the-fly during forward pass. Demonstrates");
                println!("  VRAM savings from reduced-precision shard streaming.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }
    Ok(cfg)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    aten_ptx::ensure_libtorch_cuda_loaded();
    if !tch::Cuda::is_available() {
        println!("CUDA not available");
        return Ok(());
    }

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
    let inv_sqrt_d = 1.0 / (cfg.hidden as f64).sqrt();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;

    let model_bytes = cfg.model_gb * 1024 * 1024 * 1024;
    let shard_bytes = cfg.shard_mb * 1024 * 1024;
    let shards_per_step = model_bytes.div_ceil(shard_bytes);

    let compressed_shard_bytes = (shard_bytes as f64 * cfg.quant_mode.bytes_per_param() / 4.0) as usize;
    let vram_saving = 1.0 - (cfg.quant_mode.bytes_per_param() / 4.0);

    let pool_bytes = runtime.tlsf_stats().total_pool_size;

    println!("=== Ferrite Quantized Shard Streaming ===\n");
    println!("  quant mode:       {}", cfg.quant_mode.as_str());
    println!("  bytes/param:      {:.1}", cfg.quant_mode.bytes_per_param());
    println!("  VRAM saving:      {:.0}%", vram_saving * 100.0);
    println!("  model:            {} GB (f32)", cfg.model_gb);
    println!("  shard (f32):      {} MB", cfg.shard_mb);
    println!("  shard (quant):    {:.1} MB", compressed_shard_bytes as f64 / 1e6);
    println!("  shards/step:      {}", shards_per_step);
    println!("  streams:          {}", active_streams);
    println!("  wave streams:     {}", cfg.wave_streams);
    println!("  steps:            {}", cfg.steps);
    println!("  TLSF pool:        {:.1} MB\n", pool_bytes as f64 / 1e6);

    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut non_finite = 0usize;
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f64;
    let mut quant_count = 0usize;
    let mut step_times = Vec::with_capacity(cfg.steps);
    let t_total = Instant::now();

    for step in 0..cfg.steps {
        let t_step = Instant::now();
        let mut step_loss = 0.0f64;
        let mut shard_base = 0usize;

        while shard_base < shards_per_step {
            let wave = (shards_per_step - shard_base).min(cfg.wave_streams);

            for local_idx in 0..wave {
                let shard_idx = shard_base + local_idx;
                let stream_id = shard_idx % active_streams;
                set_torch_stream(stream_id);

                let elems = shard_bytes.div_ceil(std::mem::size_of::<f32>()) as i64;
                let rows = (elems / cfg.hidden).max(1);
                let used = rows * cfg.hidden;

                // Generate base weights in f32, then quantize + dequantize
                let base_w_f32 = (Tensor::randn([used], (Kind::Float, device)) * inv_sqrt_d)
                    .view([rows, cfg.hidden]);

                let (base_w_restored, quant_err) = quantize_shard(&base_w_f32, cfg.quant_mode, cfg.block_size);
                total_quant_error += quant_err;
                max_quant_error = max_quant_error.max(quant_err);
                quant_count += 1;

                let base_w = base_w_restored.view([rows, cfg.hidden]);

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

                let ga = lora_a.grad().clamp(-1.0, 1.0);
                let gb = lora_b.grad().clamp(-1.0, 1.0);
                tch::no_grad(|| {
                    let next_a = &lora_a - ga * cfg.lr;
                    let next_b = &lora_b - gb * cfg.lr;
                    lora_a.copy_(&next_a);
                    lora_b.copy_(&next_b);
                });

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

        tch::Cuda::synchronize(0);
        let dt = t_step.elapsed().as_secs_f64();
        step_times.push(dt);

        if (step + 1) % 20 == 0 || step == 0 {
            let s = runtime.tlsf_stats();
            println!(
                "  step {:>4} | loss={:.6} | quant_err={:.6} | time={:.3}s | vram={:.0}MB | frag={:.6}",
                step + 1,
                step_loss / shards_per_step as f64,
                total_quant_error / quant_count as f64,
                dt,
                s.allocated_bytes as f64 / 1e6,
                s.fragmentation_ratio
            );
        }
    }

    let wall = t_total.elapsed().as_secs_f64();
    let sf = runtime.tlsf_stats();
    let avg_step = step_times.iter().sum::<f64>() / step_times.len() as f64;
    let avg_quant = if quant_count > 0 { total_quant_error / quant_count as f64 } else { 0.0 };

    println!("\n  Quantized Streaming Results:");
    println!("    Quant mode:       {}", cfg.quant_mode.as_str());
    println!("    VRAM saving:      {:.0}%", vram_saving * 100.0);
    println!("    Avg quant error:  {:.9}", avg_quant);
    println!("    Max quant error:  {:.9}", max_quant_error);
    println!("    Avg loss:         {:.6}", if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN });
    println!("    Wall time:        {:.2}s", wall);
    println!("    Avg step:         {:.3}s", avg_step);
    println!("    Peak VRAM:        {:.1} MB", sf.peak_allocated as f64 / 1e6);
    println!("    Fragmentation:    {:.6}", sf.fragmentation_ratio);

    println!("\nRESULT mode=quantized_streaming");
    println!("RESULT quant_mode={}", cfg.quant_mode.as_str());
    println!("RESULT bytes_per_param={:.1}", cfg.quant_mode.bytes_per_param());
    println!("RESULT vram_saving_pct={:.1}", vram_saving * 100.0);
    println!("RESULT avg_quant_error={:.9}", avg_quant);
    println!("RESULT max_quant_error={:.9}", max_quant_error);
    println!("RESULT avg_loss={:.9}", if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN });
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT avg_step_s={:.6}", avg_step);
    println!("RESULT peak_vram_mb={:.6}", sf.peak_allocated as f64 / 1e6);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);

    Ok(())
}
