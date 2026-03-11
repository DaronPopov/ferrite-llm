use std::env;
use std::time::Instant;

use anyhow::{anyhow, Result};
use aten_ptx::{init_pytorch_tlsf_ex, reset_torch_stream, set_torch_stream, sync_all_streams};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{no_grad, Device, Kind, Tensor};

/// Tiny Jetson-friendly fine-tune configuration.
#[derive(Clone)]
struct Config {
    steps: usize,
    batch_size: i64,
    hidden: i64,
    streams: u32,
    learning_rate: f64,
    pool_fraction: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            steps: 3,
            batch_size: 16,
            hidden: 128,
            streams: 8,
            learning_rate: 1e-2,
            pool_fraction: 0.15,
        }
    }
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                cfg.steps = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --steps"))?
                    .parse()?;
            }
            "--batch-size" => {
                cfg.batch_size = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --batch-size"))?
                    .parse()?;
            }
            "--hidden" => {
                cfg.hidden = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --hidden"))?
                    .parse()?;
            }
            "--streams" => {
                cfg.streams = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --streams"))?
                    .parse()?;
            }
            "--lr" => {
                cfg.learning_rate = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --lr"))?
                    .parse()?;
            }
            "--pool" => {
                cfg.pool_fraction = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --pool"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(anyhow!("unknown argument: {arg}")),
        }
    }

    if cfg.streams == 0 {
        cfg.streams = 1;
    }
    if !(0.01..=0.9).contains(&cfg.pool_fraction) {
        cfg.pool_fraction = cfg.pool_fraction.clamp(0.01, 0.9);
    }

    Ok(cfg)
}

fn print_help() {
    println!("Usage: scripting_finetune.rs [options]");
    println!();
    println!("  --steps N        Number of training steps (default: 3)");
    println!("  --batch-size N   Synthetic batch size (default: 16)");
    println!("  --hidden N       Hidden dimension (default: 128)");
    println!("  --streams N      PTX stream pool size (default: 8)");
    println!("  --lr F           Learning rate (default: 1e-2)");
    println!("  --pool F         TLSF pool fraction (0.01-0.9, default: 0.15)");
    println!("  -h, --help       Show this help");
}

fn describe_runs(start: Instant, cfg: &Config, loss: f64) {
    let elapsed = start.elapsed().as_secs_f64();
    println!();
    println!("Jetson scripting_finetune summary:");
    println!("  steps:         {}", cfg.steps);
    println!("  batch size:    {}", cfg.batch_size);
    println!("  hidden dim:    {}", cfg.hidden);
    println!("  streams:       {}", cfg.streams);
    println!("  pool fraction: {:.2}", cfg.pool_fraction);
    println!("  avg loss:      {:.6}", loss / cfg.steps as f64);
    println!("  duration:      {:.3}s", elapsed);
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: cfg.pool_fraction,
        fixed_pool_size: 0,
        reserve_vram: 0,
        max_streams: cfg.streams,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };

    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, cfg.pool_fraction as f64, cfg.streams)
        .map_err(|e| anyhow!(e))?;

    let device = Device::Cuda(0);
    let mut weights = Tensor::zeros(&[cfg.hidden, 1], (Kind::Float, device)).set_requires_grad(true);
    let mut bias = Tensor::zeros(&[1], (Kind::Float, device)).set_requires_grad(true);

    println!("Training a lightweight Jetson-friendly model with TLSF-backed PyTorch allocator...");
    let start = Instant::now();
    let mut total_loss = 0.0;

    for step in 0..cfg.steps {
        let stream_id = (step as usize) % cfg.streams as usize;
        set_torch_stream(stream_id);

        let inputs = Tensor::randn(&[cfg.batch_size, cfg.hidden], (Kind::Float, device));
        let targets = Tensor::randn(&[cfg.batch_size, 1], (Kind::Float, device));
        let output = inputs.matmul(&weights) + &bias;
        let loss = (output - targets).pow_tensor_scalar(2.0).mean(Kind::Float);

        total_loss += f64::try_from(&loss)?;
        loss.backward();

        no_grad(|| {
            let w_grad = weights.grad();
            let b_grad = bias.grad();
            weights -= &(w_grad * cfg.learning_rate);
            bias -= &(b_grad * cfg.learning_rate);
            weights.zero_grad();
            bias.zero_grad();
        });

        sync_all_streams();
        reset_torch_stream();
        println!("  step {}/{} - loss: {:.4}", step + 1, cfg.steps, f64::try_from(&loss)?);
    }

    let _ = runtime.sync_all();
    describe_runs(start, &cfg, total_loss);
    Ok(())
}
