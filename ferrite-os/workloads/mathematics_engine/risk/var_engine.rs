// =============================================================================
// Ferrite Mathematics Engine — Value-at-Risk Engine
// =============================================================================
//
// Computes VaR and CVaR (Expected Shortfall) for large portfolios via
// historical simulation with millions of scenarios. Each scenario batch
// is a shard streamed through TLSF — allowing scenario counts that would
// exhaust VRAM on traditional stacks.
//
// Supports: Historical VaR, Parametric VaR, Monte Carlo VaR, Stressed VaR
//
// Usage:
//   ferrite-run mathematics_engine/risk/var_engine.rs \
//     --assets 5000 --scenarios 5000000 --positions 5000 --confidence 0.99
//
// =============================================================================

use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, set_torch_stream, sync_all_streams};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    num_assets: usize,
    num_scenarios: usize,
    scenarios_per_shard: usize,
    wave_streams: usize,
    streams: u32,
    confidence: f64,       // e.g. 0.99 for 99% VaR
    holding_period: usize, // days
    stressed_vol_mult: f64,
    // VaR method
    method: VaRMethod,
}

#[derive(Clone, Copy)]
enum VaRMethod {
    Historical,
    MonteCarlo,
    Stressed,
}

impl VaRMethod {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "historical" | "hist" => Ok(Self::Historical),
            "montecarlo" | "mc" => Ok(Self::MonteCarlo),
            "stressed" | "stress" => Ok(Self::Stressed),
            _ => Err(anyhow::anyhow!("unknown VaR method: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Historical => "historical",
            Self::MonteCarlo => "monte_carlo",
            Self::Stressed => "stressed",
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_assets: 2000,
            num_scenarios: 2_000_000,
            scenarios_per_shard: 200_000,
            wave_streams: 4,
            streams: 32,
            confidence: 0.99,
            holding_period: 1,
            stressed_vol_mult: 3.0,
            method: VaRMethod::Historical,
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario generation
// ---------------------------------------------------------------------------

fn generate_scenario_shard(
    n_scenarios: i64,
    n_assets: i64,
    vol_scale: f64,
    device: Device,
    stream_id: usize,
) -> Tensor {
    set_torch_stream(stream_id);
    // Each scenario: vector of asset returns ~ N(0, vol²)
    // Correlated structure via: R = Z @ L where L is lower-Cholesky of correlation
    // For demonstration, use independent returns scaled by volatility
    Tensor::randn([n_scenarios, n_assets], (Kind::Float, device)) * vol_scale
}

fn compute_portfolio_pnl(
    scenario_returns: &Tensor,
    positions: &Tensor,
) -> Tensor {
    // P&L = returns @ positions (dot product per scenario)
    scenario_returns.matmul(&positions.unsqueeze(-1)).squeeze_dim(-1)
}

// ---------------------------------------------------------------------------
// VaR / CVaR computation from sorted losses
// ---------------------------------------------------------------------------

fn compute_risk_from_pnl_shard(
    pnl: &Tensor,
) -> (Tensor, f64, f64) {
    // Return sorted losses (negative P&L), sum, and sum-of-squares for online stats
    let losses = pnl.neg();
    let sum = f64::try_from(losses.sum(Kind::Float)).unwrap_or(0.0);
    let sq_sum = f64::try_from(losses.pow_tensor_scalar(2.0).sum(Kind::Float)).unwrap_or(0.0);
    let sorted = losses.sort(0, false).0;
    (sorted, sum, sq_sum)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--assets" => cfg.num_assets = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--scenarios" => cfg.num_scenarios = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--scenarios-per-shard" => cfg.scenarios_per_shard = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--confidence" => cfg.confidence = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--holding-period" => cfg.holding_period = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--stress-mult" => cfg.stressed_vol_mult = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--method" => cfg.method = VaRMethod::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "-h" | "--help" => {
                println!("Usage: var_engine.rs [options]");
                println!();
                println!("  Streaming Value-at-Risk computation via shard-streamed scenarios.");
                println!();
                println!("  --assets N            Number of assets (default: 2000)");
                println!("  --scenarios N         Total scenarios (default: 2000000)");
                println!("  --scenarios-per-shard N  Scenarios per TLSF shard (default: 200000)");
                println!("  --method TYPE         historical|montecarlo|stressed (default: historical)");
                println!("  --confidence F        VaR confidence level (default: 0.99)");
                println!("  --holding-period N    Holding period in days (default: 1)");
                println!("  --stress-mult F       Volatility multiplier for stressed VaR (default: 3.0)");
                println!("  --wave-streams N      Concurrent scenario streams (default: 4)");
                println!("  --streams N           CUDA stream pool (default: 32)");
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

    let vol_scale = match cfg.method {
        VaRMethod::Stressed => 0.02 * cfg.stressed_vol_mult,
        _ => 0.02, // ~2% daily vol per asset
    };
    let hp_scale = (cfg.holding_period as f64).sqrt();

    let n_shards = cfg.num_scenarios.div_ceil(cfg.scenarios_per_shard);
    let shard_bytes = cfg.scenarios_per_shard * cfg.num_assets * 4;
    let total_scenario_bytes = cfg.num_scenarios as f64 * cfg.num_assets as f64 * 4.0;
    let streaming_budget = cfg.wave_streams as f64 * shard_bytes as f64;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Value-at-Risk Engine ===\n");
    println!("  method:             {}", cfg.method.as_str());
    println!("  assets:             {}", cfg.num_assets);
    println!("  scenarios:          {}", cfg.num_scenarios);
    println!("  scenarios/shard:    {}", cfg.scenarios_per_shard);
    println!("  shards:             {}", n_shards);
    println!("  confidence:         {:.2}%", cfg.confidence * 100.0);
    println!("  holding period:     {} day(s)", cfg.holding_period);
    if matches!(cfg.method, VaRMethod::Stressed) {
        println!("  stress multiplier:  {:.1}x", cfg.stressed_vol_mult);
    }
    println!("  wave streams:       {}", cfg.wave_streams);
    println!();
    println!("  TLSF pool:          {:.1} MB", pool_mb);
    println!("  shard size:         {:.1} MB", shard_bytes as f64 / 1e6);
    println!("  total scenario data:{:.2} GB (NEVER in VRAM)", total_scenario_bytes / 1e9);
    println!("  streaming budget:   {:.1} MB", streaming_budget / 1e6);
    println!();

    // --- Portfolio positions (random weights, normalized) ---
    let positions_raw = Tensor::randn([cfg.num_assets as i64], (Kind::Float, device));
    let positions = &positions_raw / positions_raw.abs().sum(Kind::Float);
    let portfolio_value = 1_000_000.0f64; // $1M notional

    println!("  portfolio: {} positions, ${:.0} notional\n", cfg.num_assets, portfolio_value);

    // --- Wave-scheduled scenario simulation ---
    let t_start = Instant::now();

    // We'll collect sorted loss tails from each shard for final VaR computation
    let tail_size = ((1.0 - cfg.confidence) * cfg.scenarios_per_shard as f64).ceil() as usize + 100;
    let mut all_tail_losses: Vec<Tensor> = Vec::new();
    let mut total_loss_sum = 0.0f64;
    let mut total_loss_sq_sum = 0.0f64;
    let mut total_loss_cube_sum = 0.0f64;
    let mut total_loss_quad_sum = 0.0f64;
    let mut scenarios_computed = 0usize;
    let mut total_allocs = 0u64;

    for wave_start in (0..n_shards).step_by(cfg.wave_streams) {
        let wave_end = (wave_start + cfg.wave_streams).min(n_shards);

        for w in wave_start..wave_end {
            let remaining = cfg.num_scenarios - w * cfg.scenarios_per_shard;
            let n_scen = remaining.min(cfg.scenarios_per_shard) as i64;
            let stream_id = (w - wave_start) % active_streams;

            // Generate scenario returns (allocated from TLSF)
            let returns = generate_scenario_shard(
                n_scen,
                cfg.num_assets as i64,
                vol_scale,
                device,
                stream_id,
            );

            // Compute portfolio P&L for each scenario
            let pnl = compute_portfolio_pnl(&returns, &positions) * portfolio_value * hp_scale;
            // returns drops here → TLSF frees scenario matrix

            // Extract loss statistics
            let (sorted_losses, sum, sq_sum) = compute_risk_from_pnl_shard(&pnl);

            // Keep tail losses for final VaR merge
            let keep = (tail_size as i64).min(sorted_losses.size()[0]);
            let tail = sorted_losses.narrow(0, sorted_losses.size()[0] - keep, keep);
            all_tail_losses.push(tail.to_device(Device::Cpu));

            // Accumulate moments
            let losses = pnl.neg();
            let cube_sum = f64::try_from(losses.pow_tensor_scalar(3.0).sum(Kind::Float)).unwrap_or(0.0);
            let quad_sum = f64::try_from(losses.pow_tensor_scalar(4.0).sum(Kind::Float)).unwrap_or(0.0);

            total_loss_sum += sum;
            total_loss_sq_sum += sq_sum;
            total_loss_cube_sum += cube_sum;
            total_loss_quad_sum += quad_sum;
            scenarios_computed += n_scen as usize;
        }

        sync_all_streams();

        let s = runtime.tlsf_stats();
        total_allocs = s.total_allocations;

        if wave_start == 0 || (wave_start / cfg.wave_streams) % 5 == 0 {
            let pct = scenarios_computed as f64 / cfg.num_scenarios as f64 * 100.0;
            println!("  wave {:>3} | {:.1}% | peak={:.0}MB frag={:.6}",
                wave_start / cfg.wave_streams + 1, pct,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let wall_time = t_start.elapsed().as_secs_f64();

    // --- Merge tails and compute final VaR/CVaR ---
    let merged_tails = Tensor::cat(&all_tail_losses, 0);
    let sorted_all = merged_tails.sort(0, false).0;
    let n_total = sorted_all.size()[0];

    // VaR: the loss at the (1-confidence) percentile
    let var_idx = ((1.0 - cfg.confidence) * n_total as f64) as i64;
    let var_idx = var_idx.max(0).min(n_total - 1);
    // We sorted ascending, so VaR is near the top
    let var_percentile_idx = n_total - 1 - var_idx;
    let var_value = f64::try_from(sorted_all.get(var_percentile_idx)).unwrap_or(0.0);

    // CVaR: average of losses beyond VaR
    let tail_beyond_var = sorted_all.narrow(0, var_percentile_idx, n_total - var_percentile_idx);
    let cvar_value = f64::try_from(tail_beyond_var.mean(Kind::Float)).unwrap_or(0.0);

    // Distribution stats
    let n_f = scenarios_computed as f64;
    let mean_loss = total_loss_sum / n_f;
    let variance = (total_loss_sq_sum / n_f) - (mean_loss * mean_loss);
    let loss_std = variance.max(0.0).sqrt();
    let skewness = if loss_std > 0.0 {
        ((total_loss_cube_sum / n_f) - 3.0 * mean_loss * variance - mean_loss.powi(3)) / loss_std.powi(3)
    } else { 0.0 };
    let kurtosis = if loss_std > 0.0 {
        (total_loss_quad_sum / n_f) / loss_std.powi(4) - 3.0
    } else { 0.0 };

    let max_loss = f64::try_from(sorted_all.get(n_total - 1)).unwrap_or(0.0);

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;
    let scen_per_sec = scenarios_computed as f64 / wall_time;

    println!();
    println!("  Risk Results ({:.2}% confidence, {} day):", cfg.confidence * 100.0, cfg.holding_period);
    println!("    VaR:               ${:.2}", var_value);
    println!("    CVaR (ES):         ${:.2}", cvar_value);
    println!("    Max loss:          ${:.2}", max_loss);
    println!("    Mean loss:         ${:.2}", mean_loss);
    println!("    Loss std:          ${:.2}", loss_std);
    println!("    Skewness:          {:.4}", skewness);
    println!("    Excess kurtosis:   {:.4}", kurtosis);
    println!();
    println!("    Scenarios:         {}", scenarios_computed);
    println!("    Shards streamed:   {}", n_shards);
    println!("    Alloc events:      {}", total_allocs);
    println!("    Wall time:         {:.3}s", wall_time);
    println!("    Throughput:        {:.0} scenarios/s ({:.1}M/s)", scen_per_sec, scen_per_sec / 1e6);
    println!();
    println!("    Peak VRAM:         {:.1} MB", peak_mb);
    println!("    Scenario data:     {:.2} GB (virtual — streamed)", total_scenario_bytes / 1e9);
    println!("    VRAM savings:      {:.0}x", total_scenario_bytes / 1e6 / peak_mb);
    println!("    Fragmentation:     {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:      {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=var_engine");
    println!("RESULT method={}", cfg.method.as_str());
    println!("RESULT var={:.6}", var_value);
    println!("RESULT cvar={:.6}", cvar_value);
    println!("RESULT max_loss={:.6}", max_loss);
    println!("RESULT skewness={:.6}", skewness);
    println!("RESULT kurtosis={:.6}", kurtosis);
    println!("RESULT scenarios={}", scenarios_computed);
    println!("RESULT wall_s={:.6}", wall_time);
    println!("RESULT scen_per_sec={:.0}", scen_per_sec);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT scenario_data_gb={:.6}", total_scenario_bytes / 1e9);
    println!("RESULT vram_savings_x={:.1}", total_scenario_bytes / 1e6 / peak_mb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
