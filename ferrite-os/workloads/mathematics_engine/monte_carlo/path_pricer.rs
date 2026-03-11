// =============================================================================
// Ferrite Mathematics Engine — Monte Carlo Path Pricer
// =============================================================================
//
// Streams millions of Monte Carlo paths through the TLSF pool in wave-scheduled
// batches. Each wave allocates a shard of paths, simulates them, reduces payoffs,
// then frees the shard — so we can price options with path counts that would
// normally exceed total VRAM.
//
// Supports: European, Asian (arithmetic avg), Barrier (up-and-out), Lookback
//
// Usage:
//   ferrite-run mathematics_engine/monte_carlo/path_pricer.rs \
//     --paths 50000000 --steps 252 --option asian --strike 100.0 --spot 100.0
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
    total_paths: usize,
    time_steps: usize,
    wave_streams: usize,
    streams: u32,
    // Market parameters
    spot: f64,
    strike: f64,
    risk_free: f64,
    volatility: f64,
    maturity: f64,
    // Option type
    option_type: OptionType,
    barrier: f64,
    // Shard sizing
    paths_per_shard: usize,
}

#[derive(Clone, Copy)]
enum OptionType {
    European,
    Asian,
    BarrierUpOut,
    Lookback,
}

impl OptionType {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "european" => Ok(Self::European),
            "asian" => Ok(Self::Asian),
            "barrier" | "barrier-up-out" => Ok(Self::BarrierUpOut),
            "lookback" => Ok(Self::Lookback),
            _ => Err(anyhow::anyhow!("unknown option type: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::European => "european",
            Self::Asian => "asian",
            Self::BarrierUpOut => "barrier_up_out",
            Self::Lookback => "lookback",
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            total_paths: 10_000_000,
            time_steps: 252,
            wave_streams: 4,
            streams: 32,
            spot: 100.0,
            strike: 100.0,
            risk_free: 0.05,
            volatility: 0.2,
            maturity: 1.0,
            option_type: OptionType::European,
            barrier: 130.0,
            paths_per_shard: 500_000,
        }
    }
}

// ---------------------------------------------------------------------------
// GBM path simulation on a shard of paths
// ---------------------------------------------------------------------------

fn simulate_paths(
    n_paths: i64,
    n_steps: i64,
    spot: f64,
    _dt: f64,
    drift: f64,
    vol_sqrt_dt: f64,
    device: Device,
    stream_id: usize,
) -> Tensor {
    set_torch_stream(stream_id);

    // Z ~ N(0,1) for each path × step
    let z = Tensor::randn([n_paths, n_steps], (Kind::Float, device));

    // log returns: (r - 0.5σ²)dt + σ√dt·Z
    let log_returns = &z * vol_sqrt_dt + drift;

    // Cumulative sum of log returns → log(S/S0)
    let cum_log = log_returns.cumsum(1, Kind::Float);

    // Prepend log(1) = 0 for the initial spot
    let zero_col = Tensor::zeros([n_paths, 1], (Kind::Float, device));
    let full_log = Tensor::cat(&[&zero_col, &cum_log], 1);

    // S(t) = S0 * exp(cumulative log returns)
    full_log.exp() * spot
}

// ---------------------------------------------------------------------------
// Payoff computation per option type
// ---------------------------------------------------------------------------

fn compute_payoffs(
    paths: &Tensor,
    option_type: OptionType,
    strike: f64,
    barrier: f64,
) -> Tensor {
    match option_type {
        OptionType::European => {
            // payoff = max(S_T - K, 0)
            let terminal = paths.select(1, paths.size()[1] - 1);
            (terminal - strike).clamp_min(0.0)
        }
        OptionType::Asian => {
            // payoff = max(avg(S) - K, 0) — arithmetic average
            let avg_price = paths.mean_dim(1, false, Kind::Float);
            (avg_price - strike).clamp_min(0.0)
        }
        OptionType::BarrierUpOut => {
            // European call that knocks out if S ever exceeds barrier
            let terminal = paths.select(1, paths.size()[1] - 1);
            let max_price = paths.amax(1, false);
            let alive = max_price.le(barrier).to_kind(Kind::Float);
            (terminal - strike).clamp_min(0.0) * alive
        }
        OptionType::Lookback => {
            // Floating strike lookback call: payoff = S_T - min(S)
            let terminal = paths.select(1, paths.size()[1] - 1);
            let min_price = paths.amin(1, false);
            (terminal - min_price).clamp_min(0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--paths" => cfg.total_paths = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--steps" => cfg.time_steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--spot" => cfg.spot = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--strike" => cfg.strike = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--rate" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--vol" => cfg.volatility = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--maturity" => cfg.maturity = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--option" => cfg.option_type = OptionType::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--barrier" => cfg.barrier = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--paths-per-shard" => cfg.paths_per_shard = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: path_pricer.rs [options]");
                println!();
                println!("  Monte Carlo option pricing with shard-streamed paths through TLSF.");
                println!();
                println!("  --paths N           Total simulation paths (default: 10000000)");
                println!("  --steps N           Time steps per path (default: 252)");
                println!("  --option TYPE       european|asian|barrier|lookback (default: european)");
                println!("  --spot F            Spot price (default: 100.0)");
                println!("  --strike F          Strike price (default: 100.0)");
                println!("  --rate F            Risk-free rate (default: 0.05)");
                println!("  --vol F             Volatility (default: 0.2)");
                println!("  --maturity F        Time to maturity in years (default: 1.0)");
                println!("  --barrier F         Barrier level for barrier options (default: 130.0)");
                println!("  --paths-per-shard N Paths per streaming shard (default: 500000)");
                println!("  --wave-streams N    Concurrent shard streams (default: 4)");
                println!("  --streams N         CUDA stream pool size (default: 32)");
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

    // Derived parameters
    let dt = cfg.maturity / cfg.time_steps as f64;
    let drift = (cfg.risk_free - 0.5 * cfg.volatility * cfg.volatility) * dt;
    let vol_sqrt_dt = cfg.volatility * dt.sqrt();
    let discount = (-cfg.risk_free * cfg.maturity).exp();

    let n_shards = cfg.total_paths.div_ceil(cfg.paths_per_shard);
    let shard_bytes = cfg.paths_per_shard * (cfg.time_steps + 1) * 4; // float32
    let total_path_bytes = cfg.total_paths as f64 * (cfg.time_steps + 1) as f64 * 4.0;
    let streaming_budget = cfg.wave_streams as f64 * shard_bytes as f64;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Monte Carlo Path Pricer ===\n");
    println!("  option type:      {}", cfg.option_type.as_str());
    println!("  total paths:      {}", cfg.total_paths);
    println!("  time steps:       {}", cfg.time_steps);
    println!("  paths/shard:      {}", cfg.paths_per_shard);
    println!("  total shards:     {}", n_shards);
    println!("  wave streams:     {}", cfg.wave_streams);
    println!("  active streams:   {}", active_streams);
    println!();
    println!("  Market: S0={:.2} K={:.2} r={:.4} σ={:.4} T={:.2}", cfg.spot, cfg.strike, cfg.risk_free, cfg.volatility, cfg.maturity);
    if matches!(cfg.option_type, OptionType::BarrierUpOut) {
        println!("  Barrier:  {:.2} (up-and-out)", cfg.barrier);
    }
    println!();
    println!("  TLSF pool:        {:.1} MB", pool_mb);
    println!("  shard size:       {:.1} MB ({} paths × {} steps × 4B)",
        shard_bytes as f64 / 1e6, cfg.paths_per_shard, cfg.time_steps + 1);
    println!("  total path data:  {:.2} GB (NEVER fully in VRAM)",
        total_path_bytes / 1e9);
    println!("  streaming budget: {:.1} MB (wave_streams × shard)",
        streaming_budget / 1e6);
    println!();

    // --- Wave-scheduled Monte Carlo ---
    let t_start = Instant::now();
    let mut payoff_sum = 0.0f64;
    let mut payoff_sq_sum = 0.0f64;
    let mut paths_computed = 0usize;
    let mut total_allocs = 0u64;

    for wave_start in (0..n_shards).step_by(cfg.wave_streams) {
        let wave_end = (wave_start + cfg.wave_streams).min(n_shards);
        let wave_size = wave_end - wave_start;

        // Launch concurrent path simulations across streams
        let mut shard_payoffs: Vec<(f64, f64, usize)> = Vec::with_capacity(wave_size);

        for w in 0..wave_size {
            let shard_idx = wave_start + w;
            let remaining = cfg.total_paths - shard_idx * cfg.paths_per_shard;
            let n_paths = remaining.min(cfg.paths_per_shard) as i64;
            let stream_id = w % active_streams;

            // Simulate paths (allocated from TLSF)
            let paths = simulate_paths(
                n_paths,
                cfg.time_steps as i64,
                cfg.spot,
                dt,
                drift,
                vol_sqrt_dt,
                device,
                stream_id,
            );

            // Compute payoffs
            let payoffs = compute_payoffs(&paths, cfg.option_type, cfg.strike, cfg.barrier);

            // Reduce to scalar stats
            let sum = f64::try_from(payoffs.sum(Kind::Float)).unwrap_or(0.0);
            let sq_sum = f64::try_from(payoffs.pow_tensor_scalar(2.0).sum(Kind::Float)).unwrap_or(0.0);

            shard_payoffs.push((sum, sq_sum, n_paths as usize));
            // paths tensor drops here → TLSF frees the shard
        }

        sync_all_streams();

        // Accumulate
        for (sum, sq, n) in &shard_payoffs {
            payoff_sum += sum;
            payoff_sq_sum += sq;
            paths_computed += n;
        }

        let s = runtime.tlsf_stats();
        total_allocs = s.total_allocations;

        if wave_start == 0 || (wave_start + cfg.wave_streams) % (cfg.wave_streams * 10) == 0 {
            let pct = (paths_computed as f64 / cfg.total_paths as f64) * 100.0;
            println!("  wave {:>4} | {:.1}% done | peak={:.0}MB frag={:.6}",
                wave_start / cfg.wave_streams + 1, pct,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let wall_time = t_start.elapsed().as_secs_f64();

    // --- Compute results ---
    let n = paths_computed as f64;
    let mean_payoff = payoff_sum / n;
    let price = discount * mean_payoff;

    // Standard error via variance of payoffs
    let variance = (payoff_sq_sum / n) - (mean_payoff * mean_payoff);
    let std_err = discount * (variance / n).sqrt();

    // 95% confidence interval
    let ci_low = price - 1.96 * std_err;
    let ci_high = price + 1.96 * std_err;

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;
    let paths_per_sec = paths_computed as f64 / wall_time;

    println!();
    println!("  Pricing Results:");
    println!("    Option type:     {}", cfg.option_type.as_str());
    println!("    Price:           {:.6}", price);
    println!("    Std error:       {:.6}", std_err);
    println!("    95% CI:          [{:.6}, {:.6}]", ci_low, ci_high);
    println!("    Mean payoff:     {:.6}", mean_payoff);
    println!("    Discount:        {:.6}", discount);
    println!();
    println!("    Paths computed:  {}", paths_computed);
    println!("    Shards streamed: {}", n_shards);
    println!("    Alloc events:    {}", total_allocs);
    println!("    Wall time:       {:.3}s", wall_time);
    println!("    Throughput:      {:.0} paths/s ({:.1}M paths/s)", paths_per_sec, paths_per_sec / 1e6);
    println!();
    println!("    Peak VRAM:       {:.1} MB", peak_mb);
    println!("    Path data total: {:.2} GB (virtual — streamed)", total_path_bytes / 1e9);
    println!("    VRAM savings:    {:.0}x", total_path_bytes / 1e6 / peak_mb);
    println!("    Fragmentation:   {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:    {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=monte_carlo_path_pricer");
    println!("RESULT option_type={}", cfg.option_type.as_str());
    println!("RESULT price={:.9}", price);
    println!("RESULT std_error={:.9}", std_err);
    println!("RESULT ci_low={:.9}", ci_low);
    println!("RESULT ci_high={:.9}", ci_high);
    println!("RESULT paths={}", paths_computed);
    println!("RESULT shards={}", n_shards);
    println!("RESULT wall_s={:.6}", wall_time);
    println!("RESULT paths_per_sec={:.0}", paths_per_sec);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT total_data_gb={:.6}", total_path_bytes / 1e9);
    println!("RESULT vram_savings_x={:.1}", total_path_bytes / 1e6 / peak_mb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
