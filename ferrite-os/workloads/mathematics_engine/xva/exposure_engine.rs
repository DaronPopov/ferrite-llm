// =============================================================================
// Ferrite Mathematics Engine — XVA Exposure Engine
// =============================================================================
//
// Computes Credit Valuation Adjustment (CVA), Debit Valuation Adjustment (DVA),
// and Funding Valuation Adjustment (FVA) for a large OTC derivatives portfolio.
//
// This requires building an "exposure cube": N_trades × M_scenarios × T_timesteps
// where each cell is the Mark-to-Market value of a trade under a given scenario
// at a given time. The cube is then aggregated along trades (netting sets),
// discounted, and multiplied by default/survival probabilities.
//
// WHY THIS IS INFEASIBLE WITHOUT TLSF:
//   - Exposure cube: 10K trades × 50K scenarios × 100 time steps
//   - Full cube @ float32 = 10,000 × 50,000 × 100 × 4B = 200 GB
//   - Even one time-slice: 10K × 50K × 4B = 2 GB
//   - Must stream time-slices through VRAM, accumulating exposure profiles
//   - Each time step generates and destroys a full trade × scenario matrix
//   - 100 time steps × multiple wave shards = thousands of alloc/free cycles
//   - Each scenario shard within a time step: alloc paths, value trades, reduce, free
//   - The nested loop (time × scenario shards) creates allocation pressure
//     that only O(1) TLSF can sustain without dominating wall time
//
// Usage:
//   ferrite-run mathematics_engine/xva/exposure_engine.rs \
//     --trades 10000 --scenarios 50000 --timesteps 100
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
    num_trades: usize,
    num_scenarios: usize,
    num_timesteps: usize,
    scenarios_per_shard: usize,
    num_netting_sets: usize,
    wave_streams: usize,
    streams: u32,
    // Market/credit parameters
    risk_free: f64,
    hazard_rate_cpty: f64,   // counterparty default intensity
    hazard_rate_own: f64,    // own default intensity
    recovery_rate: f64,
    funding_spread: f64,
    max_maturity: f64,       // years
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_trades: 5000,
            num_scenarios: 20_000,
            num_timesteps: 60,
            scenarios_per_shard: 5_000,
            num_netting_sets: 50,
            wave_streams: 4,
            streams: 32,
            risk_free: 0.03,
            hazard_rate_cpty: 0.02,  // ~2% annual default probability
            hazard_rate_own: 0.01,
            recovery_rate: 0.40,
            funding_spread: 0.005,
            max_maturity: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Trade representation
// ---------------------------------------------------------------------------

struct TradeBook {
    // [num_trades] tensors
    notionals: Tensor,       // trade notionals
    maturities: Tensor,      // individual trade maturities
    strikes: Tensor,         // strike/fixed rates
    is_receiver: Tensor,     // +1 receiver, -1 payer
    netting_set_ids: Tensor, // netting set assignment (i32)
}

fn generate_trade_book(
    num_trades: usize,
    num_netting_sets: usize,
    max_maturity: f64,
    device: Device,
) -> TradeBook {
    let n = num_trades as i64;
    let notionals = Tensor::rand([n], (Kind::Float, device)) * 10_000_000.0 + 1_000_000.0;
    let maturities = Tensor::rand([n], (Kind::Float, device)) * (max_maturity - 1.0) + 1.0;
    let strikes = Tensor::rand([n], (Kind::Float, device)) * 0.06 + 0.01; // 1%-7% fixed rates
    let is_receiver = (Tensor::rand([n], (Kind::Float, device)).ge(0.5)).to_kind(Kind::Float) * 2.0 - 1.0;
    let netting_set_ids = (Tensor::rand([n], (Kind::Float, device)) * num_netting_sets as f64)
        .floor()
        .to_kind(Kind::Int);

    TradeBook { notionals, maturities, strikes, is_receiver, netting_set_ids }
}

// ---------------------------------------------------------------------------
// Forward rate simulation (simplified Hull-White style)
// ---------------------------------------------------------------------------

fn simulate_rates_shard(
    n_scenarios: i64,
    current_time: f64,
    dt: f64,
    risk_free: f64,
    device: Device,
    stream_id: usize,
) -> Tensor {
    set_torch_stream(stream_id);

    // Mean-reverting short rate: r(t) = r0 + mean_reversion * (r0 - r(t-1)) * dt + sigma * sqrt(dt) * Z
    let sigma_r = 0.01; // rate volatility
    let kappa = 0.1;    // mean reversion speed
    let sqrt_dt = dt.sqrt();

    // Generate rate perturbation for this time point
    let z = Tensor::randn([n_scenarios], (Kind::Float, device));
    let rate_shift = z * sigma_r * sqrt_dt;

    // r(t) = r0 * exp(-kappa * t) + mean * (1 - exp(-kappa*t)) + noise
    let decay = (-kappa * current_time).exp();
    Tensor::full([n_scenarios], risk_free * decay, (Kind::Float, device)) + rate_shift
}

// ---------------------------------------------------------------------------
// Trade valuation: simplified interest rate swap MtM
// ---------------------------------------------------------------------------

fn value_trades_at_time(
    book: &TradeBook,
    rates: &Tensor,       // [n_scenarios] simulated rates
    current_time: f64,
    n_scenarios: i64,
    n_trades: i64,
) -> Tensor {
    // For each trade: PV = notional * is_receiver * (strike - rate) * remaining_annuity
    // remaining_annuity ≈ (maturity - t) if t < maturity, else 0
    //
    // Output: [n_scenarios, n_trades] MtM values

    // Remaining time to maturity: [n_trades]
    let remaining = (&book.maturities - current_time).clamp_min(0.0);

    // Simplified annuity factor (linear approximation)
    let annuity = &remaining * 0.97; // rough discount

    // Rate spread per scenario: [n_scenarios, 1] - [1, n_trades] → [n_scenarios, n_trades]
    let rates_2d = rates.unsqueeze(1);  // [scen, 1]
    let strikes_2d = book.strikes.unsqueeze(0); // [1, trades]

    let spread = &strikes_2d - &rates_2d; // [scen, trades]

    // MtM: notional * is_receiver * spread * annuity
    let notionals_2d = book.notionals.unsqueeze(0);    // [1, trades]
    let direction_2d = book.is_receiver.unsqueeze(0);  // [1, trades]
    let annuity_2d = annuity.unsqueeze(0);              // [1, trades]

    let mtm = &notionals_2d * &direction_2d * &spread * &annuity_2d;

    // Zero out expired trades
    let alive = remaining.gt(0.0).unsqueeze(0).to_kind(Kind::Float); // [1, trades]
    mtm * alive
}

// ---------------------------------------------------------------------------
// Netting and exposure aggregation
// ---------------------------------------------------------------------------

fn compute_netting_set_exposure(
    mtm: &Tensor,            // [n_scenarios, n_trades]
    netting_set_ids: &Tensor, // [n_trades]
    num_netting_sets: usize,
    n_scenarios: i64,
) -> (Tensor, Tensor) {
    // Aggregate MtM by netting set, then compute positive (EE) and negative (NEE) exposure
    let mut ee_total = Tensor::zeros([n_scenarios], (Kind::Float, mtm.device()));
    let mut nee_total = Tensor::zeros([n_scenarios], (Kind::Float, mtm.device()));

    for ns in 0..num_netting_sets {
        // Mask for trades in this netting set
        let mask = netting_set_ids.eq(ns as i64).unsqueeze(0).to_kind(Kind::Float); // [1, trades]

        // Net MtM for this netting set: sum of MtM for trades in the set
        let ns_mtm = (mtm * &mask).sum_dim_intlist(1, false, Kind::Float); // [n_scenarios]

        // Positive exposure (what counterparty owes us)
        let ee = ns_mtm.clamp_min(0.0);
        // Negative exposure (what we owe counterparty)
        let nee = ns_mtm.clamp_max(0.0).neg();

        ee_total = ee_total + ee;
        nee_total = nee_total + nee;
    }

    (ee_total, nee_total)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trades" => cfg.num_trades = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--scenarios" => cfg.num_scenarios = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--timesteps" => cfg.num_timesteps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--scenarios-per-shard" => cfg.scenarios_per_shard = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--netting-sets" => cfg.num_netting_sets = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--rate" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hazard-cpty" => cfg.hazard_rate_cpty = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hazard-own" => cfg.hazard_rate_own = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--recovery" => cfg.recovery_rate = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--maturity" => cfg.max_maturity = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: exposure_engine.rs [options]");
                println!();
                println!("  XVA exposure engine: CVA/DVA/FVA for large OTC portfolios.");
                println!("  Streams the exposure cube (trades x scenarios x time) through TLSF.");
                println!();
                println!("  --trades N            Number of trades (default: 5000)");
                println!("  --scenarios N         MC scenarios (default: 20000)");
                println!("  --timesteps N         Time grid points (default: 60)");
                println!("  --scenarios-per-shard N  Per TLSF shard (default: 5000)");
                println!("  --netting-sets N      Netting set count (default: 50)");
                println!("  --rate F              Risk-free rate (default: 0.03)");
                println!("  --hazard-cpty F       Counterparty hazard rate (default: 0.02)");
                println!("  --hazard-own F        Own hazard rate (default: 0.01)");
                println!("  --recovery F          Recovery rate (default: 0.40)");
                println!("  --maturity F          Max trade maturity years (default: 10.0)");
                println!("  --wave-streams N      Concurrent shard streams (default: 4)");
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

    let dt = cfg.max_maturity / cfg.num_timesteps as f64;
    let n_scenario_shards = cfg.num_scenarios.div_ceil(cfg.scenarios_per_shard);

    // Full exposure cube size
    let cube_bytes = cfg.num_trades as f64 * cfg.num_scenarios as f64
        * cfg.num_timesteps as f64 * 4.0;
    let slice_bytes = cfg.num_trades * cfg.num_scenarios * 4; // one time-step
    let shard_bytes = cfg.scenarios_per_shard * cfg.num_trades * 4;
    let streaming_budget = cfg.wave_streams as f64 * shard_bytes as f64;

    // Each time step × each scenario shard: multiple alloc/free
    let total_shard_ops = cfg.num_timesteps * n_scenario_shards;
    let allocs_per_shard = 8; // rates, mtm, exposure tensors, intermediates
    let total_est_allocs = total_shard_ops * allocs_per_shard;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite XVA Exposure Engine ===\n");
    println!("  trades:              {}", cfg.num_trades);
    println!("  scenarios:           {}", cfg.num_scenarios);
    println!("  time steps:          {}", cfg.num_timesteps);
    println!("  netting sets:        {}", cfg.num_netting_sets);
    println!("  dt:                  {:.4} years ({:.0} days)", dt, dt * 365.0);
    println!("  scenario shards:     {}", n_scenario_shards);
    println!();
    println!("  EXPOSURE CUBE:");
    println!("    full cube:         {:.2} GB (trades x scenarios x time x 4B)", cube_bytes / 1e9);
    println!("    per time-slice:    {:.1} MB", slice_bytes as f64 / 1e6);
    println!("    per shard:         {:.1} MB", shard_bytes as f64 / 1e6);
    println!("    streaming budget:  {:.1} MB (wave_streams x shard)", streaming_budget / 1e6);
    println!("    NEVER fully in VRAM — streamed time-slice by time-slice");
    println!();
    println!("  ALLOCATION PRESSURE:");
    println!("    total shard ops:   {} (timesteps x scenario_shards)", total_shard_ops);
    println!("    est. total allocs: {}", total_est_allocs);
    println!("    cudaMalloc cost:   {:.1}s (@1ms)", total_est_allocs as f64 * 1e-3);
    println!("    TLSF cost:         {:.4}s (@0.24us)", total_est_allocs as f64 * 0.24e-6);
    println!();
    println!("  Credit: hazard_cpty={:.4} hazard_own={:.4} recovery={:.2}",
        cfg.hazard_rate_cpty, cfg.hazard_rate_own, cfg.recovery_rate);
    println!("  TLSF pool:           {:.1} MB", pool_mb);
    println!();

    // --- Generate trade book ---
    let book = generate_trade_book(cfg.num_trades, cfg.num_netting_sets, cfg.max_maturity, device);

    // --- Stream exposure cube: time-slice by time-slice ---
    let t_start = Instant::now();

    // Accumulators for exposure profile (on CPU for memory efficiency)
    let mut expected_ee_profile = vec![0.0f64; cfg.num_timesteps];
    let mut expected_nee_profile = vec![0.0f64; cfg.num_timesteps];
    let mut peak_ee = 0.0f64;
    let mut total_allocs = 0u64;

    for t_idx in 0..cfg.num_timesteps {
        let current_time = (t_idx + 1) as f64 * dt;
        let mut ee_sum = 0.0f64;
        let mut nee_sum = 0.0f64;
        let mut scenarios_at_t = 0usize;

        // Stream scenario shards for this time step
        for wave_start in (0..n_scenario_shards).step_by(cfg.wave_streams) {
            let wave_end = (wave_start + cfg.wave_streams).min(n_scenario_shards);

            for w in wave_start..wave_end {
                let remaining = cfg.num_scenarios - w * cfg.scenarios_per_shard;
                let n_scen = remaining.min(cfg.scenarios_per_shard) as i64;
                let stream_id = (w - wave_start) % active_streams;

                // 1. Simulate rates for this time step + shard (alloc from TLSF)
                let rates = simulate_rates_shard(
                    n_scen, current_time, dt, cfg.risk_free, device, stream_id,
                );

                // 2. Value all trades under these scenarios (alloc from TLSF)
                let mtm = value_trades_at_time(
                    &book, &rates, current_time,
                    n_scen, cfg.num_trades as i64,
                );

                // 3. Net by netting set and compute exposure
                let (ee, nee) = compute_netting_set_exposure(
                    &mtm, &book.netting_set_ids, cfg.num_netting_sets, n_scen,
                );

                ee_sum += f64::try_from(ee.sum(Kind::Float)).unwrap_or(0.0);
                nee_sum += f64::try_from(nee.sum(Kind::Float)).unwrap_or(0.0);
                scenarios_at_t += n_scen as usize;
                // rates, mtm, ee, nee all drop here → TLSF frees the shard
            }

            sync_all_streams();
        }

        // Expected exposure at this time point (average over scenarios)
        let mean_ee = ee_sum / scenarios_at_t as f64;
        let mean_nee = nee_sum / scenarios_at_t as f64;
        expected_ee_profile[t_idx] = mean_ee;
        expected_nee_profile[t_idx] = mean_nee;
        if mean_ee > peak_ee {
            peak_ee = mean_ee;
        }

        let s = runtime.tlsf_stats();
        total_allocs = s.total_allocations;

        if t_idx == 0 || (t_idx + 1) % 10 == 0 || t_idx == cfg.num_timesteps - 1 {
            println!("  t={:>3}/{} ({:.2}y) | EE=${:.0} NEE=${:.0} | peak={:.0}MB frag={:.6} allocs={}",
                t_idx + 1, cfg.num_timesteps, current_time,
                mean_ee, mean_nee,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio, s.total_allocations);
        }
    }

    let wall_time = t_start.elapsed().as_secs_f64();

    // --- Compute XVA metrics ---
    let lgd = 1.0 - cfg.recovery_rate;

    // CVA = LGD * integral(discount * survival_cpty * default_cpty * EE dt)
    let mut cva = 0.0f64;
    let mut dva = 0.0f64;
    let mut fva = 0.0f64;

    for t_idx in 0..cfg.num_timesteps {
        let t = (t_idx + 1) as f64 * dt;
        let disc = (-cfg.risk_free * t).exp();
        let surv_cpty = (-cfg.hazard_rate_cpty * t).exp();
        let surv_own = (-cfg.hazard_rate_own * t).exp();

        // Marginal default probability over dt
        let dp_cpty = cfg.hazard_rate_cpty * dt * surv_cpty;
        let dp_own = cfg.hazard_rate_own * dt * surv_own;

        cva += disc * lgd * dp_cpty * expected_ee_profile[t_idx];
        dva += disc * lgd * dp_own * expected_nee_profile[t_idx];
        fva += disc * cfg.funding_spread * dt * (expected_ee_profile[t_idx] - expected_nee_profile[t_idx]);
    }

    // Effective EPE (time-weighted average)
    let effective_epe: f64 = expected_ee_profile.iter().sum::<f64>() / cfg.num_timesteps as f64;

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;

    println!();
    println!("  XVA Results:");
    println!("    CVA:               ${:.2}", cva);
    println!("    DVA:               ${:.2}", dva);
    println!("    FVA:               ${:.2}", fva);
    println!("    Total XVA:         ${:.2}", cva - dva + fva);
    println!();
    println!("    Peak EE:           ${:.0}", peak_ee);
    println!("    Effective EPE:     ${:.0}", effective_epe);
    println!();
    println!("    Trades:            {}", cfg.num_trades);
    println!("    Scenarios:         {}", cfg.num_scenarios);
    println!("    Time steps:        {}", cfg.num_timesteps);
    println!("    Total allocs:      {}", sf.total_allocations);
    println!("    Wall time:         {:.3}s", wall_time);
    println!();
    println!("    Exposure cube:     {:.2} GB (virtual — NEVER in VRAM)", cube_bytes / 1e9);
    println!("    Peak VRAM:         {:.1} MB", peak_mb);
    println!("    VRAM savings:      {:.0}x", cube_bytes / 1e6 / peak_mb.max(1.0));
    println!("    Fragmentation:     {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:      {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=xva_exposure_engine");
    println!("RESULT trades={}", cfg.num_trades);
    println!("RESULT scenarios={}", cfg.num_scenarios);
    println!("RESULT timesteps={}", cfg.num_timesteps);
    println!("RESULT cva={:.6}", cva);
    println!("RESULT dva={:.6}", dva);
    println!("RESULT fva={:.6}", fva);
    println!("RESULT total_xva={:.6}", cva - dva + fva);
    println!("RESULT peak_ee={:.6}", peak_ee);
    println!("RESULT effective_epe={:.6}", effective_epe);
    println!("RESULT wall_s={:.6}", wall_time);
    println!("RESULT total_allocs={}", sf.total_allocations);
    println!("RESULT cube_gb={:.6}", cube_bytes / 1e9);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT vram_savings_x={:.1}", cube_bytes / 1e6 / peak_mb.max(1.0));
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
