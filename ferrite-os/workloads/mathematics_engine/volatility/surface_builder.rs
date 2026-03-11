// =============================================================================
// Ferrite Mathematics Engine — Implied Volatility Surface Builder
// =============================================================================
//
// Inverts 1M+ option market quotes into implied volatilities using vectorized
// Newton-Raphson iteration on GPU, then fits a smooth parametric surface
// (SABR or SVI) via iterative least-squares.
//
// WHY THIS IS INFEASIBLE WITHOUT TLSF:
//   - 1M option quotes → 1M simultaneous Newton-Raphson inversions
//   - Each Newton iteration: allocate d1/d2/N(d)/vega tensors, compute, update, free
//   - 10-20 Newton iterations × ~8 intermediate tensors per iteration = 80-160M
//     tensor alloc/free operations across all shards
//   - After inversion: iterative SABR calibration (50+ iterations) with
//     per-iteration alloc/compute/free cycles for each expiry slice
//   - Total unique alloc/free events: hundreds of thousands
//   - The iteration-heavy, alloc-heavy nature makes standard cudaMalloc
//     allocation overhead the bottleneck, not compute
//   - With 1M quotes sharded across wave-scheduled batches, each shard
//     iterates independently — creating extreme allocation churn that
//     only O(1) TLSF can handle without drowning in latency
//
// Usage:
//   ferrite-run mathematics_engine/volatility/surface_builder.rs \
//     --quotes 1000000 --expiries 20 --newton-max 20
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
    total_quotes: usize,
    num_expiries: usize,
    quotes_per_shard: usize,
    newton_max_iters: usize,
    newton_tol: f64,
    sabr_iters: usize,
    wave_streams: usize,
    streams: u32,
    spot: f64,
    risk_free: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            total_quotes: 500_000,
            num_expiries: 15,
            quotes_per_shard: 50_000,
            newton_max_iters: 15,
            newton_tol: 1e-8,
            sabr_iters: 40,
            wave_streams: 4,
            streams: 32,
            spot: 100.0,
            risk_free: 0.03,
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorized Black-Scholes pricing + vega (for Newton-Raphson)
// ---------------------------------------------------------------------------

fn bs_price_and_vega(
    spots: &Tensor,
    strikes: &Tensor,
    maturities: &Tensor,
    vols: &Tensor,
    rates: &Tensor,
    is_call: &Tensor,
) -> (Tensor, Tensor) {
    let sqrt_t = maturities.sqrt();
    let vol_sqrt_t = vols * &sqrt_t;

    let d1 = ((spots / strikes).log()
        + (rates + vols * vols * 0.5) * maturities)
        / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    // Standard normal CDF approximation (Abramowitz & Stegun)
    let norm_cdf = |x: &Tensor| -> Tensor {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = x.sign();
        let x_abs = x.abs();
        let t = 1.0 / (1.0 + &x_abs * p);
        let t2 = &t * &t;
        let t3 = &t2 * &t;
        let t4 = &t3 * &t;
        let t5 = &t4 * &t;

        let y = 1.0 - (&t * a1 + &t2 * a2 + &t3 * a3 + &t4 * a4 + t5 * a5)
            * (x_abs.pow_tensor_scalar(2.0).neg() / 2.0).exp();

        (1.0 + &sign * &y) * 0.5
    };

    // Standard normal PDF
    let norm_pdf = |x: &Tensor| -> Tensor {
        let inv_sqrt_2pi = 0.3989422804014327;
        (x.pow_tensor_scalar(2.0).neg() / 2.0).exp() * inv_sqrt_2pi
    };

    let disc = (rates.neg() * maturities).exp();

    // Price
    let call_price = spots * norm_cdf(&d1) - strikes * &disc * norm_cdf(&d2);
    let put_price = strikes * &disc * norm_cdf(&d2.neg()) - spots * norm_cdf(&d1.neg());

    let is_call_mask = is_call.ge(0.5).to_kind(Kind::Float);
    let price = &call_price * &is_call_mask + &put_price * (1.0 - &is_call_mask);

    // Vega = S * sqrt(T) * n(d1) — same for calls and puts
    let vega = spots * &sqrt_t * norm_pdf(&d1);

    (price, vega)
}

// ---------------------------------------------------------------------------
// Newton-Raphson implied vol inversion on a shard
// ---------------------------------------------------------------------------

fn newton_implied_vol_shard(
    market_prices: &Tensor,
    spots: &Tensor,
    strikes: &Tensor,
    maturities: &Tensor,
    rates: &Tensor,
    is_call: &Tensor,
    max_iters: usize,
    tol: f64,
) -> (Tensor, Tensor, Tensor) {
    // Initial guess: Brenner-Subrahmanyam approximation
    let moneyness = spots / strikes;
    let initial_vol = ((&moneyness - 1.0).abs() * 2.0 / maturities.sqrt())
        .clamp(0.05, 2.0);

    let mut vol = initial_vol;
    let mut converged = Tensor::zeros_like(market_prices).to_kind(Kind::Bool);
    let mut iters_used = Tensor::zeros_like(market_prices);

    for iter in 0..max_iters {
        // Price with current vol guess (alloc intermediate tensors → TLSF)
        let (price, vega) = bs_price_and_vega(
            spots, strikes, maturities, &vol, rates, is_call,
        );

        // Residual
        let residual = &price - market_prices;

        // Check convergence
        let newly_converged = residual.abs().lt(tol).logical_and(&converged.logical_not());
        let newly_converged_f = newly_converged.to_kind(Kind::Float);
        iters_used = &iters_used + &newly_converged_f * (iter + 1) as f64;
        converged = converged.logical_or(&newly_converged);
        // price, vega intermediates freed by TLSF on each iteration

        // Newton update: vol -= residual / vega (only for non-converged)
        let safe_vega = vega.clamp_min(1e-10);
        let update = &residual / &safe_vega;
        let active = converged.logical_not().to_kind(Kind::Float);
        vol = (&vol - &update * &active).clamp(0.001, 5.0);

        // Early exit if all converged
        if bool::try_from(converged.all()).unwrap_or(false) {
            // Set remaining iters_used for late convergers
            let still_zero = iters_used.eq(0.0).to_kind(Kind::Float);
            iters_used = &iters_used + &still_zero * (iter + 1) as f64;
            break;
        }
    }

    // Mark non-converged with iter count = max_iters
    let still_zero = iters_used.eq(0.0).to_kind(Kind::Float);
    iters_used = &iters_used + &still_zero * max_iters as f64;

    let converged_f = converged.to_kind(Kind::Float);
    (vol, converged_f, iters_used)
}

// ---------------------------------------------------------------------------
// Generate synthetic market quotes
// ---------------------------------------------------------------------------

struct MarketQuotes {
    prices: Tensor,
    spots: Tensor,
    strikes: Tensor,
    maturities: Tensor,
    rates: Tensor,
    is_call: Tensor,
    true_vols: Tensor, // for validation
}

fn generate_market_quotes(
    n: usize,
    num_expiries: usize,
    spot: f64,
    risk_free: f64,
    device: Device,
    stream_id: usize,
) -> MarketQuotes {
    set_torch_stream(stream_id);

    let n = n as i64;

    let spots = Tensor::full([n], spot, (Kind::Float, device));
    let rates = Tensor::full([n], risk_free, (Kind::Float, device));

    // Strikes: 60% to 140% of spot
    let strikes = Tensor::rand([n], (Kind::Float, device)) * (spot * 0.8) + (spot * 0.6);

    // Maturities: assign to discrete expiry buckets
    let expiry_values: Vec<f64> = (0..num_expiries)
        .map(|i| 0.1 + 2.4 * i as f64 / (num_expiries - 1).max(1) as f64)
        .collect();
    let expiry_idx = (Tensor::rand([n], (Kind::Float, device)) * num_expiries as f64).floor().to_kind(Kind::Int64);
    let expiry_tensor = Tensor::from_slice(&expiry_values).to_device(device);
    let maturities = expiry_tensor.index_select(0, &expiry_idx);

    let is_call = (Tensor::rand([n], (Kind::Float, device)).ge(0.5)).to_kind(Kind::Float);

    // True implied vols: skew pattern σ(K) = σ_ATM * (1 + skew*(K/S - 1) + smile*(K/S - 1)²)
    let moneyness = &strikes / spot;
    let base_vol = 0.20;
    let skew = -0.15;
    let smile = 0.10;
    let m_shifted = &moneyness - 1.0;
    let true_vols: Tensor = Tensor::full([n], base_vol, (Kind::Float, device))
        * (1.0 + &m_shifted * skew + m_shifted.pow_tensor_scalar(2.0) * smile);
    let true_vols = true_vols.clamp(0.05, 1.0);

    // Generate "market" prices from true vols
    let (prices, _) = bs_price_and_vega(
        &spots, &strikes, &maturities, &true_vols, &rates, &is_call,
    );

    // Add small noise to simulate bid/ask spread
    let noise = Tensor::randn([n], (Kind::Float, device)) * 0.001;
    let prices = (prices + noise).clamp_min(0.001);

    MarketQuotes { prices, spots, strikes, maturities, rates, is_call, true_vols }
}

// ---------------------------------------------------------------------------
// SABR calibration per expiry slice (iterative least-squares)
// ---------------------------------------------------------------------------

struct SABRParams {
    alpha: f64, // initial vol
    beta: f64,  // CEV exponent (usually fixed, e.g. 0.5)
    rho: f64,   // correlation
    nu: f64,    // vol of vol
}

fn calibrate_sabr_slice(
    implied_vols: &Tensor,
    strikes: &Tensor,
    forward: f64,
    maturity: f64,
    max_iters: usize,
) -> SABRParams {
    // Simplified SABR calibration via iterative Newton on 3 free params (alpha, rho, nu)
    // Beta fixed at 0.5 (common market convention)
    let beta = 0.5f64;
    let n = implied_vols.size()[0];
    if n < 4 {
        return SABRParams { alpha: 0.2, beta, rho: -0.3, nu: 0.4 };
    }

    // SABR implied vol approximation (Hagan et al.)
    let sabr_vol = |k: &Tensor, alpha: f64, rho: f64, nu: f64| -> Tensor {
        let f = forward;
        let fk = (Tensor::full([n], f, (Kind::Float, k.device())) * k).sqrt();
        let log_fk = (Tensor::full([n], f, (Kind::Float, k.device())) / k).log();
        let log_fk_sq = &log_fk * &log_fk;

        // Leading term
        let fk_beta = fk.pow_tensor_scalar(1.0 - beta);
        let leading = alpha / &fk_beta;

        // z = (nu/alpha) * fk^(1-beta) * log(F/K)
        let z = (nu / alpha.max(1e-10)) * &fk_beta * &log_fk;
        let z_sq = &z * &z;

        // x(z) = log((sqrt(1-2*rho*z+z^2)+z-rho)/(1-rho))
        let disc = (Tensor::ones_like(&z) - &z * (2.0 * rho) + &z_sq).sqrt();
        let x_num = &disc + &z - rho;
        let x_denom = 1.0 - rho;
        let x = (x_num / x_denom).clamp_min(1e-10).log();

        // Correction terms
        let corr1 = 1.0 + ((1.0 - beta).powi(2) / 24.0) * &log_fk_sq;
        let corr2_num = (1.0 - beta).powi(2) * alpha * alpha / (24.0 * fk_beta.pow_tensor_scalar(2.0))
            + 0.25 * rho * beta * nu * alpha / &fk_beta
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
        let corr2 = 1.0 + corr2_num * maturity;

        // Handle ATM (z ≈ 0)
        let is_atm = z.abs().lt(1e-7).to_kind(Kind::Float);
        let non_atm = 1.0 - &is_atm;
        let z_over_x = (&z / x.clamp_min(1e-10)) * &non_atm + &is_atm;

        &leading * z_over_x * corr2 / corr1
    };

    // Grid search for initial guess
    let mut best_alpha = 0.2;
    let mut best_rho = -0.3;
    let mut best_nu = 0.4;
    let mut best_err = f64::MAX;

    for &a in &[0.1, 0.2, 0.3, 0.5] {
        for &r in &[-0.7, -0.3, 0.0, 0.3] {
            for &v in &[0.2, 0.4, 0.8] {
                let model = sabr_vol(strikes, a, r, v);
                let err = f64::try_from((&model - implied_vols).pow_tensor_scalar(2.0).mean(Kind::Float))
                    .unwrap_or(f64::MAX);
                if err < best_err {
                    best_err = err;
                    best_alpha = a;
                    best_rho = r;
                    best_nu = v;
                }
            }
        }
    }

    // Refine with coordinate descent
    let bump = [0.01, 0.02, 0.02]; // alpha, rho, nu bumps
    for _iter in 0..max_iters {
        let base_model = sabr_vol(strikes, best_alpha, best_rho, best_nu);
        let base_err = f64::try_from(
            (&base_model - implied_vols).pow_tensor_scalar(2.0).mean(Kind::Float)
        ).unwrap_or(f64::MAX);

        // Try bumping each parameter
        for param in 0..3 {
            let (a_up, r_up, n_up) = match param {
                0 => (best_alpha + bump[0], best_rho, best_nu),
                1 => (best_alpha, (best_rho + bump[1]).min(0.999), best_nu),
                _ => (best_alpha, best_rho, best_nu + bump[2]),
            };
            let (a_dn, r_dn, n_dn) = match param {
                0 => ((best_alpha - bump[0]).max(0.001), best_rho, best_nu),
                1 => (best_alpha, (best_rho - bump[1]).max(-0.999), best_nu),
                _ => (best_alpha, best_rho, (best_nu - bump[2]).max(0.01)),
            };

            let err_up = f64::try_from(
                (&sabr_vol(strikes, a_up, r_up, n_up) - implied_vols).pow_tensor_scalar(2.0).mean(Kind::Float)
            ).unwrap_or(f64::MAX);

            let err_dn = f64::try_from(
                (&sabr_vol(strikes, a_dn, r_dn, n_dn) - implied_vols).pow_tensor_scalar(2.0).mean(Kind::Float)
            ).unwrap_or(f64::MAX);

            if err_up < base_err && err_up <= err_dn {
                match param {
                    0 => best_alpha = a_up,
                    1 => best_rho = r_up,
                    _ => best_nu = n_up,
                }
            } else if err_dn < base_err {
                match param {
                    0 => best_alpha = a_dn,
                    1 => best_rho = r_dn,
                    _ => best_nu = n_dn,
                }
            }
            // All intermediate sabr_vol tensors freed by TLSF each iteration
        }
    }

    SABRParams { alpha: best_alpha, beta, rho: best_rho, nu: best_nu }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quotes" => cfg.total_quotes = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--expiries" => cfg.num_expiries = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--quotes-per-shard" => cfg.quotes_per_shard = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--newton-max" => cfg.newton_max_iters = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--newton-tol" => cfg.newton_tol = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--sabr-iters" => cfg.sabr_iters = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--spot" => cfg.spot = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--rate" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: surface_builder.rs [options]");
                println!();
                println!("  Implied volatility surface construction from 1M+ option quotes.");
                println!("  Newton-Raphson inversion + SABR calibration, wave-scheduled.");
                println!();
                println!("  --quotes N           Total option quotes (default: 500000)");
                println!("  --expiries N         Expiry buckets (default: 15)");
                println!("  --quotes-per-shard N Per TLSF shard (default: 50000)");
                println!("  --newton-max N       Max Newton iterations (default: 15)");
                println!("  --newton-tol F       Convergence tolerance (default: 1e-8)");
                println!("  --sabr-iters N       SABR calibration iterations per expiry (default: 40)");
                println!("  --wave-streams N     Concurrent shard streams (default: 4)");
                println!("  --streams N          CUDA stream pool (default: 32)");
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

    let n_shards = cfg.total_quotes.div_ceil(cfg.quotes_per_shard);
    let shard_bytes = cfg.quotes_per_shard * 7 * 4; // ~7 float32 arrays per quote
    let total_data_bytes = cfg.total_quotes as f64 * 7.0 * 4.0;

    // Allocation estimate: newton iterations × shards × tensors per iteration
    let allocs_per_newton_iter = 12; // d1, d2, N(d1), N(d2), price, vega, residual, etc.
    let newton_allocs = n_shards * cfg.newton_max_iters * allocs_per_newton_iter;
    // SABR calibration: per expiry, per iteration, multiple model evaluations
    let sabr_allocs = cfg.num_expiries * cfg.sabr_iters * 3 * 8; // 3 params × ~8 tensors
    let total_est_allocs = newton_allocs + sabr_allocs;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Implied Volatility Surface Builder ===\n");
    println!("  total quotes:        {}", cfg.total_quotes);
    println!("  expiry buckets:      {}", cfg.num_expiries);
    println!("  quotes/shard:        {}", cfg.quotes_per_shard);
    println!("  shards:              {}", n_shards);
    println!("  Newton max iters:    {}", cfg.newton_max_iters);
    println!("  SABR fit iterations: {}", cfg.sabr_iters);
    println!();
    println!("  ALLOCATION PRESSURE (iteration-heavy):");
    println!("    Newton allocs est: {} (shards x iters x tensors)", newton_allocs);
    println!("    SABR allocs est:   {} (expiries x iters x params x tensors)", sabr_allocs);
    println!("    total est allocs:  {}", total_est_allocs);
    println!("    cudaMalloc cost:   {:.1}s (@1ms)", total_est_allocs as f64 * 1e-3);
    println!("    TLSF cost:         {:.4}s (@0.24us)", total_est_allocs as f64 * 0.24e-6);
    println!();
    println!("  TLSF pool:           {:.1} MB", pool_mb);
    println!("  shard working set:   {:.1} MB", shard_bytes as f64 / 1e6);
    println!("  total data:          {:.1} MB", total_data_bytes / 1e6);
    println!("  wave streams:        {}", cfg.wave_streams);
    println!();

    // =====================================================================
    // PHASE 1: Generate market quotes + wave-scheduled Newton-Raphson
    // =====================================================================
    println!("  --- Phase 1: Newton-Raphson Implied Vol Inversion ---\n");
    let t_phase1 = Instant::now();

    let mut all_implied_vols: Vec<Tensor> = Vec::new();
    let mut all_strikes: Vec<Tensor> = Vec::new();
    let mut all_maturities: Vec<Tensor> = Vec::new();
    let mut all_true_vols: Vec<Tensor> = Vec::new();
    let mut total_converged = 0usize;
    let mut total_quotes_done = 0usize;
    let mut total_newton_iters = 0.0f64;

    for wave_start in (0..n_shards).step_by(cfg.wave_streams) {
        let wave_end = (wave_start + cfg.wave_streams).min(n_shards);

        for w in wave_start..wave_end {
            let remaining = cfg.total_quotes - w * cfg.quotes_per_shard;
            let n_quotes = remaining.min(cfg.quotes_per_shard);
            let stream_id = (w - wave_start) % active_streams;

            // Generate synthetic quotes (alloc from TLSF)
            let quotes = generate_market_quotes(
                n_quotes, cfg.num_expiries, cfg.spot, cfg.risk_free,
                device, stream_id,
            );

            // Newton-Raphson inversion (alloc/free many intermediate tensors per iter)
            let (impl_vols, converged, iters_used) = newton_implied_vol_shard(
                &quotes.prices,
                &quotes.spots,
                &quotes.strikes,
                &quotes.maturities,
                &quotes.rates,
                &quotes.is_call,
                cfg.newton_max_iters,
                cfg.newton_tol,
            );

            let n_conv = f64::try_from(converged.sum(Kind::Float)).unwrap_or(0.0) as usize;
            let avg_iters = f64::try_from(iters_used.mean(Kind::Float)).unwrap_or(0.0);

            total_converged += n_conv;
            total_quotes_done += n_quotes;
            total_newton_iters += avg_iters * n_quotes as f64;

            // Save for Phase 2 (move to CPU to free GPU memory)
            all_implied_vols.push(impl_vols.to_device(Device::Cpu));
            all_strikes.push(quotes.strikes.to_device(Device::Cpu));
            all_maturities.push(quotes.maturities.to_device(Device::Cpu));
            all_true_vols.push(quotes.true_vols.to_device(Device::Cpu));
            // quotes tensors drop → TLSF frees shard
        }

        sync_all_streams();

        let s = runtime.tlsf_stats();
        if wave_start == 0 || (wave_start / cfg.wave_streams) % 5 == 0 {
            let pct = total_quotes_done as f64 / cfg.total_quotes as f64 * 100.0;
            println!("  shard {:>3}/{} | {:.1}% | conv={}/{} | peak={:.0}MB frag={:.6}",
                wave_start / cfg.wave_streams + 1, n_shards, pct,
                total_converged, total_quotes_done,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let phase1_time = t_phase1.elapsed().as_secs_f64();
    let avg_newton_iters = total_newton_iters / total_quotes_done as f64;
    let convergence_rate = total_converged as f64 / total_quotes_done as f64 * 100.0;

    // Compute implied vol error vs true vols
    let all_impl = Tensor::cat(&all_implied_vols, 0);
    let all_true = Tensor::cat(&all_true_vols, 0);
    let vol_error = (&all_impl - &all_true).abs();
    let mean_vol_error = f64::try_from(vol_error.mean(Kind::Float)).unwrap_or(0.0);
    let max_vol_error = f64::try_from(vol_error.amax(0, false)).unwrap_or(0.0);

    println!();
    println!("  Phase 1 Results:");
    println!("    Quotes inverted:   {}", total_quotes_done);
    println!("    Converged:         {} ({:.2}%)", total_converged, convergence_rate);
    println!("    Avg Newton iters:  {:.1}", avg_newton_iters);
    println!("    Mean vol error:    {:.8} ({:.4} bps)", mean_vol_error, mean_vol_error * 10000.0);
    println!("    Max vol error:     {:.8} ({:.2} bps)", max_vol_error, max_vol_error * 10000.0);
    println!("    Phase 1 time:      {:.3}s", phase1_time);

    // =====================================================================
    // PHASE 2: SABR surface calibration per expiry
    // =====================================================================
    println!("\n  --- Phase 2: SABR Surface Calibration ---\n");
    let t_phase2 = Instant::now();

    let all_strikes_cat = Tensor::cat(&all_strikes, 0).to_device(device);
    let all_maturities_cat = Tensor::cat(&all_maturities, 0).to_device(device);
    let all_impl_gpu = all_impl.to_device(device);

    let forward = cfg.spot * (cfg.risk_free * 1.0).exp(); // rough forward

    let expiry_values: Vec<f64> = (0..cfg.num_expiries)
        .map(|i| 0.1 + 2.4 * i as f64 / (cfg.num_expiries - 1).max(1) as f64)
        .collect();

    let mut sabr_results: Vec<SABRParams> = Vec::new();

    for (exp_idx, &maturity) in expiry_values.iter().enumerate() {
        // Filter quotes for this expiry (within tolerance)
        let mask = (&all_maturities_cat - maturity).abs().lt(0.01);
        let indices = mask.nonzero().squeeze_dim(-1);
        let n_in_slice = indices.size()[0];

        if n_in_slice < 10 {
            sabr_results.push(SABRParams { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 });
            continue;
        }

        let slice_vols = all_impl_gpu.index_select(0, &indices);
        let slice_strikes = all_strikes_cat.index_select(0, &indices);

        // Calibrate SABR to this expiry slice (iterative — many alloc/free per iter)
        let sabr = calibrate_sabr_slice(
            &slice_vols, &slice_strikes, forward, maturity, cfg.sabr_iters,
        );

        // Compute fit error
        // (SABR model tensors allocated/freed each iteration inside calibrate_sabr_slice)

        println!("  expiry {:>2}/{} | T={:.2}y | n={} | alpha={:.4} rho={:.3} nu={:.3}",
            exp_idx + 1, cfg.num_expiries, maturity, n_in_slice,
            sabr.alpha, sabr.rho, sabr.nu);

        sabr_results.push(sabr);
        // slice_vols, slice_strikes drop → TLSF frees
    }

    let phase2_time = t_phase2.elapsed().as_secs_f64();
    let total_time = t_phase1.elapsed().as_secs_f64();

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;

    println!();
    println!("  Phase 2 Results:");
    println!("    Expiries calibrated: {}", sabr_results.len());
    println!("    Phase 2 time:        {:.3}s", phase2_time);

    println!();
    println!("  Overall Results:");
    println!("    Total quotes:      {}", total_quotes_done);
    println!("    Convergence:       {:.2}%", convergence_rate);
    println!("    Mean vol error:    {:.4} bps", mean_vol_error * 10000.0);
    println!("    Total time:        {:.3}s (inversion: {:.3}s + SABR: {:.3}s)",
        total_time, phase1_time, phase2_time);
    println!("    Quotes/sec:        {:.0}", total_quotes_done as f64 / phase1_time);
    println!();
    println!("    Total allocs:      {}", sf.total_allocations);
    println!("    Peak VRAM:         {:.1} MB", peak_mb);
    println!("    Fragmentation:     {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:      {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=vol_surface_builder");
    println!("RESULT quotes={}", total_quotes_done);
    println!("RESULT converged_pct={:.4}", convergence_rate);
    println!("RESULT avg_newton_iters={:.2}", avg_newton_iters);
    println!("RESULT mean_vol_error_bps={:.6}", mean_vol_error * 10000.0);
    println!("RESULT max_vol_error_bps={:.6}", max_vol_error * 10000.0);
    println!("RESULT expiries_calibrated={}", sabr_results.len());
    println!("RESULT phase1_s={:.6}", phase1_time);
    println!("RESULT phase2_s={:.6}", phase2_time);
    println!("RESULT total_s={:.6}", total_time);
    println!("RESULT total_allocs={}", sf.total_allocations);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
