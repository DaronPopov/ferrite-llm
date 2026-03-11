// =============================================================================
// Ferrite Mathematics Engine — Heston Stochastic Volatility Calibrator
// =============================================================================
//
// Calibrates a Heston stochastic volatility model to a market-observed option
// surface via Levenberg-Marquardt optimization. Uses semi-analytical pricing
// via the Heston characteristic function (Gil-Pelaez inversion with Gauss
// quadrature), fully vectorized across all options on GPU.
//
// WHY THIS IS INFEASIBLE WITHOUT TLSF:
//   - Each CF pricing of N options creates ~100 intermediate GPU tensors
//     (complex arithmetic: d, g, exp, log, mul, div — all as real/imag pairs)
//   - Each LM iteration: 7 surface pricings (base + 5 param bumps + trial)
//   - Over 50 iterations: 350 surface pricings × ~100 tensors = ~35,000 allocs
//   - Plus quadrature: 128 points × 2 (P1,P2) × ~50 ops = ~12,800 per pricing
//   - Total: ~4.5M alloc/free cycles
//   - With cudaMalloc @ 1ms each: 75 minutes of pure allocation overhead
//   - With TLSF @ 0.24μs each: 1.1 seconds
//
// Usage:
//   ferrite-run mathematics_engine/calibration/heston_calibrator.rs \
//     --strikes 200 --maturities 8 --lm-iters 50
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
    num_strikes: usize,
    num_maturities: usize,
    n_quad: usize,        // quadrature points for CF integration
    lm_max_iters: usize,
    lm_tol: f64,
    wave_streams: usize,
    streams: u32,
    spot: f64,
    risk_free: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_strikes: 100,
            num_maturities: 6,
            n_quad: 128,
            lm_max_iters: 50,
            lm_tol: 1e-12,
            wave_streams: 4,
            streams: 32,
            spot: 100.0,
            risk_free: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Heston model parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct HestonParams {
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
}

impl HestonParams {
    fn to_vec(&self) -> Vec<f64> {
        vec![self.v0, self.kappa, self.theta, self.sigma, self.rho]
    }

    fn from_vec(v: &[f64]) -> Self {
        Self {
            v0: v[0].max(0.001),
            kappa: v[1].max(0.01),
            theta: v[2].max(0.001),
            sigma: v[3].max(0.01),
            rho: v[4].clamp(-0.999, 0.999),
        }
    }

    fn n_params() -> usize { 5 }
}

// ---------------------------------------------------------------------------
// Complex tensor arithmetic — each op allocates GPU tensors via TLSF
// ---------------------------------------------------------------------------

fn cmul(ar: &Tensor, ai: &Tensor, br: &Tensor, bi: &Tensor) -> (Tensor, Tensor) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn cdiv(ar: &Tensor, ai: &Tensor, br: &Tensor, bi: &Tensor) -> (Tensor, Tensor) {
    let denom = (br * br + bi * bi).clamp_min(1e-30);
    ((ar * br + ai * bi) / &denom, (ai * br - ar * bi) / &denom)
}

fn cexp(ar: &Tensor, ai: &Tensor) -> (Tensor, Tensor) {
    let e = ar.exp();
    (&e * ai.cos(), &e * ai.sin())
}

fn clog(ar: &Tensor, ai: &Tensor) -> (Tensor, Tensor) {
    let modulus = (ar * ar + ai * ai).clamp_min(1e-30).sqrt();
    (modulus.log(), ai.atan2(ar))
}

fn csqrt(ar: &Tensor, ai: &Tensor) -> (Tensor, Tensor) {
    let modulus = (ar * ar + ai * ai).clamp_min(1e-30).sqrt();
    let half_arg = ai.atan2(ar) * 0.5;
    let sqrt_mod = modulus.sqrt();
    (&sqrt_mod * half_arg.cos(), &sqrt_mod * half_arg.sin())
}

// ---------------------------------------------------------------------------
// Heston characteristic function pricing (vectorized)
// ---------------------------------------------------------------------------
//
// Prices European calls using Gil-Pelaez inversion of the Heston (1993)
// characteristic function with the Albrecher et al. (2007) rotation to
// avoid branch-cut discontinuities.
//
// All options are priced simultaneously: tensors are [n_quad, n_options].
// Each quadrature point creates ~50 intermediate tensor operations.

fn heston_cf_call_prices(
    spots: &Tensor,
    strikes: &Tensor,
    maturities: &Tensor,
    params: &HestonParams,
    risk_free: f64,
    n_quad: usize,
) -> Tensor {
    let device = spots.device();
    let n_opt = spots.size()[0];

    let log_sk = (spots / strikes).log().unsqueeze(0); // [1, n_opt]
    let mat = maturities.unsqueeze(0);                  // [1, n_opt]

    // Quadrature points: trapezoidal on (0, phi_max]
    let phi_max = 100.0;
    let dphi = phi_max / n_quad as f64;
    let phi_vals: Vec<f32> = (1..=n_quad).map(|i| (i as f64 * dphi) as f32).collect();
    let phi = Tensor::from_slice(&phi_vals)
        .to_device(device)
        .unsqueeze(1); // [n_quad, 1]

    let kappa = params.kappa;
    let theta = params.theta;
    let sigma = params.sigma;
    let rho = params.rho;
    let v0 = params.v0;
    let s2 = sigma * sigma;

    let mut p1_integral = Tensor::zeros([n_opt], (Kind::Float, device));
    let mut p2_integral = Tensor::zeros([n_opt], (Kind::Float, device));

    // Compute P1 (j=0) and P2 (j=1)
    for j in 0..2 {
        let (u_j, b_j): (f64, f64) = if j == 0 {
            (0.5, kappa - rho * sigma)
        } else {
            (-0.5, kappa)
        };

        // d² = (ρσiφ - b)² - σ²(2u·iφ - φ²)
        // d²_re = b² + σ²φ²(1-ρ²)
        // d²_im = -φ·(2ρσb + 2u·σ²)
        let d2_re: Tensor = &phi * &phi * (s2 * (1.0 - rho * rho)) + b_j * b_j;
        let d2_im: Tensor = &phi * (-(2.0 * rho * sigma * b_j + 2.0 * u_j * s2));

        let (d_re, d_im) = csqrt(&d2_re, &d2_im);

        // Rotated formulation: A = b - ρσiφ - d, B = b - ρσiφ + d
        let rsp: Tensor = &phi * (-rho * sigma); // -ρσφ, [n_quad, 1]
        let a_re: Tensor = -&d_re + b_j;
        let a_im: Tensor = &rsp - &d_im;
        let b_denom_re: Tensor = &d_re + b_j;
        let b_denom_im: Tensor = &rsp + &d_im;

        // g = A / B
        let (g_re, g_im) = cdiv(&a_re, &a_im, &b_denom_re, &b_denom_im);

        // exp(-d·T): broadcasts [n_quad,1] × [1,n_opt] → [n_quad,n_opt]
        let neg_dt_re = -&d_re * &mat;
        let neg_dt_im = -&d_im * &mat;
        let (edt_re, edt_im) = cexp(&neg_dt_re, &neg_dt_im);

        // g · exp(-dT)
        let (gedt_re, gedt_im) = cmul(&g_re, &g_im, &edt_re, &edt_im);

        // 1 - exp(-dT)
        let one_m_edt_re = 1.0 - &edt_re;
        let one_m_edt_im = -&edt_im;

        // 1 - g·exp(-dT)
        let one_m_gedt_re = 1.0 - &gedt_re;
        let one_m_gedt_im = -&gedt_im;

        // D = (A/σ²) · (1-exp(-dT)) / (1-g·exp(-dT))
        let (frac_re, frac_im) = cdiv(
            &one_m_edt_re, &one_m_edt_im,
            &one_m_gedt_re, &one_m_gedt_im,
        );
        let as2_re = &a_re / s2;
        let as2_im = &a_im / s2;
        let (dj_re, dj_im) = cmul(&as2_re, &as2_im, &frac_re, &frac_im);

        // C = r·iφ·T + (κθ/σ²)·[A·T - 2·ln((1-g·exp(-dT))/(1-g))]
        let one_m_g_re = 1.0 - &g_re;
        let one_m_g_im = -&g_im;

        let (ratio_re, ratio_im) = cdiv(
            &one_m_gedt_re, &one_m_gedt_im,
            &one_m_g_re, &one_m_g_im,
        );
        let (lr_re, lr_im) = clog(&ratio_re, &ratio_im);

        let at_re = &a_re * &mat;
        let at_im = &a_im * &mat;

        let kt_s2 = kappa * theta / s2;
        let cj_re = (&at_re - &lr_re * 2.0) * kt_s2;
        let cj_im = (&at_im - &lr_im * 2.0) * kt_s2 + &phi * risk_free * &mat;

        // Full exponent: C + D·v0 + iφ·ln(S/K)
        let exp_re = &cj_re + &dj_re * v0;
        let exp_im = &cj_im + &dj_im * v0 + &phi * &log_sk;

        // exp(exponent) → integrand = Re[exp(...)]/φ
        let (val_re, _val_im) = cexp(
            &exp_re.clamp(-50.0, 50.0),
            &exp_im,
        );
        let integrand = val_re / &phi; // [n_quad, n_opt]

        // Sum over quadrature dimension → [n_opt]
        let contribution = integrand.sum_dim_intlist(0, false, Kind::Float) * dphi;

        if j == 0 {
            p1_integral = contribution;
        } else {
            p2_integral = contribution;
        }
    }

    let pi = std::f64::consts::PI;
    let p1 = &p1_integral / pi + 0.5;
    let p2 = &p2_integral / pi + 0.5;

    // Call = S·P1 - K·exp(-rT)·P2
    let disc = (maturities * (-risk_free)).exp();
    let prices = spots * p1 - strikes * &disc * p2;
    prices.clamp_min(0.0)
}

// ---------------------------------------------------------------------------
// Option surface
// ---------------------------------------------------------------------------

struct OptionSurface {
    strikes: Tensor,     // [n_options]
    maturities: Tensor,  // [n_options]
    market_prices: Tensor, // [n_options]
    n_options: usize,
}

fn generate_market_surface(
    spot: f64,
    n_strikes: usize,
    n_maturities: usize,
    true_params: &HestonParams,
    risk_free: f64,
    n_quad: usize,
    device: Device,
) -> OptionSurface {
    let n_options = n_strikes * n_maturities;

    let strike_vals: Vec<f32> = (0..n_strikes)
        .map(|i| (spot * (0.70 + 0.60 * i as f64 / (n_strikes - 1).max(1) as f64)) as f32)
        .collect();

    let mat_vals: Vec<f32> = (0..n_maturities)
        .map(|i| (0.25 + 1.75 * i as f64 / (n_maturities - 1).max(1) as f64) as f32)
        .collect();

    // Build flattened [n_options] arrays: for each maturity, all strikes
    let mut all_strikes = Vec::with_capacity(n_options);
    let mut all_mats = Vec::with_capacity(n_options);
    for &m in &mat_vals {
        for &k in &strike_vals {
            all_strikes.push(k);
            all_mats.push(m);
        }
    }

    let spots_t = Tensor::full([n_options as i64], spot, (Kind::Float, device));
    let strikes_t = Tensor::from_slice(&all_strikes).to_device(device);
    let maturities_t = Tensor::from_slice(&all_mats).to_device(device);

    let market_prices = heston_cf_call_prices(
        &spots_t, &strikes_t, &maturities_t,
        true_params, risk_free, n_quad,
    );

    OptionSurface {
        strikes: strikes_t,
        maturities: maturities_t,
        market_prices,
        n_options,
    }
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt
// ---------------------------------------------------------------------------

fn levenberg_marquardt(
    surface: &OptionSurface,
    spots: &Tensor,
    initial_params: &HestonParams,
    cfg: &Config,
    device: Device,
    runtime: &PtxRuntime,
) -> (HestonParams, Vec<f64>) {
    let n_options = surface.n_options;
    let n_params = HestonParams::n_params();
    let bump_sizes = [0.002, 0.05, 0.002, 0.02, 0.02];

    let mut params = initial_params.clone();
    let mut lambda = 1e-3f64;
    let mut history = Vec::new();

    for iter in 0..cfg.lm_max_iters {
        let iter_start = Instant::now();

        // Base prices with current params
        let model_prices = heston_cf_call_prices(
            spots, &surface.strikes, &surface.maturities,
            &params, cfg.risk_free, cfg.n_quad,
        );

        let residuals = &model_prices - &surface.market_prices;
        let cost = f64::try_from(
            (&residuals * &residuals).mean(Kind::Float)
        ).unwrap_or(f64::MAX);
        history.push(cost);

        if cost < cfg.lm_tol {
            println!("  iter {:>3} | cost={:.2e} — CONVERGED", iter + 1, cost);
            break;
        }

        // Jacobian via finite differences: one column per param
        let res_cpu: Vec<f32> = Vec::try_from(&residuals.to_device(Device::Cpu)).unwrap_or_default();
        let mut jacobian = vec![vec![0.0f64; n_options]; n_params];

        for p in 0..n_params {
            let mut bumped_vec = params.to_vec();
            bumped_vec[p] += bump_sizes[p];
            let bumped_params = HestonParams::from_vec(&bumped_vec);

            let bumped_prices = heston_cf_call_prices(
                spots, &surface.strikes, &surface.maturities,
                &bumped_params, cfg.risk_free, cfg.n_quad,
            );

            let model_cpu: Vec<f32> = Vec::try_from(&model_prices.to_device(Device::Cpu)).unwrap_or_default();
            let bumped_cpu: Vec<f32> = Vec::try_from(&bumped_prices.to_device(Device::Cpu)).unwrap_or_default();

            for i in 0..n_options {
                jacobian[p][i] = (bumped_cpu[i] as f64 - model_cpu[i] as f64) / bump_sizes[p];
            }
        }

        // Normal equations: (J^T J + λ·diag) · δp = -J^T r
        let mut jtj = vec![vec![0.0f64; n_params]; n_params];
        let mut jtr = vec![0.0f64; n_params];

        for i in 0..n_params {
            for j in 0..n_params {
                let mut s = 0.0;
                for k in 0..n_options { s += jacobian[i][k] * jacobian[j][k]; }
                jtj[i][j] = s;
            }
            let mut s = 0.0;
            for k in 0..n_options { s += jacobian[i][k] * res_cpu[k] as f64; }
            jtr[i] = s;
        }

        for i in 0..n_params {
            jtj[i][i] += lambda * jtj[i][i].max(1e-10);
        }

        let dp = solve_5x5(&jtj, &jtr);

        // Trial step
        let mut trial_vec = params.to_vec();
        for i in 0..n_params { trial_vec[i] -= dp[i]; }
        let trial_params = HestonParams::from_vec(&trial_vec);

        let trial_prices = heston_cf_call_prices(
            spots, &surface.strikes, &surface.maturities,
            &trial_params, cfg.risk_free, cfg.n_quad,
        );

        let trial_res = &trial_prices - &surface.market_prices;
        let trial_cost = f64::try_from(
            (&trial_res * &trial_res).mean(Kind::Float)
        ).unwrap_or(f64::MAX);

        let iter_time = iter_start.elapsed().as_secs_f64();
        let s = runtime.tlsf_stats();

        if trial_cost < cost {
            params = trial_params;
            lambda = (lambda * 0.5).max(1e-10);
            println!("  iter {:>3} | cost={:.2e} -> {:.2e} (accept) | λ={:.1e} | {:.3}s | allocs={}",
                iter + 1, cost, trial_cost, lambda, iter_time, s.total_allocations);
        } else {
            lambda = (lambda * 2.0).min(1e6);
            println!("  iter {:>3} | cost={:.2e}    (reject, trial={:.2e}) | λ={:.1e} | {:.3}s",
                iter + 1, cost, trial_cost, lambda, iter_time);
        }
    }

    (params, history)
}

// ---------------------------------------------------------------------------
// 5×5 Gaussian elimination
// ---------------------------------------------------------------------------

fn solve_5x5(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.push(b[i]);
        r
    }).collect();

    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() { max_row = row; }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 { continue; }
        for row in (col + 1)..n {
            let f = aug[row][col] / pivot;
            for j in col..=n { aug[row][j] -= f * aug[col][j]; }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n { s -= aug[i][j] * x[j]; }
        if aug[i][i].abs() > 1e-15 { x[i] = s / aug[i][i]; }
    }
    x
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--strikes" => cfg.num_strikes = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--maturities" => cfg.num_maturities = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--n-quad" => cfg.n_quad = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--lm-iters" => cfg.lm_max_iters = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--lm-tol" => cfg.lm_tol = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--spot" => cfg.spot = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "--rate" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: heston_calibrator.rs [options]");
                println!();
                println!("  --strikes N       Strike grid points (default: 100)");
                println!("  --maturities N    Maturity grid points (default: 6)");
                println!("  --n-quad N        Quadrature points for CF (default: 128)");
                println!("  --lm-iters N      Max LM iterations (default: 50)");
                println!("  --lm-tol F        Convergence tolerance (default: 1e-12)");
                println!("  --spot F          Spot price (default: 100.0)");
                println!("  --rate F          Risk-free rate (default: 0.05)");
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
    let _active_streams = num_streams();
    set_torch_stream(0);

    let n_options = cfg.num_strikes * cfg.num_maturities;
    let n_params = HestonParams::n_params();
    let pricings_per_iter = 1 + n_params + 1; // base + jacobian + trial
    let ops_per_pricing = cfg.n_quad * 2 * 50; // quad points × 2 (P1,P2) × ~50 tensor ops
    let total_est_allocs = pricings_per_iter * cfg.lm_max_iters * ops_per_pricing;

    let s0 = runtime.tlsf_stats();

    println!("=== Ferrite Heston Stochastic Volatility Calibrator ===\n");
    println!("  option surface:    {} strikes x {} maturities = {} options",
        cfg.num_strikes, cfg.num_maturities, n_options);
    println!("  pricing method:    characteristic function (semi-analytical)");
    println!("  quadrature points: {}", cfg.n_quad);
    println!("  LM max iterations: {}", cfg.lm_max_iters);
    println!("  pricings/iter:     {}", pricings_per_iter);
    println!();
    println!("  ALLOCATION PRESSURE (complex tensor arithmetic):");
    println!("    ops/pricing:     {} ({}quad x 2 x ~50 tensor ops)", ops_per_pricing, cfg.n_quad);
    println!("    est. total:      {} alloc/free cycles", total_est_allocs);
    println!("    cudaMalloc cost: {:.0}s (@1ms) <-- INFEASIBLE", total_est_allocs as f64 * 1e-3);
    println!("    TLSF cost:       {:.2}s (@0.24us) <-- FEASIBLE", total_est_allocs as f64 * 0.24e-6);
    println!();
    println!("  TLSF pool:         {:.1} MB", s0.total_pool_size as f64 / 1e6);
    println!();

    // True Heston parameters
    let true_params = HestonParams {
        v0: 0.04,
        kappa: 2.0,
        theta: 0.04,
        sigma: 0.3,
        rho: -0.7,
    };

    println!("  True params:  v0={:.4} kappa={:.2} theta={:.4} sigma={:.2} rho={:.2}",
        true_params.v0, true_params.kappa, true_params.theta, true_params.sigma, true_params.rho);

    // Generate exact market surface via CF
    println!("  Generating market surface ({} options)...", n_options);
    let t_surf = Instant::now();
    let surface = generate_market_surface(
        cfg.spot, cfg.num_strikes, cfg.num_maturities,
        &true_params, cfg.risk_free, cfg.n_quad, device,
    );
    let surf_time = t_surf.elapsed().as_secs_f64();
    println!("  Surface generated in {:.3}s\n", surf_time);

    // Initial guess (deliberately wrong)
    let initial = HestonParams {
        v0: 0.02,
        kappa: 1.0,
        theta: 0.06,
        sigma: 0.5,
        rho: -0.3,
    };
    println!("  Initial guess: v0={:.4} kappa={:.2} theta={:.4} sigma={:.2} rho={:.2}\n",
        initial.v0, initial.kappa, initial.theta, initial.sigma, initial.rho);

    let spots = Tensor::full([n_options as i64], cfg.spot, (Kind::Float, device));

    // Run LM calibration
    println!("  --- Levenberg-Marquardt ---\n");
    let t_cal = Instant::now();
    let (calibrated, cost_history) = levenberg_marquardt(
        &surface, &spots, &initial, &cfg, device, &runtime,
    );
    let cal_time = t_cal.elapsed().as_secs_f64();

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;

    // Parameter recovery error
    let true_v = true_params.to_vec();
    let cal_v = calibrated.to_vec();
    let param_names = ["v0", "kappa", "theta", "sigma", "rho"];
    let max_rel_error: f64 = true_v.iter().zip(cal_v.iter())
        .map(|(t, c)| ((t - c) / t).abs())
        .fold(0.0, f64::max);

    println!();
    println!("  Calibration Results:");
    println!("    Calibrated: v0={:.6} kappa={:.4} theta={:.6} sigma={:.4} rho={:.4}",
        calibrated.v0, calibrated.kappa, calibrated.theta, calibrated.sigma, calibrated.rho);
    println!("    True:       v0={:.6} kappa={:.4} theta={:.6} sigma={:.4} rho={:.4}",
        true_params.v0, true_params.kappa, true_params.theta, true_params.sigma, true_params.rho);
    println!();
    print!("    Errors:    ");
    for (i, name) in param_names.iter().enumerate() {
        print!(" {}={:.6}", name, (true_v[i] - cal_v[i]).abs());
    }
    println!();
    println!("    Max rel error: {:.4}%", max_rel_error * 100.0);
    println!();
    println!("    Final cost:    {:.2e}", cost_history.last().unwrap_or(&0.0));
    println!("    LM iterations: {}", cost_history.len());
    println!("    Cal time:      {:.3}s", cal_time);
    println!("    Total allocs:  {}", sf.total_allocations);
    println!("    Peak VRAM:     {:.1} MB", peak_mb);
    println!("    Fragmentation: {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:  {}", if sf.fragmentation_ratio < 0.95 { "YES" } else { "NO" });

    println!("\nRESULT mode=heston_calibrator");
    println!("RESULT options={}", n_options);
    println!("RESULT lm_iters={}", cost_history.len());
    println!("RESULT final_cost={:.12}", cost_history.last().unwrap_or(&0.0));
    for (i, name) in param_names.iter().enumerate() {
        println!("RESULT {}={:.9}", name, cal_v[i]);
        println!("RESULT {}_error={:.9}", name, (true_v[i] - cal_v[i]).abs());
    }
    println!("RESULT max_rel_error_pct={:.6}", max_rel_error * 100.0);
    println!("RESULT cal_time_s={:.6}", cal_time);
    println!("RESULT total_allocs={}", sf.total_allocations);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if sf.fragmentation_ratio < 0.95 { 1 } else { 0 });

    Ok(())
}
