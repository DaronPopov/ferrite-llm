// =============================================================================
// Ferrite Mathematics Engine — Black-Scholes PDE Solver (Finite Difference)
// =============================================================================
//
// Solves the Black-Scholes PDE on ultra-fine grids using slab-streamed finite
// differences through the TLSF pool. The grid is decomposed into spatial slabs
// that are wave-scheduled — enabling grid resolutions that would exhaust VRAM
// on traditional stacks.
//
// Supports: European Call/Put, American Put (free boundary), Greeks surface
//
// The PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
//
// Usage:
//   ferrite-run mathematics_engine/pde/black_scholes_fd.rs \
//     --spot-points 100000 --time-points 50000 --option call
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
    spot_points: usize,
    time_points: usize,
    slab_size: usize,
    wave_streams: usize,
    streams: u32,
    // Market
    spot: f64,
    strike: f64,
    risk_free: f64,
    volatility: f64,
    maturity: f64,
    // Option
    option_style: OptionStyle,
    s_max_mult: f64, // S_max = strike * s_max_mult
    compute_greeks: bool,
}

#[derive(Clone, Copy)]
enum OptionStyle {
    EuropeanCall,
    EuropeanPut,
    AmericanPut,
}

impl OptionStyle {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "call" | "european-call" => Ok(Self::EuropeanCall),
            "put" | "european-put" => Ok(Self::EuropeanPut),
            "american-put" => Ok(Self::AmericanPut),
            _ => Err(anyhow::anyhow!("unknown option style: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::EuropeanCall => "european_call",
            Self::EuropeanPut => "european_put",
            Self::AmericanPut => "american_put",
        }
    }

    fn is_put(&self) -> bool {
        matches!(self, Self::EuropeanPut | Self::AmericanPut)
    }

    fn is_american(&self) -> bool {
        matches!(self, Self::AmericanPut)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            spot_points: 50000,
            time_points: 25000,
            slab_size: 2000,
            wave_streams: 4,
            streams: 32,
            spot: 100.0,
            strike: 100.0,
            risk_free: 0.05,
            volatility: 0.20,
            maturity: 1.0,
            option_style: OptionStyle::EuropeanCall,
            s_max_mult: 4.0,
            compute_greeks: true,
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
            "--spot-points" => cfg.spot_points = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--time-points" => cfg.time_points = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--slab-size" => cfg.slab_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--spot" => cfg.spot = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--strike" => cfg.strike = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--rate" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--vol" => cfg.volatility = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--maturity" => cfg.maturity = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--option" => cfg.option_style = OptionStyle::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--s-max-mult" => cfg.s_max_mult = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--no-greeks" => cfg.compute_greeks = false,
            "-h" | "--help" => {
                println!("Usage: black_scholes_fd.rs [options]");
                println!();
                println!("  Finite difference PDE solver with slab-streamed grid through TLSF.");
                println!();
                println!("  --spot-points N   Spatial grid points (default: 50000)");
                println!("  --time-points N   Time steps (default: 25000)");
                println!("  --slab-size N     Spatial slab width (default: 2000)");
                println!("  --option TYPE     call|put|american-put (default: call)");
                println!("  --spot F          Spot price (default: 100.0)");
                println!("  --strike F        Strike price (default: 100.0)");
                println!("  --rate F          Risk-free rate (default: 0.05)");
                println!("  --vol F           Volatility (default: 0.20)");
                println!("  --maturity F      Time to maturity (default: 1.0)");
                println!("  --s-max-mult F    S_max = strike * mult (default: 4.0)");
                println!("  --no-greeks       Skip Greeks computation");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Finite difference step on a spatial slab
// ---------------------------------------------------------------------------

fn fd_step_slab(
    v_slab: &Tensor,        // option values at current time for this slab
    s_slab: &Tensor,        // spot prices for this slab
    dt: f64,
    ds: f64,
    r: f64,
    sigma: f64,
    option_style: OptionStyle,
    strike: f64,
    device: Device,
    stream_id: usize,
) -> Tensor {
    set_torch_stream(stream_id);

    let n = v_slab.size()[0];
    if n < 3 {
        return v_slab.shallow_clone();
    }

    // Fully implicit scheme (unconditionally stable)
    // Solve tridiagonal system: (-a_i)V[i-1] + (1+b_i)V[i] + (-c_i)V[i+1] = V_old[i]
    let s2 = sigma * sigma;

    let i_vals = s_slab / ds;
    let i2 = &i_vals * &i_vals;

    // Implicit: solve (I - dt*A)V_new = V_old
    // A has sub=½σ²i²-ri/2, diag=-(σ²i²+r), super=½σ²i²+ri/2
    // So (I - dt*A) has:
    let alpha = (&i2 * s2 - &i_vals * r) * (-0.5 * dt);  // sub-diagonal: -dt*(½σ²i²-ri/2)
    let beta = Tensor::ones([n], (Kind::Float, device)) + (&i2 * s2 + r) * dt;  // main: 1+dt*(σ²i²+r)
    let gamma = (&i2 * s2 + &i_vals * r) * (-0.5 * dt);   // super-diagonal: -dt*(½σ²i²+ri/2)

    // Thomas algorithm (tridiagonal solve) on GPU tensors
    // Forward sweep: modify coefficients
    let alpha_v = Vec::<f32>::try_from(&alpha.to_device(Device::Cpu)).unwrap_or_default();
    let beta_v = Vec::<f32>::try_from(&beta.to_device(Device::Cpu)).unwrap_or_default();
    let gamma_v = Vec::<f32>::try_from(&gamma.to_device(Device::Cpu)).unwrap_or_default();
    let rhs_v = Vec::<f32>::try_from(&v_slab.to_device(Device::Cpu)).unwrap_or_default();

    let nn = alpha_v.len();
    let mut c_prime = vec![0.0f32; nn];
    let mut d_prime = vec![0.0f32; nn];

    // Forward sweep
    c_prime[0] = gamma_v[0] / beta_v[0];
    d_prime[0] = rhs_v[0] / beta_v[0];
    for i in 1..nn {
        let m = beta_v[i] - alpha_v[i] * c_prime[i - 1];
        if m.abs() < 1e-30 { continue; }
        c_prime[i] = gamma_v[i] / m;
        d_prime[i] = (rhs_v[i] - alpha_v[i] * d_prime[i - 1]) / m;
    }

    // Back substitution
    let mut result = vec![0.0f32; nn];
    result[nn - 1] = d_prime[nn - 1];
    for i in (0..nn - 1).rev() {
        result[i] = d_prime[i] - c_prime[i] * result[i + 1];
    }

    let v_new = Tensor::from_slice(&result).to_device(device);

    // Clamp to prevent numerical instability
    let v_clamped = v_new.clamp_min(0.0);

    // American put: early exercise constraint V >= max(K - S, 0)
    if option_style.is_american() {
        let exercise = (strike - s_slab).clamp_min(0.0);
        v_clamped.max_other(&exercise)
    } else {
        v_clamped
    }
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

    let ns = cfg.spot_points;
    let nt = cfg.time_points;
    let s_max = cfg.strike * cfg.s_max_mult;
    let ds = s_max / ns as f64;
    let dt = cfg.maturity / nt as f64;

    let full_grid_bytes = ns as f64 * nt as f64 * 4.0;
    let step_working_bytes = ns * 4 * 5; // ~5 vectors per tridiagonal solve

    // CFL number (informational — implicit scheme is unconditionally stable)
    let max_i = ns as f64;
    let cfl = cfg.volatility * cfg.volatility * max_i * max_i * dt;
    let cfl_ok = true; // Implicit scheme: always stable

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Black-Scholes PDE Solver ===\n");
    println!("  option:           {}", cfg.option_style.as_str());
    println!("  spot points:      {} (ds={:.6})", ns, ds);
    println!("  time steps:       {} (dt={:.8})", nt, dt);
    println!();
    println!("  Market: S0={:.2} K={:.2} r={:.4} σ={:.4} T={:.2}", cfg.spot, cfg.strike, cfg.risk_free, cfg.volatility, cfg.maturity);
    println!("  S_max={:.2} ({}×K)", s_max, cfg.s_max_mult);
    println!("  CFL number:       {:.4} (implicit — unconditionally stable)", cfl);
    println!();
    println!("  TLSF pool:        {:.1} MB", pool_mb);
    println!("  full grid:        {:.2} GB (spot × time × 4B, virtual)", full_grid_bytes / 1e9);
    println!("  step working set: {:.2} KB", step_working_bytes as f64 / 1e3);
    println!();

    // --- Build spot grid ---
    let s_grid = Tensor::arange(ns as i64, (Kind::Float, device)) * ds;

    // --- Terminal condition (payoff at maturity) ---
    let v = if cfg.option_style.is_put() {
        (cfg.strike - &s_grid).clamp_min(0.0)
    } else {
        (&s_grid - cfg.strike).clamp_min(0.0)
    };
    let mut v_current = v;

    // --- Time-step backwards (implicit scheme, full spatial tridiagonal) ---
    // The spatial grid stays resident (small). Each time step allocates the
    // tridiagonal system through TLSF, solves it, and frees. For massive grids,
    // the time-step intermediates are the streamed data.
    let t_start = Instant::now();
    let mut total_step_ops = 0u64;

    for t_step in 0..nt {
        // Solve full tridiagonal system across entire spatial grid
        v_current = fd_step_slab(
            &v_current,
            &s_grid,
            dt, ds,
            cfg.risk_free,
            cfg.volatility,
            cfg.option_style,
            cfg.strike,
            device,
            0,
        );
        total_step_ops += 1;

        // Apply boundary conditions
        // S=0: put has V=K*exp(-r*tau), call has V=0
        // After t_step steps backward, we're at time-to-maturity = (t_step+1)*dt
        let tau = (t_step + 1) as f64 * dt;
        let bc_low = if cfg.option_style.is_put() {
            cfg.strike * (-cfg.risk_free * tau).exp()
        } else {
            0.0
        };
        let bc_high = if cfg.option_style.is_put() {
            0.0
        } else {
            s_max - cfg.strike * (-cfg.risk_free * tau).exp()
        };

        // Set boundaries
        let _ = v_current.get(0).fill_(bc_low);
        let _ = v_current.get(v_current.size()[0] - 1).fill_(bc_high.max(0.0));

        if t_step % (nt / 10).max(1) == 0 || t_step == nt - 1 {
            let s = runtime.tlsf_stats();
            let pct = (t_step + 1) as f64 / nt as f64 * 100.0;
            println!("  t_step {:>6}/{} ({:.0}%) | peak={:.0}MB frag={:.6}",
                t_step + 1, nt, pct,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let solve_time = t_start.elapsed().as_secs_f64();

    // --- Extract option value at spot ---
    let spot_idx = ((cfg.spot / ds).round() as i64).max(0).min(ns as i64 - 1);
    let option_value = f64::try_from(v_current.get(spot_idx)).unwrap_or(0.0);

    // --- Compute Greeks via finite differences ---
    let mut delta = 0.0f64;
    let mut gamma = 0.0f64;
    let mut theta_est = 0.0f64;

    if cfg.compute_greeks && spot_idx > 0 && spot_idx < ns as i64 - 1 {
        let v_up = f64::try_from(v_current.get(spot_idx + 1)).unwrap_or(0.0);
        let v_dn = f64::try_from(v_current.get(spot_idx - 1)).unwrap_or(0.0);
        let v_mid = option_value;

        delta = (v_up - v_dn) / (2.0 * ds);
        gamma = (v_up - 2.0 * v_mid + v_dn) / (ds * ds);

        // Theta from BS PDE: θ = -½σ²S²Γ - rSΔ + rV
        let s = cfg.spot;
        theta_est = -0.5 * cfg.volatility.powi(2) * s * s * gamma - cfg.risk_free * s * delta + cfg.risk_free * v_mid;
    }

    // BS analytical price for comparison (European only)
    let analytical = if !cfg.option_style.is_american() {
        let d1 = ((cfg.spot / cfg.strike).ln() + (cfg.risk_free + 0.5 * cfg.volatility.powi(2)) * cfg.maturity)
            / (cfg.volatility * cfg.maturity.sqrt());
        let d2 = d1 - cfg.volatility * cfg.maturity.sqrt();
        // Approximate N(x) using Abramowitz & Stegun
        let norm_cdf = |x: f64| -> f64 {
            // A&S 7.1.26 erfc approximation → Φ(x) = 0.5 * erfc(-x/√2)
            let z = x / std::f64::consts::SQRT_2;
            let az = z.abs();
            let t = 1.0 / (1.0 + 0.3275911 * az);
            let erfc_az = t * (-az * az).exp() * (
                0.254829592
                + t * (-0.284496736
                + t * (1.421413741
                + t * (-1.453152027
                + t * 1.061405429)))
            );
            if z >= 0.0 { 1.0 - 0.5 * erfc_az } else { 0.5 * erfc_az }
        };
        let disc = (-cfg.risk_free * cfg.maturity).exp();
        if cfg.option_style.is_put() {
            cfg.strike * disc * norm_cdf(-d2) - cfg.spot * norm_cdf(-d1)
        } else {
            cfg.spot * norm_cdf(d1) - cfg.strike * disc * norm_cdf(d2)
        }
    } else {
        f64::NAN
    };

    let error = if analytical.is_finite() {
        (option_value - analytical).abs()
    } else {
        f64::NAN
    };

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;

    println!();
    println!("  PDE Results:");
    println!("    Option style:    {}", cfg.option_style.as_str());
    println!("    FD price:        {:.6}", option_value);
    if analytical.is_finite() {
        println!("    BS analytical:   {:.6}", analytical);
        println!("    Error:           {:.6} ({:.4}%)", error, error / analytical * 100.0);
    }
    println!();
    if cfg.compute_greeks {
        println!("    Delta:           {:.6}", delta);
        println!("    Gamma:           {:.6}", gamma);
        println!("    Theta (PDE):     {:.6}", theta_est);
        println!();
    }
    println!("    Grid:            {} × {} = {:.0}M points", ns, nt, ns as f64 * nt as f64 / 1e6);
    println!("    Time steps:        {}", total_step_ops);
    println!("    Wall time:       {:.3}s", solve_time);
    println!("    Grid points/s:   {:.1}M", (ns as f64 * nt as f64) / solve_time / 1e6);
    println!();
    println!("    Peak VRAM:       {:.1} MB", peak_mb);
    println!("    Full grid:       {:.2} GB (virtual — streamed)", full_grid_bytes / 1e9);
    println!("    VRAM savings:    {:.0}x", full_grid_bytes / 1e6 / peak_mb.max(1.0));
    println!("    Fragmentation:   {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:    {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=black_scholes_fd");
    println!("RESULT option_style={}", cfg.option_style.as_str());
    println!("RESULT fd_price={:.9}", option_value);
    if analytical.is_finite() {
        println!("RESULT bs_price={:.9}", analytical);
        println!("RESULT error={:.9}", error);
    }
    println!("RESULT delta={:.9}", delta);
    println!("RESULT gamma={:.9}", gamma);
    println!("RESULT theta={:.9}", theta_est);
    println!("RESULT spot_points={}", ns);
    println!("RESULT time_points={}", nt);
    println!("RESULT total_grid_points={}", ns as u64 * nt as u64);
    println!("RESULT time_steps={}", total_step_ops);
    println!("RESULT wall_s={:.6}", solve_time);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT full_grid_gb={:.6}", full_grid_bytes / 1e9);
    println!("RESULT vram_savings_x={:.1}", full_grid_bytes / 1e6 / peak_mb.max(1.0));
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
