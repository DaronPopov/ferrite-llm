// =============================================================================
// Ferrite Mathematics Engine — Streaming Greeks Engine (Adjoint/Bump-and-Reval)
// =============================================================================
//
// Computes all risk sensitivities (Greeks) for a large derivative book by
// streaming instrument batches through the TLSF pool. Each batch is priced,
// bumped, re-priced, and freed — enabling books of 100K+ instruments where
// the full sensitivity matrix would exceed VRAM.
//
// Greeks computed: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
//
// Usage:
//   ferrite-run mathematics_engine/greeks/adjoint_greeks.rs \
//     --instruments 100000 --underlyings 5000 --batch-size 10000
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
    num_instruments: usize,
    num_underlyings: usize,
    batch_size: usize,
    wave_streams: usize,
    streams: u32,
    // Bump sizes
    spot_bump: f64,     // relative (1% = 0.01)
    vol_bump: f64,      // absolute (1 vol point = 0.01)
    rate_bump: f64,     // absolute (1bp = 0.0001)
    time_bump: f64,     // days
    // Greeks to compute
    compute_gamma: bool,
    compute_cross: bool, // vanna, volga
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_instruments: 50000,
            num_underlyings: 2000,
            batch_size: 5000,
            wave_streams: 4,
            streams: 32,
            spot_bump: 0.01,
            vol_bump: 0.01,
            rate_bump: 0.0001,
            time_bump: 1.0 / 365.0,
            compute_gamma: true,
            compute_cross: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Instrument representation (vectorized)
// ---------------------------------------------------------------------------

struct InstrumentBatch {
    // [batch] tensors for each parameter
    spots: Tensor,
    strikes: Tensor,
    vols: Tensor,
    rates: Tensor,
    maturities: Tensor,
    is_call: Tensor, // 1.0 for call, -1.0 for put
    notionals: Tensor,
}

fn generate_instrument_batch(
    n: usize,
    device: Device,
    stream_id: usize,
) -> InstrumentBatch {
    set_torch_stream(stream_id);

    let n = n as i64;
    let spots = Tensor::rand([n], (Kind::Float, device)) * 200.0 + 50.0; // 50-250
    let strikes = &spots * (Tensor::rand([n], (Kind::Float, device)) * 0.4 + 0.8); // 80%-120% moneyness
    let vols = Tensor::rand([n], (Kind::Float, device)) * 0.4 + 0.1; // 10%-50% vol
    let rates = Tensor::rand([n], (Kind::Float, device)) * 0.08 + 0.01; // 1%-9%
    let maturities = Tensor::rand([n], (Kind::Float, device)) * 2.0 + 0.1; // 0.1-2.1 years
    let is_call = (Tensor::rand([n], (Kind::Float, device)).ge(0.5)).to_kind(Kind::Float) * 2.0 - 1.0;
    let notionals = Tensor::rand([n], (Kind::Float, device)) * 1_000_000.0 + 100_000.0;

    InstrumentBatch { spots, strikes, vols, rates, maturities, is_call, notionals }
}

// ---------------------------------------------------------------------------
// Vectorized BS pricing
// ---------------------------------------------------------------------------

fn bs_price_batch(
    spots: &Tensor,
    strikes: &Tensor,
    vols: &Tensor,
    rates: &Tensor,
    maturities: &Tensor,
    is_call: &Tensor,
) -> Tensor {
    let sqrt_t = maturities.sqrt();
    let vol_sqrt_t = vols * &sqrt_t;

    let d1 = ((spots / strikes).log()
        + (rates + vols * vols * 0.5) * maturities)
        / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;

    // Approximate N(x) vectorized — Abramowitz & Stegun
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

    let disc = (rates.neg() * maturities).exp();

    // Call = S*N(d1) - K*disc*N(d2)
    // Put  = K*disc*N(-d2) - S*N(-d1)
    let call_price = spots * norm_cdf(&d1) - strikes * &disc * norm_cdf(&d2);
    let put_price = strikes * &disc * norm_cdf(&d2.neg()) - spots * norm_cdf(&d1.neg());

    // Select based on is_call (+1 = call, -1 = put)
    let is_call_mask = is_call.ge(0.0).to_kind(Kind::Float);
    &call_price * &is_call_mask + &put_price * (1.0 - &is_call_mask)
}

// ---------------------------------------------------------------------------
// Greeks via bump-and-reval on a batch
// ---------------------------------------------------------------------------

struct GreeksResult {
    delta: Tensor,
    gamma: Tensor,
    vega: Tensor,
    theta: Tensor,
    rho: Tensor,
    #[allow(dead_code)]
    vanna: Tensor,
    #[allow(dead_code)]
    volga: Tensor,
    base_pv: f64,
    delta_pv: f64, // Σ delta × notional × spot_bump
}

fn compute_greeks_batch(
    batch: &InstrumentBatch,
    cfg: &Config,
) -> GreeksResult {
    let base_price = bs_price_batch(
        &batch.spots, &batch.strikes, &batch.vols,
        &batch.rates, &batch.maturities, &batch.is_call,
    );

    let base_pv = f64::try_from((&base_price * &batch.notionals).sum(Kind::Float)).unwrap_or(0.0);

    // Delta: bump spot up
    let spots_up = &batch.spots * (1.0 + cfg.spot_bump);
    let price_up = bs_price_batch(
        &spots_up, &batch.strikes, &batch.vols,
        &batch.rates, &batch.maturities, &batch.is_call,
    );

    // Delta: bump spot down
    let spots_dn = &batch.spots * (1.0 - cfg.spot_bump);
    let price_dn = bs_price_batch(
        &spots_dn, &batch.strikes, &batch.vols,
        &batch.rates, &batch.maturities, &batch.is_call,
    );

    let bump_abs = &batch.spots * cfg.spot_bump;
    let delta = (&price_up - &price_dn) / (&bump_abs * 2.0);

    // Gamma: (V_up - 2*V_base + V_dn) / bump²
    let gamma = if cfg.compute_gamma {
        (&price_up - &base_price * 2.0 + &price_dn) / bump_abs.pow_tensor_scalar(2.0)
    } else {
        Tensor::zeros_like(&delta)
    };

    // Vega: bump vol
    let vols_up = &batch.vols + cfg.vol_bump;
    let price_vol_up = bs_price_batch(
        &batch.spots, &batch.strikes, &vols_up,
        &batch.rates, &batch.maturities, &batch.is_call,
    );
    let vega = (&price_vol_up - &base_price) / cfg.vol_bump;

    // Theta: bump time
    let mat_dn = (&batch.maturities - cfg.time_bump).clamp_min(1e-6);
    let price_theta = bs_price_batch(
        &batch.spots, &batch.strikes, &batch.vols,
        &batch.rates, &mat_dn, &batch.is_call,
    );
    let theta = (&price_theta - &base_price) / cfg.time_bump;

    // Rho: bump rate
    let rates_up = &batch.rates + cfg.rate_bump;
    let price_rho = bs_price_batch(
        &batch.spots, &batch.strikes, &batch.vols,
        &rates_up, &batch.maturities, &batch.is_call,
    );
    let rho = (&price_rho - &base_price) / cfg.rate_bump;

    // Cross-greeks
    let (vanna, volga) = if cfg.compute_cross {
        // Vanna: d(delta)/d(vol) — bump both spot and vol
        let price_up_vol = bs_price_batch(
            &spots_up, &batch.strikes, &vols_up,
            &batch.rates, &batch.maturities, &batch.is_call,
        );
        let price_dn_vol = bs_price_batch(
            &spots_dn, &batch.strikes, &vols_up,
            &batch.rates, &batch.maturities, &batch.is_call,
        );
        let delta_vol_up = (&price_up_vol - &price_dn_vol) / (&bump_abs * 2.0);
        let vanna = (&delta_vol_up - &delta) / cfg.vol_bump;

        // Volga: d²V/d(vol)² — bump vol up and down
        let vols_dn = (&batch.vols - cfg.vol_bump).clamp_min(0.01);
        let price_vol_dn = bs_price_batch(
            &batch.spots, &batch.strikes, &vols_dn,
            &batch.rates, &batch.maturities, &batch.is_call,
        );
        let volga = (&price_vol_up - &base_price * 2.0 + &price_vol_dn) / (cfg.vol_bump * cfg.vol_bump);

        (vanna, volga)
    } else {
        (Tensor::zeros_like(&delta), Tensor::zeros_like(&delta))
    };

    let delta_pv = f64::try_from(
        (&delta * &batch.notionals * &batch.spots * cfg.spot_bump).sum(Kind::Float)
    ).unwrap_or(0.0);

    GreeksResult {
        delta, gamma, vega, theta, rho, vanna, volga,
        base_pv, delta_pv,
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
            "--instruments" => cfg.num_instruments = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--underlyings" => cfg.num_underlyings = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--batch-size" => cfg.batch_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--spot-bump" => cfg.spot_bump = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--vol-bump" => cfg.vol_bump = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--rate-bump" => cfg.rate_bump = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--no-gamma" => cfg.compute_gamma = false,
            "--no-cross" => cfg.compute_cross = false,
            "-h" | "--help" => {
                println!("Usage: adjoint_greeks.rs [options]");
                println!();
                println!("  Streaming Greeks computation for large derivative books via TLSF.");
                println!();
                println!("  --instruments N    Total instruments in book (default: 50000)");
                println!("  --underlyings N    Unique underlyings (default: 2000)");
                println!("  --batch-size N     Instruments per TLSF batch (default: 5000)");
                println!("  --wave-streams N   Concurrent batch streams (default: 4)");
                println!("  --spot-bump F      Relative spot bump (default: 0.01)");
                println!("  --vol-bump F       Absolute vol bump (default: 0.01)");
                println!("  --rate-bump F      Absolute rate bump (default: 0.0001)");
                println!("  --no-gamma         Skip gamma computation");
                println!("  --no-cross         Skip vanna/volga computation");
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

    let n_batches = cfg.num_instruments.div_ceil(cfg.batch_size);
    let n_revals = if cfg.compute_cross { 11 } else if cfg.compute_gamma { 7 } else { 5 };
    let total_pricings = cfg.num_instruments * n_revals;
    // Each instrument needs ~7 float32 params + bump variants
    let batch_bytes = cfg.batch_size * 7 * 4 * n_revals; // params × revals
    let total_data_bytes = cfg.num_instruments as f64 * 7.0 * 4.0 * n_revals as f64;
    let streaming_budget = cfg.wave_streams as f64 * batch_bytes as f64;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Streaming Greeks Engine ===\n");
    println!("  instruments:      {}", cfg.num_instruments);
    println!("  underlyings:      {}", cfg.num_underlyings);
    println!("  batch size:       {}", cfg.batch_size);
    println!("  batches:          {}", n_batches);
    println!("  wave streams:     {}", cfg.wave_streams);
    println!("  revals/instr:     {} (delta,gamma,vega,theta,rho{})",
        n_revals, if cfg.compute_cross { ",vanna,volga" } else { "" });
    println!("  total pricings:   {}", total_pricings);
    println!();
    println!("  Bumps: spot={:.2}% vol={:.0}bps rate={:.1}bps time={:.0}d",
        cfg.spot_bump * 100.0, cfg.vol_bump * 10000.0, cfg.rate_bump * 10000.0, cfg.time_bump * 365.0);
    println!();
    println!("  TLSF pool:        {:.1} MB", pool_mb);
    println!("  batch working set: {:.1} MB", batch_bytes as f64 / 1e6);
    println!("  total data:       {:.2} GB (virtual — streamed)", total_data_bytes / 1e9);
    println!("  streaming budget: {:.1} MB", streaming_budget / 1e6);
    println!();

    // --- Wave-scheduled Greeks computation ---
    let t_start = Instant::now();
    let mut instruments_processed = 0usize;
    let mut total_pv = 0.0f64;
    let mut total_delta_pv = 0.0f64;
    let mut agg_delta_abs = 0.0f64;
    let mut agg_gamma_abs = 0.0f64;
    let mut agg_vega_abs = 0.0f64;
    let mut agg_theta_sum = 0.0f64;
    let mut agg_rho_abs = 0.0f64;
    let mut total_allocs = 0u64;

    for wave_start in (0..n_batches).step_by(cfg.wave_streams) {
        let wave_end = (wave_start + cfg.wave_streams).min(n_batches);

        for w in wave_start..wave_end {
            let remaining = cfg.num_instruments - w * cfg.batch_size;
            let n_instr = remaining.min(cfg.batch_size);
            let stream_id = (w - wave_start) % active_streams;

            // Generate instrument batch (allocated from TLSF)
            let batch = generate_instrument_batch(n_instr, device, stream_id);

            // Compute all Greeks via bump-and-reval
            let greeks = compute_greeks_batch(&batch, &cfg);

            // Aggregate risk measures
            total_pv += greeks.base_pv;
            total_delta_pv += greeks.delta_pv;
            agg_delta_abs += f64::try_from(greeks.delta.abs().sum(Kind::Float)).unwrap_or(0.0);
            agg_gamma_abs += f64::try_from(greeks.gamma.abs().sum(Kind::Float)).unwrap_or(0.0);
            agg_vega_abs += f64::try_from(greeks.vega.abs().sum(Kind::Float)).unwrap_or(0.0);
            agg_theta_sum += f64::try_from(greeks.theta.sum(Kind::Float)).unwrap_or(0.0);
            agg_rho_abs += f64::try_from(greeks.rho.abs().sum(Kind::Float)).unwrap_or(0.0);

            instruments_processed += n_instr;
            // batch + greeks tensors drop here → TLSF frees everything
        }

        sync_all_streams();

        let s = runtime.tlsf_stats();
        total_allocs = s.total_allocations;

        if wave_start == 0 || (wave_start / cfg.wave_streams) % 5 == 0 {
            let pct = instruments_processed as f64 / cfg.num_instruments as f64 * 100.0;
            println!("  batch {:>3} | {:.1}% | {} instruments | peak={:.0}MB frag={:.6}",
                wave_start / cfg.wave_streams + 1, pct, instruments_processed,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let wall_time = t_start.elapsed().as_secs_f64();

    let n_f = instruments_processed as f64;
    let pricings_per_sec = total_pricings as f64 / wall_time;
    let instruments_per_sec = instruments_processed as f64 / wall_time;

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;

    println!();
    println!("  Greeks Results:");
    println!("    Book PV:           ${:.2}", total_pv);
    println!("    Delta P&L (1%):    ${:.2}", total_delta_pv);
    println!("    Avg |delta|:       {:.6}", agg_delta_abs / n_f);
    println!("    Avg |gamma|:       {:.6}", agg_gamma_abs / n_f);
    println!("    Avg |vega|:        {:.6}", agg_vega_abs / n_f);
    println!("    Total theta:       ${:.2}/day", agg_theta_sum);
    println!("    Avg |rho|:         {:.6}", agg_rho_abs / n_f);
    println!();
    println!("    Instruments:       {}", instruments_processed);
    println!("    Total pricings:    {}", total_pricings);
    println!("    Alloc events:      {}", total_allocs);
    println!("    Wall time:         {:.3}s", wall_time);
    println!("    Pricings/sec:      {:.0} ({:.1}M/s)", pricings_per_sec, pricings_per_sec / 1e6);
    println!("    Instruments/sec:   {:.0}", instruments_per_sec);
    println!();
    println!("    Peak VRAM:         {:.1} MB", peak_mb);
    println!("    Working set total: {:.2} GB (virtual — streamed)", total_data_bytes / 1e9);
    println!("    VRAM savings:      {:.0}x", total_data_bytes / 1e6 / peak_mb.max(1.0));
    println!("    Fragmentation:     {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:      {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=adjoint_greeks");
    println!("RESULT instruments={}", instruments_processed);
    println!("RESULT total_pricings={}", total_pricings);
    println!("RESULT book_pv={:.6}", total_pv);
    println!("RESULT delta_pnl_1pct={:.6}", total_delta_pv);
    println!("RESULT avg_abs_delta={:.9}", agg_delta_abs / n_f);
    println!("RESULT avg_abs_gamma={:.9}", agg_gamma_abs / n_f);
    println!("RESULT avg_abs_vega={:.9}", agg_vega_abs / n_f);
    println!("RESULT total_theta={:.6}", agg_theta_sum);
    println!("RESULT wall_s={:.6}", wall_time);
    println!("RESULT pricings_per_sec={:.0}", pricings_per_sec);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT total_data_gb={:.6}", total_data_bytes / 1e9);
    println!("RESULT vram_savings_x={:.1}", total_data_bytes / 1e6 / peak_mb.max(1.0));
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
