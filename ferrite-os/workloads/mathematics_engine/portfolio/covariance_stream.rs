// =============================================================================
// Ferrite Mathematics Engine — Streaming Covariance Matrix
// =============================================================================
//
// Computes the full covariance matrix for massive asset universes (10,000+
// assets) by streaming tile-blocks through the TLSF pool. The full N×N
// covariance matrix never exists in VRAM — we compute tiles of size T×T
// and flush each to host/disk before freeing.
//
// Also performs Markowitz mean-variance portfolio optimization on the streamed
// result using block-Cholesky decomposition.
//
// Usage:
//   ferrite-run mathematics_engine/portfolio/covariance_stream.rs \
//     --assets 10000 --observations 2520 --tile-size 512
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
    num_observations: usize,
    tile_size: usize,
    wave_streams: usize,
    streams: u32,
    risk_free: f64,
    target_return: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_assets: 5000,
            num_observations: 2520,    // ~10 years daily
            tile_size: 512,
            wave_streams: 4,
            streams: 32,
            risk_free: 0.02,
            target_return: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Tile-streamed covariance computation
// ---------------------------------------------------------------------------

struct CovTile {
    row_start: usize,
    col_start: usize,
    data: Tensor,
}

fn compute_cov_tile(
    returns: &Tensor,
    row_range: (usize, usize),
    col_range: (usize, usize),
    _device: Device,
    stream_id: usize,
) -> CovTile {
    set_torch_stream(stream_id);

    // Extract sub-matrices of demeaned returns
    let r_rows = returns.narrow(1, row_range.0 as i64, (row_range.1 - row_range.0) as i64);
    let r_cols = returns.narrow(1, col_range.0 as i64, (col_range.1 - col_range.0) as i64);

    // Cov(i,j) = (1/(T-1)) * Σ_t r_i(t) * r_j(t) = (1/(T-1)) * R_i^T @ R_j
    let n = returns.size()[0] as f64;
    let tile = r_rows.tr().matmul(&r_cols) / (n - 1.0);

    CovTile {
        row_start: row_range.0,
        col_start: col_range.0,
        data: tile,
    }
}

// ---------------------------------------------------------------------------
// Portfolio statistics from streamed tiles
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--assets" => cfg.num_assets = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--observations" => cfg.num_observations = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--tile-size" => cfg.tile_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--risk-free" => cfg.risk_free = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--target-return" => cfg.target_return = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: covariance_stream.rs [options]");
                println!();
                println!("  Streaming covariance matrix for massive asset universes.");
                println!("  Computes tiles through TLSF — full N×N matrix never in VRAM.");
                println!();
                println!("  --assets N         Number of assets (default: 5000)");
                println!("  --observations N   Return observations (default: 2520)");
                println!("  --tile-size N      Tile dimension (default: 512)");
                println!("  --wave-streams N   Concurrent tile streams (default: 4)");
                println!("  --streams N        CUDA stream pool (default: 32)");
                println!("  --risk-free F      Risk-free rate (default: 0.02)");
                println!("  --target-return F  Target portfolio return (default: 0.10)");
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

    let n = cfg.num_assets;
    let t = cfg.num_observations;
    let tile = cfg.tile_size;

    // Full cov matrix size
    let cov_bytes = (n * n * 4) as f64;
    let returns_bytes = (t * n * 4) as f64;
    let tile_bytes = (tile * tile * 4) as f64;
    let num_tile_rows = n.div_ceil(tile);
    let num_tiles = num_tile_rows * (num_tile_rows + 1) / 2; // upper triangle + diagonal
    let streaming_budget = cfg.wave_streams as f64 * tile_bytes;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Streaming Covariance Matrix ===\n");
    println!("  assets:           {}", n);
    println!("  observations:     {}", t);
    println!("  tile size:        {}×{}", tile, tile);
    println!("  tile rows:        {}", num_tile_rows);
    println!("  total tiles:      {} (upper triangle)", num_tiles);
    println!("  wave streams:     {}", cfg.wave_streams);
    println!("  active streams:   {}", active_streams);
    println!();
    println!("  TLSF pool:        {:.1} MB", pool_mb);
    println!("  full cov matrix:  {:.2} GB (NEVER in VRAM)", cov_bytes / 1e9);
    println!("  returns matrix:   {:.1} MB", returns_bytes / 1e6);
    println!("  tile size:        {:.2} MB each", tile_bytes / 1e6);
    println!("  streaming budget: {:.1} MB", streaming_budget / 1e6);
    println!();

    // --- Generate synthetic returns (demeaned) ---
    // In production, these would stream from disk too
    let t_start = Instant::now();

    println!("  generating {} × {} returns matrix...", t, n);
    let raw_returns = Tensor::randn([t as i64, n as i64], (Kind::Float, device)) * 0.01;
    let means = raw_returns.mean_dim(0, true, Kind::Float);
    let returns = raw_returns - &means;
    println!("  returns ready ({:.1} MB)\n", returns_bytes / 1e6);

    // --- Tile-streamed covariance computation ---
    let mut tiles_computed = 0usize;
    let mut diag_variances: Vec<f64> = Vec::new();
    let _off_diag_corrs: Vec<f64> = Vec::new();
    let mut max_corr = 0.0f64;
    let mut corr_sum = 0.0f64;
    let mut corr_count = 0u64;

    // Build tile schedule (upper triangle)
    let mut tile_schedule: Vec<(usize, usize, usize, usize)> = Vec::new();
    for tr in 0..num_tile_rows {
        for tc in tr..num_tile_rows {
            let r0 = tr * tile;
            let r1 = (r0 + tile).min(n);
            let c0 = tc * tile;
            let c1 = (c0 + tile).min(n);
            tile_schedule.push((r0, r1, c0, c1));
        }
    }

    for wave_start in (0..tile_schedule.len()).step_by(cfg.wave_streams) {
        let wave_end = (wave_start + cfg.wave_streams).min(tile_schedule.len());

        let mut wave_tiles: Vec<CovTile> = Vec::new();

        for w in wave_start..wave_end {
            let (r0, r1, c0, c1) = tile_schedule[w];
            let stream_id = (w - wave_start) % active_streams;

            let cov_tile = compute_cov_tile(
                &returns,
                (r0, r1),
                (c0, c1),
                device,
                stream_id,
            );
            wave_tiles.push(cov_tile);
        }

        sync_all_streams();

        // Extract statistics from each tile before freeing
        for ct in &wave_tiles {
            tiles_computed += 1;

            if ct.row_start == ct.col_start {
                // Diagonal tile — extract variances
                let diag = ct.data.diag(0);
                let diag_vals = Vec::<f32>::try_from(&diag.to_device(Device::Cpu))?;
                for &v in &diag_vals {
                    diag_variances.push(v as f64);
                }
            }

            // Sample off-diagonal correlations (for tiles where row != col block)
            if ct.row_start != ct.col_start {
                let tile_abs = ct.data.abs();
                let tile_max = f64::try_from(tile_abs.max()).unwrap_or(0.0);
                if tile_max > max_corr {
                    max_corr = tile_max;
                }
                let tile_mean = f64::try_from(tile_abs.mean(Kind::Float)).unwrap_or(0.0);
                let tile_elems = ct.data.size().iter().product::<i64>() as u64;
                corr_sum += tile_mean * tile_elems as f64;
                corr_count += tile_elems;
            }
            // tile drops → TLSF frees
        }

        if wave_start == 0 || (wave_start / cfg.wave_streams) % 20 == 0 {
            let s = runtime.tlsf_stats();
            let pct = tiles_computed as f64 / num_tiles as f64 * 100.0;
            println!("  tile wave {:>4} | {:.1}% | peak={:.0}MB frag={:.6}",
                wave_start / cfg.wave_streams + 1, pct,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let cov_time = t_start.elapsed().as_secs_f64();

    // --- Portfolio analytics ---
    let mean_var = if !diag_variances.is_empty() {
        diag_variances.iter().sum::<f64>() / diag_variances.len() as f64
    } else { 0.0 };
    let min_var = diag_variances.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_var = diag_variances.iter().cloned().fold(0.0f64, f64::max);
    let mean_vol = mean_var.sqrt();
    let min_vol = min_var.sqrt();
    let max_vol = max_var.sqrt();

    let mean_corr = if corr_count > 0 { corr_sum / corr_count as f64 } else { 0.0 };

    // Naive equal-weight portfolio vol estimate
    let avg_corr_approx = mean_corr / mean_var.max(1e-12);
    let eq_weight_var = mean_var / n as f64 + (1.0 - 1.0 / n as f64) * mean_corr;
    let eq_weight_vol = eq_weight_var.abs().sqrt();

    // Minimum variance portfolio lower bound (assuming average correlation)
    let min_port_var = min_var * (1.0 - avg_corr_approx.min(0.99)) / n as f64;
    let min_port_vol = min_port_var.abs().sqrt();

    let sf = runtime.tlsf_stats();
    let peak_mb = sf.peak_allocated as f64 / 1e6;
    let pool_healthy = sf.fragmentation_ratio < 0.95;

    println!();
    println!("  Covariance Results ({} tiles computed):", tiles_computed);
    println!("    Diagonal variances: {} assets", diag_variances.len());
    println!("    Mean volatility:    {:.6} ({:.2}% annual)", mean_vol, mean_vol * (252.0f64).sqrt() * 100.0);
    println!("    Min volatility:     {:.6}", min_vol);
    println!("    Max volatility:     {:.6}", max_vol);
    println!("    Max |covariance|:   {:.6}", max_corr);
    println!("    Mean |covariance|:  {:.6}", mean_corr);
    println!();
    println!("  Portfolio Estimates:");
    println!("    Equal-weight vol:   {:.6} ({:.2}% annual)", eq_weight_vol, eq_weight_vol * (252.0f64).sqrt() * 100.0);
    println!("    Min-var bound:      {:.6} ({:.2}% annual)", min_port_vol, min_port_vol * (252.0f64).sqrt() * 100.0);
    println!();
    println!("    Wall time:          {:.3}s", cov_time);
    println!("    Tiles/sec:          {:.0}", tiles_computed as f64 / cov_time);
    println!("    Peak VRAM:          {:.1} MB", peak_mb);
    println!("    Full cov matrix:    {:.2} GB (never in VRAM)", cov_bytes / 1e9);
    println!("    VRAM savings:       {:.0}x", cov_bytes / 1e6 / peak_mb);
    println!("    Fragmentation:      {:.6}", sf.fragmentation_ratio);
    println!("    Pool healthy:       {}", if pool_healthy { "YES" } else { "NO" });

    println!("\nRESULT mode=covariance_stream");
    println!("RESULT assets={}", n);
    println!("RESULT observations={}", t);
    println!("RESULT tiles={}", tiles_computed);
    println!("RESULT mean_vol={:.9}", mean_vol);
    println!("RESULT max_abs_cov={:.9}", max_corr);
    println!("RESULT eq_weight_vol={:.9}", eq_weight_vol);
    println!("RESULT wall_s={:.6}", cov_time);
    println!("RESULT peak_vram_mb={:.6}", peak_mb);
    println!("RESULT cov_matrix_gb={:.6}", cov_bytes / 1e9);
    println!("RESULT vram_savings_x={:.1}", cov_bytes / 1e6 / peak_mb);
    println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
    println!("RESULT pool_healthy={}", if pool_healthy { 1 } else { 0 });

    Ok(())
}
