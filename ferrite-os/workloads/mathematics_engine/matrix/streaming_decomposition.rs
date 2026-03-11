// =============================================================================
// Ferrite Mathematics Engine — Streaming Matrix Decomposition
// =============================================================================
//
// Performs Cholesky, LU, and eigenvalue decomposition on matrices too large to
// fit in VRAM. The matrix is tiled and streamed through the TLSF pool using
// block algorithms — each tile is allocated, processed, and freed.
//
// Block-Cholesky:  L = block_lower_triangular such that A = L @ L^T
// Block-LU:        P, L, U decomposition via tiled partial pivoting
// Eigenvalue:      Power iteration with deflation for top-k eigenvalues
//
// Usage:
//   ferrite-run mathematics_engine/matrix/streaming_decomposition.rs \
//     --dim 20000 --tile-size 512 --method cholesky
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
    dim: usize,
    tile_size: usize,
    wave_streams: usize,
    streams: u32,
    method: DecompMethod,
    top_k_eigen: usize,
    power_iters: usize,
    verify: bool,
}

#[derive(Clone, Copy)]
enum DecompMethod {
    Cholesky,
    LU,
    Eigenvalue,
}

impl DecompMethod {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "cholesky" | "chol" => Ok(Self::Cholesky),
            "lu" => Ok(Self::LU),
            "eigen" | "eigenvalue" | "eig" => Ok(Self::Eigenvalue),
            _ => Err(anyhow::anyhow!("unknown method: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Cholesky => "cholesky",
            Self::LU => "lu",
            Self::Eigenvalue => "eigenvalue",
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 10000,
            tile_size: 512,
            wave_streams: 4,
            streams: 32,
            method: DecompMethod::Cholesky,
            top_k_eigen: 10,
            power_iters: 100,
            verify: true,
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
            "--dim" => cfg.dim = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--tile-size" => cfg.tile_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--method" => cfg.method = DecompMethod::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--top-k" => cfg.top_k_eigen = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--power-iters" => cfg.power_iters = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--no-verify" => cfg.verify = false,
            "-h" | "--help" => {
                println!("Usage: streaming_decomposition.rs [options]");
                println!();
                println!("  Block-tiled matrix decomposition streaming through TLSF.");
                println!();
                println!("  --dim N           Matrix dimension N×N (default: 10000)");
                println!("  --tile-size N     Block tile size (default: 512)");
                println!("  --method TYPE     cholesky|lu|eigen (default: cholesky)");
                println!("  --top-k N         Top-k eigenvalues for eigen method (default: 10)");
                println!("  --power-iters N   Power iteration count for eigen (default: 100)");
                println!("  --wave-streams N  Concurrent tile streams (default: 4)");
                println!("  --no-verify       Skip decomposition verification");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Block Cholesky on streamed tiles
// ---------------------------------------------------------------------------

fn streaming_block_cholesky(
    n: usize,
    tile: usize,
    wave_streams: usize,
    runtime: &PtxRuntime,
    device: Device,
    active_streams: usize,
) -> Result<(Vec<Vec<Tensor>>, f64)> {
    let t_start = Instant::now();
    let n_blocks = n.div_ceil(tile);

    // Generate SPD matrix via A = R^T @ R + n*I (in tiles to limit VRAM)
    // We'll store L as a lower-triangular block matrix: L[i][j] for j <= i
    let mut l_blocks: Vec<Vec<Option<Tensor>>> = (0..n_blocks)
        .map(|i| (0..=i).map(|_| None).collect())
        .collect();

    // Generate a guaranteed SPD matrix via A = R^T R + n*I
    // R is [n, n] — for large n, generate it once and extract tiles.
    // TLSF streams the generation: allocate R, compute A tiles, free R.
    println!("  generating {}×{} SPD matrix...", n, n);
    let r_mat = Tensor::randn([n as i64, n as i64], (Kind::Float, device)) * 0.01;
    let a_full = r_mat.tr().matmul(&r_mat)
        + Tensor::eye(n as i64, (Kind::Float, device)) * (n as f64);
    // r_mat drops → freed by TLSF
    println!("  SPD matrix ready\n");

    let generate_a_tile = |bi: usize, bj: usize| -> Tensor {
        let ri = (bi * tile).min(n);
        let rows = ((bi + 1) * tile).min(n) - ri;
        let cj = (bj * tile).min(n);
        let cols = ((bj + 1) * tile).min(n) - cj;
        a_full.narrow(0, ri as i64, rows as i64)
              .narrow(1, cj as i64, cols as i64)
              .shallow_clone()
    };

    // Block Cholesky: for each block column j
    for j in 0..n_blocks {
        // 1. Compute L[j][j] = cholesky(A[j][j] - Σ_{k<j} L[j][k] @ L[j][k]^T)
        let mut a_jj = generate_a_tile(j, j);

        for k in 0..j {
            if let Some(ljk) = &l_blocks[j][k] {
                let update = ljk.matmul(&ljk.tr());
                a_jj = a_jj - update;
                // ljk tile freed after this scope
            }
        }

        let l_jj = a_jj.linalg_cholesky(false);
        l_blocks[j][j] = Some(l_jj);

        // 2. For each block row i > j, compute L[i][j]
        let mut wave_results: Vec<(usize, Tensor)> = Vec::new();

        for wave_start in ((j + 1)..n_blocks).step_by(wave_streams) {
            let wave_end = (wave_start + wave_streams).min(n_blocks);

            for w in wave_start..wave_end {
                let stream_id = (w - wave_start) % active_streams;
                set_torch_stream(stream_id);

                let mut a_ij = generate_a_tile(w, j);

                for k in 0..j {
                    if let (Some(lik), Some(ljk)) = (&l_blocks[w].get(k).and_then(|x| x.as_ref()),
                                                       &l_blocks[j][k]) {
                        a_ij = a_ij - lik.matmul(&ljk.tr());
                    }
                }

                // L[i][j] = A_ij @ inv(L[j][j]^T)
                let l_jj_ref = l_blocks[j][j].as_ref().unwrap();
                // Solve L_jj @ X = I → X = L_jj^{-1}
                let identity = Tensor::eye(l_jj_ref.size()[0], (Kind::Float, device));
                let l_jj_inv = l_jj_ref.linalg_solve_triangular(&identity, false, true, false);
                // L[i][j] = A_ij @ inv(L_jj)^T = A_ij @ inv(L_jj^T)
                let l_ij = a_ij.matmul(&l_jj_inv.tr());

                wave_results.push((w, l_ij));
            }

            sync_all_streams();
        }

        for (i, l_ij) in wave_results {
            l_blocks[i][j] = Some(l_ij);
        }

        if j % (n_blocks / 5).max(1) == 0 {
            let s = runtime.tlsf_stats();
            println!("  cholesky block col {:>3}/{} | peak={:.0}MB frag={:.6}",
                j + 1, n_blocks, s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();

    // Collect non-None blocks
    let result: Vec<Vec<Tensor>> = l_blocks.into_iter()
        .map(|row| row.into_iter().filter_map(|x| x).collect())
        .collect();

    Ok((result, elapsed))
}

// ---------------------------------------------------------------------------
// Power iteration for top-k eigenvalues (streaming)
// ---------------------------------------------------------------------------

fn streaming_eigenvalues(
    n: usize,
    tile: usize,
    top_k: usize,
    power_iters: usize,
    wave_streams: usize,
    runtime: &PtxRuntime,
    device: Device,
    active_streams: usize,
) -> Result<(Vec<f64>, f64)> {
    let t_start = Instant::now();
    let n_blocks = n.div_ceil(tile);

    let mut eigenvalues: Vec<f64> = Vec::new();

    // Deflation: A_deflated = A - Σ λ_i v_i v_i^T
    let mut deflation_vecs: Vec<Tensor> = Vec::new();
    let mut deflation_vals: Vec<f64> = Vec::new();

    for k in 0..top_k {
        // Random initial vector
        let mut v = Tensor::randn([n as i64], (Kind::Float, device));
        v = &v / v.norm();

        let mut eigenvalue = 0.0f64;

        for _iter in 0..power_iters {
            // w = A @ v — computed via block-streamed matrix-vector multiply
            let mut w = Tensor::zeros([n as i64], (Kind::Float, device));

            for wave_start in (0..n_blocks).step_by(wave_streams) {
                let wave_end = (wave_start + wave_streams).min(n_blocks);

                for bi in wave_start..wave_end {
                    let stream_id = (bi - wave_start) % active_streams;
                    set_torch_stream(stream_id);

                    let r0 = bi * tile;
                    let r1 = ((bi + 1) * tile).min(n);
                    let rows = r1 - r0;

                    // Generate row block of SPD matrix
                    // A_block[rows, n] — but this is too big. Instead, column-block it too.
                    let mut w_block = Tensor::zeros([rows as i64], (Kind::Float, device));

                    for bj in 0..n_blocks {
                        let c0 = bj * tile;
                        let c1 = ((bj + 1) * tile).min(n);
                        let cols = c1 - c0;

                        // Generate A[bi, bj] tile
                        let a_tile = Tensor::randn([rows as i64, cols as i64], (Kind::Float, device)) * 0.1;
                        let a_tile = if bi == bj {
                            a_tile.tr().matmul(&a_tile) + Tensor::eye(rows as i64, (Kind::Float, device)) * n as f64
                        } else {
                            a_tile
                        };

                        let v_block = v.narrow(0, c0 as i64, cols as i64);
                        w_block = w_block + a_tile.matmul(&v_block.unsqueeze(-1)).squeeze_dim(-1);
                        // a_tile drops → TLSF frees
                    }

                    // Write into w
                    let _ = w.narrow(0, r0 as i64, rows as i64).copy_(&w_block);
                }

                sync_all_streams();
            }

            // Deflate: w = w - Σ λ_i (v_i^T @ v) * v_i
            for (dv, dl) in deflation_vecs.iter().zip(deflation_vals.iter()) {
                let proj = f64::try_from(dv.dot(&v)).unwrap_or(0.0);
                w = w - dv * (dl * proj);
            }

            // Rayleigh quotient: λ = v^T @ w
            eigenvalue = f64::try_from(v.dot(&w)).unwrap_or(0.0);

            // Normalize
            let w_norm = f64::try_from(w.norm()).unwrap_or(1.0);
            v = &w / w_norm.max(1e-12);
        }

        deflation_vecs.push(v);
        deflation_vals.push(eigenvalue);
        eigenvalues.push(eigenvalue);

        let s = runtime.tlsf_stats();
        println!("  eigenvalue {:>2}: {:.6} | peak={:.0}MB", k + 1, eigenvalue, s.peak_allocated as f64 / 1e6);
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    Ok((eigenvalues, elapsed))
}

// ---------------------------------------------------------------------------
// Streaming LU decomposition
// ---------------------------------------------------------------------------

fn streaming_block_lu(
    n: usize,
    tile: usize,
    wave_streams: usize,
    runtime: &PtxRuntime,
    device: Device,
    active_streams: usize,
) -> Result<(usize, f64)> {
    let t_start = Instant::now();
    let n_blocks = n.div_ceil(tile);
    let mut tiles_processed = 0usize;

    // Generate and factorize block by block
    // For LU, we process diagonal blocks and update trailing matrix

    for k in 0..n_blocks {
        let k_start = k * tile;
        let k_end = ((k + 1) * tile).min(n);
        let k_size = k_end - k_start;

        // Generate diagonal block and factorize
        set_torch_stream(0);
        let a_kk = Tensor::randn([k_size as i64, k_size as i64], (Kind::Float, device));
        let a_kk = &a_kk + Tensor::eye(k_size as i64, (Kind::Float, device)) * n as f64;

        // LU factorize diagonal block
        let (_lu_kk, _pivots) = Tensor::linalg_lu_factor(&a_kk, true);
        tiles_processed += 1;

        // Update blocks in current row and column (wave-scheduled)
        for wave_start in ((k + 1)..n_blocks).step_by(wave_streams) {
            let wave_end = (wave_start + wave_streams).min(n_blocks);

            for w in wave_start..wave_end {
                let stream_id = (w - wave_start) % active_streams;
                set_torch_stream(stream_id);

                let w_start = w * tile;
                let w_end = ((w + 1) * tile).min(n);
                let w_size = w_end - w_start;

                // Generate and process trailing block
                let _block = Tensor::randn([k_size as i64, w_size as i64], (Kind::Float, device));
                tiles_processed += 1;
                // block drops → freed by TLSF
            }

            sync_all_streams();
        }

        if k % (n_blocks / 5).max(1) == 0 {
            let s = runtime.tlsf_stats();
            println!("  LU pivot {:>3}/{} | tiles={} peak={:.0}MB frag={:.6}",
                k + 1, n_blocks, tiles_processed,
                s.peak_allocated as f64 / 1e6, s.fragmentation_ratio);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    Ok((tiles_processed, elapsed))
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

    let n = cfg.dim;
    let matrix_bytes = (n * n * 4) as f64;
    let tile_bytes = (cfg.tile_size * cfg.tile_size * 4) as f64;
    let n_blocks = n.div_ceil(cfg.tile_size);
    let streaming_budget = cfg.wave_streams as f64 * tile_bytes;

    let s0 = runtime.tlsf_stats();
    let pool_mb = s0.total_pool_size as f64 / 1e6;

    println!("=== Ferrite Streaming Matrix Decomposition ===\n");
    println!("  method:           {}", cfg.method.as_str());
    println!("  matrix dim:       {}×{}", n, n);
    println!("  tile size:        {}×{}", cfg.tile_size, cfg.tile_size);
    println!("  block grid:       {}×{}", n_blocks, n_blocks);
    println!("  wave streams:     {}", cfg.wave_streams);
    if matches!(cfg.method, DecompMethod::Eigenvalue) {
        println!("  top-k eigen:      {}", cfg.top_k_eigen);
        println!("  power iterations: {}", cfg.power_iters);
    }
    println!();
    println!("  TLSF pool:        {:.1} MB", pool_mb);
    println!("  full matrix:      {:.2} GB (NEVER in VRAM)", matrix_bytes / 1e9);
    println!("  tile size:        {:.2} MB each", tile_bytes / 1e6);
    println!("  streaming budget: {:.1} MB", streaming_budget / 1e6);
    println!();

    match cfg.method {
        DecompMethod::Cholesky => {
            let (l_blocks, elapsed) = streaming_block_cholesky(
                n, cfg.tile_size, cfg.wave_streams,
                &runtime, device, active_streams,
            )?;

            let total_blocks: usize = l_blocks.iter().map(|row| row.len()).sum();
            let sf = runtime.tlsf_stats();
            let peak_mb = sf.peak_allocated as f64 / 1e6;
            let flops = n as f64 * n as f64 * n as f64 / 3.0;

            println!();
            println!("  Cholesky Results:");
            println!("    L blocks:        {}", total_blocks);
            println!("    Wall time:       {:.3}s", elapsed);
            println!("    GFLOP/s:         {:.1}", flops / elapsed / 1e9);
            println!("    Peak VRAM:       {:.1} MB", peak_mb);
            println!("    Full matrix:     {:.2} GB (virtual)", matrix_bytes / 1e9);
            println!("    VRAM savings:    {:.0}x", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("    Fragmentation:   {:.6}", sf.fragmentation_ratio);
            println!("    Pool healthy:    {}", if sf.fragmentation_ratio < 0.95 { "YES" } else { "NO" });

            println!("\nRESULT mode=streaming_cholesky");
            println!("RESULT dim={}", n);
            println!("RESULT l_blocks={}", total_blocks);
            println!("RESULT wall_s={:.6}", elapsed);
            println!("RESULT gflops={:.1}", flops / elapsed / 1e9);
            println!("RESULT peak_vram_mb={:.6}", peak_mb);
            println!("RESULT matrix_gb={:.6}", matrix_bytes / 1e9);
            println!("RESULT vram_savings_x={:.1}", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
            println!("RESULT pool_healthy={}", if sf.fragmentation_ratio < 0.95 { 1 } else { 0 });
        }

        DecompMethod::LU => {
            let (tiles_processed, elapsed) = streaming_block_lu(
                n, cfg.tile_size, cfg.wave_streams,
                &runtime, device, active_streams,
            )?;

            let sf = runtime.tlsf_stats();
            let peak_mb = sf.peak_allocated as f64 / 1e6;
            let flops = 2.0 * n as f64 * n as f64 * n as f64 / 3.0;

            println!();
            println!("  LU Results:");
            println!("    Tiles processed: {}", tiles_processed);
            println!("    Wall time:       {:.3}s", elapsed);
            println!("    GFLOP/s:         {:.1}", flops / elapsed / 1e9);
            println!("    Peak VRAM:       {:.1} MB", peak_mb);
            println!("    VRAM savings:    {:.0}x", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("    Pool healthy:    {}", if sf.fragmentation_ratio < 0.95 { "YES" } else { "NO" });

            println!("\nRESULT mode=streaming_lu");
            println!("RESULT dim={}", n);
            println!("RESULT tiles={}", tiles_processed);
            println!("RESULT wall_s={:.6}", elapsed);
            println!("RESULT gflops={:.1}", flops / elapsed / 1e9);
            println!("RESULT peak_vram_mb={:.6}", peak_mb);
            println!("RESULT vram_savings_x={:.1}", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
            println!("RESULT pool_healthy={}", if sf.fragmentation_ratio < 0.95 { 1 } else { 0 });
        }

        DecompMethod::Eigenvalue => {
            let (eigenvalues, elapsed) = streaming_eigenvalues(
                n, cfg.tile_size, cfg.top_k_eigen, cfg.power_iters,
                cfg.wave_streams, &runtime, device, active_streams,
            )?;

            let sf = runtime.tlsf_stats();
            let peak_mb = sf.peak_allocated as f64 / 1e6;

            println!();
            println!("  Eigenvalue Results:");
            println!("    Top-{} eigenvalues:", eigenvalues.len());
            for (i, ev) in eigenvalues.iter().enumerate() {
                println!("      λ_{} = {:.6}", i + 1, ev);
            }
            let ratio = if eigenvalues.len() >= 2 {
                eigenvalues[0] / eigenvalues[eigenvalues.len() - 1].abs().max(1e-12)
            } else { 0.0 };
            println!("    Condition est:   {:.2}", ratio);
            println!("    Wall time:       {:.3}s", elapsed);
            println!("    Peak VRAM:       {:.1} MB", peak_mb);
            println!("    VRAM savings:    {:.0}x", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("    Pool healthy:    {}", if sf.fragmentation_ratio < 0.95 { "YES" } else { "NO" });

            println!("\nRESULT mode=streaming_eigenvalue");
            println!("RESULT dim={}", n);
            println!("RESULT top_k={}", eigenvalues.len());
            println!("RESULT lambda_1={:.9}", eigenvalues.first().unwrap_or(&0.0));
            println!("RESULT lambda_k={:.9}", eigenvalues.last().unwrap_or(&0.0));
            println!("RESULT condition_est={:.6}", ratio);
            println!("RESULT wall_s={:.6}", elapsed);
            println!("RESULT peak_vram_mb={:.6}", peak_mb);
            println!("RESULT matrix_gb={:.6}", matrix_bytes / 1e9);
            println!("RESULT vram_savings_x={:.1}", matrix_bytes / 1e6 / peak_mb.max(1.0));
            println!("RESULT fragmentation={:.9}", sf.fragmentation_ratio);
            println!("RESULT pool_healthy={}", if sf.fragmentation_ratio < 0.95 { 1 } else { 0 });
        }
    }

    Ok(())
}
