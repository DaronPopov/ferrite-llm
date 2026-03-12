use ferrite_core::{create_ug_cuda_device, ug, ug_cuda};
use ferrite_core::ug::Slice;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let n_rows = env_usize("FERRITE_UG_ROWS", 4096);
    let n_cols = env_usize("FERRITE_UG_COLS", 1024);
    let warmup = env_usize("FERRITE_UG_WARMUP", 5);
    let iters = env_usize("FERRITE_UG_ITERS", 50);

    if n_cols == 0 || n_cols > 1024 {
        anyhow::bail!("FERRITE_UG_COLS must be between 1 and 1024, got {n_cols}");
    }

    let kernel = ug::samples::ssa::exp(n_cols)?;
    let mut buf = Vec::new();
    ug_cuda::code_gen::gen(&mut buf, "ferrite_ug_exp", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;

    let device = create_ug_cuda_device(0)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (n_cols as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = device.compile_cu(&cuda_code, "ferrite_ug_cuda_bench", "ferrite_ug_exp")?;
    let func = ug_cuda::runtime::Func::new(func, cfg);

    let n_elements = n_rows * n_cols;
    let input_host: Vec<f32> = (0..n_elements)
        .map(|i| ((i % 251) as f32) * 0.001 - 0.125)
        .collect();
    let input = device.slice_from_values(&input_host)?;
    let output = device.zeros(n_elements)?;

    for _ in 0..warmup {
        unsafe { func.launch2((input.slice::<f32>()?, output.slice::<f32>()?))? };
        device.synchronize()?;
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { func.launch2((input.slice::<f32>()?, output.slice::<f32>()?))? };
    }
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();

    let result: Vec<f32> = output.to_vec()?;
    let checksum: f64 = result.iter().take(16).map(|&v| v as f64).sum();
    let bytes_per_iter = (n_elements * std::mem::size_of::<f32>() * 2) as f64;
    let gbps = (bytes_per_iter * iters as f64) / elapsed / 1e9;
    let avg_ms = elapsed * 1e3 / iters as f64;

    println!("Ferrite ug-cuda benchmark");
    println!("  Kernel: exp");
    println!("  Shape: rows={n_rows} cols={n_cols}");
    println!("  Warmup: {warmup}  Iterations: {iters}");
    println!("  Avg latency: {:.3} ms", avg_ms);
    println!("  Estimated throughput: {:.3} GB/s", gbps);
    println!("  Output checksum (first 16): {:.6}", checksum);

    Ok(())
}
