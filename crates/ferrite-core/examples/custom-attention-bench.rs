#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn make_input(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 97) as f32) * 0.013 + 0.1) * scale)
        .collect()
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), DriverError> {
    let batch = env_usize("FERRITE_ATTN_BATCH", 1);
    let heads = env_usize("FERRITE_ATTN_HEADS", 8);
    let seq = env_usize("FERRITE_ATTN_SEQ", 256);
    let dim = env_usize("FERRITE_ATTN_DIM", 64);
    let warmup = env_usize("FERRITE_ATTN_WARMUP", 3);
    let iters = env_usize("FERRITE_ATTN_ITERS", 20);

    if dim > 64 {
        eprintln!("This kernel currently assumes head_dim <= 64. Got {dim}.");
        std::process::exit(2);
    }

    let ptx_path = PathBuf::from(env!("KERNEL_OUTPUT_DIR")).join("flash_attention.ptx");
    let dev = CudaDevice::new(0)?;

    dev.load_ptx(
        Ptx::from_file(&ptx_path),
        "ferrite_flash_attention",
        &["flash_attention_causal_forward"],
    )?;
    let func = dev
        .get_func("ferrite_flash_attention", "flash_attention_causal_forward")
        .expect("flash attention kernel missing after PTX load");

    let elems = batch * heads * seq * dim;
    let q_host = make_input(elems, 1.0);
    let k_host = make_input(elems, 0.5);
    let v_host = make_input(elems, 0.25);
    let mut o_host = vec![0.0f32; elems];

    let q_dev = dev.htod_sync_copy(&q_host)?;
    let k_dev = dev.htod_sync_copy(&k_host)?;
    let v_dev = dev.htod_sync_copy(&v_host)?;
    let mut o_dev = dev.htod_sync_copy(&o_host)?;

    let scale = 1.0f32 / (dim as f32).sqrt();
    let cfg = LaunchConfig {
        block_dim: (64, 1, 1),
        grid_dim: (seq.div_ceil(64) as u32, heads as u32, batch as u32),
        shared_mem_bytes: 0,
    };

    for _ in 0..warmup {
        unsafe {
            func.clone().launch(
                cfg,
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut o_dev,
                    batch as i32,
                    heads as i32,
                    seq as i32,
                    dim as i32,
                    scale,
                ),
            )
        }?;
        dev.synchronize()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            func.clone().launch(
                cfg,
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut o_dev,
                    batch as i32,
                    heads as i32,
                    seq as i32,
                    dim as i32,
                    scale,
                ),
            )
        }?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    dev.dtoh_sync_copy_into(&o_dev, &mut o_host)?;

    let pairs = seq * (seq + 1) / 2;
    let flops_per_iter = 4usize * batch * heads * pairs * dim;
    let total_flops = flops_per_iter as f64 * iters as f64;
    let avg_ms = elapsed.as_secs_f64() * 1e3 / iters as f64;
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;

    let checksum: f64 = o_host.iter().take(dim.min(16)).map(|&x| x as f64).sum();

    println!("Ferrite custom attention benchmark");
    println!("  Kernel: flash_attention_causal_forward");
    println!("  PTX: {}", ptx_path.display());
    println!("  Shape: batch={batch} heads={heads} seq={seq} dim={dim}");
    println!("  Warmup: {warmup}  Iterations: {iters}");
    println!("  Avg latency: {:.3} ms", avg_ms);
    println!("  Estimated throughput: {:.2} GFLOP/s", gflops);
    println!("  Output checksum (first slice): {:.6}", checksum);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Enable the `cuda` feature to run this benchmark.");
    std::process::exit(1);
}
