//! Tiled matrix multiplication example.
//!
//! Demonstrates custom tiling configuration for optimal GPU performance,
//! similar to NVIDIA's CuTe library.
//!
//! Run:
//!   cargo run --release -p ptx-compute --example tiled_matmul

use std::sync::Arc;
use ptx_runtime::PtxRuntime;
use ptx_compute::tiling::{TileConfig, TiledMatmul, suggest_tile_config, TileIterator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Tiled Matrix Multiplication Example                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let m = 2048;
    let n = 2048;
    let k = 2048;

    println!("Problem size:");
    println!("  A: {}×{}", m, k);
    println!("  B: {}×{}", k, n);
    println!("  C: {}×{}", m, n);
    println!();

    // Initialize runtime
    let runtime = Arc::new(PtxRuntime::new(0)?);
    println!("✓ Runtime initialized");
    println!();

    // Get suggested tile configuration
    let suggested = suggest_tile_config(m, n, k);
    println!("📐 Suggested tile configuration:");
    println!("  Block tile: {}×{} (k={})",
             suggested.block_tile_m,
             suggested.block_tile_n,
             suggested.block_tile_k);
    println!("  Thread tile: {}×{}",
             suggested.thread_tile_m,
             suggested.thread_tile_n);
    println!("  Threads/block: {}×{} = {} threads",
             suggested.threads_x,
             suggested.threads_y,
             suggested.threads_x * suggested.threads_y);
    println!();

    // Calculate grid dimensions
    let (grid_m, grid_n) = suggested.grid_dims(m, n);
    println!("  Grid dimensions: {}×{} blocks", grid_m, grid_n);
    println!("  Total blocks: {}", grid_m * grid_n);
    println!();

    // Calculate memory usage
    let shmem = suggested.shared_memory_bytes(4);
    println!("  Shared memory/block: {:.1} KB", shmem as f64 / 1024.0);
    println!();

    // Create custom configuration
    println!("🔧 Creating custom tile configuration...");
    let config = TileConfig::new()
        .block_tile(128, 128)
        .block_tile_k(16)
        .thread_tile(8, 8)
        .threads(16, 16);

    config.validate()?;
    println!("✓ Custom configuration validated");
    println!();

    // Create tiled matmul
    let tiled_mm = TiledMatmul::new(&runtime, config)?;
    println!("✓ Tiled matmul initialized");
    println!();

    // Allocate matrices
    let a_bytes = m * k * std::mem::size_of::<f32>();
    let b_bytes = k * n * std::mem::size_of::<f32>();
    let c_bytes = m * n * std::mem::size_of::<f32>();

    let a = runtime.alloc(a_bytes)?;
    let b = runtime.alloc(b_bytes)?;
    let c = runtime.alloc(c_bytes)?;

    println!("📦 Allocated matrices:");
    println!("   Total: {:.2} GB", (a_bytes + b_bytes + c_bytes) as f64 / 1e9);
    println!();

    // Show tiling info
    let (grid_m_custom, grid_n_custom) = tiled_mm.grid_dims(m, n);
    println!("🎯 Tiling strategy:");
    println!("   Grid: {}×{} = {} blocks", grid_m_custom, grid_n_custom,
             grid_m_custom * grid_n_custom);
    println!("   Shared memory: {:.1} KB/block", tiled_mm.shared_memory_bytes() as f64 / 1024.0);
    println!("   Total shared memory: {:.2} MB",
             (tiled_mm.shared_memory_bytes() * grid_m_custom * grid_n_custom) as f64 / 1e6);
    println!();

    // Perform tiled matmul
    let stream = runtime.stream(0)?;
    println!("🚀 Launching tiled matrix multiplication...");

    unsafe {
        tiled_mm.multiply_f32(
            a.as_ptr() as *const f32,
            b.as_ptr() as *const f32,
            c.as_ptr() as *mut f32,
            m, n, k,
            &stream,
        )?;
    }
    runtime.sync_all()?;

    println!("✓ Tiled matmul complete!");
    println!();

    // Show FLOPS
    let flops = TiledMatmul::flops(m, n, k);
    println!("💡 Compute:");
    println!("   {:.2} GFLOPS", flops / 1e9);
    println!();

    // Demonstrate tile iterator for large problems
    println!("📊 Tile Iterator Demo:");
    println!("   Processing problem in tiles for memory efficiency");

    let mut iter = TileIterator::new(m, n, 512, 512);
    println!("   Total tiles to process: {}", iter.num_tiles());

    let mut tile_count = 0;
    while let Some((start_m, start_n, extent_m, extent_n)) = iter.next() {
        tile_count += 1;
        if tile_count <= 3 {
            println!("   Tile {}: [{}, {}] size {}×{}",
                     tile_count, start_m, start_n, extent_m, extent_n);
        }
    }
    if tile_count > 3 {
        println!("   ... and {} more tiles", tile_count - 3);
    }
    println!();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Custom tiling enables optimal GPU memory hierarchy usage!      ║");
    println!("║  - Registers (thread tiles) - fastest                           ║");
    println!("║  - Shared memory (block tiles) - 100x faster than global        ║");
    println!("║  - Global memory - largest but slowest                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
