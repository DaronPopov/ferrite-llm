//! Simple matrix multiplication example using ptx-compute API.
//!
//! This demonstrates the ergonomic high-level API for matrix operations.
//!
//! Run:
//!   cargo run --release -p ptx-compute --example simple_matmul

use std::sync::Arc;
use ptx_runtime::PtxRuntime;
use ptx_compute::gemm::Matmul;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = 1024;
    let n = 1024;
    let k = 1024;

    println!("Matrix Multiplication Example");
    println!("  A: {}×{}", m, k);
    println!("  B: {}×{}", k, n);
    println!("  C: {}×{}", m, n);
    println!();

    // Initialize runtime
    let runtime = Arc::new(PtxRuntime::new(0)?);
    println!("✓ Runtime initialized");

    // Create matmul helper
    let matmul = Matmul::new(&runtime)?;
    println!("✓ cuBLAS initialized");
    println!();

    // Allocate matrices
    let a_bytes = m * k * std::mem::size_of::<f32>();
    let b_bytes = k * n * std::mem::size_of::<f32>();
    let c_bytes = m * n * std::mem::size_of::<f32>();

    let a = runtime.alloc(a_bytes)?;
    let b = runtime.alloc(b_bytes)?;
    let c = runtime.alloc(c_bytes)?;

    println!("✓ Allocated matrices ({} MB total)",
             (a_bytes + b_bytes + c_bytes) as f64 / 1e6);
    println!();

    // Perform matrix multiplication: C = A @ B
    println!("🚀 Computing C = A @ B...");
    unsafe {
        matmul.multiply_f32(
            a.as_ptr() as *const f32,
            b.as_ptr() as *const f32,
            c.as_ptr() as *mut f32,
            m, n, k,
        )?;
    }
    runtime.sync_all()?;

    println!("✓ Matrix multiplication complete!");
    println!();

    // Calculate FLOPS
    let flops = Matmul::flops(m, n, k);
    println!("  Compute: {:.2} GFLOPS", flops / 1e9);

    Ok(())
}
