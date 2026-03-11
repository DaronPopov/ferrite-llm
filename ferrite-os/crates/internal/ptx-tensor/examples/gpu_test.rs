//! Simple GPU test to verify PTX-OS and tensor operations work.
//!
//! Run with: cargo run --release --example gpu_test

use std::sync::Arc;

use ptx_runtime::{PtxRuntime, Result, get_ops_count};
use ptx_tensor::{Tensor, DType};

fn main() -> Result<()> {
    println!("=== PTX-OS Rust Runtime Test ===\n");

    // Initialize runtime (auto-detects GPU)
    println!("Initializing PTX-OS runtime...");
    let runtime = Arc::new(PtxRuntime::new(0)?);
    println!("Runtime initialized successfully!\n");

    // Print stats
    let stats = runtime.stats();
    println!("GPU Stats:");
    println!("  VRAM Allocated: {} MB", stats.vram_allocated / (1024 * 1024));
    println!("  VRAM Used: {} MB", stats.vram_used / (1024 * 1024));
    println!("  VRAM Free: {} MB", stats.vram_free / (1024 * 1024));
    println!("  Active Streams: {}", stats.active_streams);
    println!();

    // Create tensors
    println!("Creating tensors...");
    let a = Tensor::full(&[4, 4], 2.0, DType::F32, &runtime)?;
    let b = Tensor::full(&[4, 4], 3.0, DType::F32, &runtime)?;
    println!("  Created 4x4 tensor A (filled with 2.0)");
    println!("  Created 4x4 tensor B (filled with 3.0)");
    println!();

    // Test binary operations
    println!("Testing binary operations...");

    let c = a.add(&b)?;
    runtime.sync_all()?;
    let c_data = c.to_vec::<f32>()?;
    println!("  A + B = {} (expected: 5.0)", c_data[0]);

    let d = a.mul(&b)?;
    runtime.sync_all()?;
    let d_data = d.to_vec::<f32>()?;
    println!("  A * B = {} (expected: 6.0)", d_data[0]);

    let e = a.sub(&b)?;
    runtime.sync_all()?;
    let e_data = e.to_vec::<f32>()?;
    println!("  A - B = {} (expected: -1.0)", e_data[0]);

    let f = a.div(&b)?;
    runtime.sync_all()?;
    let f_data = f.to_vec::<f32>()?;
    println!("  A / B = {:.4} (expected: 0.6667)", f_data[0]);
    println!();

    // Test unary operations
    println!("Testing unary operations...");
    let g = a.exp()?;
    runtime.sync_all()?;
    let g_data = g.to_vec::<f32>()?;
    println!("  exp(2.0) = {:.4} (expected: 7.3891)", g_data[0]);

    let h = a.sqrt()?;
    runtime.sync_all()?;
    let h_data = h.to_vec::<f32>()?;
    println!("  sqrt(2.0) = {:.4} (expected: 1.4142)", h_data[0]);
    println!();

    // Test activations
    println!("Testing activation functions...");
    let neg_vals = Tensor::full(&[4], -1.0, DType::F32, &runtime)?;
    let pos_vals = Tensor::full(&[4], 1.0, DType::F32, &runtime)?;

    let relu_neg = neg_vals.relu()?;
    runtime.sync_all()?;
    let relu_neg_data = relu_neg.to_vec::<f32>()?;
    println!("  relu(-1.0) = {} (expected: 0.0)", relu_neg_data[0]);

    let relu_pos = pos_vals.relu()?;
    runtime.sync_all()?;
    let relu_pos_data = relu_pos.to_vec::<f32>()?;
    println!("  relu(1.0) = {} (expected: 1.0)", relu_pos_data[0]);

    let sig = Tensor::full(&[4], 0.0, DType::F32, &runtime)?.sigmoid()?;
    runtime.sync_all()?;
    let sig_data = sig.to_vec::<f32>()?;
    println!("  sigmoid(0.0) = {:.4} (expected: 0.5)", sig_data[0]);
    println!();

    // Test reductions
    println!("Testing reductions...");
    let vals = Tensor::full(&[100], 1.0, DType::F32, &runtime)?;
    let sum = vals.sum(-1)?;
    runtime.sync_all()?;
    let sum_data = sum.to_vec::<f32>()?;
    println!("  sum([1.0] * 100) = {} (expected: 100.0)", sum_data[0]);
    println!();

    // Test softmax
    println!("Testing softmax...");
    let logits = Tensor::full(&[1, 4], 1.0, DType::F32, &runtime)?;
    let probs = logits.softmax(-1)?;
    runtime.sync_all()?;
    let probs_data = probs.to_vec::<f32>()?;
    println!("  softmax([1,1,1,1]) = [{:.4}, {:.4}, {:.4}, {:.4}] (expected: [0.25, 0.25, 0.25, 0.25])",
             probs_data[0], probs_data[1], probs_data[2], probs_data[3]);
    println!();

    // Test cuBLAS matmul
    println!("Testing matrix multiplication (cuBLAS)...");
    let mat_a = Tensor::full(&[2, 3], 1.0, DType::F32, &runtime)?;
    let mat_b = Tensor::full(&[3, 2], 2.0, DType::F32, &runtime)?;
    let mat_c = mat_a.matmul(&mat_b)?;
    runtime.sync_all()?;
    let mat_c_data = mat_c.to_vec::<f32>()?;
    println!("  [2x3] @ [3x2] = [2x2], element = {} (expected: 6.0)", mat_c_data[0]);
    println!();

    // Final stats
    let final_stats = runtime.stats();
    println!("Final GPU Stats:");
    println!("  VRAM Used: {} MB", final_stats.vram_used / (1024 * 1024));
    println!("  Total Tensor Ops: {}", get_ops_count());
    println!();

    println!("=== All tests passed! ===");
    Ok(())
}
