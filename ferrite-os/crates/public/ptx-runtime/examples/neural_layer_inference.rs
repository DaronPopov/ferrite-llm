//! Neural Network Layer Inference Example
//!
//! Demonstrates a complete forward pass through a neural network layer using:
//! - Stream-ordered TLSF allocations
//! - Candle kernels via guard layer
//! - Realistic ML workload (matrix multiply + activation)

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, test_kernels};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Network Layer Inference");
    println!("====================================\n");

    // Initialize runtime with production config
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.75;  // Use 75% for production workloads
    config.max_streams = 32;
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    println!("✓ PTX-OS Runtime initialized");
    println!("  Device ID: {}", runtime.device().id());

    let stats = runtime.tlsf_stats();
    println!("  TLSF Pool: {:.2} GB", stats.total_pool_size as f64 / 1e9);

    // Layer configuration (typical small layer)
    const BATCH_SIZE: usize = 32;
    const INPUT_DIM: usize = 512;
    const HIDDEN_DIM: usize = 2048;

    let input_size = BATCH_SIZE * INPUT_DIM;
    let weights_size = INPUT_DIM * HIDDEN_DIM;
    let output_size = BATCH_SIZE * HIDDEN_DIM;

    println!("\n📐 Layer Configuration:");
    println!("  Input: [{}x{}]", BATCH_SIZE, INPUT_DIM);
    println!("  Weights: [{}x{}]", INPUT_DIM, HIDDEN_DIM);
    println!("  Output: [{}x{}]", BATCH_SIZE, HIDDEN_DIM);

    // Get dedicated stream for this workload
    let stream = runtime.next_stream();
    println!("\n🌊 Using stream {}", stream.id());

    // Stream-ordered allocations
    println!("\n📦 Allocating buffers (stream-ordered)...");
    let start = Instant::now();

    let input_ptr = runtime.alloc_async(input_size * 4, &stream)?;
    let weights_ptr = runtime.alloc_async(weights_size * 4, &stream)?;
    let matmul_out_ptr = runtime.alloc_async(output_size * 4, &stream)?;
    let final_out_ptr = runtime.alloc_async(output_size * 4, &stream)?;

    let alloc_time = start.elapsed();
    println!("  ✓ 4 buffers allocated in {:?}", alloc_time);
    println!("  Total GPU memory: {:.2} MB",
        (input_size + weights_size + output_size * 2) * 4 / 1024 / 1024);

    // Prepare host data
    println!("\n🔧 Preparing input data...");
    let host_input: Vec<f32> = (0..input_size)
        .map(|i| ((i % 100) as f32) / 100.0)
        .collect();
    let host_weights: Vec<f32> = (0..weights_size)
        .map(|i| ((i % 50) as f32) / 50.0 - 0.5)
        .collect();
    let mut host_output = vec![0.0f32; output_size];

    // Stream-ordered data transfer
    println!("\n📤 Transferring data to GPU (stream-ordered)...");
    let start = Instant::now();

    unsafe {
        ptx_sys::cudaMemcpyAsync(
            input_ptr,
            host_input.as_ptr() as *const _,
            input_size * 4,
            ptx_sys::cudaMemcpyHostToDevice,
            stream.raw(),
        );
        ptx_sys::cudaMemcpyAsync(
            weights_ptr,
            host_weights.as_ptr() as *const _,
            weights_size * 4,
            ptx_sys::cudaMemcpyHostToDevice,
            stream.raw(),
        );
    }

    let transfer_time = start.elapsed();
    println!("  ✓ Data transferred in {:?}", transfer_time);

    // Create guarded buffers
    println!("\n🛡️  Creating guarded buffers...");
    let input_buf = unsafe {
        GuardedBuffer::new(input_ptr, input_size * 4, runtime.raw())?
    };
    let matmul_buf = unsafe {
        GuardedBuffer::new(matmul_out_ptr, output_size * 4, runtime.raw())?
    };
    let output_buf = unsafe {
        GuardedBuffer::new(final_out_ptr, output_size * 4, runtime.raw())?
    };

    let context = KernelContext::new(runtime.raw(), stream.raw())?;

    // Inference: [Input × Weights] → GELU → Output
    println!("\n🚀 Running inference pipeline...");
    let start = Instant::now();

    // For now, simulate with element-wise ops (real matmul would use cuBLAS)
    // Step 1: "Matrix multiply" (simulated with multiply for demo)
    println!("  1. Matrix multiplication (simulated)...");
    unsafe {
        test_kernels::test_launch_mul_f32(
            input_buf.as_ptr_typed::<f32>(),
            input_buf.as_ptr_typed::<f32>(),
            matmul_buf.as_ptr_typed::<f32>(),
            output_size.min(input_size),
            context.stream(),
        );
    }

    // Step 2: GELU activation
    println!("  2. GELU activation...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            matmul_buf.as_ptr_typed::<f32>(),
            output_buf.as_ptr_typed::<f32>(),
            output_size.min(input_size),
            context.stream(),
        );
    }

    context.sync()?;
    let inference_time = start.elapsed();
    println!("  ✓ Inference completed in {:?}", inference_time);

    // Copy results back
    println!("\n📥 Retrieving results...");
    unsafe {
        ptx_sys::cudaMemcpyAsync(
            host_output.as_mut_ptr() as *mut _,
            final_out_ptr,
            output_size.min(input_size) * 4,
            ptx_sys::cudaMemcpyDeviceToHost,
            stream.raw(),
        );
    }
    stream.synchronize()?;

    // Display results
    println!("\n📊 Results (first 10 elements):");
    for i in 0..10.min(output_size) {
        println!("  output[{}] = {:.6}", i, host_output[i]);
    }

    // Stream-ordered cleanup
    println!("\n🗑️  Cleaning up (stream-ordered)...");
    unsafe {
        runtime.free_async(input_ptr, &stream)?;
        runtime.free_async(weights_ptr, &stream)?;
        runtime.free_async(matmul_out_ptr, &stream)?;
        runtime.free_async(final_out_ptr, &stream)?;
    }
    stream.synchronize()?;

    // Final stats
    println!("\n📈 Performance Summary:");
    println!("  Allocation:  {:?}", alloc_time);
    println!("  Transfer:    {:?}", transfer_time);
    println!("  Inference:   {:?}", inference_time);
    println!("  Total:       {:?}", alloc_time + transfer_time + inference_time);

    let final_stats = runtime.tlsf_stats();
    println!("\n📊 TLSF Statistics:");
    println!("  Allocated: {:.2} MB", final_stats.allocated_bytes as f64 / 1e6);
    println!("  Free: {:.2} MB", final_stats.free_bytes as f64 / 1e6);
    println!("  Utilization: {:.1}%", final_stats.utilization_percent);
    println!("  Fragmentation: {:.2}%", final_stats.fragmentation_ratio * 100.0);

    println!("\n✅ Neural layer inference completed successfully!");
    Ok(())
}
