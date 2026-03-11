//! Transformer Attention Layer Example
//!
//! Demonstrates a complete multi-head attention forward pass:
//! - Stream-ordered TLSF allocations for all intermediate tensors
//! - Kernel chaining for Q/K/V projections
//! - Attention score computation
//! - Output projection
//! - Shows realistic ML workload pattern

use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, test_kernels};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Transformer Multi-Head Attention Layer");
    println!("============================================\n");

    // Production config
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.75;
    config.max_streams = 64;
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    // Typical transformer configuration
    const BATCH_SIZE: usize = 8;
    const SEQ_LEN: usize = 512;
    const D_MODEL: usize = 512;
    const NUM_HEADS: usize = 8;
    const D_K: usize = D_MODEL / NUM_HEADS;  // 64

    println!("📐 Attention Configuration:");
    println!("  Batch size:      {}", BATCH_SIZE);
    println!("  Sequence length: {}", SEQ_LEN);
    println!("  Model dim:       {}", D_MODEL);
    println!("  Heads:           {}", NUM_HEADS);
    println!("  Head dim:        {}", D_K);

    let input_size = BATCH_SIZE * SEQ_LEN * D_MODEL;
    let qkv_size = input_size;  // Each of Q, K, V
    let attn_scores_size = BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN;

    println!("\n💾 Memory Requirements:");
    println!("  Input:  {:.2} MB", input_size * 4 / 1024 / 1024);
    println!("  Q/K/V:  {:.2} MB each", qkv_size * 4 / 1024 / 1024);
    println!("  Attention scores: {:.2} MB", attn_scores_size * 4 / 1024 / 1024);

    // Get stream for attention computation
    let stream = runtime.stream(0)?;
    println!("\n🌊 Using stream {} for attention layer", stream.id());

    // Allocate all tensors (stream-ordered)
    println!("\n📦 Allocating tensors (stream-ordered)...");
    let start = Instant::now();

    let input_ptr = runtime.alloc_async(input_size * 4, &stream)?;
    let q_ptr = runtime.alloc_async(qkv_size * 4, &stream)?;
    let k_ptr = runtime.alloc_async(qkv_size * 4, &stream)?;
    let v_ptr = runtime.alloc_async(qkv_size * 4, &stream)?;
    let attn_scores_ptr = runtime.alloc_async(attn_scores_size * 4, &stream)?;
    let attn_output_ptr = runtime.alloc_async(input_size * 4, &stream)?;
    let final_output_ptr = runtime.alloc_async(input_size * 4, &stream)?;

    let alloc_time = start.elapsed();
    println!("  ✓ 7 tensors allocated in {:?}", alloc_time);
    println!("  Total memory: {:.2} MB",
        (input_size * 4 + qkv_size * 12 + attn_scores_size * 4) / 1024 / 1024);

    // Prepare input
    println!("\n🔧 Preparing input...");
    let host_input: Vec<f32> = (0..input_size)
        .map(|i| ((i % 1000) as f32) / 1000.0)
        .collect();

    // Upload input
    unsafe {
        ptx_sys::cudaMemcpyAsync(
            input_ptr,
            host_input.as_ptr() as *const _,
            input_size * 4,
            ptx_sys::cudaMemcpyHostToDevice,
            stream.raw(),
        );
    }

    // Create guarded buffers
    println!("\n🛡️  Creating guarded buffers...");
    let input_buf = unsafe { GuardedBuffer::new(input_ptr, input_size * 4, runtime.raw())? };
    let q_buf = unsafe { GuardedBuffer::new(q_ptr, qkv_size * 4, runtime.raw())? };
    let k_buf = unsafe { GuardedBuffer::new(k_ptr, qkv_size * 4, runtime.raw())? };
    let v_buf = unsafe { GuardedBuffer::new(v_ptr, qkv_size * 4, runtime.raw())? };
    let attn_scores_buf = unsafe {
        GuardedBuffer::new(attn_scores_ptr, attn_scores_size * 4, runtime.raw())?
    };
    let attn_output_buf = unsafe {
        GuardedBuffer::new(attn_output_ptr, input_size * 4, runtime.raw())?
    };
    let output_buf = unsafe {
        GuardedBuffer::new(final_output_ptr, input_size * 4, runtime.raw())?
    };

    let context = KernelContext::new(runtime.raw(), stream.raw())?;

    // Forward pass: Multi-head attention
    println!("\n🚀 Running attention forward pass...");
    let start = Instant::now();

    // Step 1: Compute Q, K, V projections (simulated with GELU for demo)
    println!("  1. Computing Q projection...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            input_buf.as_ptr_typed::<f32>(),
            q_buf.as_ptr_typed::<f32>(),
            input_size,
            context.stream(),
        );
    }

    println!("  2. Computing K projection...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            input_buf.as_ptr_typed::<f32>(),
            k_buf.as_ptr_typed::<f32>(),
            input_size,
            context.stream(),
        );
    }

    println!("  3. Computing V projection...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            input_buf.as_ptr_typed::<f32>(),
            v_buf.as_ptr_typed::<f32>(),
            input_size,
            context.stream(),
        );
    }

    // Step 2: Compute attention scores (simulated with mul)
    println!("  4. Computing attention scores (Q @ K^T)...");
    let score_size = input_size.min(attn_scores_size);
    unsafe {
        test_kernels::test_launch_mul_f32(
            q_buf.as_ptr_typed::<f32>(),
            k_buf.as_ptr_typed::<f32>(),
            attn_scores_buf.as_ptr_typed::<f32>(),
            score_size,
            context.stream(),
        );
    }

    // Step 3: Apply softmax (simulated with GELU for now)
    println!("  5. Applying softmax...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            attn_scores_buf.as_ptr_typed::<f32>(),
            attn_scores_buf.as_ptr_typed::<f32>(),
            score_size,
            context.stream(),
        );
    }

    // Step 4: Apply attention to values
    println!("  6. Applying attention to values...");
    unsafe {
        test_kernels::test_launch_mul_f32(
            attn_scores_buf.as_ptr_typed::<f32>(),
            v_buf.as_ptr_typed::<f32>(),
            attn_output_buf.as_ptr_typed::<f32>(),
            input_size.min(score_size),
            context.stream(),
        );
    }

    // Step 5: Output projection
    println!("  7. Output projection...");
    unsafe {
        test_kernels::test_launch_gelu_f32(
            attn_output_buf.as_ptr_typed::<f32>(),
            output_buf.as_ptr_typed::<f32>(),
            input_size,
            context.stream(),
        );
    }

    context.sync()?;
    let forward_time = start.elapsed();
    println!("  ✓ Forward pass completed in {:?}", forward_time);

    // Download results
    println!("\n📥 Retrieving attention output...");
    let mut host_output = vec![0.0f32; input_size];
    unsafe {
        ptx_sys::cudaMemcpyAsync(
            host_output.as_mut_ptr() as *mut _,
            final_output_ptr,
            input_size * 4,
            ptx_sys::cudaMemcpyDeviceToHost,
            stream.raw(),
        );
    }
    stream.synchronize()?;

    // Display sample results
    println!("\n📊 Sample Outputs:");
    for batch in 0..2 {
        println!("  Batch {}:", batch);
        let offset = batch * SEQ_LEN * D_MODEL;
        for i in 0..5 {
            println!("    pos[{}][{}] = {:.6}", 0, i, host_output[offset + i]);
        }
    }

    // Cleanup (stream-ordered)
    println!("\n🗑️  Cleaning up (stream-ordered)...");
    unsafe {
        runtime.free_async(input_ptr, &stream)?;
        runtime.free_async(q_ptr, &stream)?;
        runtime.free_async(k_ptr, &stream)?;
        runtime.free_async(v_ptr, &stream)?;
        runtime.free_async(attn_scores_ptr, &stream)?;
        runtime.free_async(attn_output_ptr, &stream)?;
        runtime.free_async(final_output_ptr, &stream)?;
    }
    stream.synchronize()?;

    // Performance metrics
    println!("\n📈 Performance Metrics:");
    println!("  Allocation:   {:?}", alloc_time);
    println!("  Forward pass: {:?}", forward_time);
    println!("  Throughput:   {:.2} tokens/sec",
        (BATCH_SIZE * SEQ_LEN) as f64 / forward_time.as_secs_f64());

    let stats = runtime.tlsf_stats();
    println!("\n📊 TLSF Statistics:");
    println!("  Peak allocated: {:.2} MB", stats.peak_allocated as f64 / 1e6);
    println!("  Total allocs:   {}", stats.total_allocations);
    println!("  Total frees:    {}", stats.total_frees);
    println!("  Fragmentation:  {:.2}%", stats.fragmentation_ratio * 100.0);

    println!("\n✅ Transformer attention layer completed!");
    Ok(())
}
