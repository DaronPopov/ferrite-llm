// Ferrite Stream Demo - Zero-Copy Execution Pipeline
//
// Demonstrates zero-copy GPU execution:
// - No host-device copies during compute
// - All operations on GPU memory handles
// - Only copy to host when results needed

use ferrite::compute::{Stream, Pipeline};
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║          Ferrite Zero-Copy Streaming Pipeline               ║");
    println!("╚═════════════════════════════════════════════════════════════╝\n");

    // Initialize GPU and allocator
    let device = CudaDevice::new(0).expect("No CUDA device");
    let allocator = Arc::new(TlsfAllocator::new(device, 256 * 1024 * 1024));

    let mut stream = Stream::new(allocator.clone());
    stream.init().expect("Failed to init spcpp");

    println!("Memory pool: {} MB", allocator.capacity() / (1024 * 1024));
    println!("Zero-copy mode: All ops on GPU handles, no intermediate copies\n");

    // =========================================================================
    // Demo 1: Zero-copy MLP
    // =========================================================================
    println!("═══ Demo 1: Zero-Copy MLP ═══\n");

    let batch = 64;
    let input_dim = 784;
    let hidden = 256;
    let output_dim = 10;

    // Allocate all tensors on GPU (no copy, just offset allocation)
    stream.alloc("input", batch * input_dim);
    stream.alloc("w1", input_dim * hidden);
    stream.alloc("b1", hidden);
    stream.alloc("h1", batch * hidden);
    stream.alloc("w2", hidden * output_dim);
    stream.alloc("b2", output_dim);
    stream.alloc("output", batch * output_dim);

    // Initialize on GPU (no host involvement)
    stream.init_weights("w1", input_dim as i32, hidden as i32, "xavier");
    stream.init_weights("w2", hidden as i32, output_dim as i32, "xavier");
    stream.fill("b1", 0.0, hidden as i32);
    stream.fill("b2", 0.0, output_dim as i32);
    stream.fill("input", 0.5, (batch * input_dim) as i32);

    println!("Allocated and initialized on GPU (zero host copies)");
    println!("Network: {} -> {} -> {}", input_dim, hidden, output_dim);
    println!("Batch: {}\n", batch);

    // Warmup (all zero-copy)
    for _ in 0..5 {
        stream.linear("input", "w1", Some("b1"), "h1", batch as i32, input_dim as i32, hidden as i32);
        stream.relu("h1", (batch * hidden) as i32);
        stream.linear("h1", "w2", Some("b2"), "output", batch as i32, hidden as i32, output_dim as i32);
        stream.softmax("output", batch as i32, output_dim as i32);
    }
    stream.sync();

    // Benchmark (zero-copy throughout)
    let start = Instant::now();
    let iters = 1000;

    for _ in 0..iters {
        // All these operations are zero-copy:
        // - linear: reads GPU, writes GPU
        // - relu: in-place on GPU
        // - softmax: in-place on GPU
        stream.linear("input", "w1", Some("b1"), "h1", batch as i32, input_dim as i32, hidden as i32);
        stream.relu("h1", (batch * hidden) as i32);
        stream.linear("h1", "w2", Some("b2"), "output", batch as i32, hidden as i32, output_dim as i32);
        stream.softmax("output", batch as i32, output_dim as i32);
    }
    stream.sync();

    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let throughput = iters as f64 / elapsed.as_secs_f64();

    println!("Forward pass: {:.2} µs/iter", per_iter_us);
    println!("Throughput: {:.0} inferences/sec", throughput);

    // Only now do we copy to host (first copy in entire pipeline)
    let output_data = stream.download("output");
    let sum: f32 = output_data[0..output_dim].iter().sum();
    println!("Softmax sum (verify): {:.4}\n", sum);

    // =========================================================================
    // Demo 2: Zero-copy Training Loop
    // =========================================================================
    println!("═══ Demo 2: Zero-Copy Training ═══\n");

    // Allocate gradients on GPU
    stream.alloc("grad_w1", input_dim * hidden);
    stream.alloc("grad_w2", hidden * output_dim);

    stream.set_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0);

    // Warmup
    for _ in 0..5 {
        stream.linear("input", "w1", Some("b1"), "h1", batch as i32, input_dim as i32, hidden as i32);
        stream.relu("h1", (batch * hidden) as i32);
        stream.linear("h1", "w2", Some("b2"), "output", batch as i32, hidden as i32, output_dim as i32);
        stream.fill("grad_w1", 0.001, (input_dim * hidden) as i32);
        stream.fill("grad_w2", 0.001, (hidden * output_dim) as i32);
        stream.adam_step("w1", "w1", "grad_w1", (input_dim * hidden) as i32);
        stream.adam_step("w2", "w2", "grad_w2", (hidden * output_dim) as i32);
    }
    stream.sync();

    // Benchmark training loop (all zero-copy)
    let start = Instant::now();
    let iters = 1000;

    for _ in 0..iters {
        // Forward (zero-copy)
        stream.linear("input", "w1", Some("b1"), "h1", batch as i32, input_dim as i32, hidden as i32);
        stream.relu("h1", (batch * hidden) as i32);
        stream.linear("h1", "w2", Some("b2"), "output", batch as i32, hidden as i32, output_dim as i32);

        // Fake gradients (zero-copy fill on GPU)
        stream.fill("grad_w1", 0.001, (input_dim * hidden) as i32);
        stream.fill("grad_w2", 0.001, (hidden * output_dim) as i32);

        // Adam update (zero-copy, in-place)
        stream.adam_step("w1", "w1", "grad_w1", (input_dim * hidden) as i32);
        stream.adam_step("w2", "w2", "grad_w2", (hidden * output_dim) as i32);
    }
    stream.sync();

    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let throughput = iters as f64 / elapsed.as_secs_f64();

    println!("Training step: {:.2} µs/iter", per_iter_us);
    println!("Throughput: {:.0} steps/sec\n", throughput);

    // =========================================================================
    // Demo 3: Pipeline Builder (declarative zero-copy)
    // =========================================================================
    println!("═══ Demo 3: Pipeline Builder ═══\n");

    // Build a pipeline declaratively
    let pipeline = Pipeline::new()
        .linear("input", "w1", Some("b1"), "h1", batch as i32, input_dim as i32, hidden as i32)
        .relu("h1", (batch * hidden) as i32)
        .linear("h1", "w2", Some("b2"), "output", batch as i32, hidden as i32, output_dim as i32)
        .softmax("output", batch as i32, output_dim as i32)
        .sync();

    let ops = pipeline.build();
    println!("Pipeline has {} operations", ops.len());

    // Execute pipeline
    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        stream.execute(&ops);
    }

    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    println!("Pipeline execution: {:.2} µs/iter\n", per_iter_us);

    // =========================================================================
    // Demo 4: Large MatMul Streaming
    // =========================================================================
    println!("═══ Demo 4: Large MatMul Stream ═══\n");

    let sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)];

    for (m, k, n) in sizes {
        let name_a = format!("mat_a_{}", m);
        let name_b = format!("mat_b_{}", m);
        let name_c = format!("mat_c_{}", m);

        // Leak the strings to get static lifetimes (ugly but works for demo)
        // In real code you'd use indices or a different approach
        stream.alloc(&name_a, m * k);
        stream.alloc(&name_b, k * n);
        stream.alloc(&name_c, m * n);

        stream.fill(&name_a, 0.5, (m * k) as i32);
        stream.fill(&name_b, 0.5, (k * n) as i32);

        // Warmup
        for _ in 0..5 {
            stream.matmul(&name_a, &name_b, &name_c, m as i32, k as i32, n as i32);
        }
        stream.sync();

        // Benchmark
        let start = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            stream.matmul(&name_a, &name_b, &name_c, m as i32, k as i32, n as i32);
        }
        stream.sync();

        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tflops = flops / elapsed / 1e9;

        println!("{}x{}x{}: {:.3} ms ({:.1} TFLOPS) - zero-copy", m, k, n, elapsed, tflops);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n═══ Zero-Copy Summary ═══\n");
    println!("Host-to-Device copies: 0 during compute");
    println!("Device-to-Host copies: 1 (only to verify result)");
    println!("All operations: Direct GPU pointer manipulation");
    println!("\nPool used: {} MB / {} MB",
             allocator.consumed() / (1024 * 1024),
             allocator.capacity() / (1024 * 1024));

    println!("\n╔═════════════════════════════════════════════════════════════╗");
    println!("║                    Stream Demo Complete                      ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
}
