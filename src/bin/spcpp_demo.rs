// SPCPP Backend Demo
//
// Demonstrates: Rust orchestrates memory safety, spcpp provides GPU performance
//
// Architecture:
//   Rust (Brain)     -> Memory safety, lifetimes, ownership
//   TLSF Allocator   -> O(1) GPU memory allocation
//   spcpp/cuBLAS     -> Raw GPU compute (JIT CUDA + cuBLAS)

use ferrite::compute::Backend;
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     Ferrite - spcpp Backend Demo                 ║");
    println!("║     Rust orchestrates, spcpp executes                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA device and TLSF allocator (Rust manages memory)
    let device = CudaDevice::new(0).expect("No CUDA device found");
    let allocator = Arc::new(TlsfAllocator::new(device, 256 * 1024 * 1024)); // 256MB pool

    println!("GPU Memory Pool: {} MB", allocator.capacity() / (1024 * 1024));
    println!("Allocator: TLSF (O(1) alloc/free)\n");

    // Create backend (Rust wrapper around spcpp)
    let mut backend = Backend::new(allocator.clone());
    if let Err(e) = backend.init() {
        eprintln!("Failed to initialize backend: {}", e);
        eprintln!("Run ./build_spcpp.sh first!");
        return;
    }

    println!("Backend: spcpp (cuBLAS + JIT CUDA kernels)");
    println!("Memory Safety: Rust ownership + lifetimes\n");

    // =========================================================================
    // Demo 1: Simple MLP Forward Pass
    // =========================================================================
    println!("═══ Demo 1: MLP Forward Pass ═══\n");

    let batch_size = 64;
    let input_dim = 784;
    let hidden_dim = 256;
    let output_dim = 10;

    // Allocate tensors (Rust ensures they live long enough)
    let input = backend.tensor(vec![batch_size, input_dim]).unwrap();
    let w1 = backend.tensor(vec![input_dim, hidden_dim]).unwrap();
    let b1 = backend.tensor(vec![hidden_dim]).unwrap();
    let h1 = backend.tensor(vec![batch_size, hidden_dim]).unwrap();
    let w2 = backend.tensor(vec![hidden_dim, output_dim]).unwrap();
    let b2 = backend.tensor(vec![output_dim]).unwrap();
    let output = backend.tensor(vec![batch_size, output_dim]).unwrap();

    // Initialize weights (cuBLAS operations)
    backend.init_xavier("w1", &w1);
    backend.init_xavier("w2", &w2);
    backend.fill(&b1, 0.0);
    backend.fill(&b2, 0.0);

    // Random input
    let input_data: Vec<f32> = (0..batch_size * input_dim)
        .map(|i| ((i * 17) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    input.from_host(&input_data);

    println!("Network: {} -> {} -> {}", input_dim, hidden_dim, output_dim);
    println!("Batch size: {}\n", batch_size);

    // Forward pass with timing
    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        // h1 = relu(input @ w1 + b1)
        backend.linear(&input, &w1, Some(&b1), &h1);
        backend.relu(&h1);

        // output = h1 @ w2 + b2
        backend.linear(&h1, &w2, Some(&b2), &output);
        backend.softmax(&output);
    }
    backend.sync();

    let elapsed = start.elapsed();
    let per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("Forward pass: {:.3} ms/iteration ({} iterations)", per_iter, iterations);

    // Verify output (Rust safely reads from GPU)
    let out_data = output.to_host();
    let sum: f32 = out_data[0..output_dim].iter().sum();
    println!("Output sum (should be ~1.0 after softmax): {:.4}\n", sum);

    // =========================================================================
    // Demo 2: Training Step with Adam
    // =========================================================================
    println!("═══ Demo 2: Training with Adam ═══\n");

    // Gradients (Rust tracks these separately from weights)
    let grad_w1 = backend.zeros(vec![input_dim, hidden_dim]).unwrap();
    let grad_w2 = backend.zeros(vec![hidden_dim, output_dim]).unwrap();

    // Set optimizer params
    backend.set_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0);

    // Simulate training steps
    let start = Instant::now();
    let train_iters = 100;

    for _ in 0..train_iters {
        // Forward
        backend.linear(&input, &w1, Some(&b1), &h1);
        backend.relu(&h1);
        backend.linear(&h1, &w2, Some(&b2), &output);

        // Fake gradients (normally computed via backward pass)
        backend.fill(&grad_w1, 0.001);
        backend.fill(&grad_w2, 0.001);

        // Adam update
        backend.adam_step("w1", &w1, &grad_w1);
        backend.adam_step("w2", &w2, &grad_w2);
    }
    backend.sync();

    let elapsed = start.elapsed();
    let per_iter = elapsed.as_secs_f64() * 1000.0 / train_iters as f64;

    println!("Training step: {:.3} ms/iteration ({} iterations)", per_iter, train_iters);

    // =========================================================================
    // Demo 3: Large Matrix Multiplication
    // =========================================================================
    println!("\n═══ Demo 3: Large MatMul Benchmark ═══\n");

    let sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)];

    for (m, k, n) in sizes {
        let a = backend.tensor(vec![m, k]).unwrap();
        let b = backend.tensor(vec![k, n]).unwrap();
        let c = backend.tensor(vec![m, n]).unwrap();

        backend.fill(&a, 0.5);
        backend.fill(&b, 0.5);

        // Warmup
        for _ in 0..5 {
            backend.matmul(&a, &b, &c);
        }
        backend.sync();

        // Benchmark
        let start = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            backend.matmul(&a, &b, &c);
        }
        backend.sync();

        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tflops = flops / elapsed / 1e9;

        println!("{}x{}x{}: {:.3} ms ({:.1} TFLOPS)", m, k, n, elapsed, tflops);
    }

    // =========================================================================
    // Memory Safety Demo
    // =========================================================================
    println!("\n═══ Memory Safety ═══\n");
    println!("Pool capacity:  {} MB", allocator.capacity() / (1024 * 1024));
    println!("Pool consumed:  {} MB", allocator.consumed() / (1024 * 1024));
    println!("Pool available: {} MB", allocator.available() / (1024 * 1024));
    println!("\nAll tensors are bound to the allocator's lifetime.");
    println!("When the allocator drops, all GPU memory is freed.");
    println!("Rust's ownership system prevents use-after-free.\n");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    Demo Complete                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}
