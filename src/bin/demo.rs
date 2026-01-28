// Ferrite Demo - Unified Runtime
//
// Auto-selects GPU (CUDA) or CPU fallback

use ferrite::compute::{Runtime, DeviceType};
use std::time::Instant;

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║           Ferrite Runtime               ║");
    println!("║     Rust safety, GPU speed              ║");
    println!("╚═════════════════════════════════════════╝\n");

    // Auto-select best device (GPU if available, else CPU)
    let mut runtime = Runtime::auto(256); // 256MB pool for GPU

    match runtime.device_type() {
        DeviceType::Cuda(idx) => println!("Device: CUDA GPU {}", idx),
        DeviceType::Cpu => println!("Device: CPU (fallback)"),
    }
    println!();

    // =========================================================================
    // Test 1: Matrix Multiplication
    // =========================================================================
    println!("═══ Matrix Multiplication ═══\n");

    let sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];

    for (m, k, n) in sizes {
        let a = runtime.tensor(vec![m, k]);
        let b = runtime.tensor(vec![k, n]);
        let c = runtime.zeros(vec![m, n]);

        runtime.fill(&a, 0.5);
        runtime.fill(&b, 0.5);

        // Warmup
        for _ in 0..3 {
            runtime.matmul(&a, &b, &c);
        }
        runtime.sync();

        // Benchmark
        let start = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            runtime.matmul(&a, &b, &c);
        }
        runtime.sync();

        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = flops / elapsed / 1e6;

        println!("{}x{}x{}: {:.3} ms ({:.1} GFLOPS)", m, k, n, elapsed, gflops);
    }

    // =========================================================================
    // Test 2: MLP Forward Pass
    // =========================================================================
    println!("\n═══ MLP Forward Pass ═══\n");

    let batch = 64;
    let input_dim = 784;
    let hidden = 256;
    let output_dim = 10;

    let input = runtime.tensor(vec![batch, input_dim]);
    let w1 = runtime.tensor(vec![input_dim, hidden]);
    let b1 = runtime.tensor(vec![hidden]);
    let h1 = runtime.zeros(vec![batch, hidden]);
    let w2 = runtime.tensor(vec![hidden, output_dim]);
    let b2 = runtime.tensor(vec![output_dim]);
    let output = runtime.zeros(vec![batch, output_dim]);

    runtime.init_xavier("w1", &w1);
    runtime.init_xavier("w2", &w2);
    runtime.fill(&b1, 0.0);
    runtime.fill(&b2, 0.0);
    runtime.fill(&input, 0.5);

    // Warmup
    for _ in 0..5 {
        runtime.linear(&input, &w1, Some(&b1), &h1);
        runtime.relu(&h1);
        runtime.linear(&h1, &w2, Some(&b2), &output);
        runtime.softmax(&output);
    }
    runtime.sync();

    // Benchmark
    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        runtime.linear(&input, &w1, Some(&b1), &h1);
        runtime.relu(&h1);
        runtime.linear(&h1, &w2, Some(&b2), &output);
        runtime.softmax(&output);
    }
    runtime.sync();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Network: {} -> {} -> {}", input_dim, hidden, output_dim);
    println!("Batch: {}", batch);
    println!("Forward: {:.3} ms/iter\n", elapsed);

    // Verify softmax
    let out_data = output.to_vec();
    let sum: f32 = out_data[0..output_dim].iter().sum();
    println!("Softmax sum (should be ~1.0): {:.4}", sum);

    // =========================================================================
    // Test 3: Training Step
    // =========================================================================
    println!("\n═══ Training (Adam) ═══\n");

    let grad_w1 = runtime.zeros(vec![input_dim, hidden]);
    let grad_w2 = runtime.zeros(vec![hidden, output_dim]);

    runtime.set_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0);

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        // Forward
        runtime.linear(&input, &w1, Some(&b1), &h1);
        runtime.relu(&h1);
        runtime.linear(&h1, &w2, Some(&b2), &output);

        // Fake gradients
        runtime.fill(&grad_w1, 0.001);
        runtime.fill(&grad_w2, 0.001);

        // Adam update
        runtime.adam_step("w1", &w1, &grad_w1);
        runtime.adam_step("w2", &w2, &grad_w2);
    }
    runtime.sync();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Training step: {:.3} ms/iter", elapsed);

    println!("\n╔═════════════════════════════════════════╗");
    println!("║             Demo Complete               ║");
    println!("╚═════════════════════════════════════════╝");
}
