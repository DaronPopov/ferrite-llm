// Ferrite CPU Demo - Test CPU fallback
//
// Forces CPU mode to verify fallback works

use ferrite::compute::Runtime;
use std::time::Instant;

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║        Ferrite CPU Fallback Test        ║");
    println!("╚═════════════════════════════════════════╝\n");

    // Force CPU mode
    let mut runtime = Runtime::cpu();
    println!("Device: {:?}\n", runtime.device_type());

    // =========================================================================
    // Test 1: Matrix Multiplication
    // =========================================================================
    println!("═══ Matrix Multiplication (CPU) ═══\n");

    let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)];

    for (m, k, n) in sizes {
        let a = runtime.tensor(vec![m, k]);
        let b = runtime.tensor(vec![k, n]);
        let c = runtime.zeros(vec![m, n]);

        runtime.fill(&a, 0.5);
        runtime.fill(&b, 0.5);

        // Benchmark
        let start = Instant::now();
        let iters = 5;
        for _ in 0..iters {
            runtime.matmul(&a, &b, &c);
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = flops / elapsed / 1e6;

        println!("{}x{}x{}: {:.3} ms ({:.2} GFLOPS)", m, k, n, elapsed, gflops);
    }

    // =========================================================================
    // Test 2: Activations
    // =========================================================================
    println!("\n═══ Activations (CPU) ═══\n");

    let x = runtime.tensor_from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    println!("Input: {:?}", x.to_vec());

    let relu_x = runtime.tensor_from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    runtime.relu(&relu_x);
    println!("ReLU:  {:?}", relu_x.to_vec());

    let sig_x = runtime.tensor_from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    runtime.sigmoid(&sig_x);
    println!("Sigmoid: {:?}", sig_x.to_vec());

    let tanh_x = runtime.tensor_from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    runtime.tanh(&tanh_x);
    println!("Tanh: {:?}", tanh_x.to_vec());

    // =========================================================================
    // Test 3: Softmax
    // =========================================================================
    println!("\n═══ Softmax (CPU) ═══\n");

    let logits = runtime.tensor_from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
    runtime.softmax(&logits);
    let probs = logits.to_vec();
    println!("Probs: {:?}", probs);
    println!("Sum: {:.6}", probs.iter().sum::<f32>());

    // =========================================================================
    // Test 4: MLP
    // =========================================================================
    println!("\n═══ MLP Forward (CPU) ═══\n");

    let batch = 4;
    let input_dim = 8;
    let hidden = 4;
    let output_dim = 2;

    let input = runtime.tensor(vec![batch, input_dim]);
    let w1 = runtime.tensor(vec![input_dim, hidden]);
    let h1 = runtime.zeros(vec![batch, hidden]);
    let w2 = runtime.tensor(vec![hidden, output_dim]);
    let output = runtime.zeros(vec![batch, output_dim]);

    runtime.fill(&input, 0.5);
    runtime.init_xavier("w1", &w1);
    runtime.init_xavier("w2", &w2);

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        runtime.linear(&input, &w1, None, &h1);
        runtime.relu(&h1);
        runtime.linear(&h1, &w2, None, &output);
        runtime.softmax(&output);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Forward: {:.3} ms/iter", elapsed);
    println!("Output shape: {:?}", output.shape());

    // =========================================================================
    // Test 5: Training
    // =========================================================================
    println!("\n═══ Training (Adam, CPU) ═══\n");

    let grad_w1 = runtime.zeros(vec![input_dim, hidden]);
    let grad_w2 = runtime.zeros(vec![hidden, output_dim]);

    runtime.set_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0);

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        runtime.fill(&grad_w1, 0.01);
        runtime.fill(&grad_w2, 0.01);
        runtime.adam_step("w1", &w1, &grad_w1);
        runtime.adam_step("w2", &w2, &grad_w2);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Adam step: {:.3} ms/iter", elapsed);

    println!("\n╔═════════════════════════════════════════╗");
    println!("║         CPU Fallback Works!             ║");
    println!("╚═════════════════════════════════════════╝");
}
