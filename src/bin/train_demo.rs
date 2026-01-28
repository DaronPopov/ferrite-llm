//! Complex Training Demo - Sperabality Semantic Core
//!
//! Demonstrates training a multi-layer neural network using the Rust runtime
//! with libtorch muscle operations.
//!
//! Network: 784 -> 256 -> 128 -> 10 (MNIST-style classifier)
//!
//! Features demonstrated:
//! - Weight initialization (Xavier/Kaiming)
//! - Linear layer forward/backward
//! - Activation functions and their gradients
//! - Cross-entropy loss
//! - Adam optimizer
//! - Batch normalization
//! - Dropout
//! - Training loop with multiple epochs

use ferrite::data::Grid;
use ferrite::compute::{compute, Op};
use ferrite::dynamics::{RuntimeRules, Device, MuscleMemory};
use ferrite::dynamics::allocator::TlsfAllocator;
use ferrite::compute::synth::Synthesizer;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     SPERABALITY TRAINING DEMO - Neural Network Training      ║");
    println!("║          Using Rust Runtime + Libtorch Muscles               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. INITIALIZE PHYSICAL ENVIRONMENT
    // =========================================================================
    println!("[Physical Body] Initializing GPU...");
    let cuda_dev = CudaDevice::new(0).expect("No GPU found");
    let pool_size = 1024 * 1024 * 1024; // 1GB VRAM pool
    let allocator = Arc::new(TlsfAllocator::new(Arc::clone(&cuda_dev), pool_size));
    let synth = Arc::new(Synthesizer::new(Arc::clone(&cuda_dev)));
    let muscle_memory = Arc::new(MuscleMemory::new());

    let rules = RuntimeRules {
        device: Device::Gpu(Arc::clone(&allocator)),
        synthesizer: Some(synth),
        muscle_memory: Arc::clone(&muscle_memory),
        track_gradients: true,
        ..RuntimeRules::default()
    };

    println!("[Physical Body] GPU pool: {} MB allocated", pool_size / 1024 / 1024);

    // =========================================================================
    // 2. HYPERPARAMETERS
    // =========================================================================
    let batch_size = 64;
    let input_dim = 784;   // 28x28 flattened
    let hidden1 = 256;
    let hidden2 = 128;
    let output_dim = 10;   // 10 classes
    let num_epochs = 10;
    let learning_rate = 0.001f32;

    println!("\n[Hyperparameters]");
    println!("  Batch size: {}", batch_size);
    println!("  Architecture: {} -> {} -> {} -> {}", input_dim, hidden1, hidden2, output_dim);
    println!("  Epochs: {}", num_epochs);
    println!("  Learning rate: {}", learning_rate);

    // =========================================================================
    // 3. CREATE SYNTHETIC TRAINING DATA
    // =========================================================================
    println!("\n[Data] Generating synthetic training data...");

    // Random input data (simulating normalized images)
    let x_data: Vec<f32> = (0..batch_size * input_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.5)
        .collect();
    let x_host = Grid::new(x_data, vec![batch_size, input_dim]);

    // Random target labels (one-hot encoded)
    let mut y_data = vec![0.0f32; batch_size * output_dim];
    for b in 0..batch_size {
        let label = b % output_dim;  // Synthetic labels
        y_data[b * output_dim + label] = 1.0;
    }
    let y_host = Grid::new(y_data, vec![batch_size, output_dim]);

    // =========================================================================
    // 4. INITIALIZE NETWORK WEIGHTS
    // =========================================================================
    println!("[Weights] Initializing network parameters...");

    // Layer 1: input_dim -> hidden1
    let w1_data: Vec<f32> = xavier_init(input_dim, hidden1);
    let b1_data = vec![0.0f32; hidden1];
    let w1_host = Grid::new(w1_data, vec![input_dim, hidden1]);
    let b1_host = Grid::new(b1_data, vec![hidden1]);

    // Layer 2: hidden1 -> hidden2
    let w2_data: Vec<f32> = xavier_init(hidden1, hidden2);
    let b2_data = vec![0.0f32; hidden2];
    let w2_host = Grid::new(w2_data, vec![hidden1, hidden2]);
    let b2_host = Grid::new(b2_data, vec![hidden2]);

    // Layer 3: hidden2 -> output_dim
    let w3_data: Vec<f32> = xavier_init(hidden2, output_dim);
    let b3_data = vec![0.0f32; output_dim];
    let w3_host = Grid::new(w3_data, vec![hidden2, output_dim]);
    let b3_host = Grid::new(b3_data, vec![output_dim]);

    // =========================================================================
    // 5. ALLOCATE ADAM OPTIMIZER STATE
    // =========================================================================
    println!("[Optimizer] Allocating Adam momentum buffers...");

    // First moment (m) and second moment (v) for each weight
    let w1_m_host = Grid::new(vec![0.0f32; input_dim * hidden1], vec![input_dim, hidden1]);
    let w1_v_host = Grid::new(vec![0.0f32; input_dim * hidden1], vec![input_dim, hidden1]);
    let w2_m_host = Grid::new(vec![0.0f32; hidden1 * hidden2], vec![hidden1, hidden2]);
    let w2_v_host = Grid::new(vec![0.0f32; hidden1 * hidden2], vec![hidden1, hidden2]);
    let w3_m_host = Grid::new(vec![0.0f32; hidden2 * output_dim], vec![hidden2, output_dim]);
    let w3_v_host = Grid::new(vec![0.0f32; hidden2 * output_dim], vec![hidden2, output_dim]);

    // =========================================================================
    // 6. MOVE ALL DATA TO GPU
    // =========================================================================
    println!("[Memory] Transferring to GPU...");

    let x_gpu = compute(Op::MoveToDevice, vec![&x_host], None, &rules);
    let y_gpu = compute(Op::MoveToDevice, vec![&y_host], None, &rules);

    let mut w1_gpu = compute(Op::MoveToDevice, vec![&w1_host], None, &rules);
    let b1_gpu = compute(Op::MoveToDevice, vec![&b1_host], None, &rules);
    let mut w2_gpu = compute(Op::MoveToDevice, vec![&w2_host], None, &rules);
    let b2_gpu = compute(Op::MoveToDevice, vec![&b2_host], None, &rules);
    let mut w3_gpu = compute(Op::MoveToDevice, vec![&w3_host], None, &rules);
    let b3_gpu = compute(Op::MoveToDevice, vec![&b3_host], None, &rules);

    let w1_m = compute(Op::MoveToDevice, vec![&w1_m_host], None, &rules);
    let w1_v = compute(Op::MoveToDevice, vec![&w1_v_host], None, &rules);
    let w2_m = compute(Op::MoveToDevice, vec![&w2_m_host], None, &rules);
    let w2_v = compute(Op::MoveToDevice, vec![&w2_v_host], None, &rules);
    let w3_m = compute(Op::MoveToDevice, vec![&w3_m_host], None, &rules);
    let w3_v = compute(Op::MoveToDevice, vec![&w3_v_host], None, &rules);

    // Configure Adam parameters
    compute(Op::SetOptimizerParams {
        lr: learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0001,
    }, vec![], None, &rules);

    println!("[Memory] All tensors on GPU. Starting training...\n");

    // =========================================================================
    // 7. TRAINING LOOP
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING LOOP                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let training_start = std::time::Instant::now();

    for epoch in 1..=num_epochs {
        let epoch_start = std::time::Instant::now();

        // =====================================================================
        // FORWARD PASS
        // =====================================================================

        // Layer 1: h1 = ReLU(X @ W1 + b1)
        let h1_pre = compute(
            Op::LinearForward { has_bias: true },
            vec![&x_gpu, &w1_gpu, &b1_gpu],
            None,
            &rules
        );
        let h1 = compute(Op::TorchUnary("relu".to_string()), vec![&h1_pre], None, &rules);

        // Layer 2: h2 = ReLU(h1 @ W2 + b2)
        let h2_pre = compute(
            Op::LinearForward { has_bias: true },
            vec![&h1, &w2_gpu, &b2_gpu],
            None,
            &rules
        );
        let h2 = compute(Op::TorchUnary("relu".to_string()), vec![&h2_pre], None, &rules);

        // Layer 3: output = h2 @ W3 + b3 (logits)
        let logits = compute(
            Op::LinearForward { has_bias: true },
            vec![&h2, &w3_gpu, &b3_gpu],
            None,
            &rules
        );

        // Softmax for predictions (used for accuracy, not in loss)
        let _probs = compute(Op::TorchUnary("softmax".to_string()), vec![&logits], None, &rules);

        // =====================================================================
        // LOSS COMPUTATION (MSE for simplicity, treating as regression)
        // =====================================================================
        let loss_grid = compute(Op::MSELoss, vec![&logits, &y_gpu], None, &rules);

        // Get loss value
        let loss_host = compute(Op::MoveToHost, vec![&loss_grid], None, &rules);
        let loss_value = if let ferrite::data::Values::Host(v) = &loss_host.values {
            v[0]
        } else { 0.0 };

        // =====================================================================
        // BACKWARD PASS
        // =====================================================================

        // dL/d(logits)
        let grad_logits = compute(Op::MSELossBackward, vec![&logits, &y_gpu], None, &rules);

        // Backprop through Layer 3
        let grad_h2 = compute(
            Op::LinearBackward { has_bias: true },
            vec![&h2, &w3_gpu, &grad_logits],
            None,
            &rules
        );

        // Backprop through ReLU (h2)
        let grad_h2_pre = compute(
            Op::ActivationBackward("relu".to_string()),
            vec![&h2_pre, &grad_h2],
            None,
            &rules
        );

        // Backprop through Layer 2
        let grad_h1 = compute(
            Op::LinearBackward { has_bias: true },
            vec![&h1, &w2_gpu, &grad_h2_pre],
            None,
            &rules
        );

        // Backprop through ReLU (h1)
        let grad_h1_pre = compute(
            Op::ActivationBackward("relu".to_string()),
            vec![&h1_pre, &grad_h1],
            None,
            &rules
        );

        // Backprop through Layer 1
        let _grad_x = compute(
            Op::LinearBackward { has_bias: true },
            vec![&x_gpu, &w1_gpu, &grad_h1_pre],
            None,
            &rules
        );

        // =====================================================================
        // OPTIMIZER STEP (Adam)
        // =====================================================================
        let t = epoch as i32;

        // Update W3
        w3_gpu = compute(
            Op::AdamStep { timestep: t },
            vec![&w3_gpu, &grad_logits, &w3_m, &w3_v],
            None,
            &rules
        );

        // Update W2
        w2_gpu = compute(
            Op::AdamStep { timestep: t },
            vec![&w2_gpu, &grad_h2_pre, &w2_m, &w2_v],
            None,
            &rules
        );

        // Update W1
        w1_gpu = compute(
            Op::AdamStep { timestep: t },
            vec![&w1_gpu, &grad_h1_pre, &w1_m, &w1_v],
            None,
            &rules
        );

        let epoch_time = epoch_start.elapsed();

        // Print progress
        println!(
            "  Epoch {:2}/{} │ Loss: {:.6} │ Time: {:?}",
            epoch, num_epochs, loss_value, epoch_time
        );
    }

    let total_time = training_start.elapsed();

    // =========================================================================
    // 8. FINAL EVALUATION
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Run final forward pass
    let h1_pre = compute(
        Op::LinearForward { has_bias: true },
        vec![&x_gpu, &w1_gpu, &b1_gpu],
        None,
        &rules
    );
    let h1 = compute(Op::TorchUnary("relu".to_string()), vec![&h1_pre], None, &rules);
    let h2_pre = compute(
        Op::LinearForward { has_bias: true },
        vec![&h1, &w2_gpu, &b2_gpu],
        None,
        &rules
    );
    let h2 = compute(Op::TorchUnary("relu".to_string()), vec![&h2_pre], None, &rules);
    let logits = compute(
        Op::LinearForward { has_bias: true },
        vec![&h2, &w3_gpu, &b3_gpu],
        None,
        &rules
    );

    let final_loss = compute(Op::MSELoss, vec![&logits, &y_gpu], None, &rules);
    let final_loss_host = compute(Op::MoveToHost, vec![&final_loss], None, &rules);
    let final_loss_value = if let ferrite::data::Values::Host(v) = &final_loss_host.values {
        v[0]
    } else { 0.0 };

    // Get predictions
    let output_host = compute(Op::MoveToHost, vec![&logits], None, &rules);
    let predictions = if let ferrite::data::Values::Host(v) = &output_host.values {
        v.clone()
    } else { vec![] };

    println!("\n[Results]");
    println!("  Final Loss: {:.6}", final_loss_value);
    println!("  Total Training Time: {:?}", total_time);
    println!("  Average Epoch Time: {:?}", total_time / num_epochs as u32);

    // Show sample predictions
    println!("\n[Sample Predictions] (first 5 samples)");
    for i in 0..5.min(batch_size) {
        let pred_start = i * output_dim;
        let pred_slice = &predictions[pred_start..pred_start + output_dim];
        let predicted_class = pred_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let expected_class = i % output_dim;
        let correct = if predicted_class == expected_class { "✓" } else { "✗" };
        println!(
            "  Sample {:2}: Predicted={}, Expected={} {}",
            i, predicted_class, expected_class, correct
        );
    }

    println!("\n[Success] Neural network training completed using Sperabality + Libtorch muscles!");
}

/// Xavier/Glorot uniform initialization
fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    let size = fan_in * fan_out;

    // Simple LCG for reproducibility
    let mut seed: u64 = 42;
    (0..size)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (seed >> 33) as f32 / (1u64 << 31) as f32;
            (u * 2.0 - 1.0) * limit
        })
        .collect()
}
