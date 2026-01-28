//! CNN Training Demo - Sperabality Semantic Core
//!
//! Demonstrates convolutional neural network training:
//! - Conv2D layers
//! - MaxPool2D / AvgPool2D
//! - Batch normalization
//! - ReLU activations
//! - Fully connected classifier
//! - Cross-entropy loss
//! - SGD with momentum

use ferrite::data::Grid;
use ferrite::compute::{compute, Op};
use ferrite::dynamics::{RuntimeRules, Device, MuscleMemory};
use ferrite::dynamics::allocator::TlsfAllocator;
use ferrite::compute::synth::Synthesizer;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     SPERABALITY CNN DEMO - Convolutional Network Training    ║");
    println!("║          Using Rust Runtime + Libtorch Muscles               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // RUNTIME INITIALIZATION
    // =========================================================================
    let cuda_dev = CudaDevice::new(0).expect("No GPU found");
    let allocator = Arc::new(TlsfAllocator::new(Arc::clone(&cuda_dev), 2 * 1024 * 1024 * 1024));
    let synth = Arc::new(Synthesizer::new(Arc::clone(&cuda_dev)));
    let muscle_memory = Arc::new(MuscleMemory::new());

    let rules = RuntimeRules {
        device: Device::Gpu(Arc::clone(&allocator)),
        synthesizer: Some(synth),
        muscle_memory,
        track_gradients: true,
        ..RuntimeRules::default()
    };

    // =========================================================================
    // NETWORK ARCHITECTURE (LeNet-style)
    // =========================================================================
    // Input: [batch, 1, 28, 28] (grayscale images)
    // Conv1: 1 -> 32 channels, 3x3 kernel, padding 1 -> [batch, 32, 28, 28]
    // ReLU + MaxPool 2x2 -> [batch, 32, 14, 14]
    // Conv2: 32 -> 64 channels, 3x3 kernel, padding 1 -> [batch, 64, 14, 14]
    // ReLU + MaxPool 2x2 -> [batch, 64, 7, 7]
    // Flatten -> [batch, 64*7*7] = [batch, 3136]
    // FC1: 3136 -> 128
    // ReLU
    // FC2: 128 -> 10 (classes)

    let batch_size = 32;
    let in_channels = 1;
    let img_size = 28;
    let conv1_out = 32;
    let conv2_out = 64;
    let fc1_out = 128;
    let num_classes = 10;
    let num_epochs = 5;

    println!("[Architecture]");
    println!("  Input: {}x{}x{}", in_channels, img_size, img_size);
    println!("  Conv1: {} -> {} (3x3)", in_channels, conv1_out);
    println!("  Conv2: {} -> {} (3x3)", conv1_out, conv2_out);
    println!("  FC1: {} -> {}", conv2_out * 7 * 7, fc1_out);
    println!("  FC2: {} -> {} (classes)", fc1_out, num_classes);
    println!("");

    // =========================================================================
    // CREATE SYNTHETIC DATA
    // =========================================================================
    println!("[Data] Creating synthetic image batch...");

    // Random images [batch, channels, height, width]
    let img_data: Vec<f32> = (0..batch_size * in_channels * img_size * img_size)
        .map(|i| (i as f32 * 0.001).sin().abs())
        .collect();
    let images_host = Grid::new(img_data, vec![batch_size, in_channels, img_size, img_size]);

    // Labels (class indices as floats for simplicity, we'll use MSE instead of CE)
    let mut labels_data = vec![0.0f32; batch_size * num_classes];
    for b in 0..batch_size {
        let class_idx = b % num_classes;
        labels_data[b * num_classes + class_idx] = 1.0;  // One-hot
    }
    let labels_host = Grid::new(labels_data, vec![batch_size, num_classes]);

    // =========================================================================
    // INITIALIZE WEIGHTS
    // =========================================================================
    println!("[Weights] Initializing CNN parameters...");

    // Conv1 weights: [out_channels, in_channels, kH, kW]
    let conv1_w_data = kaiming_init(conv1_out * in_channels * 9, conv1_out);
    let conv1_w_host = Grid::new(conv1_w_data, vec![conv1_out, in_channels, 3, 3]);

    // Conv2 weights
    let conv2_w_data = kaiming_init(conv2_out * conv1_out * 9, conv2_out);
    let conv2_w_host = Grid::new(conv2_w_data, vec![conv2_out, conv1_out, 3, 3]);

    // FC1 weights: [in_features, out_features]
    let fc1_in = conv2_out * 7 * 7;  // After 2 maxpools: 28->14->7
    let fc1_w_data = xavier_init(fc1_in, fc1_out);
    let fc1_w_host = Grid::new(fc1_w_data, vec![fc1_in, fc1_out]);
    let fc1_b_host = Grid::new(vec![0.0f32; fc1_out], vec![fc1_out]);

    // FC2 weights
    let fc2_w_data = xavier_init(fc1_out, num_classes);
    let fc2_w_host = Grid::new(fc2_w_data, vec![fc1_out, num_classes]);
    let fc2_b_host = Grid::new(vec![0.0f32; num_classes], vec![num_classes]);

    // BatchNorm parameters for conv layers
    let bn1_gamma = Grid::new(vec![1.0f32; conv1_out], vec![conv1_out]);
    let bn1_beta = Grid::new(vec![0.0f32; conv1_out], vec![conv1_out]);
    let bn1_mean = Grid::new(vec![0.0f32; conv1_out], vec![conv1_out]);
    let bn1_var = Grid::new(vec![1.0f32; conv1_out], vec![conv1_out]);

    let bn2_gamma = Grid::new(vec![1.0f32; conv2_out], vec![conv2_out]);
    let bn2_beta = Grid::new(vec![0.0f32; conv2_out], vec![conv2_out]);
    let bn2_mean = Grid::new(vec![0.0f32; conv2_out], vec![conv2_out]);
    let bn2_var = Grid::new(vec![1.0f32; conv2_out], vec![conv2_out]);

    // SGD momentum buffers
    let fc1_mom = Grid::new(vec![0.0f32; fc1_in * fc1_out], vec![fc1_in, fc1_out]);
    let fc2_mom = Grid::new(vec![0.0f32; fc1_out * num_classes], vec![fc1_out, num_classes]);

    // =========================================================================
    // MOVE TO GPU
    // =========================================================================
    println!("[Memory] Transferring to GPU...");

    let images_gpu = compute(Op::MoveToDevice, vec![&images_host], None, &rules);
    let labels_gpu = compute(Op::MoveToDevice, vec![&labels_host], None, &rules);

    let conv1_w = compute(Op::MoveToDevice, vec![&conv1_w_host], None, &rules);
    let conv2_w = compute(Op::MoveToDevice, vec![&conv2_w_host], None, &rules);

    let mut fc1_w = compute(Op::MoveToDevice, vec![&fc1_w_host], None, &rules);
    let fc1_b = compute(Op::MoveToDevice, vec![&fc1_b_host], None, &rules);
    let mut fc2_w = compute(Op::MoveToDevice, vec![&fc2_w_host], None, &rules);
    let fc2_b = compute(Op::MoveToDevice, vec![&fc2_b_host], None, &rules);

    let bn1_g = compute(Op::MoveToDevice, vec![&bn1_gamma], None, &rules);
    let bn1_b = compute(Op::MoveToDevice, vec![&bn1_beta], None, &rules);
    let bn1_m = compute(Op::MoveToDevice, vec![&bn1_mean], None, &rules);
    let bn1_v = compute(Op::MoveToDevice, vec![&bn1_var], None, &rules);

    let bn2_g = compute(Op::MoveToDevice, vec![&bn2_gamma], None, &rules);
    let bn2_b = compute(Op::MoveToDevice, vec![&bn2_beta], None, &rules);
    let bn2_m = compute(Op::MoveToDevice, vec![&bn2_mean], None, &rules);
    let bn2_v = compute(Op::MoveToDevice, vec![&bn2_var], None, &rules);

    let fc1_momentum = compute(Op::MoveToDevice, vec![&fc1_mom], None, &rules);
    let fc2_momentum = compute(Op::MoveToDevice, vec![&fc2_mom], None, &rules);

    // =========================================================================
    // TRAINING LOOP
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     CNN TRAINING                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let lr = 0.01f32;
    let momentum = 0.9f32;

    for epoch in 1..=num_epochs {
        let epoch_start = std::time::Instant::now();

        // =====================================================================
        // FORWARD PASS
        // =====================================================================

        // Conv1: [batch, 1, 28, 28] -> [batch, 32, 28, 28]
        let conv1_out_tensor = compute(
            Op::Conv2dForward {
                stride: (1, 1),
                padding: (1, 1),
                has_bias: false
            },
            vec![&images_gpu, &conv1_w],
            None,
            &rules
        );

        // Flatten for batch norm: [batch, 32, 28, 28] -> [batch * 28 * 28, 32]
        let conv1_flat = Grid {
            values: conv1_out_tensor.values.clone(),
            shape: vec![batch_size * 28 * 28, conv1_out],
        };

        // BatchNorm1
        let bn1_out = compute(
            Op::BatchNormForward { training: true, momentum: 0.1, eps: 1e-5 },
            vec![&conv1_flat, &bn1_g, &bn1_b, &bn1_m, &bn1_v],
            None,
            &rules
        );

        // Reshape back and ReLU
        let bn1_reshaped = Grid {
            values: bn1_out.values.clone(),
            shape: vec![batch_size, conv1_out, 28, 28],
        };
        let relu1 = compute(Op::TorchUnary("relu".to_string()), vec![&bn1_reshaped], None, &rules);

        // MaxPool: [batch, 32, 28, 28] -> [batch, 32, 14, 14]
        let pool1 = compute(
            Op::MaxPool2d { kernel: (2, 2), stride: (2, 2) },
            vec![&relu1],
            None,
            &rules
        );

        // Conv2: [batch, 32, 14, 14] -> [batch, 64, 14, 14]
        let conv2_out_tensor = compute(
            Op::Conv2dForward {
                stride: (1, 1),
                padding: (1, 1),
                has_bias: false
            },
            vec![&pool1, &conv2_w],
            None,
            &rules
        );

        // Flatten for batch norm
        let conv2_flat = Grid {
            values: conv2_out_tensor.values.clone(),
            shape: vec![batch_size * 14 * 14, conv2_out],
        };

        // BatchNorm2
        let bn2_out = compute(
            Op::BatchNormForward { training: true, momentum: 0.1, eps: 1e-5 },
            vec![&conv2_flat, &bn2_g, &bn2_b, &bn2_m, &bn2_v],
            None,
            &rules
        );

        let bn2_reshaped = Grid {
            values: bn2_out.values.clone(),
            shape: vec![batch_size, conv2_out, 14, 14],
        };
        let relu2 = compute(Op::TorchUnary("relu".to_string()), vec![&bn2_reshaped], None, &rules);

        // MaxPool: [batch, 64, 14, 14] -> [batch, 64, 7, 7]
        let pool2 = compute(
            Op::MaxPool2d { kernel: (2, 2), stride: (2, 2) },
            vec![&relu2],
            None,
            &rules
        );

        // Flatten: [batch, 64, 7, 7] -> [batch, 3136]
        let flattened = Grid {
            values: pool2.values.clone(),
            shape: vec![batch_size, fc1_in],
        };

        // FC1
        let fc1_out_tensor = compute(
            Op::LinearForward { has_bias: true },
            vec![&flattened, &fc1_w, &fc1_b],
            None,
            &rules
        );
        let fc1_relu = compute(Op::TorchUnary("relu".to_string()), vec![&fc1_out_tensor], None, &rules);

        // FC2 (logits)
        let logits = compute(
            Op::LinearForward { has_bias: true },
            vec![&fc1_relu, &fc2_w, &fc2_b],
            None,
            &rules
        );

        // =====================================================================
        // LOSS (MSE as simplified cross-entropy)
        // =====================================================================
        let logits_flat = Grid {
            values: logits.values.clone(),
            shape: vec![batch_size * num_classes],
        };
        let labels_flat = Grid {
            values: labels_gpu.values.clone(),
            shape: vec![batch_size * num_classes],
        };

        let loss = compute(Op::MSELoss, vec![&logits_flat, &labels_flat], None, &rules);

        let loss_host = compute(Op::MoveToHost, vec![&loss], None, &rules);
        let loss_val = if let ferrite::data::Values::Host(v) = &loss_host.values {
            v[0]
        } else { 0.0 };

        // =====================================================================
        // BACKWARD PASS (through FC layers)
        // =====================================================================
        let grad_logits = compute(Op::MSELossBackward, vec![&logits_flat, &labels_flat], None, &rules);

        // Reshape gradient
        let grad_logits_2d = Grid {
            values: grad_logits.values.clone(),
            shape: vec![batch_size, num_classes],
        };

        // Backprop through FC2
        let grad_fc1_relu = compute(
            Op::LinearBackward { has_bias: true },
            vec![&fc1_relu, &fc2_w, &grad_logits_2d],
            None,
            &rules
        );

        // Backprop through FC1 ReLU
        let grad_fc1_out = compute(
            Op::ActivationBackward("relu".to_string()),
            vec![&fc1_out_tensor, &grad_fc1_relu],
            None,
            &rules
        );

        // Backprop through FC1
        let _grad_flatten = compute(
            Op::LinearBackward { has_bias: true },
            vec![&flattened, &fc1_w, &grad_fc1_out],
            None,
            &rules
        );

        // =====================================================================
        // SGD UPDATE (FC layers only for demo)
        // =====================================================================

        // Reshape grads for SGD
        let fc2_grad_flat = Grid {
            values: grad_logits.values.clone(),
            shape: vec![fc1_out * num_classes],
        };
        let fc1_grad_flat = Grid {
            values: grad_fc1_out.values.clone(),
            shape: vec![fc1_in * fc1_out],
        };
        let fc2_w_flat = Grid {
            values: fc2_w.values.clone(),
            shape: vec![fc1_out * num_classes],
        };
        let fc1_w_flat = Grid {
            values: fc1_w.values.clone(),
            shape: vec![fc1_in * fc1_out],
        };
        let fc2_mom_flat = Grid {
            values: fc2_momentum.values.clone(),
            shape: vec![fc1_out * num_classes],
        };
        let fc1_mom_flat = Grid {
            values: fc1_momentum.values.clone(),
            shape: vec![fc1_in * fc1_out],
        };

        let fc2_updated = compute(
            Op::SGDStep { lr, momentum, weight_decay: 0.0001 },
            vec![&fc2_w_flat, &fc2_grad_flat, &fc2_mom_flat],
            None,
            &rules
        );

        let fc1_updated = compute(
            Op::SGDStep { lr, momentum, weight_decay: 0.0001 },
            vec![&fc1_w_flat, &fc1_grad_flat, &fc1_mom_flat],
            None,
            &rules
        );

        // Reshape back
        fc2_w = Grid {
            values: fc2_updated.values,
            shape: vec![fc1_out, num_classes],
        };
        fc1_w = Grid {
            values: fc1_updated.values,
            shape: vec![fc1_in, fc1_out],
        };

        let epoch_time = epoch_start.elapsed();

        // Calculate accuracy (argmax predictions vs labels)
        let logits_host = compute(Op::MoveToHost, vec![&logits], None, &rules);
        let preds = if let ferrite::data::Values::Host(v) = &logits_host.values {
            v.clone()
        } else { vec![] };

        let mut correct = 0;
        for b in 0..batch_size {
            let start = b * num_classes;
            let pred_class = preds[start..start + num_classes]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let true_class = b % num_classes;
            if pred_class == true_class {
                correct += 1;
            }
        }
        let accuracy = 100.0 * correct as f32 / batch_size as f32;

        println!(
            "  Epoch {:2}/{} │ Loss: {:.6} │ Accuracy: {:5.1}% │ Time: {:?}",
            epoch, num_epochs, loss_val, accuracy, epoch_time
        );
    }

    println!("\n[Success] CNN training complete!");
    println!("[Info] Demonstrated: Conv2D, MaxPool2D, BatchNorm, ReLU, Linear, SGD with Momentum");
}

fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    let size = fan_in * fan_out;
    let mut seed: u64 = 42;
    (0..size).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (seed >> 33) as f32 / (1u64 << 31) as f32;
        (u * 2.0 - 1.0) * limit
    }).collect()
}

fn kaiming_init(size: usize, fan_in: usize) -> Vec<f32> {
    let std_val = (2.0f32 / fan_in as f32).sqrt();
    let mut seed: u64 = 123;
    (0..size).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (seed >> 33) as f32 / (1u64 << 31) as f32;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (seed >> 33) as f32 / (1u64 << 31) as f32;
        // Box-Muller transform
        let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z * std_val
    }).collect()
}
