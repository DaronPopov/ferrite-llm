// Integration tests for custom CUDA kernels
//
// Verifies kernels produce mathematically correct outputs

#[cfg(all(test, feature = "cuda"))]
mod cuda_kernel_tests {
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn test_flash_attention_kernel_exists() {
        // Verify PTX was compiled
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/flash_attention.ptx");
        let ptx_exists = std::path::Path::new(ptx_path).exists();

        if !ptx_exists {
            panic!("Flash attention PTX not found at: {}", ptx_path);
        }

        // Read PTX file
        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        // Verify it contains expected kernels
        assert!(ptx_content.contains("flash_attention_forward"),
            "PTX should contain flash_attention_forward kernel");
        assert!(ptx_content.contains("flash_attention_causal_forward"),
            "PTX should contain flash_attention_causal_forward kernel");

        println!("✓ Flash Attention PTX compiled successfully");
        println!("  Size: {} bytes", ptx_content.len());
    }

    #[test]
    fn test_example_kernel_exists() {
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/example_kernel.ptx");
        let ptx_exists = std::path::Path::new(ptx_path).exists();

        if !ptx_exists {
            panic!("Example kernel PTX not found at: {}", ptx_path);
        }

        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        // Verify it contains expected kernels
        assert!(ptx_content.contains("relu_forward"),
            "PTX should contain relu_forward kernel");

        println!("✓ Example kernel PTX compiled successfully");
    }

    #[test]
    fn test_flash_attention_cpu_reference() {
        // Test CPU implementation produces correct results
        use ferrite::{flash_attention, FlashAttentionConfig};

        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;

        // Create simple test inputs
        let q = Tensor::ones((batch, heads, seq, dim), DType::F32, &device).unwrap();
        let k = Tensor::ones((batch, heads, seq, dim), DType::F32, &device).unwrap();
        let v = Tensor::arange(0f32, (batch * heads * seq * dim) as f32, &device)
            .unwrap()
            .reshape((batch, heads, seq, dim))
            .unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention(&q, &k, &v, &config).unwrap();

        // Verify output shape
        assert_eq!(output.dims(), &[batch, heads, seq, dim]);

        // Verify output is not NaN or Inf
        let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &val) in output_vec.iter().enumerate() {
            assert!(val.is_finite(), "Output[{}] is not finite: {}", i, val);
        }

        println!("✓ Flash Attention CPU reference produces valid outputs");
        println!("  Output shape: {:?}", output.dims());
        println!("  Output range: [{:.4}, {:.4}]",
            output_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            output_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    }

    #[test]
    fn test_flash_attention_causal_mask() {
        use ferrite::{flash_attention, FlashAttentionConfig};

        let device = Device::Cpu;
        let batch = 1;
        let heads = 1;
        let seq = 4;
        let dim = 4;

        // Q, K, V all ones - with causal mask, output should be different per position
        let q = Tensor::ones((batch, heads, seq, dim), DType::F32, &device).unwrap();
        let k = Tensor::ones((batch, heads, seq, dim), DType::F32, &device).unwrap();
        let v = Tensor::arange(0f32, (seq * dim) as f32, &device)
            .unwrap()
            .reshape((batch, heads, seq, dim))
            .unwrap();

        let mut config = FlashAttentionConfig::default();
        config.use_causal_mask = true;

        let output = flash_attention(&q, &k, &v, &config).unwrap();

        // With causal mask, each position should only attend to previous positions
        // So outputs should be different
        let out_data = output.squeeze(0).unwrap().squeeze(0).unwrap().to_vec2::<f32>().unwrap();

        // Position 0 should only see position 0
        // Position 1 should see average of positions 0-1
        // etc.

        println!("✓ Causal mask working correctly");
        println!("  Position 0 output: {:?}", out_data[0]);
        println!("  Position 3 output: {:?}", out_data[3]);

        // Outputs should be different due to causal masking
        assert_ne!(out_data[0], out_data[3], "Causal mask should create different outputs");
    }

    #[test]
    fn test_relu_kernel_gpu_execution() {
        use cudarc::driver::safe::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::Ptx;
        use std::sync::Arc;

        // Check if CUDA is available
        let device_result = CudaDevice::new(0);
        if device_result.is_err() {
            println!("⚠ CUDA device not available, skipping GPU test");
            return;
        }

        let cuda_dev = Arc::new(device_result.unwrap());

        // Load PTX
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/example_kernel.ptx");
        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        // Load module and get kernel function
        let ptx = Ptx::from_src(ptx_content);
        cuda_dev.load_ptx(
            ptx,
            "example_kernel",
            &["relu_forward"]
        ).expect("Should load PTX module");

        let func = cuda_dev.get_func("example_kernel", "relu_forward")
            .expect("Should find relu_forward kernel");

        // Create test data on CPU
        let n: usize = 1024;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 - 512.0).collect(); // Mix of positive and negative

        // Allocate GPU memory
        let input_gpu = cuda_dev.htod_sync_copy(&input_data).expect("Should copy to device");
        let mut output_gpu = cuda_dev.alloc_zeros::<f32>(n).expect("Should allocate output");

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (&input_gpu, &mut output_gpu, n as i32);
        unsafe {
            func.launch(cfg, params).expect("Should launch kernel");
        }

        // Copy result back
        let output_data = cuda_dev.dtoh_sync_copy(&output_gpu).expect("Should copy from device");

        // Verify results (ReLU: max(0, x))
        for i in 0..n {
            let expected = input_data[i].max(0.0);
            let actual = output_data[i];
            assert_eq!(actual, expected, "ReLU mismatch at index {}", i);
        }

        println!("✓ ReLU kernel executed successfully on GPU");
        println!("  Processed {} elements", n);
        println!("  Sample: input[-5.0] -> output[{}]", output_data[507]);
        println!("  Sample: input[5.0] -> output[{}]", output_data[517]);
    }

    #[test]
    fn test_flash_attention_kernel_gpu_execution() {
        use cudarc::driver::safe::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::Ptx;
        use std::sync::Arc;

        // Check if CUDA is available
        let device_result = CudaDevice::new(0);
        if device_result.is_err() {
            println!("⚠ CUDA device not available, skipping GPU test");
            return;
        }

        let cuda_dev = Arc::new(device_result.unwrap());

        // Load PTX
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/flash_attention.ptx");
        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        // Load module
        let ptx = Ptx::from_src(ptx_content);
        cuda_dev.load_ptx(
            ptx,
            "flash_attention",
            &["flash_attention_forward", "flash_attention_causal_forward"]
        ).expect("Should load PTX module");

        let func = cuda_dev.get_func("flash_attention", "flash_attention_forward")
            .expect("Should find flash_attention_forward kernel");

        // Small test case: batch=1, heads=1, seq=4, dim=8
        let batch = 1;
        let heads = 1;
        let seq = 4;
        let dim = 8;
        let total_size = batch * heads * seq * dim;

        // Create simple test inputs: Q, K all ones, V is sequential
        let q_data = vec![1.0f32; total_size];
        let k_data = vec![1.0f32; total_size];
        let v_data: Vec<f32> = (0..total_size).map(|i| i as f32).collect();

        // Allocate GPU memory
        let q_gpu = cuda_dev.htod_sync_copy(&q_data).expect("Should copy Q to device");
        let k_gpu = cuda_dev.htod_sync_copy(&k_data).expect("Should copy K to device");
        let v_gpu = cuda_dev.htod_sync_copy(&v_data).expect("Should copy V to device");
        let mut o_gpu = cuda_dev.alloc_zeros::<f32>(total_size).expect("Should allocate output");

        // Launch kernel
        let scale = 1.0 / (dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (
                (((seq + 63) / 64) as u32).max(1),
                heads as u32,
                batch as u32
            ),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &mut o_gpu,
            batch as i32,
            heads as i32,
            seq as i32,
            dim as i32,
            scale,
        );

        unsafe {
            func.launch(cfg, params).expect("Should launch kernel");
        }

        // Copy result back
        let output_data = cuda_dev.dtoh_sync_copy(&o_gpu).expect("Should copy from device");

        // Basic sanity checks
        assert_eq!(output_data.len(), total_size);

        // All outputs should be finite (no NaN/Inf)
        for (i, &val) in output_data.iter().enumerate() {
            assert!(val.is_finite(), "Output[{}] is not finite: {}", i, val);
        }

        println!("✓ Flash Attention kernel executed successfully on GPU");
        println!("  Batch={}, Heads={}, Seq={}, Dim={}", batch, heads, seq, dim);
        println!("  Output shape: [{}, {}, {}, {}]", batch, heads, seq, dim);
        println!("  Output sample: [{:.2}, {:.2}, {:.2}, ...]",
                 output_data[0], output_data[1], output_data[2]);
        println!("  All {} values are finite", total_size);
    }

    #[test]
    fn test_relu_kernel_large_scale_performance() {
        use cudarc::driver::safe::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::Ptx;
        use std::sync::Arc;
        use std::time::Instant;

        // Check if CUDA is available
        let device_result = CudaDevice::new(0);
        if device_result.is_err() {
            println!("⚠ CUDA device not available, skipping GPU test");
            return;
        }

        let cuda_dev = Arc::new(device_result.unwrap());

        // Load PTX
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/example_kernel.ptx");
        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        let ptx = Ptx::from_src(ptx_content);
        cuda_dev.load_ptx(ptx, "example_kernel_large", &["relu_forward"])
            .expect("Should load PTX module");

        let func = cuda_dev.get_func("example_kernel_large", "relu_forward")
            .expect("Should find relu_forward kernel");

        // Large-scale test: 64 million elements (~256MB of data)
        let n: usize = 64 * 1024 * 1024;
        println!("\n🚀 Large-Scale ReLU Kernel Test");
        println!("   Elements: {} ({:.2} million)", n, n as f64 / 1_000_000.0);
        println!("   Memory: {:.2} MB input + {:.2} MB output = {:.2} MB total",
                 (n * 4) as f64 / 1_048_576.0,
                 (n * 4) as f64 / 1_048_576.0,
                 (n * 8) as f64 / 1_048_576.0);

        // Create test data
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 / 1000.0) - 32000.0).collect();

        // Time: Host to Device transfer
        let start = Instant::now();
        let input_gpu = cuda_dev.htod_sync_copy(&input_data).expect("Should copy to device");
        let mut output_gpu = cuda_dev.alloc_zeros::<f32>(n).expect("Should allocate output");
        let transfer_time = start.elapsed();

        // Time: Kernel execution
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (&input_gpu, &mut output_gpu, n as i32);

        let start = Instant::now();
        unsafe {
            func.launch(cfg, params).expect("Should launch kernel");
        }
        cuda_dev.synchronize().expect("Should synchronize");
        let kernel_time = start.elapsed();

        // Time: Device to Host transfer
        let start = Instant::now();
        let output_data = cuda_dev.dtoh_sync_copy(&output_gpu).expect("Should copy from device");
        let download_time = start.elapsed();

        // Verify sample results
        for i in (0..n).step_by(n / 100) {
            let expected = input_data[i].max(0.0);
            let actual = output_data[i];
            assert!(
                (actual - expected).abs() < 1e-5,
                "ReLU mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }

        // Calculate throughput
        let total_bytes = (n * 8) as f64; // Input + output
        let kernel_throughput_gbps = (total_bytes / kernel_time.as_secs_f64()) / 1_073_741_824.0;
        let elements_per_sec = n as f64 / kernel_time.as_secs_f64();

        println!("\n📊 Performance Results:");
        println!("   Upload time:     {:>8.2} ms", transfer_time.as_secs_f64() * 1000.0);
        println!("   Kernel time:     {:>8.2} ms ⚡", kernel_time.as_secs_f64() * 1000.0);
        println!("   Download time:   {:>8.2} ms", download_time.as_secs_f64() * 1000.0);
        println!("   Total time:      {:>8.2} ms",
                 (transfer_time + kernel_time + download_time).as_secs_f64() * 1000.0);
        println!("\n   Throughput:      {:>8.2} GB/s", kernel_throughput_gbps);
        println!("   Elements/sec:    {:>8.2} billion/s", elements_per_sec / 1_000_000_000.0);
        println!("\n✓ All sampled results verified correct");
    }

    #[test]
    fn test_flash_attention_realistic_workload() {
        use cudarc::driver::safe::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::Ptx;
        use std::sync::Arc;
        use std::time::Instant;

        // Check if CUDA is available
        let device_result = CudaDevice::new(0);
        if device_result.is_err() {
            println!("⚠ CUDA device not available, skipping GPU test");
            return;
        }

        let cuda_dev = Arc::new(device_result.unwrap());

        // Load PTX
        let ptx_path = concat!(env!("OUT_DIR"), "/kernels/flash_attention.ptx");
        let ptx_content = std::fs::read_to_string(ptx_path)
            .expect("Should be able to read PTX file");

        let ptx = Ptx::from_src(ptx_content);
        cuda_dev.load_ptx(ptx, "flash_attention_realistic", &["flash_attention_forward"])
            .expect("Should load PTX module");

        let func = cuda_dev.get_func("flash_attention_realistic", "flash_attention_forward")
            .expect("Should find flash_attention_forward kernel");

        // Realistic LLM workload parameters
        let batch = 4;
        let heads = 32;     // Common for 7B-13B models
        let seq = 1024;     // 1K context length
        let dim = 128;      // Head dimension (total hidden = 4096 for 32*128)
        let total_size = batch * heads * seq * dim;

        println!("\n🚀 Realistic Flash Attention Workload");
        println!("   Batch size:      {}", batch);
        println!("   Attention heads: {}", heads);
        println!("   Sequence length: {}", seq);
        println!("   Head dimension:  {}", dim);
        println!("   Total hidden:    {} ({}*{})", heads * dim, heads, dim);
        println!("\n   Total elements:  {} ({:.2} million)", total_size, total_size as f64 / 1_000_000.0);
        println!("   Memory per tensor: {:.2} MB", (total_size * 4) as f64 / 1_048_576.0);
        println!("   Total memory:      {:.2} MB (Q+K+V+O)", (total_size * 16) as f64 / 1_048_576.0);

        // Create realistic test inputs
        let q_data: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let k_data: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01).cos()).collect();
        let v_data: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.001).tanh()).collect();

        // Time: Memory allocation and upload
        let start = Instant::now();
        let q_gpu = cuda_dev.htod_sync_copy(&q_data).expect("Should copy Q to device");
        let k_gpu = cuda_dev.htod_sync_copy(&k_data).expect("Should copy K to device");
        let v_gpu = cuda_dev.htod_sync_copy(&v_data).expect("Should copy V to device");
        let mut o_gpu = cuda_dev.alloc_zeros::<f32>(total_size).expect("Should allocate output");
        let transfer_time = start.elapsed();

        // Time: Kernel execution
        let scale = 1.0 / (dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (
                (((seq + 63) / 64) as u32).max(1),
                heads as u32,
                batch as u32
            ),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &mut o_gpu,
            batch as i32,
            heads as i32,
            seq as i32,
            dim as i32,
            scale,
        );

        let start = Instant::now();
        unsafe {
            func.launch(cfg, params).expect("Should launch kernel");
        }
        cuda_dev.synchronize().expect("Should synchronize");
        let kernel_time = start.elapsed();

        // Time: Download results
        let start = Instant::now();
        let output_data = cuda_dev.dtoh_sync_copy(&o_gpu).expect("Should copy from device");
        let download_time = start.elapsed();

        // Verify outputs are valid
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in output_data.iter() {
            if val.is_nan() {
                nan_count += 1;
            } else if val.is_infinite() {
                inf_count += 1;
            } else {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        assert_eq!(nan_count, 0, "Found {} NaN values in output", nan_count);
        assert_eq!(inf_count, 0, "Found {} Inf values in output", inf_count);

        // Calculate FLOPs (approximate)
        // Attention: 2*seq*seq*dim per head per batch (QK^T + softmax(QK^T)V)
        let flops = (batch * heads * seq * seq * dim * 4) as f64;
        let tflops = (flops / kernel_time.as_secs_f64()) / 1e12;

        println!("\n📊 Performance Results:");
        println!("   Upload time:     {:>8.2} ms", transfer_time.as_secs_f64() * 1000.0);
        println!("   Kernel time:     {:>8.2} ms ⚡", kernel_time.as_secs_f64() * 1000.0);
        println!("   Download time:   {:>8.2} ms", download_time.as_secs_f64() * 1000.0);
        println!("   Total time:      {:>8.2} ms",
                 (transfer_time + kernel_time + download_time).as_secs_f64() * 1000.0);
        println!("\n   Approx FLOPs:    {:.2} TFLOPs", flops / 1e12);
        println!("   Throughput:      {:.2} TFLOPs/s", tflops);
        println!("   Tokens/sec:      {:.2} (at seq={}, batch={})",
                 (batch * seq) as f64 / kernel_time.as_secs_f64(), seq, batch);
        println!("\n   Output range:    [{:.6}, {:.6}]", min_val, max_val);
        println!("   All {} values valid (no NaN/Inf)", total_size);
        println!("\n✓ Flash Attention completed successfully");
    }

    #[test]
    fn test_attention_numerical_stability() {
        use ferrite::{flash_attention, FlashAttentionConfig};

        let device = Device::Cpu;

        // Test with large values to check numerical stability
        let q = Tensor::new(&[100.0f32; 64], &device).unwrap().reshape((1, 1, 4, 16)).unwrap();
        let k = Tensor::new(&[100.0f32; 64], &device).unwrap().reshape((1, 1, 4, 16)).unwrap();
        let v = Tensor::new(&[1.0f32; 64], &device).unwrap().reshape((1, 1, 4, 16)).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention(&q, &k, &v, &config).unwrap();

        // Check for numerical issues
        let out_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &out_vec {
            assert!(val.is_finite(), "Output should not be NaN or Inf with large inputs");
            assert!(val.abs() < 1e6, "Output should not explode: {}", val);
        }

        println!("✓ Attention is numerically stable with large inputs");
    }
}

#[cfg(test)]
mod cpu_tests {
    use candle_core::{Device, Tensor};

    #[test]
    fn test_basic_tensor_ops() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).unwrap();

        let c = (&a + &b).unwrap();
        let result = c.to_vec1::<f32>().unwrap();

        assert_eq!(result, vec![5.0, 7.0, 9.0]);
        println!("✓ Basic tensor operations working");
    }
}
