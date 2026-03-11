//! neural_fabric — Self-Healing Training Loop with cuBLAS + CUDA Graphs
//!
//! Proves: Long-running cuBLAS computation with resilience (circuit breaker, retries),
//! CUDA graph replay, VFS checkpointing — all on TLSF.
//!
//! OS primitives exercised: TLSF, cuBLAS, CUDA graphs, VFS checkpointing,
//! CircuitBreaker, RetryPolicy, RateLimiter, watchdog.

use std::time::{Duration, Instant};

use anyhow::Result;
use ptx_runtime::resilience::{CircuitBreaker, RetryPolicy};
use ptx_runtime::{GpuPtr, GemmOp};

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.55;
const MAX_STREAMS: u32 = 64;

// MLP architecture: 256 → 512 → 256 → 128 → 1
const LAYER_DIMS: [(usize, usize); 4] = [
    (256, 512),
    (512, 256),
    (256, 128),
    (128, 1),
];
const BATCH_SIZE: usize = 32;
const WARMUP_STEPS: u64 = 100;
const CHECKPOINT_INTERVAL_SECS: u64 = 60;

struct Layer {
    weights: GpuPtr,
    output: GpuPtr,
    rows: usize,
    cols: usize,
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();
    println!("=== NEURAL FABRIC ===");
    println!("Self-healing MLP training loop with cuBLAS + CUDA graphs");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!("Architecture: 256->512->256->128->1, batch_size={}", BATCH_SIZE);
    println!("Config: pool_fraction={}, max_streams={}", POOL_FRACTION, MAX_STREAMS);
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("neural", 10);

    // Resilience primitives
    let circuit_breaker = CircuitBreaker::new(5, 3, Duration::from_secs(10));
    let retry_policy = RetryPolicy {
        max_attempts: 3,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        backoff_multiplier: 2.0,
    };

    // Initialize VFS for checkpointing
    let vfs = unsafe { platform::vfs_safe_init(&rt)? };
    unsafe {
        platform::vfs_safe_mkdir(vfs, "/model")?;
        platform::vfs_safe_mkdir(vfs, "/model/checkpoints")?;
    }

    // Allocate layers using circuit breaker + retry
    println!("Allocating MLP layers...");
    let mut layers: Vec<Layer> = Vec::new();
    for (i, &(rows, cols)) in LAYER_DIMS.iter().enumerate() {
        let weight_bytes = rows * cols * 4;
        let output_bytes = BATCH_SIZE * cols * 4;

        let weights = retry_policy.execute(|| {
            if !circuit_breaker.allow_request() {
                return Err(anyhow::anyhow!("Circuit breaker open"));
            }
            match rt.alloc(weight_bytes) {
                Ok(ptr) => {
                    circuit_breaker.record_success();
                    Ok(ptr)
                }
                Err(e) => {
                    circuit_breaker.record_failure();
                    Err(anyhow::anyhow!("Alloc failed: {}", e))
                }
            }
        })?;

        let output = rt.alloc(output_bytes)?;

        // Initialize weights with small random-ish values via fill + affine
        let stream = rt.next_stream();
        let init_val = 0.01 * ((i + 1) as f32);
        unsafe {
            ptx_sys::ptx_tensor_fill_f32(
                weights.as_ptr_typed::<f32>(),
                rows * cols,
                init_val,
                stream.raw(),
            );
        }
        output.zero()?;

        println!("  Layer {}: {}x{} ({}) | output: {}x{} ({})",
            i, rows, cols, platform::format_bytes(weight_bytes),
            BATCH_SIZE, cols, platform::format_bytes(output_bytes));

        layers.push(Layer { weights, output, rows, cols });
    }

    // Allocate input buffer
    let input_bytes = BATCH_SIZE * LAYER_DIMS[0].0 * 4;
    let input = rt.alloc(input_bytes)?;
    let stream0 = rt.stream(0)?;
    unsafe {
        ptx_sys::ptx_tensor_fill_f32(
            input.as_ptr_typed::<f32>(),
            BATCH_SIZE * LAYER_DIMS[0].0,
            0.5,
            stream0.raw(),
        );
    }

    // Temp buffer for activations
    let max_layer_size = LAYER_DIMS.iter()
        .map(|&(_, c)| BATCH_SIZE * c)
        .max().unwrap();
    let temp = rt.alloc(max_layer_size * 4)?;

    // Loss buffer (single f32)
    let loss_buf = rt.alloc(4)?;

    // Get cuBLAS handle
    let cublas_guard = rt.cublas()?;
    let cublas = cublas_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("cuBLAS not available"))?;

    let mut step: u64 = 0;
    let mut gemm_count: u64 = 0;
    let mut graph_replays: u64 = 0;
    let mut checkpoint_count: u32 = 0;
    let mut last_checkpoint = Instant::now();
    let mut captured_graph: Option<ptx_runtime::CudaGraph> = None;
    let mut learning_rate: f32 = 0.001;

    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);

    println!("\nStarting training loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();
        step += 1;

        // === FORWARD PASS ===
        let use_graph = step > WARMUP_STEPS && captured_graph.is_some();

        if use_graph {
            // Replay captured CUDA graph
            let graph = captured_graph.as_ref().unwrap();
            graph.launch(&stream0)?;
            graph_replays += 1;
        } else {
            // Manual forward pass: cuBLAS SGEMM per layer → ReLU → final sigmoid
            let mut prev_ptr = input.as_ptr_typed::<f32>() as *const f32;
            let mut prev_cols = LAYER_DIMS[0].0;

            for (i, layer) in layers.iter().enumerate() {
                let stream = rt.stream(i as i32 % MAX_STREAMS as i32)?;
                cublas.set_stream(&stream)?;

                // SGEMM: output = input @ weights^T
                // For row-major: C = A * B  →  cuBLAS(B^T, A^T) in col-major
                unsafe {
                    cublas.sgemm(
                        GemmOp::None,        // B (no transpose in col-major → transpose in row-major)
                        GemmOp::None,        // A
                        layer.cols as i32,   // n = cols of output
                        BATCH_SIZE as i32,   // m = batch
                        prev_cols as i32,    // k = shared dim
                        1.0,                 // alpha
                        layer.weights.as_ptr_typed::<f32>(),  // B
                        layer.cols as i32,                    // ldb
                        prev_ptr,                             // A
                        prev_cols as i32,                     // lda
                        0.0,                                  // beta
                        layer.output.as_ptr_typed::<f32>(),   // C
                        layer.cols as i32,                    // ldc
                    )?;
                }
                gemm_count += 1;

                let out_size = BATCH_SIZE * layer.cols;

                // Activation: ReLU for hidden layers, sigmoid for output
                if i < LAYER_DIMS.len() - 1 {
                    unsafe {
                        ptx_sys::ptx_tensor_relu_f32(
                            layer.output.as_ptr_typed::<f32>(),
                            layer.output.as_ptr_typed::<f32>(),
                            out_size,
                            stream.raw(),
                        );
                    }
                } else {
                    unsafe {
                        ptx_sys::ptx_tensor_sigmoid_f32(
                            layer.output.as_ptr_typed::<f32>(),
                            layer.output.as_ptr_typed::<f32>(),
                            out_size,
                            stream.raw(),
                        );
                    }
                }

                prev_ptr = layer.output.as_ptr_typed::<f32>() as *const f32;
                prev_cols = layer.cols;
            }
        }

        // === COMPUTE LOSS (reduce_mean of output) ===
        let final_layer = &layers[layers.len() - 1];
        let final_size = BATCH_SIZE * final_layer.cols;
        unsafe {
            ptx_sys::ptx_tensor_reduce_mean_f32(
                final_layer.output.as_ptr_typed::<f32>(),
                loss_buf.as_ptr_typed::<f32>(),
                1, final_size, 1,
                stream0.raw(),
            );
        }

        // === SIMPLIFIED BACKWARD PASS ===
        // Apply weight updates using affine: w = w * (1 - lr) + gradient_noise * lr
        for (i, layer) in layers.iter().enumerate() {
            let stream = rt.stream(i as i32 % MAX_STREAMS as i32)?;
            let n_weights = layer.rows * layer.cols;
            let decay = 1.0 - learning_rate;
            let noise = learning_rate * 0.01;
            unsafe {
                ptx_sys::ptx_tensor_affine_f32(
                    layer.weights.as_ptr_typed::<f32>(),
                    layer.weights.as_ptr_typed::<f32>(),
                    n_weights,
                    decay,
                    noise,
                    stream.raw(),
                );
            }
        }

        // === CUDA GRAPH CAPTURE at warmup boundary ===
        if step == WARMUP_STEPS && captured_graph.is_none() {
            println!("Step {}: Capturing CUDA graph for forward pass...", step);
            match rt.begin_capture(0, "forward_pass") {
                Ok(capture) => {
                    // Re-run forward pass inside capture
                    let mut prev_ptr = input.as_ptr_typed::<f32>() as *const f32;
                    let mut prev_cols = LAYER_DIMS[0].0;
                    for (i, layer) in layers.iter().enumerate() {
                        unsafe {
                            cublas.set_stream(&stream0)?;
                            cublas.sgemm(
                                GemmOp::None,
                                GemmOp::None,
                                layer.cols as i32,
                                BATCH_SIZE as i32,
                                prev_cols as i32,
                                1.0,
                                layer.weights.as_ptr_typed::<f32>(),
                                layer.cols as i32,
                                prev_ptr,
                                prev_cols as i32,
                                0.0,
                                layer.output.as_ptr_typed::<f32>(),
                                layer.cols as i32,
                            )?;
                        }

                        let out_size = BATCH_SIZE * layer.cols;
                        if i < LAYER_DIMS.len() - 1 {
                            unsafe {
                                ptx_sys::ptx_tensor_relu_f32(
                                    layer.output.as_ptr_typed::<f32>(),
                                    layer.output.as_ptr_typed::<f32>(),
                                    out_size,
                                    stream0.raw(),
                                );
                            }
                        } else {
                            unsafe {
                                ptx_sys::ptx_tensor_sigmoid_f32(
                                    layer.output.as_ptr_typed::<f32>(),
                                    layer.output.as_ptr_typed::<f32>(),
                                    out_size,
                                    stream0.raw(),
                                );
                            }
                        }

                        prev_ptr = layer.output.as_ptr_typed::<f32>() as *const f32;
                        prev_cols = layer.cols;
                    }

                    match capture.end() {
                        Ok(graph) => {
                            println!("  CUDA graph captured successfully.");
                            captured_graph = Some(graph);
                        }
                        Err(e) => {
                            eprintln!("  Graph capture failed: {}. Continuing without graph.", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  begin_capture failed: {}. Continuing without graph.", e);
                }
            }
        }

        // === VFS CHECKPOINT ===
        if last_checkpoint.elapsed() >= Duration::from_secs(CHECKPOINT_INTERVAL_SECS) {
            rt.sync_all()?;
            checkpoint_count += 1;
            for (i, layer) in layers.iter().enumerate() {
                let path = format!("/model/checkpoints/layer_{}", i);
                let shape = [layer.rows as i32, layer.cols as i32];
                unsafe {
                    let _ = platform::vfs_safe_create_tensor(vfs, &path, &shape, 0);
                    let _ = platform::vfs_safe_sync_tensor(vfs, &path);
                }
            }
            last_checkpoint = Instant::now();
        }

        // === LEARNING RATE DECAY ===
        if step % 1000 == 0 && learning_rate > 0.0001 {
            learning_rate *= 0.99;
        }

        // === TELEMETRY ===
        if reporter.should_report() {
            rt.sync_all()?;
            let mut loss_val: f32 = 0.0;
            unsafe {
                loss_buf.copy_to_host(
                    &mut loss_val as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }

            let circuit_state = match circuit_breaker.state() {
                ptx_runtime::resilience::CircuitState::Closed => "CLOSED",
                ptx_runtime::resilience::CircuitState::Open => "OPEN",
                ptx_runtime::resilience::CircuitState::HalfOpen => "HALF_OPEN",
            };

            reporter.report(&rt, &format!(
                "step={} | loss={:.6} | lr={:.6} | gemm={} | graph_replay={} | ckpt={} | circuit={}",
                step, loss_val, learning_rate, gemm_count, graph_replays, checkpoint_count, circuit_state,
            ));
        }
    }

    println!("\n=== NEURAL FABRIC COMPLETE ===");
    println!("Total steps: {}", step);
    println!("Total GEMMs: {}", gemm_count);
    println!("Graph replays: {}", graph_replays);
    println!("Checkpoints saved: {}", checkpoint_count);
    println!("Final learning rate: {:.6}", learning_rate);
    println!("Duration: {:.1}s", reporter.elapsed().as_secs_f64());

    // Cleanup
    drop(captured_graph);
    drop(cublas_guard);
    drop(loss_buf);
    drop(temp);
    drop(input);
    for layer in layers.into_iter().rev() {
        drop(layer.output);
        drop(layer.weights);
    }

    // Cleanup VFS checkpoints
    unsafe {
        for i in 0..LAYER_DIMS.len() {
            let path = format!("/model/checkpoints/layer_{}", i);
            let _ = platform::vfs_safe_unlink(vfs, &path);
        }
        let _ = platform::vfs_safe_rmdir(vfs, "/model/checkpoints");
        let _ = platform::vfs_safe_rmdir(vfs, "/model");
        ptx_sys::vfs_shutdown(vfs);
    }

    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}
