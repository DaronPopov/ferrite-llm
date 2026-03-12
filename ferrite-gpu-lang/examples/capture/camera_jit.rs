/// Camera → JIT processing loop.
///
/// Captures frames from the default webcam, converts to f32 tensor,
/// and runs a JIT-compiled script through the GPU runtime.
///
/// Usage: cargo run --example camera_jit --features capture

use std::time::Instant;

use ferrite_gpu_lang::capture::camera::{Camera, CameraConfig, CameraSource};
use ferrite_gpu_lang::capture::convert::{frame_to_host_tensor, Normalize};
use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{cpu, GpuLangRuntime, Result};

const MAX_FRAMES: u64 = 100;

fn main() -> Result<()> {
    // Open webcam.
    let config = CameraConfig::default().with_resolution(640, 480).with_fps(30.0);
    let mut camera = Camera::open(CameraSource::Device(0), config).map_err(|e| {
        eprintln!("Failed to open camera: {e}");
        eprintln!("Hint: ensure a webcam is connected, or use a video file.");
        e
    })?;

    let (w, h) = camera.resolution();
    println!("camera opened: {}x{} @ {:.1} fps", w, h, camera.fps());

    // Compile JIT script.
    let script = r#"
        x = input([1, 3, 480, 640])
        h = relu(x)
        h = sigmoid(h)
        return h
    "#;

    let mut jit = JitEngine::new();
    let compiled = jit.compile(script)?;
    println!(
        "JIT compiled: input={:?} → output={:?}",
        compiled.input_shapes(),
        compiled.output_shape()
    );

    // GPU runtime.
    let runtime = GpuLangRuntime::new(0)?;

    // Run the loop inside cpu() to have access to CpuCtx for TLSF allocations.
    cpu(|ctx| -> Result<()> {
        let mut total_capture_us = 0u128;
        let mut total_convert_us = 0u128;
        let mut total_execute_us = 0u128;

        for i in 0..MAX_FRAMES {
            // 1. Capture
            let t0 = Instant::now();
            let frame = camera.read(ctx)?;
            let capture_us = t0.elapsed().as_micros();
            total_capture_us += capture_us;

            // 2. Convert to HostTensor (fused HWC→CHW + u8→f32 /255)
            let t1 = Instant::now();
            let tensor = frame_to_host_tensor(&frame, &Normalize::UnitRange)?;
            let convert_us = t1.elapsed().as_micros();
            total_convert_us += convert_us;

            // 3. Execute on GPU
            let t2 = Instant::now();
            let output = runtime.execute(compiled, &[tensor])?;
            let execute_us = t2.elapsed().as_micros();
            total_execute_us += execute_us;

            if i % 10 == 0 {
                println!(
                    "frame {:>3}: capture={:.1}ms convert={:.1}ms execute={:.1}ms shape={:?} head={:.4?}",
                    i,
                    capture_us as f64 / 1000.0,
                    convert_us as f64 / 1000.0,
                    execute_us as f64 / 1000.0,
                    output.shape(),
                    &output.data()[..4.min(output.data().len())],
                );
            }
        }

        let n = MAX_FRAMES as f64;
        println!("\n── Summary ({MAX_FRAMES} frames) ──");
        println!(
            "avg capture:  {:.2} ms",
            total_capture_us as f64 / n / 1000.0
        );
        println!(
            "avg convert:  {:.2} ms",
            total_convert_us as f64 / n / 1000.0
        );
        println!(
            "avg execute:  {:.2} ms",
            total_execute_us as f64 / n / 1000.0
        );
        let total_ms = (total_capture_us + total_convert_us + total_execute_us) as f64 / 1000.0;
        println!("effective fps: {:.1}", n / (total_ms / 1000.0));

        let stats = ctx.allocator_stats();
        println!(
            "TLSF: allocs={} frees={} peak={}KB current={}KB",
            stats.alloc_calls,
            stats.free_calls,
            stats.peak_allocated_bytes / 1024,
            stats.current_allocated_bytes / 1024,
        );

        Ok(())
    })
}
