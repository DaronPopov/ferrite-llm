/// Camera → CvBuilder processing loop.
///
/// Captures frames from the default webcam, converts to tch::Tensor via
/// the capture→torch bridge, and runs through a CvBuilder conv2d+relu pipeline.
///
/// Usage: cargo run --example camera_cv --features "capture,torch"

use std::time::Instant;

use ferrite_gpu_lang::capture::bridge::FrameToTensor;
use ferrite_gpu_lang::capture::camera::{Camera, CameraConfig, CameraSource};
use ferrite_gpu_lang::capture::convert::Normalize;
use ferrite_gpu_lang::torch::cv::Conv2dCfg;
use ferrite_gpu_lang::torch_bridge::{init_torch_tlsf, torch_cuda_available};
use ferrite_gpu_lang::{cpu, gpu_anyhow};
use tch::{Device, Kind, Tensor};

const MAX_FRAMES: u64 = 60;

fn main() -> anyhow::Result<()> {
    if !torch_cuda_available() {
        println!("CUDA not available — exiting.");
        return Ok(());
    }

    // Open webcam.
    let config = CameraConfig::default().with_resolution(640, 480).with_fps(30.0);
    let mut camera = Camera::open(CameraSource::Device(0), config).map_err(|e| {
        eprintln!("Failed to open camera: {e}");
        anyhow::Error::msg(format!("{e}"))
    })?;

    let (w, h) = camera.resolution();
    println!("camera opened: {}x{} @ {:.1} fps", w, h, camera.fps());

    gpu_anyhow(0, |g| {
        init_torch_tlsf(0, 0.70)?;

        let device = Device::Cuda(0);

        // Random 3×3 conv weight: [out_ch=3, in_ch=3, kH=3, kW=3].
        let weight = Tensor::randn([3, 3, 3, 3], (Kind::Float, device));

        let mut total_capture_us = 0u128;
        let mut total_bridge_us = 0u128;
        let mut total_pipeline_us = 0u128;

        // Run capture + CvBuilder loop inside cpu() for TLSF access.
        cpu(|ctx| -> anyhow::Result<()> {
            for i in 0..MAX_FRAMES {
                // 1. Capture frame (TLSF-backed).
                let t0 = Instant::now();
                let frame = camera.read(ctx).map_err(anyhow::Error::msg)?;
                let capture_us = t0.elapsed().as_micros();
                total_capture_us += capture_us;

                // 2. Frame → tch::Tensor via bridge (ImageNet normalisation).
                let t1 = Instant::now();
                let tensor = frame
                    .to_tch_tensor(&Normalize::imagenet(), device)
                    .map_err(anyhow::Error::msg)?;
                let bridge_us = t1.elapsed().as_micros();
                total_bridge_us += bridge_us;

                // 3. CvBuilder pipeline: conv2d → relu.
                let t2 = Instant::now();
                let output = g
                    .cv()
                    .input(tensor)
                    .conv2d(weight.shallow_clone(), Conv2dCfg::same())?
                    .relu()?
                    .run()?;
                let pipeline_us = t2.elapsed().as_micros();
                total_pipeline_us += pipeline_us;

                if i % 10 == 0 {
                    println!(
                        "frame {:>3}: capture={:.1}ms bridge={:.1}ms pipeline={:.1}ms output={:?}",
                        i,
                        capture_us as f64 / 1000.0,
                        bridge_us as f64 / 1000.0,
                        pipeline_us as f64 / 1000.0,
                        output.size(),
                    );
                }
            }

            let n = MAX_FRAMES as f64;
            println!("\n── Summary ({MAX_FRAMES} frames) ──");
            println!(
                "avg capture:   {:.2} ms",
                total_capture_us as f64 / n / 1000.0
            );
            println!(
                "avg bridge:    {:.2} ms",
                total_bridge_us as f64 / n / 1000.0
            );
            println!(
                "avg pipeline:  {:.2} ms",
                total_pipeline_us as f64 / n / 1000.0
            );
            let total_ms =
                (total_capture_us + total_bridge_us + total_pipeline_us) as f64 / 1000.0;
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
        })?;

        Ok(())
    })
}
