#![cfg(feature = "torch")]

use anyhow::Result;
use ferrite_gpu_lang::gpu_anyhow;
use ferrite_gpu_lang::torch_bridge::{init_torch_tlsf, torch_cuda_available};
use tch::{Device, Kind, Tensor};

fn main() -> Result<()> {
    if !torch_cuda_available() {
        println!("CUDA not available");
        return Ok(());
    }

    gpu_anyhow(0, |g| {
        init_torch_tlsf(0, 0.70)?;

        let pred = Tensor::rand([1, 256, 85], (Kind::Float, Device::Cuda(0)));
        let dets = g
            .cv()
            .input(pred)
            .yolo_decode(0.25)?
            .nms(0.45, 0.25, 50)?
            .run()?;

        println!("script=cv_detect");
        println!("detections={}", dets.size()[0]);
        Ok(())
    })
}
