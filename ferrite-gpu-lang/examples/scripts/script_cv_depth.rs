#![cfg(feature = "torch")]

use anyhow::Result;
use ferrite_gpu_lang::cv::Conv2dCfg;
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

        let device = Device::Cuda(0);
        let image = Tensor::rand([1, 3, 128, 128], (Kind::Float, device));
        let w1 = Tensor::randn([8, 3, 3, 3], (Kind::Float, device)) * 0.02;
        let w2 = Tensor::randn([1, 8, 3, 3], (Kind::Float, device)) * 0.02;

        let depth = g
            .cv()
            .input(image)
            .conv2d(w1, Conv2dCfg::same())?
            .relu()?
            .conv2d(w2, Conv2dCfg::same())?
            .upsample2x()?
            .sigmoid()?
            .run()?;

        println!("script=cv_depth");
        println!("shape={:?}", depth.size());
        Ok(())
    })
}
