use std::time::Instant;

use anyhow::Result;
use tch::{Device, Kind, Tensor};

use super::policy::TorchTlsfInit;

#[derive(Debug, Clone)]
pub struct TorchSmokeMetrics {
    pub avg_alloc_us: f64,
    pub matmul_ms: f64,
    pub sum: f64,
}

pub fn init_torch_tlsf(device_id: i32, pool_fraction: f32) -> Result<()> {
    let req = TorchTlsfInit {
        device_id,
        pool_fraction,
    }
    .validate()?;
    aten_ptx::init_pytorch_tlsf(req.device_id, req.pool_fraction as f64)
        .map_err(anyhow::Error::msg)?;
    unsafe {
        let p = cudarc::driver::result::malloc_sync(256)?;
        cudarc::driver::result::free_sync(p)?;
    }
    Ok(())
}

pub fn torch_cuda_available() -> bool {
    aten_ptx::ensure_libtorch_cuda_loaded();
    tch::Cuda::is_available()
}

pub fn torch_smoke_benchmark(device_id: usize, n: i64, iters: usize) -> Result<TorchSmokeMetrics> {
    let device = Device::Cuda(device_id);

    let mut alloc_total_us = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        let _x = Tensor::zeros([n, n], (Kind::Float, device));
        alloc_total_us += t0.elapsed().as_secs_f64() * 1e6;
    }

    let a = Tensor::randn([n, n], (Kind::Float, device));
    let b = Tensor::randn([n, n], (Kind::Float, device));

    let t1 = Instant::now();
    let c = a.matmul(&b);
    let matmul_ms = t1.elapsed().as_secs_f64() * 1e3;

    let sum = f64::try_from(c.sum(Kind::Float))?;
    tch::Cuda::synchronize(device_id as i64);

    Ok(TorchSmokeMetrics {
        avg_alloc_us: alloc_total_us / iters as f64,
        matmul_ms,
        sum,
    })
}
