#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;

#[cfg(feature = "cuda")]
const PTX_SRC: &str = r#"
extern "C" __global__ void ferrite_axpy(
    const float *x,
    float *y,
    float alpha,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
"#;

#[cfg(feature = "cuda")]
fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;
    let ptx = compile_ptx(PTX_SRC).expect("NVRTC compilation failed");

    dev.load_ptx(ptx, "ferrite_axpy_module", &["ferrite_axpy"])?;
    let func = dev
        .get_func("ferrite_axpy_module", "ferrite_axpy")
        .expect("kernel missing after PTX load");

    let x_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut y_host = [10.0f32, 20.0, 30.0, 40.0];

    let x_dev = dev.htod_sync_copy(&x_host)?;
    let mut y_dev = dev.htod_sync_copy(&y_host)?;

    let cfg = LaunchConfig::for_num_elems(x_host.len() as u32);
    unsafe { func.launch(cfg, (&x_dev, &mut y_dev, 2.0f32, x_host.len() as i32)) }?;

    dev.dtoh_sync_copy_into(&y_dev, &mut y_host)?;
    assert_eq!(y_host, [12.0, 24.0, 36.0, 48.0]);
    println!("Ferrite custom kernel smoke test passed: {y_host:?}");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Enable the `cuda` feature to run this example.");
    std::process::exit(1);
}
