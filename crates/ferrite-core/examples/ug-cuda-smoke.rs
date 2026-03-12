use ferrite_core::{create_ug_cuda_device, ug, ug_cuda};
use ferrite_core::ug::Slice;

fn main() -> anyhow::Result<()> {
    let kernel = ug::samples::ssa::simple_dotprod(1024)?;
    let mut buf = Vec::new();
    ug_cuda::code_gen::gen(&mut buf, "ferrite_dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;

    let device = create_ug_cuda_device(0)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
    let func = device.compile_cu(&cuda_code, "ferrite_ug_cuda", "ferrite_dotprod")?;
    let func = ug_cuda::runtime::Func::new(func, cfg);

    let lhs_host: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let rhs_host: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected: f32 = lhs_host
        .iter()
        .zip(rhs_host.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum();

    let lhs = device.slice_from_values(&lhs_host)?;
    let rhs = device.slice_from_values(&rhs_host)?;
    let out = device.zeros(1)?;

    unsafe {
        func.launch3((out.slice::<f32>()?, lhs.slice::<f32>()?, rhs.slice::<f32>()?))?;
    }
    device.synchronize()?;

    let result: Vec<f32> = out.to_vec()?;
    anyhow::ensure!(
        (result[0] - expected).abs() < 1e-2_f32,
        "ug-cuda mismatch: got {}, expected {expected}",
        result[0]
    );

    println!("Ferrite ug-cuda smoke test passed: result={}", result[0]);
    Ok(())
}
