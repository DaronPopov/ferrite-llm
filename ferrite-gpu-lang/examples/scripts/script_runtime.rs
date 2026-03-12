use ferrite_gpu_lang::{GpuLangRuntime, HostTensor, Program};

fn main() -> ferrite_gpu_lang::Result<()> {
    let mut p = Program::new();
    let x = p.input(&[1, 1, 1, 2048])?;
    let y = p.relu(x);
    let z = p.sigmoid(y);
    p.set_output(z);
    let c = p.compile()?;

    let host: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.002 - 1.0).collect();
    let input = HostTensor::new(vec![1, 1, 1, 2048], host)?;

    let rt = GpuLangRuntime::new(0)?;
    let out = rt.execute(&c, &[input])?;

    println!("script=runtime");
    println!("shape={:?}", out.shape());
    println!("head={:?}", &out.data()[0..8]);
    Ok(())
}
