use ferrite_gpu_lang::{cpu, gpu, CpuTensor, HostTensor, Program, ToCpu, ToGpu};

fn main() -> ferrite_gpu_lang::Result<()> {
    println!("=== Ferrite Runtime Script: Whole OS Execution ===");

    // CPU algorithm stage: normal Rust data prep.
    let cpu_input = cpu(|_| {
        let data: Vec<f32> = (0..8192).map(|i| (i as f32) * 0.001 - 2.0).collect();
        CpuTensor::new(vec![1, 1, 1, 8192], data)
    })?;

    // GPU stage: explicit handoff + runtime graph execution on Ferrite OS.
    let (output, tlsf, hot) = gpu(0, |g| {
        // 1) Typed transfer CPU -> GPU
        let gpu_in = cpu_input.to_gpu(g)?;

        // 2) Direct GPU op (Candle-backed kernel) through Ferrite runtime
        let gpu_relu = gpu_in.relu(g)?;

        // 3) Runtime graph execution (same allocator/runtime)
        let mut p = Program::new();
        let x = p.input(gpu_relu.shape())?;
        let y = p.sigmoid(x);
        p.set_output(y);
        let c = p.compile()?;

        let host_relu = gpu_relu.to_cpu()?;
        let (shape, data) = host_relu.into_inner();
        let host = HostTensor::new(shape, data)?;
        let out = g.runtime().execute(&c, &[host])?;

        let tlsf = g.runtime().runtime().tlsf_stats();
        let hot = g.runtime().runtime().stats();
        Ok((out, tlsf, hot))
    })?;

    // CPU post-stage: consume result with normal Rust code.
    let (sum, positives) = cpu(|_| {
        let sum: f32 = output.data().iter().sum();
        let positives = output.data().iter().filter(|x| **x > 0.5).count();
        Ok::<(f32, usize), ferrite_gpu_lang::LangError>((sum, positives))
    })?;

    println!("output_shape={:?}", output.shape());
    println!("output_head={:?}", &output.data()[0..8]);
    println!("cpu_post: sum={:.6} positives={}", sum, positives);

    println!("\n--- Why this runtime is unique ---");
    println!("1) Unified CPU/GPU scripting in one Rust flow (no context switch language).");
    println!("2) Explicit typed handoff boundaries: CpuTensor <-> GpuTensor.");
    println!("3) Single allocator/runtime foundation for all GPU work (TLSF-backed).");
    println!("4) Deterministic allocator metrics available at script level.");

    println!("\n--- Ferrite OS Runtime Stats ---");
    println!("streams_active={}", hot.active_streams);
    println!("ops_total={}", hot.total_ops);
    println!(
        "tlsf: allocs={} frees={} fallback={} frag={:.6} util={:.2}%",
        tlsf.total_allocations,
        tlsf.total_frees,
        tlsf.fallback_count,
        tlsf.fragmentation_ratio,
        tlsf.utilization_percent
    );

    Ok(())
}
