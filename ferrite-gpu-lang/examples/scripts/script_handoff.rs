use ferrite_gpu_lang::{cpu, gpu, CpuTensor, ToCpu, ToGpu};

fn main() -> ferrite_gpu_lang::Result<()> {
    let cpu_data =
        cpu(|c| CpuTensor::with_allocator(c, vec![1, 1, 1, 1024], |i| (i as f32) * 0.01 - 2.5))?;

    let back_on_cpu = gpu(0, |g| {
        let gbuf = cpu_data.to_gpu(g)?;
        let gbuf = gbuf.relu(g)?;
        gbuf.to_cpu()
    })?;

    let (sum, positives, stats) = cpu(|c| {
        let sum: f32 = back_on_cpu.data().iter().sum();
        let positives = back_on_cpu.data().iter().filter(|x| **x > 0.0).count();
        let stats = c.allocator_stats();
        Ok::<(f32, usize, ferrite_gpu_lang::CpuTlsfStats), ferrite_gpu_lang::LangError>((
            sum, positives, stats,
        ))
    })?;

    println!("script=handoff");
    println!("shape={:?}", back_on_cpu.shape());
    println!("sum={:.4} positives={}", sum, positives);
    println!(
        "cpu_tlsf: arena={} allocs={} frees={} failed={} current={} peak={}",
        stats.arena_bytes,
        stats.alloc_calls,
        stats.free_calls,
        stats.failed_alloc_calls,
        stats.current_allocated_bytes,
        stats.peak_allocated_bytes
    );
    Ok(())
}
