use ferrite_gpu_lang::{fer, CpuTensor, HostTensor, Program};

fn main() -> ferrite_gpu_lang::Result<()> {
    println!("=== FerCtx: Unified Hardware OS Runtime ===\n");

    fer(0, |ctx| {
        // ── Stage 1: CPU TLSF allocation ────────────────────────────
        let cpu_data = CpuTensor::with_allocator(ctx, vec![1, 1, 1, 8192], |i| {
            (i as f32) * 0.001 - 4.0
        })?;
        println!("[1] cpu tensor allocated via TLSF: shape={:?} len={}", cpu_data.shape(), cpu_data.len());

        // ── Stage 2: CPU → GPU transfer ─────────────────────────────
        let gpu_data = ctx.cpu_to_gpu(&cpu_data)?;
        println!("[2] cpu_to_gpu transfer: shape={:?} bytes={}", gpu_data.shape(), gpu_data.bytes_len());

        // ── Stage 3: GPU kernel (ReLU via Candle PTX) ───────────────
        let gpu_relu = gpu_data.relu(ctx)?;
        println!("[3] gpu relu kernel executed");

        // ── Stage 4: GPU → CPU transfer ─────────────────────────────
        let cpu_relu = ctx.gpu_to_cpu(&gpu_relu)?;
        let negatives = cpu_relu.data().iter().filter(|x| **x == 0.0).count();
        let positives = cpu_relu.data().iter().filter(|x| **x > 0.0).count();
        println!("[4] gpu_to_cpu transfer: zeroed={negatives} positive={positives}");

        // ── Stage 5: Graph execution through runtime ────────────────
        let mut p = Program::new();
        let x = p.input(gpu_relu.shape())?;
        let y = p.sigmoid(x);
        p.set_output(y);
        let compiled = p.compile()?;

        let host_relu = ctx.gpu_to_cpu(&gpu_relu)?;
        let (shape, data) = host_relu.into_inner();
        let host = HostTensor::new(shape, data)?;
        let output = ctx.runtime().execute(&compiled, &[host])?;
        let sum: f32 = output.data().iter().sum();
        println!("[5] graph sigmoid executed: output_shape={:?} sum={sum:.4}", output.shape());

        // ── Stage 6: Unified stats — both TLSF pools in one call ────
        let stats = ctx.stats();

        println!("\n--- CPU TLSF ---");
        println!(
            "  arena={} allocs={} frees={} failed={} current={} peak={}",
            stats.cpu.arena_bytes,
            stats.cpu.alloc_calls,
            stats.cpu.free_calls,
            stats.cpu.failed_alloc_calls,
            stats.cpu.current_allocated_bytes,
            stats.cpu.peak_allocated_bytes,
        );

        println!("\n--- GPU TLSF ---");
        println!(
            "  pool={} alloc={} free={} peak={}",
            stats.gpu.total_pool_size,
            stats.gpu.allocated_bytes,
            stats.gpu.free_bytes,
            stats.gpu.peak_allocated,
        );
        println!(
            "  allocs={} frees={} splits={} merges={}",
            stats.gpu.total_allocations,
            stats.gpu.total_frees,
            stats.gpu.total_splits,
            stats.gpu.total_merges,
        );
        println!(
            "  frag={:.6} util={:.2}% healthy={} needs_defrag={}",
            stats.gpu.fragmentation_ratio,
            stats.gpu.utilization_percent,
            stats.gpu.is_healthy,
            stats.gpu.needs_defrag,
        );

        // ── Stage 7: GPU health + hot stats ─────────────────────────
        let hot = ctx.gpu_hot_stats();
        println!("\n--- GPU Hot ---");
        println!(
            "  vram_alloc={} vram_used={} vram_free={}",
            hot.vram_allocated, hot.vram_used, hot.vram_free,
        );
        println!(
            "  streams={} kernels={} ops={} avg_latency={:.2}us",
            hot.active_streams, hot.registered_kernels, hot.total_ops, hot.avg_latency_us,
        );

        let health = ctx.gpu_health();
        println!("\n--- GPU Pool Health ---");
        println!(
            "  valid={} leaks={} corrupted={} broken_chains={} hash_errors={}",
            health.is_valid,
            health.has_memory_leaks,
            health.has_corrupted_blocks,
            health.has_broken_chains,
            health.has_hash_errors,
        );

        println!("\n=== All stages passed — unified context works. ===");
        Ok(())
    })
}
