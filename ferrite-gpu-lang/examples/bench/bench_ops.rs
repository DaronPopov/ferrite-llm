/// Benchmark all tensor operations.
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let config = ptx_runtime::PTXStableConfig {
        struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
        abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.50,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: 16,
        quiet_init: 1,
        enable_leak_detection: 0,
        enable_pool_health: 0,
        _reserved0: 0,
    };
    let runtime = Arc::new(
        ptx_runtime::PtxRuntime::with_stable_config(0, Some(config))
            .expect("runtime init failed"),
    );
    runtime.export_for_hook();
    runtime.export_context();

    let warmup = 10;
    let iters = 200;
    let n = 10_000_000usize; // 10M elements for elementwise ops

    // Build a 10M f32 input with values in [0.1, 5.0] to avoid domain issues
    let data_vec: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32 % 4.9)).collect();
    let x = ptx_tensor::Tensor::from_slice(&data_vec, &[n], ptx_tensor::DType::F32, &runtime).unwrap();

    // Second input for binary ops
    let data2_vec: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 % 3.0)).collect();
    let y = ptx_tensor::Tensor::from_slice(&data2_vec, &[n], ptx_tensor::DType::F32, &runtime).unwrap();

    println!("=== Elementwise ops (10M f32) — {} warmup, {} iters ===", warmup, iters);
    println!("{:<20} {:>10} {:>12}", "op", "us/iter", "GB/s");
    println!("{}", "-".repeat(44));

    macro_rules! bench_unary {
        ($name:expr, $op:expr) => {{
            for _ in 0..warmup { let _ = $op; }
            runtime.sync_all();
            let t0 = Instant::now();
            for _ in 0..iters { let _ = $op; }
            runtime.sync_all();
            let us = t0.elapsed().as_micros() as f64 / iters as f64;
            // read 4B + write 4B = 8B per element
            let gbps = (n as f64 * 8.0) / (us * 1e-6) / 1e9;
            println!("{:<20} {:>8.1} us {:>10.1} GB/s", $name, us, gbps);
        }};
    }

    macro_rules! bench_binary {
        ($name:expr, $op:expr) => {{
            for _ in 0..warmup { let _ = $op; }
            runtime.sync_all();
            let t0 = Instant::now();
            for _ in 0..iters { let _ = $op; }
            runtime.sync_all();
            let us = t0.elapsed().as_micros() as f64 / iters as f64;
            // read 4B + 4B + write 4B = 12B per element
            let gbps = (n as f64 * 12.0) / (us * 1e-6) / 1e9;
            println!("{:<20} {:>8.1} us {:>10.1} GB/s", $name, us, gbps);
        }};
    }

    // --- Existing unary ops ---
    bench_unary!("neg", x.neg().unwrap());
    bench_unary!("abs", x.abs().unwrap());
    bench_unary!("exp", x.exp().unwrap());
    bench_unary!("log", x.log().unwrap());
    bench_unary!("sqrt", x.sqrt().unwrap());
    bench_unary!("rsqrt", x.rsqrt().unwrap());
    bench_unary!("sin", x.sin().unwrap());
    bench_unary!("cos", x.cos().unwrap());
    bench_unary!("tanh", x.tanh().unwrap());
    bench_unary!("ceil", x.ceil().unwrap());
    bench_unary!("floor", x.floor().unwrap());
    bench_unary!("round", x.round().unwrap());
    bench_unary!("sqr", x.sqr().unwrap());
    bench_unary!("recip", x.recip().unwrap());

    // --- New unary ops ---
    bench_unary!("log2", x.log2().unwrap());
    bench_unary!("log10", x.log10().unwrap());
    bench_unary!("tan", x.tan().unwrap());
    bench_unary!("sinh", x.sinh().unwrap());
    bench_unary!("cosh", x.cosh().unwrap());
    bench_unary!("sign", x.sign().unwrap());
    bench_unary!("erf", x.erf().unwrap());

    println!();

    // --- Existing binary ops ---
    bench_binary!("add", x.add(&y).unwrap());
    bench_binary!("sub", x.sub(&y).unwrap());
    bench_binary!("mul", x.mul(&y).unwrap());
    bench_binary!("div", x.div(&y).unwrap());
    bench_binary!("maximum", x.maximum(&y).unwrap());
    bench_binary!("minimum", x.minimum(&y).unwrap());

    // --- New binary op ---
    bench_binary!("fmod", x.fmod(&y).unwrap());

    println!();

    // --- Existing activations ---
    bench_unary!("relu", x.relu().unwrap());
    bench_unary!("relu6", x.relu6().unwrap());
    bench_unary!("selu", x.selu().unwrap());
    bench_unary!("gelu", x.gelu().unwrap());
    bench_unary!("sigmoid", x.sigmoid().unwrap());
    bench_unary!("silu", x.silu().unwrap());
    bench_unary!("softplus", x.softplus().unwrap());
    bench_unary!("mish", x.mish().unwrap());

    // --- New activations ---
    bench_unary!("gelu_tanh", x.gelu_tanh().unwrap());
    bench_unary!("hardswish", x.hardswish().unwrap());
    bench_unary!("hardsigmoid", x.hardsigmoid().unwrap());

    println!();

    // === Reduction ops ===
    println!("=== Reduction ops (2K x 5K f32) ===");
    println!("{:<20} {:>10} {:>12}", "op", "us/iter", "GB/s");
    println!("{}", "-".repeat(44));
    {
        let rows = 2000usize;
        let cols = 5000usize;
        let red_vec: Vec<f32> = (0..rows * cols).map(|i| 0.5 + (i as f32 % 2.0)).collect();
        let r = ptx_tensor::Tensor::from_slice(&red_vec, &[rows, cols], ptx_tensor::DType::F32, &runtime).unwrap();

        macro_rules! bench_reduce {
            ($name:expr, $op:expr) => {{
                for _ in 0..warmup { let _ = $op; }
                runtime.sync_all();
                let t0 = Instant::now();
                for _ in 0..iters { let _ = $op; }
                runtime.sync_all();
                let us = t0.elapsed().as_micros() as f64 / iters as f64;
                let gbps = (rows * cols * 4) as f64 / (us * 1e-6) / 1e9;
                println!("{:<20} {:>8.1} us {:>10.1} GB/s", $name, us, gbps);
            }};
        }

        bench_reduce!("sum(dim=-1)", r.sum(-1).unwrap());
        bench_reduce!("mean(dim=-1)", r.mean(-1).unwrap());
        bench_reduce!("max(dim=-1)", r.max(-1).unwrap());
        bench_reduce!("min(dim=-1)", r.min(-1).unwrap());
        bench_reduce!("prod(dim=-1)", r.prod(-1).unwrap());
        bench_reduce!("argmax(dim=-1)", r.argmax(-1).unwrap());
        bench_reduce!("argmin(dim=-1)", r.argmin(-1).unwrap());
    }

    println!();

    // === Structured ops ===
    println!("=== Structured ops ===");
    println!("{:<20} {:>10} {:>12}", "op", "us/iter", "metric");
    println!("{}", "-".repeat(44));

    // index_select: 1M → 100K
    {
        let sn = 1_000_000usize;
        let k = 100_000usize;
        let dv: Vec<f32> = (0..sn).map(|i| i as f32).collect();
        let iv: Vec<i32> = (0..k).map(|i| (i * 7 % sn) as i32).collect();
        let d = ptx_tensor::Tensor::from_slice(&dv, &[sn], ptx_tensor::DType::F32, &runtime).unwrap();
        let ids = ptx_tensor::Tensor::from_slice(&iv, &[k], ptx_tensor::DType::I32, &runtime).unwrap();
        for _ in 0..warmup { let _ = d.index_select(0, &ids).unwrap(); }
        runtime.sync_all();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = d.index_select(0, &ids).unwrap(); }
        runtime.sync_all();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let gbps = (k * 4) as f64 / (us * 1e-6) / 1e9;
        println!("{:<20} {:>8.1} us {:>7.1} GB/s out", "index_select 1M→100K", us, gbps);
    }

    // scatter_add: 1M → 100K
    {
        let sn = 1_000_000usize;
        let dst = 100_000usize;
        let sv: Vec<f32> = (0..sn).map(|i| (i % 100) as f32).collect();
        let iv: Vec<i32> = (0..sn).map(|i| (i % dst) as i32).collect();
        let s = ptx_tensor::Tensor::from_slice(&sv, &[sn], ptx_tensor::DType::F32, &runtime).unwrap();
        let ids = ptx_tensor::Tensor::from_slice(&iv, &[sn], ptx_tensor::DType::I32, &runtime).unwrap();
        for _ in 0..warmup { let _ = s.scatter_add(0, &ids, dst).unwrap(); }
        runtime.sync_all();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = s.scatter_add(0, &ids, dst).unwrap(); }
        runtime.sync_all();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let gbps = (sn * 4) as f64 / (us * 1e-6) / 1e9;
        println!("{:<20} {:>8.1} us {:>7.1} GB/s in", "scatter_add 1M→100K", us, gbps);
    }

    // argsort: 1K x 1K
    {
        let rows = 1000usize;
        let cols = 1024usize;
        let dv: Vec<f32> = (0..rows * cols).map(|i| ((i * 17 + 31) % 10007) as f32).collect();
        let d = ptx_tensor::Tensor::from_slice(&dv, &[rows, cols], ptx_tensor::DType::F32, &runtime).unwrap();
        for _ in 0..warmup { let _ = d.argsort(-1, true).unwrap(); }
        runtime.sync_all();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = d.argsort(-1, true).unwrap(); }
        runtime.sync_all();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let rps = rows as f64 / (us * 1e-6);
        println!("{:<20} {:>8.1} us {:>7.0} rows/s", "argsort 1Kx1K", us, rps);
    }

    // where: 10M
    {
        let wn = 10_000_000usize;
        let tv: Vec<f32> = (0..wn).map(|i| i as f32).collect();
        let fv: Vec<f32> = (0..wn).map(|i| -(i as f32)).collect();
        let cv: Vec<u8> = (0..wn).map(|i| (i % 2) as u8).collect();
        let t = ptx_tensor::Tensor::from_slice(&tv, &[wn], ptx_tensor::DType::F32, &runtime).unwrap();
        let f = ptx_tensor::Tensor::from_slice(&fv, &[wn], ptx_tensor::DType::F32, &runtime).unwrap();
        let c = ptx_tensor::Tensor::from_slice(&cv, &[wn], ptx_tensor::DType::U8, &runtime).unwrap();
        for _ in 0..warmup { let _ = t.where_cond(&c, &f).unwrap(); }
        runtime.sync_all();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = t.where_cond(&c, &f).unwrap(); }
        runtime.sync_all();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let gbps = (wn as f64 * 13.0) / (us * 1e-6) / 1e9;
        println!("{:<20} {:>8.1} us {:>7.1} GB/s eff", "where 10M", us, gbps);
    }

    println!("\nDone.");
}
