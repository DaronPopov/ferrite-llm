/// JIT pipeline — caching, hot-reload, and multi-script composition.
///
/// Demonstrates production JIT patterns:
///   1. Compile once, execute many times (cache hit)
///   2. Different scripts produce independent cached programs
///   3. Swap scripts at runtime without restart (hot-reload)
///   4. Measure JIT overhead vs execution

use std::time::Instant;

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

fn main() {
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── 1. Compile + warm-up ────────────────────────────────────

    let script_a = r#"
        x = input([1, 1, 1, 1024])
        h = relu(x)
        y = sigmoid(h)
        return y
    "#;

    let script_b = r#"
        x = input([1, 1, 1, 1024])
        h = tanh(x)
        y = relu(h)
        return y
    "#;

    // Clone compiled programs so we can release the &mut borrow on
    // the JitEngine between compiles.  In production you'd typically
    // compile once and hold the owned CompiledProgram.

    let t0 = Instant::now();
    let compiled_a = jit.compile(script_a).unwrap().clone();
    let first_compile_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let _compiled_a_again = jit.compile(script_a).unwrap().clone();
    let cache_hit_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let compiled_b = jit.compile(script_b).unwrap().clone();
    let second_compile_us = t2.elapsed().as_micros();

    println!("JIT compile timings:");
    println!("  first compile (script_a):  {} μs", first_compile_us);
    println!("  cache hit (script_a):      {} μs", cache_hit_us);
    println!("  new script (script_b):     {} μs", second_compile_us);
    println!("  cache entries:             {}", jit.cache_len());
    println!();

    // ── 2. Repeated execution ───────────────────────────────────

    let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 256.0).collect();
    let iters = 100;

    let t3 = Instant::now();
    for _ in 0..iters {
        let input = HostTensor::new(vec![1, 1, 1, 1024], data.clone()).unwrap();
        let _ = runtime.execute(&compiled_a, &[input]).unwrap();
    }
    let exec_a_ms = t3.elapsed().as_secs_f64() * 1e3;

    let t4 = Instant::now();
    for _ in 0..iters {
        let input = HostTensor::new(vec![1, 1, 1, 1024], data.clone()).unwrap();
        let _ = runtime.execute(&compiled_b, &[input]).unwrap();
    }
    let exec_b_ms = t4.elapsed().as_secs_f64() * 1e3;

    println!("Execution timings ({} iterations each):", iters);
    println!(
        "  script_a (relu→sigmoid):  {:.2} ms total, {:.1} μs/iter",
        exec_a_ms,
        exec_a_ms * 1e3 / iters as f64,
    );
    println!(
        "  script_b (tanh→relu):     {:.2} ms total, {:.1} μs/iter",
        exec_b_ms,
        exec_b_ms * 1e3 / iters as f64,
    );
    println!();

    // ── 3. Hot-reload simulation ────────────────────────────────
    //
    // In a real server, you'd watch a file and re-compile when it
    // changes.  The old CompiledProgram is dropped, the new one
    // takes its place.  TLSF frees the old buffers in O(1).

    let scripts = [
        ("v1: relu",    "x = input([1, 1, 1, 4])\nh = relu(x)\nreturn h"),
        ("v2: tanh",    "x = input([1, 1, 1, 4])\nh = tanh(x)\nreturn h"),
        ("v3: sigmoid", "x = input([1, 1, 1, 4])\nh = sigmoid(x)\nreturn h"),
    ];

    let input_data = vec![-1.0f32, 0.0, 1.0, 2.0];

    println!("Hot-reload simulation:");
    for (label, script) in &scripts {
        let compiled = jit.compile(script).unwrap().clone();
        let input = HostTensor::new(vec![1, 1, 1, 4], input_data.clone()).unwrap();
        let out = runtime.execute(&compiled, &[input]).unwrap();
        println!("  {} → {:?}", label, out.data());
    }
    println!();

    // ── 4. Verify final cache state ─────────────────────────────

    // script_a + script_b + 3 hot-reload scripts = 5 entries
    println!("Final cache entries: {}", jit.cache_len());

    println!("Done.");
}
