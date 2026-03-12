/// Heavy math scaling test for the ferrite JIT.
///
/// Pushes the JIT across four axes:
///   1. Tensor size scaling   (1K → 4M elements)
///   2. Graph depth scaling   (10 → 200 ops)
///   3. Wide parallel graphs  (8 parallel paths merging)
///   4. Simulated MLP layers  (element-wise multi-layer network)
///
/// Prints compile-time and execution-time at each scale point so
/// you can see exactly where overhead lives.

use std::time::Instant;

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

// ── Script generators ────────────────────────────────────────────

/// Build a deep chain: input → [relu → tanh → sigmoid] × depth → return
fn gen_deep_chain(size: usize, depth: usize) -> String {
    let mut s = format!("x = input([1, 1, 1, {}])\n", size);
    let mut prev = "x".to_string();
    let ops = ["relu", "tanh", "sigmoid"];
    for i in 0..depth {
        let name = format!("h{}", i);
        let op = ops[i % ops.len()];
        s.push_str(&format!("{} = {}({})\n", name, op, prev));
        prev = name;
    }
    s.push_str(&format!("return {}\n", prev));
    s
}

/// Build a wide graph: N parallel input paths, each transformed,
/// then pairwise-merged with add/mul down to a single output.
fn gen_wide_graph(size: usize, width: usize) -> String {
    let mut s = String::new();
    let mut names: Vec<String> = Vec::new();
    let ops = ["relu", "tanh", "sigmoid", "relu"];

    // N inputs, each goes through a different activation
    for i in 0..width {
        let inp = format!("x{}", i);
        s.push_str(&format!("{} = input([1, 1, 1, {}])\n", inp, size));
        let act = format!("a{}", i);
        let op = ops[i % ops.len()];
        s.push_str(&format!("{} = {}({})\n", act, op, inp));
        names.push(act);
    }

    // Pairwise reduction: add adjacent, then mul adjacent, repeat
    let mut round = 0;
    while names.len() > 1 {
        let mut next = Vec::new();
        let merge_op = if round % 2 == 0 { "add" } else { "mul" };
        let mut i = 0;
        while i + 1 < names.len() {
            let out = format!("m{}_{}", round, i / 2);
            s.push_str(&format!(
                "{} = {}({}, {})\n",
                out, merge_op, names[i], names[i + 1]
            ));
            next.push(out);
            i += 2;
        }
        if i < names.len() {
            // Odd one out — carry forward
            next.push(names[i].clone());
        }
        names = next;
        round += 1;
    }

    s.push_str(&format!("return {}\n", names[0]));
    s
}

/// Simulate a multi-layer perceptron (element-wise):
///   for each layer: h = activation(add(mul(h, w), b))
fn gen_mlp(size: usize, layers: usize) -> String {
    let mut s = format!("x = input([1, 1, 1, {}])\n", size);

    // Weight and bias inputs for each layer
    for l in 0..layers {
        s.push_str(&format!("w{} = input([1, 1, 1, {}])\n", l, size));
        s.push_str(&format!("b{} = input([1, 1, 1, {}])\n", l, size));
    }

    let activations = ["relu", "sigmoid", "tanh"];
    let mut prev = "x".to_string();
    for l in 0..layers {
        let t1 = format!("t{}a", l);
        let t2 = format!("t{}b", l);
        let h = format!("h{}", l);
        let act = activations[l % activations.len()];
        s.push_str(&format!("{} = mul({}, w{})\n", t1, prev, l));
        s.push_str(&format!("{} = add({}, b{})\n", t2, t1, l));
        s.push_str(&format!("{} = {}({})\n", h, act, t2));
        prev = h;
    }

    s.push_str(&format!("return {}\n", prev));
    s
}

/// Generate a cumulative-sum heavy workload: multiple cumsum passes
/// interleaved with activations.
fn gen_cumsum_heavy(size: usize, passes: usize) -> String {
    let mut s = format!("x = input([1, 1, 1, {}])\n", size);
    let mut prev = "x".to_string();
    for i in 0..passes {
        let cs = format!("cs{}", i);
        let act = format!("a{}", i);
        s.push_str(&format!("{} = cumsum({}, dim=3)\n", cs, prev));
        // Sigmoid to keep values bounded (cumsum grows fast)
        s.push_str(&format!("{} = sigmoid({})\n", act, cs));
        prev = act;
    }
    s.push_str(&format!("return {}\n", prev));
    s
}

// ── Helpers ──────────────────────────────────────────────────────

fn rand_data(n: usize) -> Vec<f32> {
    // Simple LCG — deterministic, fast, no deps
    let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1]
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn bench_script(
    jit: &mut JitEngine,
    runtime: &GpuLangRuntime,
    label: &str,
    script: &str,
    n_inputs: usize,
    size: usize,
    iters: usize,
) {
    // Compile
    let t0 = Instant::now();
    let compiled = match jit.compile(script) {
        Ok(c) => c.clone(),
        Err(e) => {
            println!("  {} — COMPILE ERROR: {}", label, e);
            return;
        }
    };
    let compile_us = t0.elapsed().as_micros();

    // Build inputs
    let inputs: Vec<HostTensor> = (0..n_inputs)
        .map(|_| HostTensor::new(vec![1, 1, 1, size], rand_data(size)).unwrap())
        .collect();

    // Warmup
    let _ = runtime.execute(&compiled, &inputs);

    // Bench
    let t1 = Instant::now();
    for _ in 0..iters {
        let ins: Vec<HostTensor> = inputs
            .iter()
            .map(|t| HostTensor::new(t.shape().to_vec(), t.data().to_vec()).unwrap())
            .collect();
        let _ = runtime.execute(&compiled, &ins).unwrap();
    }
    let exec_ms = t1.elapsed().as_secs_f64() * 1e3;
    let per_iter_us = exec_ms * 1e3 / iters as f64;

    println!(
        "  {:40} compile {:>6} μs | exec {:>8.2} ms ({} iters, {:.1} μs/iter)",
        label, compile_us, exec_ms, iters, per_iter_us,
    );
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");
    let iters = 50;

    // ═══════════════════════════════════════════════════════════════
    println!("=== 1. TENSOR SIZE SCALING (20-op chain) ===\n");
    // Fixed depth=20, vary tensor size
    for &size in &[1_024, 16_384, 131_072, 524_288, 1_048_576, 4_194_304] {
        let label = format!("depth=20, size={}", size);
        let script = gen_deep_chain(size, 20);
        bench_script(&mut jit, &runtime, &label, &script, 1, size, iters);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 2. GRAPH DEPTH SCALING (size=65536) ===\n");
    // Fixed size=65536, vary depth
    let size = 65_536;
    for &depth in &[10, 25, 50, 100, 200] {
        let label = format!("depth={}, size={}", depth, size);
        let script = gen_deep_chain(size, depth);
        bench_script(&mut jit, &runtime, &label, &script, 1, size, iters);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 3. WIDE PARALLEL GRAPH (size=131072) ===\n");
    // Multiple parallel input paths merging
    let size = 131_072;
    for &width in &[2, 4, 8] {
        let label = format!("width={}, size={}", width, size);
        let script = gen_wide_graph(size, width);
        bench_script(&mut jit, &runtime, &label, &script, width, size, iters);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 4. SIMULATED MLP (size=131072) ===\n");
    // Element-wise MLP: each layer = mul + add + activation
    let size = 131_072;
    for &layers in &[4, 8, 16, 32] {
        let label = format!("layers={}, size={}", layers, size);
        let script = gen_mlp(size, layers);
        let n_inputs = 1 + layers * 2; // x + (w,b) per layer
        bench_script(&mut jit, &runtime, &label, &script, n_inputs, size, iters);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 5. CUMSUM STRESS (size=65536) ===\n");
    // Cumulative sum is a scan — tests non-trivial kernel scaling
    let size = 65_536;
    for &passes in &[4, 8, 16, 32] {
        let label = format!("passes={}, size={}", passes, size);
        let script = gen_cumsum_heavy(size, passes);
        bench_script(&mut jit, &runtime, &label, &script, 1, size, iters);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 6. EXTREME: 4M elements × 50-layer MLP ===\n");
    {
        let size = 4_194_304;
        let layers = 50;
        let script = gen_mlp(size, layers);
        let n_inputs = 1 + layers * 2;
        let label = format!("layers={}, size={}", layers, size);
        bench_script(&mut jit, &runtime, &label, &script, n_inputs, size, 10);
    }

    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 7. VERIFY CORRECTNESS (spot check) ===\n");
    {
        // Manual check: 3-layer MLP with known values
        let script = r#"
            x  = input([1, 1, 1, 4])
            w0 = input([1, 1, 1, 4])
            b0 = input([1, 1, 1, 4])
            w1 = input([1, 1, 1, 4])
            b1 = input([1, 1, 1, 4])

            # Layer 0: relu(x*w0 + b0)
            t0 = mul(x, w0)
            t1 = add(t0, b0)
            h0 = relu(t1)

            # Layer 1: sigmoid(h0*w1 + b1)
            t2 = mul(h0, w1)
            t3 = add(t2, b1)
            h1 = sigmoid(t3)

            return h1
        "#;

        let compiled = jit.compile(script).unwrap().clone();

        let x  = HostTensor::new(vec![1,1,1,4], vec![ 1.0,  2.0, -1.0,  0.5]).unwrap();
        let w0 = HostTensor::new(vec![1,1,1,4], vec![ 0.5,  0.5,  0.5,  0.5]).unwrap();
        let b0 = HostTensor::new(vec![1,1,1,4], vec![-0.5, -0.5, -0.5, -0.5]).unwrap();
        let w1 = HostTensor::new(vec![1,1,1,4], vec![ 1.0,  1.0,  1.0,  1.0]).unwrap();
        let b1 = HostTensor::new(vec![1,1,1,4], vec![ 0.0,  0.0,  0.0,  0.0]).unwrap();

        let out = runtime.execute(&compiled, &[x, w0, b0, w1, b1]).unwrap();
        let d = out.data();

        // x*w0 = [0.5, 1.0, -0.5, 0.25]
        // +b0  = [0.0, 0.5, -1.0, -0.25]
        // relu = [0.0, 0.5,  0.0,  0.0]
        // *w1  = [0.0, 0.5,  0.0,  0.0]
        // +b1  = [0.0, 0.5,  0.0,  0.0]
        // sig  = [0.5, 0.622, 0.5, 0.5]
        let expected = [
            0.5,
            1.0 / (1.0 + (-0.5f32).exp()),
            0.5,
            0.5,
        ];

        let ok = d.iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 1e-3);

        if ok {
            println!("  PASS: 2-layer MLP correctness check");
            println!("        output = {:?}", d);
            println!("        expect = {:?}", expected);
        } else {
            println!("  FAIL: 2-layer MLP correctness check");
            println!("        output = {:?}", d);
            println!("        expect = {:?}", expected);
            std::process::exit(1);
        }
    }

    println!(
        "\nFinal JIT cache: {} compiled programs",
        jit.cache_len()
    );
    println!("Done.");
}
