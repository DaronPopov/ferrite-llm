/// Benchmark: infix expression syntax vs function-call syntax vs hand-built IR.
///
/// Measures three things:
///   1. Compile latency — does the precedence-climbing parser add overhead?
///   2. Execution time  — does FillLike (scalar broadcast) hurt at scale?
///   3. IR equivalence  — does JIT infix produce identical perf to hand-built Program?

use std::time::Instant;

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor, Program};

fn main() {
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");
    let mut jit = JitEngine::new();

    let warmup = 20;
    let iters = 500;

    // ─────────────────────────────────────────────────────────────
    // Section 1: Compile latency (no cache)
    // ─────────────────────────────────────────────────────────────
    println!("=== Compile latency (cold, no cache) — {} iters ===", iters);
    println!("{:<35} {:>10}", "script style", "us/compile");
    println!("{}", "-".repeat(47));

    // Old style: function-call only
    let script_fn_call = r#"
        x = input([1024])
        a = mul(x, x)
        b = add(a, x)
        c = relu(b)
        return c
    "#;

    // New style: infix equivalent
    let script_infix = r#"
        x = input([1024])
        c = relu(x * x + x)
        return c
    "#;

    // Infix with scalar broadcast (FillLike path)
    let script_infix_scalar = r#"
        x = input([1024])
        c = relu(x * 2.0 + 1.0)
        return c
    "#;

    // Larger expression
    let script_infix_complex = r#"
        x = input([1024])
        y = input([1024])
        z = (x * 2.0 + 1.0) * relu(y) - sigmoid(x) / 2.0
        return z
    "#;

    // Tile block
    let script_tile = r#"
        x = input([1024])
        y = input([1024])
        tile z over (x, y):
            t = x * y + x
            z = tanh(t) * 0.5
        end
        return z
    "#;

    for (label, script) in [
        ("fn-call (mul/add/relu)", script_fn_call),
        ("infix (x*x + x)", script_infix),
        ("infix+scalar (x*2.0+1.0)", script_infix_scalar),
        ("infix complex (5 ops)", script_infix_complex),
        ("tile block", script_tile),
    ] {
        // Measure cold compiles (fresh engine each time)
        let mut total = std::time::Duration::ZERO;
        for _ in 0..iters {
            let mut e = JitEngine::new();
            let t0 = Instant::now();
            let _ = e.compile(script).unwrap();
            total += t0.elapsed();
        }
        let us = total.as_micros() as f64 / iters as f64;
        println!("{:<35} {:>8.1} us", label, us);
    }

    // Also measure cached compile (should be ~0)
    {
        let _ = jit.compile(script_infix_scalar).unwrap();
        let t0 = Instant::now();
        for _ in 0..10_000 {
            let _ = jit.compile(script_infix_scalar).unwrap();
        }
        let us = t0.elapsed().as_nanos() as f64 / 10_000.0;
        println!("{:<35} {:>7.0} ns", "cached (hash hit)", us);
    }

    // ─────────────────────────────────────────────────────────────
    // Section 2: Execution time at various tensor sizes
    // ─────────────────────────────────────────────────────────────
    println!();
    println!("=== Execution: x * 2.0 + 1.0 — {} warmup, {} iters ===", warmup, iters);
    println!("{:<15} {:>10} {:>12} {:>10}", "N", "us/iter", "GB/s", "Melems/s");
    println!("{}", "-".repeat(50));

    for &n in &[1024usize, 16_384, 131_072, 1_048_576, 10_000_000] {
        let script = format!(
            "x = input([{}])\ny = x * 2.0 + 1.0\nreturn y",
            n
        );
        let compiled = jit.compile(&script).unwrap();
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let input = HostTensor::new(vec![n], data).unwrap();

        // warmup
        for _ in 0..warmup {
            let _ = runtime.execute(compiled, &[input.clone()]).unwrap();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runtime.execute(compiled, &[input.clone()]).unwrap();
        }
        let elapsed = t0.elapsed();
        let us = elapsed.as_micros() as f64 / iters as f64;
        // Bandwidth: read x (4B) + write fill1 (4B) + read x+fill1, write mul (12B)
        //          + write fill2 (4B) + read mul+fill2, write add (12B) = 36B/elem
        // But simpler: user sees N elems in, N elems out = 8B effective
        let gbps_effective = (n as f64 * 8.0) / (us * 1e-6) / 1e9;
        let melems = n as f64 / (us);
        println!("{:<15} {:>8.1} us {:>9.1} GB/s {:>8.1} M", format!("{}",n), us, gbps_effective, melems);
    }

    // ─────────────────────────────────────────────────────────────
    // Section 3: JIT infix vs hand-built Program IR
    // ─────────────────────────────────────────────────────────────
    println!();
    println!("=== JIT infix vs hand-built IR (N=1M) — {} warmup, {} iters ===", warmup, iters);
    println!("{:<30} {:>10}", "method", "us/iter");
    println!("{}", "-".repeat(42));

    let n = 1_048_576usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let input = HostTensor::new(vec![n], data).unwrap();

    // Hand-built Program: x * 2.0 + 1.0
    let hand_compiled = {
        let mut p = Program::new();
        let x = p.input(&[n]).unwrap();
        let two = p.fill_like(2.0, x);
        let mul = p.mul(x, two);
        let one = p.fill_like(1.0, mul);
        let add = p.add(mul, one);
        p.set_output(add);
        p.compile().unwrap()
    };

    // JIT-compiled version
    let jit_script = format!("x = input([{}])\ny = x * 2.0 + 1.0\nreturn y", n);
    let jit_compiled = jit.compile(&jit_script).unwrap();

    // Warmup both
    for _ in 0..warmup {
        let _ = runtime.execute(&hand_compiled, &[input.clone()]).unwrap();
        let _ = runtime.execute(jit_compiled, &[input.clone()]).unwrap();
    }

    // Bench hand-built
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = runtime.execute(&hand_compiled, &[input.clone()]).unwrap();
    }
    let hand_us = t0.elapsed().as_micros() as f64 / iters as f64;
    println!("{:<30} {:>8.1} us", "hand-built Program IR", hand_us);

    // Bench JIT
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = runtime.execute(jit_compiled, &[input.clone()]).unwrap();
    }
    let jit_us = t0.elapsed().as_micros() as f64 / iters as f64;
    println!("{:<30} {:>8.1} us", "JIT infix (x * 2.0 + 1.0)", jit_us);

    let diff_pct = ((jit_us - hand_us) / hand_us) * 100.0;
    println!("{:<30} {:>+7.1}%", "difference", diff_pct);

    // ─────────────────────────────────────────────────────────────
    // Section 4: FillLike overhead — scalar broadcast vs tensor-tensor
    // ─────────────────────────────────────────────────────────────
    println!();
    println!("=== FillLike overhead: x*2.0 vs x*y (N=1M) — {} warmup, {} iters ===", warmup, iters);
    println!("{:<30} {:>10}", "expression", "us/iter");
    println!("{}", "-".repeat(42));

    let data2: Vec<f32> = vec![2.0f32; n];
    let input2 = HostTensor::new(vec![n], data2).unwrap();

    // x * 2.0 (uses FillLike to broadcast scalar)
    let scalar_us = {
        let script_scalar = format!("x = input([{}])\ny = x * 2.0\nreturn y", n);
        let mut e = JitEngine::new();
        let c = e.compile(&script_scalar).unwrap();

        for _ in 0..warmup {
            let _ = runtime.execute(c, &[input.clone()]).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runtime.execute(c, &[input.clone()]).unwrap();
        }
        t0.elapsed().as_micros() as f64 / iters as f64
    };
    println!("{:<30} {:>8.1} us", "x * 2.0 (FillLike)", scalar_us);

    // x * y where y is all 2.0 (no FillLike, but extra H→D transfer)
    let tensor_us = {
        let script_tensor = format!("x = input([{}])\ny = input([{}])\nz = x * y\nreturn z", n, n);
        let mut e = JitEngine::new();
        let c = e.compile(&script_tensor).unwrap();

        for _ in 0..warmup {
            let _ = runtime.execute(c, &[input.clone(), input2.clone()]).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runtime.execute(c, &[input.clone(), input2.clone()]).unwrap();
        }
        t0.elapsed().as_micros() as f64 / iters as f64
    };
    println!("{:<30} {:>8.1} us", "x * y (tensor-tensor)", tensor_us);

    let diff_pct = ((scalar_us - tensor_us) / tensor_us) * 100.0;
    println!("{:<30} {:>+7.1}%", "FillLike overhead", diff_pct);

    // ─────────────────────────────────────────────────────────────
    // Section 5: Tile block vs flat infix (should be identical)
    // ─────────────────────────────────────────────────────────────
    println!();
    println!("=== Tile block vs flat infix (N=1M) — {} warmup, {} iters ===", warmup, iters);
    println!("{:<30} {:>10}", "style", "us/iter");
    println!("{}", "-".repeat(42));

    let flat_us = {
        let script_flat = format!(
            "x = input([{}])\ny = input([{}])\nt = x * y + x\nz = tanh(t) * 0.5\nreturn z",
            n, n
        );
        let mut e = JitEngine::new();
        let c = e.compile(&script_flat).unwrap();

        for _ in 0..warmup {
            let _ = runtime.execute(c, &[input.clone(), input.clone()]).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runtime.execute(c, &[input.clone(), input.clone()]).unwrap();
        }
        t0.elapsed().as_micros() as f64 / iters as f64
    };
    println!("{:<30} {:>8.1} us", "flat infix", flat_us);

    let tile_us = {
        let script_tile = format!(
            "x = input([{n}])\ny = input([{n}])\ntile z over (x, y):\n  t = x * y + x\n  z = tanh(t) * 0.5\nend\nreturn z"
        );
        let mut e = JitEngine::new();
        let c = e.compile(&script_tile).unwrap();

        for _ in 0..warmup {
            let _ = runtime.execute(c, &[input.clone(), input.clone()]).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runtime.execute(c, &[input.clone(), input.clone()]).unwrap();
        }
        t0.elapsed().as_micros() as f64 / iters as f64
    };
    println!("{:<30} {:>8.1} us", "tile block", tile_us);

    let diff_pct = ((tile_us - flat_us) / flat_us) * 100.0;
    println!("{:<30} {:>+7.1}%", "tile overhead", diff_pct);

    println!("\nDone.");
}
