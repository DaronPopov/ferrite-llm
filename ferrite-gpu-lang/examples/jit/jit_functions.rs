/// JIT function inlining — define reusable graph templates.
///
/// Functions in ferrite scripts are **not** runtime calls.  They are
/// graph templates that get inlined at JIT compile time:
///
///   fn activate(x):
///       h = relu(x)
///       return sigmoid(h)
///   end
///
/// When you write `y = activate(tensor)`, the JIT splices the body
/// into the flat graph with parameters bound to the caller's values.
/// The compiled output is byte-identical to writing the ops inline —
/// zero overhead.

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

fn main() {
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── Example 1: single function ──────────────────────────────

    let script_single = r#"
        fn activate(x):
            h = relu(x)
            y = sigmoid(h)
            return y
        end

        x = input([1, 1, 1, 4])
        out = activate(x)
        return out
    "#;

    let compiled = jit.compile(script_single).unwrap();
    let input = HostTensor::new(vec![1, 1, 1, 4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = runtime.execute(compiled, &[input]).unwrap();

    // relu([-1,0,1,2]) = [0,0,1,2]  →  sigmoid([0,0,1,2])
    assert!(approx(out.data()[0], 0.5));
    assert!(approx(out.data()[1], 0.5));
    assert!(approx(out.data()[2], 0.7311));
    assert!(approx(out.data()[3], 0.8808));
    println!("PASS: single function inlining");
    println!("      activate([-1,0,1,2]) = {:?}", out.data());

    // ── Example 2: multi-function pipeline ──────────────────────
    //
    // Define an encoder and a decoder as separate graph templates,
    // then chain them.  Each function inlines independently.

    let script_pipeline = r#"
        fn encoder(x):
            h = relu(x)
            return h
        end

        fn decoder(x):
            y = tanh(x)
            return y
        end

        a = input([1, 1, 1, 4])
        b = encoder(a)
        c = decoder(b)
        return c
    "#;

    let compiled = jit.compile(script_pipeline).unwrap();
    let input = HostTensor::new(vec![1, 1, 1, 4], vec![-2.0, -1.0, 1.0, 3.0]).unwrap();
    let out = runtime.execute(compiled, &[input]).unwrap();

    // relu([-2,-1,1,3]) = [0,0,1,3]  →  tanh([0,0,1,3])
    assert!(approx(out.data()[0], 0.0));
    assert!(approx(out.data()[1], 0.0));
    assert!(approx(out.data()[2], 0.7616));
    assert!(approx(out.data()[3], 0.9951));
    println!("PASS: encoder → decoder pipeline");
    println!("      encoder→decoder([-2,-1,1,3]) = {:?}", out.data());

    // ── Example 3: function with binary ops ─────────────────────

    let script_residual = r#"
        fn residual_block(x, skip):
            h = relu(x)
            out = add(h, skip)
            return out
        end

        x = input([1, 1, 1, 4])
        s = input([1, 1, 1, 4])
        out = residual_block(x, s)
        return out
    "#;

    let compiled = jit.compile(script_residual).unwrap();
    let in_x = HostTensor::new(vec![1, 1, 1, 4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let in_s = HostTensor::new(vec![1, 1, 1, 4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let out = runtime.execute(compiled, &[in_x, in_s]).unwrap();

    // relu([-1,2,-3,4]) = [0,2,0,4]  +  [1,1,1,1] = [1,3,1,5]
    assert!(approx(out.data()[0], 1.0));
    assert!(approx(out.data()[1], 3.0));
    assert!(approx(out.data()[2], 1.0));
    assert!(approx(out.data()[3], 5.0));
    println!("PASS: residual block (relu + skip connection)");
    println!("      residual([-1,2,-3,4], [1,1,1,1]) = {:?}", out.data());

    println!("\nAll function inlining examples passed.");
}
