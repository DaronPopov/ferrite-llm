/// JIT basics — compile a script string and execute it on the GPU.
///
/// Demonstrates the minimal JIT workflow:
///   1. Create a JitEngine
///   2. Write a ferrite script as a string literal
///   3. Compile it (parse → AST → Program → CompiledProgram)
///   4. Execute the CompiledProgram on a GpuLangRuntime
///
/// The first compile pays ~μs for parsing + shape-checking.
/// Subsequent calls with the same script text hit the cache and
/// return the CompiledProgram instantly.

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

fn main() {
    // ── 1. Spin up the JIT engine and GPU runtime ───────────────
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── 2. Write a ferrite script ───────────────────────────────
    //
    // The script is a flat sequence of op calls that maps 1:1 to
    // the Program IR.  No boilerplate, no imports — just ops.
    let script = r#"
        # Declare inputs with their shapes
        x = input([1, 1, 1, 8])

        # Chain unary activations
        h = relu(x)
        h = tanh(h)
        h = sigmoid(h)

        return h
    "#;

    // ── 3. Compile (first call parses; second call is a cache hit)
    let compiled = jit.compile(script).expect("compile");
    println!("input shapes:  {:?}", compiled.input_shapes());
    println!("output shape:  {:?}", compiled.output_shape());

    // ── 4. Execute on GPU ───────────────────────────────────────
    let data: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let input = HostTensor::new(vec![1, 1, 1, 8], data).unwrap();
    let output = runtime.execute(compiled, &[input]).unwrap();

    println!("input:  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]");
    println!("output: {:?}", output.data());

    // ── 5. Re-compile hits cache ────────────────────────────────
    let _ = jit.compile(script).unwrap();
    println!("\ncache size after two compiles of the same script: {}", jit.cache_len());
    assert_eq!(jit.cache_len(), 1, "should be 1 — cache hit");

    println!("\nDone.");
}
