/// JIT tiling & infix expressions — per-element math auto-parallelized on CUDA.
///
/// Demonstrates the new infix expression syntax and tile blocks:
///   - Scalar multiplication: `x * 2.0`
///   - Infix + builtin: `x * 2.0 + relu(x)`
///   - Tile block: `tile y over x: y = x * 2.0 + 1.0 end`
///   - Nested tile block: tile-in-tile for staged transforms
///   - Annotated tile block: explicit tiling + precision/quant/distributed/collective hints
///   - Symbolic shape contracts: `input([B, T, H], B=..., T=..., H=...)`
///   - Sub/div: `(x - 1.0) / 2.0`
///   - Parenthesized grouping: `(x + 1.0) * 2.0`

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

fn main() {
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── Test 1: Scalar multiply ─────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = x * 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile scalar mul");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        println!("Test 1 — scalar multiply (x * 2.0):");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[2.0, 4.0, 6.0, 8.0]);
        println!("  PASS");
    }

    // ── Test 2: Infix + builtin ─────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = x * 2.0 + relu(x)
            return y
        "#;
        let compiled = jit.compile(script).expect("compile infix+builtin");
        let input = HostTensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // x*2.0: [-2.0, 0.0, 2.0, 4.0]
        // relu(x): [0.0, 0.0, 1.0, 2.0]
        // sum: [-2.0, 0.0, 3.0, 6.0]
        println!("\nTest 2 — infix + builtin (x * 2.0 + relu(x)):");
        println!("  input:  [-1.0, 0.0, 1.0, 2.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[-2.0, 0.0, 3.0, 6.0]);
        println!("  PASS");
    }

    // ── Test 3: Tile block ──────────────────────────────────────
    {
        let script = r#"
            x = input([4])
            tile y over x:
                y = x * 2.0 + 1.0
            end
            return y
        "#;
        let compiled = jit.compile(script).expect("compile tile");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // x*2.0: [2.0, 4.0, 6.0, 8.0]
        // +1.0:  [3.0, 5.0, 7.0, 9.0]
        println!("\nTest 3 — tile block (y = x * 2.0 + 1.0):");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[3.0, 5.0, 7.0, 9.0]);
        println!("  PASS");
    }

    // ── Test 4: Sub and div ─────────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = (x - 1.0) / 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile sub/div");
        let input = HostTensor::new(vec![4], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // (x-1.0): [0.0, 2.0, 4.0, 6.0]
        // /2.0:    [0.0, 1.0, 2.0, 3.0]
        println!("\nTest 4 — sub/div ((x - 1.0) / 2.0):");
        println!("  input:  [1.0, 3.0, 5.0, 7.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[0.0, 1.0, 2.0, 3.0]);
        println!("  PASS");
    }

    // ── Test 5: Parenthesized grouping ──────────────────────────
    {
        let script = r#"
            x = input([4])
            y = (x + 1.0) * 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile parens");
        let input = HostTensor::new(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // (x+1.0): [1.0, 2.0, 3.0, 4.0]
        // *2.0:    [2.0, 4.0, 6.0, 8.0]
        println!("\nTest 5 — parenthesized ((x + 1.0) * 2.0):");
        println!("  input:  [0.0, 1.0, 2.0, 3.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[2.0, 4.0, 6.0, 8.0]);
        println!("  PASS");
    }

    // ── Test 6: Nested tile blocks ──────────────────────────────
    {
        let script = r#"
            x = input([4])
            tile y over (x):
                t = x * 2.0
                tile z over (t):
                    z = t + 1.0
                end
                y = z * 3.0
            end
            return y
        "#;
        let compiled = jit.compile(script).expect("compile nested tile");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // t = x*2: [2,4,6,8]
        // z = t+1: [3,5,7,9]
        // y = z*3: [9,15,21,27]
        println!("\nTest 6 — nested tile blocks:");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[9.0, 15.0, 21.0, 27.0]);
        println!("  PASS");
    }

    // ── Test 7: Annotated tile block ────────────────────────────
    {
        let script = r#"
            x = input([4])
            tile y over (x) with (
                tile_m=128,
                tile_n=64,
                tile_k=32,
                unroll=4,
                pipeline_stages=2,
                precision=bf16,
                quant=nf4,
                dist=shard,
                replicas=2,
                mesh_axis=0,
                layout=blocked_32x8,
                accum=bf16,
                collective=all_reduce
            ):
                y = x * 2.0 + 5.0
            end
            return y
        "#;
        let compiled = jit.compile(script).expect("compile annotated tile");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        println!("\nTest 7 — annotated tile block:");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[7.0, 9.0, 11.0, 13.0]);
        println!("  PASS");
    }

    // ── Test 8: Symbolic shape contract ─────────────────────────
    {
        let script = r#"
            x = input([B], B=4) where B >= 4
            y = x * 3.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile symbolic shape");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        println!("\nTest 8 — symbolic shape contract:");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[3.0, 6.0, 9.0, 12.0]);
        println!("  PASS");
    }

    println!("\nAll tiling/infix tests passed.");
}
