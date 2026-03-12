/// Integration test for the ferrite JIT compiler.
///
/// Tests the full pipeline: script text → lexer → parser → AST →
/// lower → Program → compile → CompiledProgram → execute on GPU.

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;
    let tol = 1e-3;

    let mut jit = ferrite_gpu_lang::jit::JitEngine::new();
    let runtime = ferrite_gpu_lang::GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── Test 1: simple unary chain ──────────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            x = input([1, 1, 1, 4])
            h = relu(x)
            y = sigmoid(h)
            return y
        "#,
            )
            .expect("compile test 1");

        let input = ferrite_gpu_lang::HostTensor::new(
            vec![1, 1, 1, 4],
            vec![-1.0, 0.0, 1.0, 2.0],
        )
        .unwrap();
        let out = runtime.execute(compiled, &[input]).unwrap();
        let d = out.data();
        // relu([-1,0,1,2]) = [0,0,1,2]
        // sigmoid([0,0,1,2]) ≈ [0.5, 0.5, 0.731, 0.881]
        let ok = approx(d[0], 0.5, tol)
            && approx(d[1], 0.5, tol)
            && approx(d[2], 0.7311, tol)
            && approx(d[3], 0.8808, tol);
        if ok {
            println!("PASS: unary chain (relu → sigmoid)");
            pass += 1;
        } else {
            println!("FAIL: unary chain — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 2: binary ops ──────────────────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            a = input([1, 1, 1, 4])
            b = input([1, 1, 1, 4])
            c = add(a, b)
            d = relu(c)
            return d
        "#,
            )
            .expect("compile test 2");

        let in_a = ferrite_gpu_lang::HostTensor::new(
            vec![1, 1, 1, 4],
            vec![1.0, -2.0, 3.0, -4.0],
        )
        .unwrap();
        let in_b = ferrite_gpu_lang::HostTensor::new(
            vec![1, 1, 1, 4],
            vec![-1.0, 3.0, -3.0, 5.0],
        )
        .unwrap();
        let out = runtime.execute(compiled, &[in_a, in_b]).unwrap();
        let d = out.data();
        // add = [0, 1, 0, 1], relu = [0, 1, 0, 1]
        let ok = approx(d[0], 0.0, tol)
            && approx(d[1], 1.0, tol)
            && approx(d[2], 0.0, tol)
            && approx(d[3], 1.0, tol);
        if ok {
            println!("PASS: binary add + relu");
            pass += 1;
        } else {
            println!("FAIL: binary add + relu — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 3: mul ─────────────────────────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            a = input([1, 1, 1, 4])
            b = input([1, 1, 1, 4])
            c = mul(a, b)
            return c
        "#,
            )
            .expect("compile test 3");

        let in_a =
            ferrite_gpu_lang::HostTensor::new(vec![1, 1, 1, 4], vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        let in_b =
            ferrite_gpu_lang::HostTensor::new(vec![1, 1, 1, 4], vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        let out = runtime.execute(compiled, &[in_a, in_b]).unwrap();
        let d = out.data();
        // [1.0, 1.5, 2.0, 2.5]
        let ok = approx(d[0], 1.0, tol)
            && approx(d[1], 1.5, tol)
            && approx(d[2], 2.0, tol)
            && approx(d[3], 2.5, tol);
        if ok {
            println!("PASS: mul");
            pass += 1;
        } else {
            println!("FAIL: mul — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 4: tanh ────────────────────────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            x = input([1, 1, 1, 4])
            y = tanh(x)
            return y
        "#,
            )
            .expect("compile test 4");

        let input =
            ferrite_gpu_lang::HostTensor::new(vec![1, 1, 1, 4], vec![0.0, 1.0, -1.0, 2.0])
                .unwrap();
        let out = runtime.execute(compiled, &[input]).unwrap();
        let d = out.data();
        // tanh([0,1,-1,2]) ≈ [0, 0.7616, -0.7616, 0.9640]
        let ok = approx(d[0], 0.0, tol)
            && approx(d[1], 0.7616, tol)
            && approx(d[2], -0.7616, tol)
            && approx(d[3], 0.9640, tol);
        if ok {
            println!("PASS: tanh");
            pass += 1;
        } else {
            println!("FAIL: tanh — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 5: function inlining ───────────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            fn activate(x):
                h = relu(x)
                y = sigmoid(h)
                return y
            end

            a = input([1, 1, 1, 4])
            b = activate(a)
            return b
        "#,
            )
            .expect("compile test 5");

        let input = ferrite_gpu_lang::HostTensor::new(
            vec![1, 1, 1, 4],
            vec![-1.0, 0.0, 1.0, 2.0],
        )
        .unwrap();
        let out = runtime.execute(compiled, &[input]).unwrap();
        let d = out.data();
        // Same as test 1: relu then sigmoid
        let ok = approx(d[0], 0.5, tol)
            && approx(d[1], 0.5, tol)
            && approx(d[2], 0.7311, tol)
            && approx(d[3], 0.8808, tol);
        if ok {
            println!("PASS: function inlining");
            pass += 1;
        } else {
            println!("FAIL: function inlining — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 6: multi-function pipeline ─────────────────────────
    {
        let compiled = jit
            .compile(
                r#"
            fn encoder(x):
                h = relu(x)
                return h
            end

            fn decoder(x):
                h = sigmoid(x)
                return h
            end

            a = input([1, 1, 1, 4])
            b = encoder(a)
            c = decoder(b)
            return c
        "#,
            )
            .expect("compile test 6");

        let input = ferrite_gpu_lang::HostTensor::new(
            vec![1, 1, 1, 4],
            vec![-1.0, 0.0, 1.0, 2.0],
        )
        .unwrap();
        let out = runtime.execute(compiled, &[input]).unwrap();
        let d = out.data();
        // relu then sigmoid — same as test 1
        let ok = approx(d[0], 0.5, tol)
            && approx(d[1], 0.5, tol)
            && approx(d[2], 0.7311, tol)
            && approx(d[3], 0.8808, tol);
        if ok {
            println!("PASS: multi-function pipeline");
            pass += 1;
        } else {
            println!("FAIL: multi-function pipeline — got {:?}", d);
            fail += 1;
        }
    }

    // ── Test 7: cache verification ──────────────────────────────
    {
        let before = jit.cache_len();
        let _ = jit
            .compile("x = input([1, 4])\ny = relu(x)\nreturn y")
            .unwrap();
        let after_first = jit.cache_len();
        let _ = jit
            .compile("x = input([1, 4])\ny = relu(x)\nreturn y")
            .unwrap();
        let after_second = jit.cache_len();

        if after_first == after_second && after_first == before + 1 {
            println!("PASS: JIT cache hit");
            pass += 1;
        } else {
            println!(
                "FAIL: JIT cache — before={}, after_first={}, after_second={}",
                before, after_first, after_second
            );
            fail += 1;
        }
    }

    // ── Test 8: complex graph (binary + unary mix) ──────────────
    {
        let compiled = jit
            .compile(
                r#"
            a = input([1, 1, 1, 4])
            b = input([1, 1, 1, 4])
            c = add(a, b)
            d = relu(c)
            e = mul(d, a)
            f = sigmoid(e)
            return f
        "#,
            )
            .expect("compile test 8");

        let in_a =
            ferrite_gpu_lang::HostTensor::new(vec![1, 1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let in_b =
            ferrite_gpu_lang::HostTensor::new(vec![1, 1, 1, 4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = runtime.execute(compiled, &[in_a, in_b]).unwrap();
        let d = out.data();
        // add=[2,3,4,5], relu=[2,3,4,5], mul(relu,a)=[2,6,12,20], sigmoid(...)
        let expected = [
            1.0 / (1.0 + (-2.0f32).exp()),
            1.0 / (1.0 + (-6.0f32).exp()),
            1.0 / (1.0 + (-12.0f32).exp()),
            1.0 / (1.0 + (-20.0f32).exp()),
        ];
        let ok = d.iter().zip(&expected).all(|(a, b)| approx(*a, *b, tol));
        if ok {
            println!("PASS: complex graph (add → relu → mul → sigmoid)");
            pass += 1;
        } else {
            println!(
                "FAIL: complex graph — expected {:?}, got {:?}",
                expected, d
            );
            fail += 1;
        }
    }

    // ── summary ─────────────────────────────────────────────────
    println!("\n{}/{} JIT tests passed", pass, pass + fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
