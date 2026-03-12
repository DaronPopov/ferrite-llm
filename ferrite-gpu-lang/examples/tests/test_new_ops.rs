/// Test the 14 newly added ops.
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

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
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = Arc::new(
        ptx_runtime::PtxRuntime::with_stable_config(0, Some(config))
            .expect("runtime init failed"),
    );
    runtime.export_for_hook();
    runtime.export_context();

    let mut pass = 0u32;
    let mut fail = 0u32;
    let tol = 1e-4;

    // Helper to make a 1D f32 tensor
    macro_rules! tensor {
        ($data:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                &[$data.len()],
                ptx_tensor::DType::F32,
                &runtime,
            )
            .unwrap()
        };
    }

    macro_rules! tensor2d {
        ($data:expr, $rows:expr, $cols:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                &[$rows, $cols],
                ptx_tensor::DType::F32,
                &runtime,
            )
            .unwrap()
        };
    }

    macro_rules! check {
        ($name:expr, $result:expr, $expected:expr) => {{
            let res = $result.to_vec_f32().unwrap();
            let exp: &[f32] = $expected;
            let ok = res.len() == exp.len()
                && res.iter().zip(exp.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
            if ok {
                println!("PASS: {}", $name);
                pass += 1;
            } else {
                println!("FAIL: {} — expected {:?}, got {:?}", $name, exp, res);
                fail += 1;
            }
        }};
    }

    // =====================================================================
    // Unary ops (7)
    // =====================================================================

    // 1. log2
    {
        let x = tensor!(&[1.0f32, 2.0, 4.0, 8.0]);
        let r = x.log2().unwrap();
        check!("log2", r, &[0.0, 1.0, 2.0, 3.0]);
    }

    // 2. log10
    {
        let x = tensor!(&[1.0f32, 10.0, 100.0, 1000.0]);
        let r = x.log10().unwrap();
        check!("log10", r, &[0.0, 1.0, 2.0, 3.0]);
    }

    // 3. tan
    {
        let x = tensor!(&[0.0f32, std::f32::consts::FRAC_PI_4]);
        let r = x.tan().unwrap();
        check!("tan", r, &[0.0, 1.0]);
    }

    // 4. sinh
    {
        let x = tensor!(&[0.0f32, 1.0]);
        let r = x.sinh().unwrap();
        check!("sinh", r, &[0.0, 1.17520]);
    }

    // 5. cosh
    {
        let x = tensor!(&[0.0f32, 1.0]);
        let r = x.cosh().unwrap();
        check!("cosh", r, &[1.0, 1.54308]);
    }

    // 6. sign
    {
        let x = tensor!(&[-5.0f32, 0.0, 3.0, -0.1]);
        let r = x.sign().unwrap();
        check!("sign", r, &[-1.0, 0.0, 1.0, -1.0]);
    }

    // 7. erf
    {
        let x = tensor!(&[0.0f32, 1.0, -1.0]);
        let r = x.erf().unwrap();
        check!("erf", r, &[0.0, 0.8427, -0.8427]);
    }

    // =====================================================================
    // Binary op (1)
    // =====================================================================

    // 8. fmod
    {
        let a = tensor!(&[7.0f32, 5.5, -3.0, 10.0]);
        let b = tensor!(&[3.0f32, 2.0, 2.0, 3.0]);
        let r = a.fmod(&b).unwrap();
        check!("fmod", r, &[1.0, 1.5, -1.0, 1.0]);
    }

    // =====================================================================
    // Activation ops (3)
    // =====================================================================

    // 9. gelu_tanh (tanh approximation)
    //    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    {
        let x = tensor!(&[0.0f32, 1.0, -1.0, 2.0]);
        let r = x.gelu_tanh().unwrap();
        // expected: 0.0, ~0.8412, ~-0.1588, ~1.9545
        let res = r.to_vec_f32().unwrap();
        let ok = approx_eq(res[0], 0.0, tol)
            && approx_eq(res[1], 0.8412, 0.01)
            && approx_eq(res[2], -0.1588, 0.01)
            && approx_eq(res[3], 1.9545, 0.01);
        if ok {
            println!("PASS: gelu_tanh");
            pass += 1;
        } else {
            println!("FAIL: gelu_tanh — got {:?}", res);
            fail += 1;
        }
    }

    // 10. hardswish: x * min(max(x+3, 0), 6) / 6
    {
        let x = tensor!(&[-4.0f32, -3.0, 0.0, 3.0, 4.0]);
        let r = x.hardswish().unwrap();
        // -4 → 0, -3 → 0, 0 → 0, 3 → 3, 4 → 4
        check!("hardswish", r, &[0.0, 0.0, 0.0, 3.0, 4.0]);
    }

    // 11. hardsigmoid: min(max(x/6 + 0.5, 0), 1)
    {
        let x = tensor!(&[-4.0f32, -3.0, 0.0, 3.0, 4.0]);
        let r = x.hardsigmoid().unwrap();
        // -4 → 0, -3 → 0, 0 → 0.5, 3 → 1, 4 → 1
        check!("hardsigmoid", r, &[0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    // =====================================================================
    // Reduction ops (3)
    // =====================================================================

    // 12. prod
    {
        let x = tensor2d!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let r = x.prod(-1).unwrap();
        // row 0: 1*2*3=6, row 1: 4*5*6=120
        check!("prod", r, &[6.0, 120.0]);
    }

    // 13. argmax
    {
        let x = tensor2d!(&[1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0, 0.0, 7.0, 1.0], 3, 3);
        let r = x.argmax(-1).unwrap();
        let out = r.to_vec_i32().unwrap();
        let expected = vec![1i32, 2, 1]; // row 0: max at 1, row 1: max at 2, row 2: max at 1
        if out == expected {
            println!("PASS: argmax");
            pass += 1;
        } else {
            println!("FAIL: argmax — expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // 14. argmin
    {
        let x = tensor2d!(&[3.0f32, 1.0, 5.0, 6.0, 2.0, 4.0, 7.0, 0.0, 9.0], 3, 3);
        let r = x.argmin(-1).unwrap();
        let out = r.to_vec_i32().unwrap();
        let expected = vec![1i32, 1, 1]; // row 0: min at 1, row 1: min at 1, row 2: min at 1
        if out == expected {
            println!("PASS: argmin");
            pass += 1;
        } else {
            println!("FAIL: argmin — expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // =====================================================================
    println!("\n{}/{} tests passed", pass, pass + fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
