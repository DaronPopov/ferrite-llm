/// Hardening stress test — validates correctness after safety fixes.
/// Tests: single-element tensors, non-contiguous inputs, NaN/Inf propagation,
/// large shapes, chained pipelines, empty tensor errors, overflow shape errors,
/// multi-dtype, and edge values.
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
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

    macro_rules! tensor_shape {
        ($data:expr, $shape:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                $shape,
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
                println!("  PASS: {}", $name);
                pass += 1;
            } else {
                println!("  FAIL: {} — expected {:?}, got {:?}", $name, exp, &res[..res.len().min(16)]);
                fail += 1;
            }
        }};
    }

    macro_rules! check_err {
        ($name:expr, $result:expr) => {{
            match $result {
                Ok(_) => {
                    println!("  FAIL: {} — expected error, got Ok", $name);
                    fail += 1;
                }
                Err(_) => {
                    println!("  PASS: {}", $name);
                    pass += 1;
                }
            }
        }};
    }

    // =====================================================================
    // 1. Single-element tensors
    // =====================================================================
    println!("\n=== Test 1: Single-element tensors ===");
    {
        let a = tensor!(&[3.0f32]);
        let b = tensor!(&[2.0f32]);

        // Unary ops on [1]
        check!("neg([3])", a.neg().unwrap(), &[-3.0]);
        check!("abs([-3])", tensor!(&[-3.0f32]).abs().unwrap(), &[3.0]);
        check!("exp([0])", tensor!(&[0.0f32]).exp().unwrap(), &[1.0]);
        check!("log([1])", tensor!(&[1.0f32]).log().unwrap(), &[0.0]);
        check!("sqrt([4])", tensor!(&[4.0f32]).sqrt().unwrap(), &[2.0]);
        check!("relu([-1])", tensor!(&[-1.0f32]).relu().unwrap(), &[0.0]);
        check!("sigmoid([0])", tensor!(&[0.0f32]).sigmoid().unwrap(), &[0.5]);

        // Binary ops on [1]
        check!("add([3],[2])", a.add(&b).unwrap(), &[5.0]);
        check!("sub([3],[2])", a.sub(&b).unwrap(), &[1.0]);
        check!("mul([3],[2])", a.mul(&b).unwrap(), &[6.0]);
        check!("div([3],[2])", a.div(&b).unwrap(), &[1.5]);

        // Reduction on [1]
        check!("sum([3])", a.sum(0).unwrap(), &[3.0]);
        check!("mean([3])", a.mean(0).unwrap(), &[3.0]);
        check!("max([3])", a.max(0).unwrap(), &[3.0]);
        check!("min([3])", a.min(0).unwrap(), &[3.0]);
        check!("prod([3])", a.prod(0).unwrap(), &[3.0]);
    }

    // =====================================================================
    // 2. Non-contiguous input (transpose + op)
    // =====================================================================
    println!("\n=== Test 2: Non-contiguous input (auto-contiguous) ===");
    {
        // Create 2x3 tensor [[1,2,3],[4,5,6]], transpose to 3x2, then run ops
        let data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = tensor_shape!(data, &[2, 3]);
        let tt = t.t().unwrap(); // 3x2: [[1,4],[2,5],[3,6]]
        assert!(!tt.is_contiguous(), "transposed tensor should not be contiguous");

        // Unary on non-contiguous
        let neg_tt = tt.neg().unwrap();
        let neg_vals = neg_tt.to_vec_f32().unwrap();
        let expected_neg: &[f32] = &[-1.0, -4.0, -2.0, -5.0, -3.0, -6.0];
        let neg_ok = neg_vals.len() == expected_neg.len()
            && neg_vals.iter().zip(expected_neg.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
        if neg_ok {
            println!("  PASS: neg(transposed)");
            pass += 1;
        } else {
            println!("  FAIL: neg(transposed) — expected {:?}, got {:?}", expected_neg, neg_vals);
            fail += 1;
        }

        // Relu on non-contiguous
        let neg_data: &[f32] = &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let nt = tensor_shape!(neg_data, &[2, 3]);
        let ntt = nt.t().unwrap(); // 3x2: [[1,-4],[-2,5],[3,-6]]
        let relu_ntt = ntt.relu().unwrap();
        let relu_vals = relu_ntt.to_vec_f32().unwrap();
        let expected_relu: &[f32] = &[1.0, 0.0, 0.0, 5.0, 3.0, 0.0];
        let relu_ok = relu_vals.len() == expected_relu.len()
            && relu_vals.iter().zip(expected_relu.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
        if relu_ok {
            println!("  PASS: relu(transposed)");
            pass += 1;
        } else {
            println!("  FAIL: relu(transposed) — expected {:?}, got {:?}", expected_relu, relu_vals);
            fail += 1;
        }

        // Binary on non-contiguous: add(transpose, transpose)
        let a_data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: &[f32] = &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a2 = tensor_shape!(a_data, &[2, 3]).t().unwrap();
        let b2 = tensor_shape!(b_data, &[2, 3]).t().unwrap();
        let sum_ab = a2.add(&b2).unwrap();
        let sum_vals = sum_ab.to_vec_f32().unwrap();
        // a2 = [[1,4],[2,5],[3,6]], b2 = [[10,40],[20,50],[30,60]]
        let expected_sum: &[f32] = &[11.0, 44.0, 22.0, 55.0, 33.0, 66.0];
        let sum_ok = sum_vals.len() == expected_sum.len()
            && sum_vals.iter().zip(expected_sum.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
        if sum_ok {
            println!("  PASS: add(transposed, transposed)");
            pass += 1;
        } else {
            println!("  FAIL: add(transposed, transposed) — expected {:?}, got {:?}", expected_sum, sum_vals);
            fail += 1;
        }

        // Reduction on non-contiguous: sum along dim 1 of 3x2 transposed tensor
        let t3x2 = tensor_shape!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).t().unwrap();
        // t3x2 = [[1,4],[2,5],[3,6]], sum(dim=1) = [5, 7, 9]
        let sum_dim1 = t3x2.sum(1).unwrap();
        check!("sum(transposed, dim=1)", sum_dim1, &[5.0, 7.0, 9.0]);
    }

    // =====================================================================
    // 3. NaN/Inf propagation
    // =====================================================================
    println!("\n=== Test 3: NaN/Inf propagation ===");
    {
        let nan_t = tensor!(&[f32::NAN]);
        let inf_t = tensor!(&[f32::INFINITY]);
        let ninf_t = tensor!(&[f32::NEG_INFINITY]);

        // NaN in → NaN out
        let r = nan_t.exp().unwrap().to_vec_f32().unwrap();
        if r[0].is_nan() { println!("  PASS: exp(NaN) = NaN"); pass += 1; }
        else { println!("  FAIL: exp(NaN) = {} (expected NaN)", r[0]); fail += 1; }

        let r = nan_t.log().unwrap().to_vec_f32().unwrap();
        if r[0].is_nan() { println!("  PASS: log(NaN) = NaN"); pass += 1; }
        else { println!("  FAIL: log(NaN) = {} (expected NaN)", r[0]); fail += 1; }

        let r = nan_t.relu().unwrap().to_vec_f32().unwrap();
        // GPU relu uses max(x, 0) which returns 0 for NaN per CUDA semantics
        if r[0] == 0.0 || r[0].is_nan() { println!("  PASS: relu(NaN) = {} (GPU max(NaN,0) behavior)", r[0]); pass += 1; }
        else { println!("  FAIL: relu(NaN) = {} (expected 0 or NaN)", r[0]); fail += 1; }

        let r = nan_t.sin().unwrap().to_vec_f32().unwrap();
        if r[0].is_nan() { println!("  PASS: sin(NaN) = NaN"); pass += 1; }
        else { println!("  FAIL: sin(NaN) = {} (expected NaN)", r[0]); fail += 1; }

        // Inf propagation
        check!("exp(Inf)", inf_t.exp().unwrap(), &[f32::INFINITY]);
        check!("exp(-Inf)", ninf_t.exp().unwrap(), &[0.0]);

        // NaN + normal = NaN
        let normal = tensor!(&[5.0f32]);
        let r = nan_t.add(&normal).unwrap().to_vec_f32().unwrap();
        if r[0].is_nan() { println!("  PASS: NaN + 5 = NaN"); pass += 1; }
        else { println!("  FAIL: NaN + 5 = {} (expected NaN)", r[0]); fail += 1; }
    }

    // =====================================================================
    // 4. Large shapes
    // =====================================================================
    println!("\n=== Test 4: Large shapes ===");
    {
        // [1, 1, 1, 10_000_000] — stress grid dispatch
        let big = ptx_tensor::Tensor::ones(&[1, 1, 1, 10_000_000], ptx_tensor::DType::F32, &runtime).unwrap();
        let big_sum = big.sum_all().unwrap();
        let val = big_sum.to_vec_f32().unwrap();
        let expected = 10_000_000.0f32;
        if approx_eq(val[0], expected, 1.0) {
            println!("  PASS: sum_all([1,1,1,10M]) = {}", val[0]);
            pass += 1;
        } else {
            println!("  FAIL: sum_all([1,1,1,10M]) = {} (expected ~{})", val[0], expected);
            fail += 1;
        }

        // Neg on large tensor, spot-check
        let big_neg = big.neg().unwrap();
        let big_neg_sum = big_neg.sum_all().unwrap();
        let neg_val = big_neg_sum.to_vec_f32().unwrap();
        if approx_eq(neg_val[0], -expected, 1.0) {
            println!("  PASS: sum_all(neg([1,1,1,10M])) = {}", neg_val[0]);
            pass += 1;
        } else {
            println!("  FAIL: sum_all(neg([1,1,1,10M])) = {} (expected ~{})", neg_val[0], -expected);
            fail += 1;
        }
    }

    // =====================================================================
    // 5. Chained pipeline — 10 ops without manual sync
    // =====================================================================
    println!("\n=== Test 5: Chained pipeline (10 ops) ===");
    {
        let x = tensor!(&[2.0f32]);
        // Chain: neg → abs → exp → log → sqrt → sqr → add_scalar(1) → mul_scalar(3) → neg → abs
        // neg(2) = -2
        // abs(-2) = 2
        // exp(2) ≈ 7.389
        // log(7.389) ≈ 2.0
        // sqrt(2) ≈ 1.4142
        // sqr(1.4142) ≈ 2.0
        // +1 = 3.0
        // *3 = 9.0
        // neg = -9.0
        // abs = 9.0
        let result = x.neg().unwrap()
            .abs().unwrap()
            .exp().unwrap()
            .log().unwrap()
            .sqrt().unwrap()
            .sqr().unwrap()
            .add_scalar(1.0).unwrap()
            .mul_scalar(3.0).unwrap()
            .neg().unwrap()
            .abs().unwrap();
        check!("chained 10-op pipeline", result, &[9.0]);
    }

    // =====================================================================
    // 6. Empty tensor errors
    // =====================================================================
    println!("\n=== Test 6: Empty tensor errors ===");
    {
        // Creating a tensor with a zero-dimension should fail (allocator rejects 0 bytes)
        check_err!("new([3,0,4]) should error", ptx_tensor::Tensor::new(&[3, 0, 4], ptx_tensor::DType::F32, &runtime));

        // Also test that reduce on a valid tensor with small dim works but errors on invalid dim
        let small = ptx_tensor::Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], ptx_tensor::DType::F32, &runtime).unwrap();
        check_err!("reduce invalid dim", small.sum(5));
        check_err!("reduce negative invalid dim", small.sum(-5));
    }

    // =====================================================================
    // 7. Overflow shape errors
    // =====================================================================
    println!("\n=== Test 7: Overflow shape errors ===");
    {
        // Shape that would overflow usize on multiply
        let result = ptx_tensor::Tensor::new(
            &[usize::MAX / 2, 3],
            ptx_tensor::DType::F32,
            &runtime,
        );
        check_err!("overflow shape [MAX/2, 3]", result);

        let result = ptx_tensor::Tensor::zeros(
            &[1_000_000, 1_000_000, 1_000_000],
            ptx_tensor::DType::F32,
            &runtime,
        );
        check_err!("overflow shape [1M, 1M, 1M]", result);
    }

    // =====================================================================
    // 8. Multi-dtype (F32, F64 paths)
    // =====================================================================
    println!("\n=== Test 8: Multi-dtype ===");
    {
        // F64 basic ops
        let f64_data: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let f64_t = ptx_tensor::Tensor::from_slice(
            f64_data,
            &[4],
            ptx_tensor::DType::F64,
            &runtime,
        ).unwrap();

        // F64 neg
        let f64_neg = f64_t.neg().unwrap();
        let f64_neg_vals: Vec<f64> = f64_neg.to_vec().unwrap();
        let f64_neg_ok = f64_neg_vals == vec![-1.0, -2.0, -3.0, -4.0];
        if f64_neg_ok { println!("  PASS: F64 neg"); pass += 1; }
        else { println!("  FAIL: F64 neg — got {:?}", f64_neg_vals); fail += 1; }

        // F64 add
        let f64_b = ptx_tensor::Tensor::from_slice(
            &[10.0f64, 20.0, 30.0, 40.0],
            &[4],
            ptx_tensor::DType::F64,
            &runtime,
        ).unwrap();
        let f64_sum = f64_t.add(&f64_b).unwrap();
        let f64_sum_vals: Vec<f64> = f64_sum.to_vec().unwrap();
        let f64_sum_ok = f64_sum_vals == vec![11.0, 22.0, 33.0, 44.0];
        if f64_sum_ok { println!("  PASS: F64 add"); pass += 1; }
        else { println!("  FAIL: F64 add — got {:?}", f64_sum_vals); fail += 1; }

        // F64 reduction (sum)
        let f64_sum_all = f64_t.sum(0).unwrap();
        let f64_sa_vals: Vec<f64> = f64_sum_all.to_vec().unwrap();
        if (f64_sa_vals[0] - 10.0).abs() < 1e-10 {
            println!("  PASS: F64 sum_all");
            pass += 1;
        } else {
            println!("  FAIL: F64 sum_all — got {:?}", f64_sa_vals);
            fail += 1;
        }

        // F64 relu
        let f64_mix = ptx_tensor::Tensor::from_slice(
            &[-1.0f64, 2.0, -3.0, 4.0],
            &[4],
            ptx_tensor::DType::F64,
            &runtime,
        ).unwrap();
        let f64_relu = f64_mix.relu().unwrap();
        let f64_relu_vals: Vec<f64> = f64_relu.to_vec().unwrap();
        let f64_relu_ok = f64_relu_vals == vec![0.0, 2.0, 0.0, 4.0];
        if f64_relu_ok { println!("  PASS: F64 relu"); pass += 1; }
        else { println!("  FAIL: F64 relu — got {:?}", f64_relu_vals); fail += 1; }
    }

    // =====================================================================
    // 9. Edge values — 0, -0, very large, very small (denormals)
    // =====================================================================
    println!("\n=== Test 9: Edge values ===");
    {
        // Zero and negative zero
        let zeros = tensor!(&[0.0f32, -0.0f32]);
        check!("abs(0, -0)", zeros.abs().unwrap(), &[0.0, 0.0]);
        check!("relu(0, -0)", zeros.relu().unwrap(), &[0.0, 0.0]);

        // Very large values
        let large = tensor!(&[1e38f32]);
        let large_neg = large.neg().unwrap();
        check!("neg(1e38)", large_neg, &[-1e38]);

        // Very small (subnormal) values
        let tiny = tensor!(&[1e-40f32]);
        let tiny_abs = tiny.abs().unwrap();
        let tiny_val = tiny_abs.to_vec_f32().unwrap();
        if tiny_val[0] > 0.0 && tiny_val[0] < 1e-30 {
            println!("  PASS: abs(1e-40) = {} (subnormal preserved)", tiny_val[0]);
            pass += 1;
        } else {
            println!("  FAIL: abs(1e-40) = {} (expected subnormal > 0)", tiny_val[0]);
            fail += 1;
        }

        // Sign of zero
        let sign_z = tensor!(&[0.0f32]).sign().unwrap().to_vec_f32().unwrap();
        if sign_z[0] == 0.0 {
            println!("  PASS: sign(0) = 0");
            pass += 1;
        } else {
            println!("  FAIL: sign(0) = {} (expected 0)", sign_z[0]);
            fail += 1;
        }

        // Clamp edge: already inside range
        check!("clamp(5, 0, 10)", tensor!(&[5.0f32]).clamp(0.0, 10.0).unwrap(), &[5.0]);
        // Clamp edge: at boundary
        check!("clamp(0, 0, 10)", tensor!(&[0.0f32]).clamp(0.0, 10.0).unwrap(), &[0.0]);
        check!("clamp(10, 0, 10)", tensor!(&[10.0f32]).clamp(0.0, 10.0).unwrap(), &[10.0]);
    }

    // =====================================================================
    // Summary
    // =====================================================================
    println!("\n========================================");
    println!("Hardening stress test: {} passed, {} failed", pass, fail);
    println!("========================================");

    if fail > 0 {
        std::process::exit(1);
    }
}
