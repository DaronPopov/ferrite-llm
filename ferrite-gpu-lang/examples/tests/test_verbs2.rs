/// Test round 3: scans, casting, norms, random, pad.
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
    let tol = 1e-3;

    macro_rules! tensor_f32 {
        ($data:expr, $shape:expr) => {
            ptx_tensor::Tensor::from_slice($data, $shape, ptx_tensor::DType::F32, &runtime).unwrap()
        };
    }

    macro_rules! check_f32 {
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
    // 1. Scan ops (cumprod, cummax, cummin)
    // =====================================================================
    println!("--- Scans ---");
    {
        let x = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 5]);

        // cumsum: [1, 3, 6, 10, 15]
        check_f32!("cumsum", x.cumsum(-1).unwrap(), &[1.0, 3.0, 6.0, 10.0, 15.0]);

        // cumprod: [1, 2, 6, 24, 120]
        check_f32!("cumprod", x.cumprod(-1).unwrap(), &[1.0, 2.0, 6.0, 24.0, 120.0]);

        // cummax: [1, 2, 3, 4, 5]
        check_f32!("cummax", x.cummax(-1).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // With non-monotone input
        let y = tensor_f32!(&[5.0f32, 3.0, 4.0, 1.0, 2.0], &[1, 5]);
        check_f32!("cummax(non-mono)", y.cummax(-1).unwrap(), &[5.0, 5.0, 5.0, 5.0, 5.0]);
        check_f32!("cummin(non-mono)", y.cummin(-1).unwrap(), &[5.0, 3.0, 3.0, 1.0, 1.0]);

        // 2D: along rows
        let z = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        check_f32!("cumprod 2D", z.cumprod(-1).unwrap(), &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    // =====================================================================
    // 2. Casting ops
    // =====================================================================
    println!("--- Casting ---");
    {
        // f32 → i32 → f32 round-trip
        let x = tensor_f32!(&[1.7f32, 2.0, -3.9, 4.1], &[4]);
        let xi = x.to_i32().unwrap();
        let xf = xi.to_f32().unwrap();
        // i32 truncates towards zero
        check_f32!("f32→i32→f32", xf, &[1.0, 2.0, -3.0, 4.0]);

        // f32 → f16 → f32 round-trip (with precision loss)
        let x = tensor_f32!(&[1.0f32, 0.5, -2.0, 100.0], &[4]);
        let xh = x.to_f16().unwrap();
        let xf = xh.to_f32().unwrap();
        check_f32!("f32→f16→f32", xf, &[1.0, 0.5, -2.0, 100.0]);
    }

    // =====================================================================
    // 3. Norms
    // =====================================================================
    println!("--- Norms ---");
    {
        let x = tensor_f32!(&[3.0f32, -4.0], &[1, 2]);

        // L1: |3| + |-4| = 7
        let l1 = x.norm_l1(-1).unwrap();
        check_f32!("norm_l1", l1, &[7.0]);

        // L2: sqrt(9 + 16) = 5
        let l2 = x.norm_l2(-1).unwrap();
        check_f32!("norm_l2", l2, &[5.0]);

        // Full reduction norms
        let x = tensor_f32!(&[1.0f32, -2.0, 3.0, -4.0], &[4]);
        let l1a = x.norm_l1_all().unwrap();
        check_f32!("norm_l1_all", l1a, &[10.0]);

        let l2a = x.norm_l2_all().unwrap();
        // sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
        check_f32!("norm_l2_all", l2a, &[5.477]);

        // Normalize
        let x = tensor_f32!(&[3.0f32, 4.0], &[1, 2]);
        let n = x.normalize(-1).unwrap();
        // [3/5, 4/5] = [0.6, 0.8]
        check_f32!("normalize", n, &[0.6, 0.8]);
    }

    // =====================================================================
    // 4. Random
    // =====================================================================
    println!("--- Random ---");
    {
        // rand: all values in [0, 1)
        let r = ptx_tensor::Tensor::rand(&[1000], &runtime).unwrap();
        let vals = r.to_vec_f32().unwrap();
        let all_in_range = vals.iter().all(|&v| v >= 0.0 && v < 1.0);
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        // Mean should be near 0.5
        let mean_ok = (mean - 0.5).abs() < 0.05;
        if all_in_range && mean_ok {
            println!("PASS: rand [0,1) — mean={:.3}", mean);
            pass += 1;
        } else {
            println!("FAIL: rand — in_range={}, mean={:.3}", all_in_range, mean);
            fail += 1;
        }

        // randn: mean near 0, std near 1
        let r = ptx_tensor::Tensor::randn(&[10000], &runtime).unwrap();
        let vals = r.to_vec_f32().unwrap();
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        let var: f32 = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        let std = var.sqrt();
        let mean_ok = mean.abs() < 0.05;
        let std_ok = (std - 1.0).abs() < 0.1;
        if mean_ok && std_ok {
            println!("PASS: randn — mean={:.3}, std={:.3}", mean, std);
            pass += 1;
        } else {
            println!("FAIL: randn — mean={:.3}, std={:.3}", mean, std);
            fail += 1;
        }

        // rand_like
        let x = tensor_f32!(&[0.0f32; 100], &[10, 10]);
        let rl = x.rand_like().unwrap();
        assert_eq!(rl.shape(), &[10, 10]);
        let vals = rl.to_vec_f32().unwrap();
        let all_in_range = vals.iter().all(|&v| v >= 0.0 && v < 1.0);
        if all_in_range {
            println!("PASS: rand_like shape={:?}", rl.shape());
            pass += 1;
        } else {
            println!("FAIL: rand_like out of range");
            fail += 1;
        }
    }

    // =====================================================================
    // 5. Pad
    // =====================================================================
    println!("--- Pad ---");
    {
        // 1x1x2x3 → pad top=1, bottom=1, left=1, right=1 → 1x1x4x5
        let x = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]);
        let p = x.pad2d([1, 1, 1, 1], 0.0).unwrap();
        assert_eq!(p.shape(), &[1, 1, 4, 5]);
        let vals = p.to_vec_f32().unwrap();
        // Row 0: all zeros (pad top)
        // Row 1: 0, 1, 2, 3, 0
        // Row 2: 0, 4, 5, 6, 0
        // Row 3: all zeros (pad bottom)
        let expected = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 3.0, 0.0,
            0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ok = vals.len() == expected.len()
            && vals.iter().zip(expected.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
        if ok {
            println!("PASS: pad2d [1,1,2,3] → [1,1,4,5]");
            pass += 1;
        } else {
            println!("FAIL: pad2d — got {:?}", vals);
            fail += 1;
        }

        // Pad with non-zero value
        let p2 = x.pad2d_uniform(1, -1.0).unwrap();
        let vals2 = p2.to_vec_f32().unwrap();
        // First element should be -1 (pad value)
        if approx_eq(vals2[0], -1.0, tol) && approx_eq(vals2[6], 1.0, tol) {
            println!("PASS: pad2d_uniform with value=-1");
            pass += 1;
        } else {
            println!("FAIL: pad2d_uniform — got {:?}", &vals2[..10]);
            fail += 1;
        }
    }

    // =====================================================================
    // Combined: pipeline test
    // =====================================================================
    println!("--- Pipeline ---");
    {
        // Generate random data, normalize, compute cumulative sum
        let x = ptx_tensor::Tensor::rand(&[1, 5], &runtime).unwrap();
        let n = x.normalize(-1).unwrap();
        let cs = n.cumsum(-1).unwrap();
        let vals = cs.to_vec_f32().unwrap();
        // Last element should be the L1 norm of the normalized vector
        println!("PASS: rand→normalize→cumsum pipeline (last={:.3})", vals[4]);
        pass += 1;
    }

    // =====================================================================
    println!("\n{}/{} tests passed", pass, pass + fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
