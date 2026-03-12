/// Test all new verbal syntax ops: comparisons, logical, arange/linspace,
/// transpose/permute, broadcast, cat/stack, repeat, masked_fill.
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

    macro_rules! tensor_f32 {
        ($data:expr, $shape:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data, $shape, ptx_tensor::DType::F32, &runtime,
            ).unwrap()
        };
    }

    macro_rules! tensor_u8 {
        ($data:expr, $shape:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data, $shape, ptx_tensor::DType::U8, &runtime,
            ).unwrap()
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

    macro_rules! check_u8 {
        ($name:expr, $result:expr, $expected:expr) => {{
            let res = $result.to_vec_u8().unwrap();
            let exp: &[u8] = $expected;
            if res == exp {
                println!("PASS: {}", $name);
                pass += 1;
            } else {
                println!("FAIL: {} — expected {:?}, got {:?}", $name, exp, res);
                fail += 1;
            }
        }};
    }

    // =====================================================================
    // 1. Comparison ops
    // =====================================================================
    println!("--- Comparisons ---");
    {
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let b = tensor_f32!(&[2.0f32, 2.0, 1.0, 5.0], &[4]);

        check_u8!("eq", a.eq(&b).unwrap(), &[0, 1, 0, 0]);
        check_u8!("ne", a.ne(&b).unwrap(), &[1, 0, 1, 1]);
        check_u8!("lt", a.lt(&b).unwrap(), &[1, 0, 0, 1]);
        check_u8!("le", a.le(&b).unwrap(), &[1, 1, 0, 1]);
        check_u8!("gt", a.gt(&b).unwrap(), &[0, 0, 1, 0]);
        check_u8!("ge", a.ge(&b).unwrap(), &[0, 1, 1, 0]);
    }

    // =====================================================================
    // 2. Logical ops
    // =====================================================================
    println!("--- Logical ---");
    {
        let a = tensor_u8!(&[1u8, 1, 0, 0], &[4]);
        let b = tensor_u8!(&[1u8, 0, 1, 0], &[4]);

        check_u8!("logical_and", a.logical_and(&b).unwrap(), &[1, 0, 0, 0]);
        check_u8!("logical_or", a.logical_or(&b).unwrap(), &[1, 1, 1, 0]);
        check_u8!("logical_not", a.logical_not().unwrap(), &[0, 0, 1, 1]);
        check_u8!("logical_xor", a.logical_xor(&b).unwrap(), &[0, 1, 1, 0]);
    }

    // =====================================================================
    // 3. Arange / Linspace
    // =====================================================================
    println!("--- Arange / Linspace ---");
    {
        let r = ptx_tensor::Tensor::arange(0.0, 5.0, 1.0, &runtime).unwrap();
        check_f32!("arange(0,5,1)", r, &[0.0, 1.0, 2.0, 3.0, 4.0]);

        let r = ptx_tensor::Tensor::arange(1.0, 4.0, 0.5, &runtime).unwrap();
        check_f32!("arange(1,4,0.5)", r, &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);

        let r = ptx_tensor::Tensor::linspace(0.0, 1.0, 5, &runtime).unwrap();
        check_f32!("linspace(0,1,5)", r, &[0.0, 0.25, 0.5, 0.75, 1.0]);

        let r = ptx_tensor::Tensor::linspace(2.0, 4.0, 3, &runtime).unwrap();
        check_f32!("linspace(2,4,3)", r, &[2.0, 3.0, 4.0]);
    }

    // =====================================================================
    // 4. Transpose / Permute
    // =====================================================================
    println!("--- Transpose / Permute ---");
    {
        // 2x3 matrix → transpose to 3x2
        let m = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mt = m.t().unwrap();
        assert_eq!(mt.shape(), &[3, 2]);

        // 2x3 = [[1,2,3],[4,5,6]]
        // transposed 3x2 = [[1,4],[2,5],[3,6]]
        // But since it's a view (non-contiguous), we can't to_vec directly.
        // Let's verify the shape is correct.
        println!("PASS: transpose shape {:?} -> {:?}", m.shape(), mt.shape());
        pass += 1;

        // Permute: [2,3,4] → [4,2,3]
        let t = tensor_f32!(&vec![0.0f32; 24], &[2, 3, 4]);
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        println!("PASS: permute [2,3,4] -> {:?}", p.shape());
        pass += 1;
    }

    // =====================================================================
    // 5. Broadcast binary ops
    // =====================================================================
    println!("--- Broadcast ---");
    {
        // [2,3] + [3] → [2,3]
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = tensor_f32!(&[10.0f32, 20.0, 30.0], &[3]);
        let r = a.broadcast_add(&b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        check_f32!("broadcast_add [2,3]+[3]", r, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

        // [3,1] * [1,4] → [3,4]
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0], &[3, 1]);
        let b = tensor_f32!(&[10.0f32, 20.0, 30.0, 40.0], &[1, 4]);
        let r = a.broadcast_mul(&b).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
        check_f32!("broadcast_mul [3,1]*[1,4]", r,
            &[10.0, 20.0, 30.0, 40.0,
              20.0, 40.0, 60.0, 80.0,
              30.0, 60.0, 90.0, 120.0]);

        // Scalar broadcast: [4] - [1] → [4]
        let a = tensor_f32!(&[10.0f32, 20.0, 30.0, 40.0], &[4]);
        let b = tensor_f32!(&[5.0f32], &[1]);
        let r = a.broadcast_sub(&b).unwrap();
        check_f32!("broadcast_sub [4]-[1]", r, &[5.0, 15.0, 25.0, 35.0]);
    }

    // =====================================================================
    // 6. Cat / Stack
    // =====================================================================
    println!("--- Cat / Stack ---");
    {
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0], &[3]);
        let b = tensor_f32!(&[4.0f32, 5.0], &[2]);
        let r = ptx_tensor::Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(r.shape(), &[5]);
        check_f32!("cat 1D", r, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // 2D cat along dim 0
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor_f32!(&[5.0f32, 6.0], &[1, 2]);
        let r = ptx_tensor::Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        check_f32!("cat 2D dim=0", r, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Stack: create new dim
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0], &[3]);
        let b = tensor_f32!(&[4.0f32, 5.0, 6.0], &[3]);
        let r = ptx_tensor::Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        check_f32!("stack dim=0", r, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // =====================================================================
    // 7. Repeat
    // =====================================================================
    println!("--- Repeat ---");
    {
        let a = tensor_f32!(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let r = a.repeat(&[3, 1]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        check_f32!("repeat [1,3] x [3,1]", r,
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let a = tensor_f32!(&[1.0f32, 2.0], &[2]);
        let r = a.repeat(&[3]).unwrap();
        assert_eq!(r.shape(), &[6]);
        check_f32!("repeat [2] x [3]", r, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    // =====================================================================
    // 8. Masked fill
    // =====================================================================
    println!("--- Masked fill ---");
    {
        let data = tensor_f32!(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
        let mask = tensor_u8!(&[0u8, 1, 0, 1, 0], &[5]);
        let r = data.masked_fill(&mask, -999.0).unwrap();
        check_f32!("masked_fill", r, &[1.0, -999.0, 3.0, -999.0, 5.0]);
    }

    // =====================================================================
    // Combined: comparison → logical → masked_fill pipeline
    // =====================================================================
    println!("--- Pipeline: compare → logic → mask ---");
    {
        let x = tensor_f32!(&[1.0f32, 5.0, 3.0, 7.0, 2.0], &[5]);
        let lo = tensor_f32!(&[2.0f32; 5], &[5]);
        let hi = tensor_f32!(&[6.0f32; 5], &[5]);

        // mask = (x >= lo) AND (x <= hi)  → elements in [2, 6]
        let ge_lo = x.ge(&lo).unwrap();
        let le_hi = x.le(&hi).unwrap();
        let in_range = ge_lo.logical_and(&le_hi).unwrap();
        check_u8!("pipeline: in_range [2,6]", in_range, &[0, 1, 1, 0, 1]);

        // Fill out-of-range with 0
        let out_range = in_range.logical_not().unwrap();
        let result = x.masked_fill(&out_range, 0.0).unwrap();
        check_f32!("pipeline: clamp-mask", result, &[0.0, 5.0, 3.0, 0.0, 2.0]);
    }

    // =====================================================================
    println!("\n{}/{} tests passed", pass, pass + fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
