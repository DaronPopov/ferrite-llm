/// Test Where (conditional select) operation.
use std::sync::Arc;

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

    let mut pass = 0;
    let mut fail = 0;

    // ---------------------------------------------------------------
    // Test 1: where with alternating condition
    //   cond  = [1, 0, 1, 0, 1]
    //   true  = [10, 20, 30, 40, 50]
    //   false = [100, 200, 300, 400, 500]
    //   expected = [10, 200, 30, 400, 50]
    // ---------------------------------------------------------------
    {
        let true_val = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let false_val = ptx_tensor::Tensor::from_slice(
            &[100.0f32, 200.0, 300.0, 400.0, 500.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let cond = ptx_tensor::Tensor::from_slice(
            &[1u8, 0, 1, 0, 1],
            &[5],
            ptx_tensor::DType::U8,
            &runtime,
        ).unwrap();
        let result = true_val.where_cond(&cond, &false_val).unwrap();
        let out = result.to_vec_f32().unwrap();
        let expected = vec![10.0, 200.0, 30.0, 400.0, 50.0];
        if out == expected {
            println!("Test 1 PASS: where alternating = {:?}", out);
            pass += 1;
        } else {
            println!("Test 1 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 2: where all true
    //   cond  = [1, 1, 1]
    //   true  = [10, 20, 30]
    //   false = [100, 200, 300]
    //   expected = [10, 20, 30]
    // ---------------------------------------------------------------
    {
        let true_val = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0],
            &[3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let false_val = ptx_tensor::Tensor::from_slice(
            &[100.0f32, 200.0, 300.0],
            &[3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let cond = ptx_tensor::Tensor::from_slice(
            &[1u8, 1, 1],
            &[3],
            ptx_tensor::DType::U8,
            &runtime,
        ).unwrap();
        let result = true_val.where_cond(&cond, &false_val).unwrap();
        let out = result.to_vec_f32().unwrap();
        let expected = vec![10.0, 20.0, 30.0];
        if out == expected {
            println!("Test 2 PASS: where all true = {:?}", out);
            pass += 1;
        } else {
            println!("Test 2 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 3: where all false
    //   cond  = [0, 0, 0]
    //   true  = [10, 20, 30]
    //   false = [100, 200, 300]
    //   expected = [100, 200, 300]
    // ---------------------------------------------------------------
    {
        let true_val = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0],
            &[3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let false_val = ptx_tensor::Tensor::from_slice(
            &[100.0f32, 200.0, 300.0],
            &[3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let cond = ptx_tensor::Tensor::from_slice(
            &[0u8, 0, 0],
            &[3],
            ptx_tensor::DType::U8,
            &runtime,
        ).unwrap();
        let result = true_val.where_cond(&cond, &false_val).unwrap();
        let out = result.to_vec_f32().unwrap();
        let expected = vec![100.0, 200.0, 300.0];
        if out == expected {
            println!("Test 3 PASS: where all false = {:?}", out);
            pass += 1;
        } else {
            println!("Test 3 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    println!("\n{}/{} tests passed", pass, pass + fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
