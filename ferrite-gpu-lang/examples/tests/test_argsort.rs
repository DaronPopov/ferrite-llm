/// Test Argsort operation.
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
    // Test 1: argsort ascending [30, 10, 50, 20, 40]
    //   sorted: [10, 20, 30, 40, 50] → indices [1, 3, 0, 4, 2]
    // ---------------------------------------------------------------
    {
        let data = ptx_tensor::Tensor::from_slice(
            &[30.0f32, 10.0, 50.0, 20.0, 40.0],
            &[1, 5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let result = data.argsort(-1, true).unwrap();
        let out = result.to_vec_u32().unwrap();
        let expected = vec![1u32, 3, 0, 4, 2];
        if out == expected {
            println!("Test 1 PASS: argsort ascending = {:?}", out);
            pass += 1;
        } else {
            println!("Test 1 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 2: argsort descending [30, 10, 50, 20, 40]
    //   sorted desc: [50, 40, 30, 20, 10] → indices [2, 4, 0, 3, 1]
    // ---------------------------------------------------------------
    {
        let data = ptx_tensor::Tensor::from_slice(
            &[30.0f32, 10.0, 50.0, 20.0, 40.0],
            &[1, 5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let result = data.argsort(-1, false).unwrap();
        let out = result.to_vec_u32().unwrap();
        let expected = vec![2u32, 4, 0, 3, 1];
        if out == expected {
            println!("Test 2 PASS: argsort descending = {:?}", out);
            pass += 1;
        } else {
            println!("Test 2 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 3: argsort 2D — two rows, ascending
    //   [[5, 3, 1, 4, 2],
    //    [1, 5, 2, 4, 3]]
    //   Row 0 sorted: [1,2,3,4,5] → indices [2,4,1,3,0]
    //   Row 1 sorted: [1,2,3,4,5] → indices [0,2,4,3,1]
    // ---------------------------------------------------------------
    {
        let data = ptx_tensor::Tensor::from_slice(
            &[5.0f32, 3.0, 1.0, 4.0, 2.0,
              1.0, 5.0, 2.0, 4.0, 3.0],
            &[2, 5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let result = data.argsort(-1, true).unwrap();
        let out = result.to_vec_u32().unwrap();
        let expected = vec![2u32, 4, 1, 3, 0,
                            0, 2, 4, 3, 1];
        if out == expected {
            println!("Test 3 PASS: argsort 2D ascending = {:?}", out);
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
