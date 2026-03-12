/// Test IndexSelect and ScatterAdd operations.
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
    // Test 1: index_select on 1D — select elements [2, 0] from [10, 20, 30, 40, 50]
    // ---------------------------------------------------------------
    {
        let data = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let ids = ptx_tensor::Tensor::from_slice(
            &[2i32, 0],
            &[2],
            ptx_tensor::DType::I32,
            &runtime,
        ).unwrap();
        let result = data.index_select(0, &ids).unwrap();
        let out = result.to_vec_f32().unwrap();
        let expected = vec![30.0, 10.0];
        if out == expected {
            println!("Test 1 PASS: index_select 1D = {:?}", out);
            pass += 1;
        } else {
            println!("Test 1 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 2: index_select on 2D dim=0 — select rows [1, 0] from [[1,2,3],[4,5,6]]
    // ---------------------------------------------------------------
    {
        let data = ptx_tensor::Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let ids = ptx_tensor::Tensor::from_slice(
            &[1i32, 0],
            &[2],
            ptx_tensor::DType::I32,
            &runtime,
        ).unwrap();
        let result = data.index_select(0, &ids).unwrap();
        let out = result.to_vec_f32().unwrap();
        // Row 1 = [4,5,6], Row 0 = [1,2,3] => [4,5,6,1,2,3]
        let expected = vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0];
        if out == expected {
            println!("Test 2 PASS: index_select 2D dim=0 = {:?}", out);
            pass += 1;
        } else {
            println!("Test 2 FAIL: expected {:?}, got {:?}", expected, out);
            fail += 1;
        }
    }

    // ---------------------------------------------------------------
    // Test 3: scatter_add — accumulate [10, 20, 30] into 5 buckets at indices [1, 3, 1]
    //   bucket 0: 0, bucket 1: 10+30=40, bucket 2: 0, bucket 3: 20, bucket 4: 0
    // ---------------------------------------------------------------
    {
        let src = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0],
            &[3],
            ptx_tensor::DType::F32,
            &runtime,
        ).unwrap();
        let ids = ptx_tensor::Tensor::from_slice(
            &[1i32, 3, 1],
            &[3],
            ptx_tensor::DType::I32,
            &runtime,
        ).unwrap();
        let result = src.scatter_add(0, &ids, 5).unwrap();
        let out = result.to_vec_f32().unwrap();
        let expected = vec![0.0, 40.0, 0.0, 20.0, 0.0];
        if out == expected {
            println!("Test 3 PASS: scatter_add = {:?}", out);
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
