use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("=== TopK OpCode (0xE2) Test ===\n");

    let config = ptx_runtime::PTXStableConfig {
        struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
        abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: 16,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = Arc::new(ptx_runtime::PtxRuntime::with_stable_config(0, Some(config))?);
    runtime.export_for_hook();
    runtime.export_context();

    // Test 1: 1D topk, k=3, largest=true
    // input = [5, 1, 9, 3, 7], top-3 largest => values=[9, 7, 5], indices=[2, 4, 0]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[5.0f32, 1.0, 9.0, 3.0, 7.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        )?;

        let (values, indices) = input.topk(3, 0, true)?;
        let vals = values.to_vec_f32()?;
        let idxs = indices.to_vec_i32()?;

        let expected_vals = vec![9.0, 7.0, 5.0];
        let expected_idxs = vec![2i32, 4, 0];

        println!("Test 1: 1D top-3 largest");
        println!("  input:         [5, 1, 9, 3, 7]");
        println!("  values:        {:?}", vals);
        println!("  expected vals: {:?}", expected_vals);
        println!("  indices:       {:?}", idxs);
        println!("  expected idx:  {:?}", expected_idxs);

        let vals_ok = vals.iter().zip(expected_vals.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        let idx_ok = idxs == expected_idxs;
        println!("  PASS: {}\n", vals_ok && idx_ok);
        assert!(vals_ok && idx_ok, "1D top-3 largest failed!");
    }

    // Test 2: 1D topk, k=2, largest=false (smallest)
    // input = [5, 1, 9, 3, 7], top-2 smallest => values=[1, 3], indices=[1, 3]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[5.0f32, 1.0, 9.0, 3.0, 7.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        )?;

        let (values, indices) = input.topk(2, 0, false)?;
        let vals = values.to_vec_f32()?;
        let idxs = indices.to_vec_i32()?;

        let expected_vals = vec![1.0, 3.0];
        let expected_idxs = vec![1i32, 3];

        println!("Test 2: 1D top-2 smallest");
        println!("  values:        {:?}", vals);
        println!("  expected vals: {:?}", expected_vals);
        println!("  indices:       {:?}", idxs);
        println!("  expected idx:  {:?}", expected_idxs);

        let vals_ok = vals.iter().zip(expected_vals.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        let idx_ok = idxs == expected_idxs;
        println!("  PASS: {}\n", vals_ok && idx_ok);
        assert!(vals_ok && idx_ok, "1D top-2 smallest failed!");
    }

    // Test 3: 2D topk along dim=1, k=2, largest=true
    // input = [[4, 2, 8, 6], [3, 9, 1, 5]], shape [2, 4]
    // Row 0: top-2 largest => [8, 6] at indices [2, 3]
    // Row 1: top-2 largest => [9, 5] at indices [1, 3]
    // output shape = [2, 2]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[4.0f32, 2.0, 8.0, 6.0, 3.0, 9.0, 1.0, 5.0],
            &[2, 4],
            ptx_tensor::DType::F32,
            &runtime,
        )?;

        let (values, indices) = input.topk(2, 1, true)?;
        let vals = values.to_vec_f32()?;
        let idxs = indices.to_vec_i32()?;

        // [8, 6, 9, 5]
        let expected_vals = vec![8.0, 6.0, 9.0, 5.0];
        // [2, 3, 1, 3]
        let expected_idxs = vec![2i32, 3, 1, 3];

        println!("Test 3: 2D top-2 largest along dim=1");
        println!("  values:        {:?}", vals);
        println!("  expected vals: {:?}", expected_vals);
        println!("  indices:       {:?}", idxs);
        println!("  expected idx:  {:?}", expected_idxs);

        let vals_ok = vals.iter().zip(expected_vals.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        let idx_ok = idxs == expected_idxs;
        println!("  PASS: {}\n", vals_ok && idx_ok);
        assert!(vals_ok && idx_ok, "2D top-2 largest dim=1 failed!");
    }

    // Test 4: 2D topk along dim=0, k=1, largest=true
    // input = [[4, 2, 8], [3, 9, 1]], shape [2, 3]
    // Column-wise top-1: [4, 9, 8] at indices [0, 1, 0]
    // output shape = [1, 3]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[4.0f32, 2.0, 8.0, 3.0, 9.0, 1.0],
            &[2, 3],
            ptx_tensor::DType::F32,
            &runtime,
        )?;

        let (values, indices) = input.topk(1, 0, true)?;
        let vals = values.to_vec_f32()?;
        let idxs = indices.to_vec_i32()?;

        let expected_vals = vec![4.0, 9.0, 8.0];
        let expected_idxs = vec![0i32, 1, 0];

        println!("Test 4: 2D top-1 largest along dim=0");
        println!("  values:        {:?}", vals);
        println!("  expected vals: {:?}", expected_vals);
        println!("  indices:       {:?}", idxs);
        println!("  expected idx:  {:?}", expected_idxs);

        let vals_ok = vals.iter().zip(expected_vals.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        let idx_ok = idxs == expected_idxs;
        println!("  PASS: {}\n", vals_ok && idx_ok);
        assert!(vals_ok && idx_ok, "2D top-1 largest dim=0 failed!");
    }

    println!("RESULT mode=test_topk");
    println!("RESULT all_tests=PASSED");
    println!("RESULT opcode=0xE2");

    Ok(())
}
