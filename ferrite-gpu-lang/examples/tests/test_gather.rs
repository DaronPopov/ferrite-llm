use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("=== Gather OpCode (0xB0) Test ===\n");

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

    // Test 1: gather from 1D
    // input = [10, 20, 30, 40, 50], indices = [1, 3, 0, 4]
    // output = [20, 40, 10, 50]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0],
            &[5],
            ptx_tensor::DType::F32,
            &runtime,
        )?;
        let indices = ptx_tensor::Tensor::from_slice(
            &[1i32, 3, 0, 4],
            &[4],
            ptx_tensor::DType::I32,
            &runtime,
        )?;

        let output = input.gather(0, &indices)?;
        let result = output.to_vec_f32()?;
        let expected = vec![20.0, 40.0, 10.0, 50.0];

        println!("Test 1: 1D gather");
        println!("  input:    [10, 20, 30, 40, 50]");
        println!("  indices:  [1, 3, 0, 4]");
        println!("  output:   {:?}", result);
        println!("  expected: {:?}", expected);

        let ok = result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "1D gather failed!");
    }

    // Test 2: 2D gather along dim=1
    // input = [[10, 20, 30], [40, 50, 60]], shape [2, 3]
    // indices = [[2, 0], [1, 2]], shape [2, 2]
    // output[0] = [input[0][2], input[0][0]] = [30, 10]
    // output[1] = [input[1][1], input[1][2]] = [50, 60]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            ptx_tensor::DType::F32,
            &runtime,
        )?;
        let indices = ptx_tensor::Tensor::from_slice(
            &[2i32, 0, 1, 2],
            &[2, 2],
            ptx_tensor::DType::I32,
            &runtime,
        )?;

        let output = input.gather(1, &indices)?;
        let result = output.to_vec_f32()?;
        let expected = vec![30.0, 10.0, 50.0, 60.0];

        println!("Test 2: 2D gather along dim=1");
        println!("  output:   {:?}", result);
        println!("  expected: {:?}", expected);

        let ok = result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "2D gather dim=1 failed!");
    }

    // Test 3: 2D gather along dim=0
    // input = [[10, 20, 30], [40, 50, 60]], shape [2, 3]
    // indices = [[1, 0, 1]], shape [1, 3]
    // output[0] = [input[1][0], input[0][1], input[1][2]] = [40, 20, 60]
    {
        let input = ptx_tensor::Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            ptx_tensor::DType::F32,
            &runtime,
        )?;
        let indices = ptx_tensor::Tensor::from_slice(
            &[1i32, 0, 1],
            &[1, 3],
            ptx_tensor::DType::I32,
            &runtime,
        )?;

        let output = input.gather(0, &indices)?;
        let result = output.to_vec_f32()?;
        let expected = vec![40.0, 20.0, 60.0];

        println!("Test 3: 2D gather along dim=0");
        println!("  output:   {:?}", result);
        println!("  expected: {:?}", expected);

        let ok = result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  PASS: {}\n", ok);
        assert!(ok, "2D gather dim=0 failed!");
    }

    println!("RESULT mode=test_gather");
    println!("RESULT all_tests=PASSED");
    println!("RESULT opcode=0xB0");

    Ok(())
}
