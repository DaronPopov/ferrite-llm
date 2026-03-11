//! Verify Candle kernels produce correct mathematical results with PTX-OS TLSF
//!
//! This test validates that kernels actually compute the right values,
//! not just run without crashing.

use ptx_kernels::{GuardedBuffer, KernelContext, safe_api};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Verifying Candle Kernel Mathematics\n");

    unsafe {
        // Initialize PTX-OS TLSF runtime
        let runtime_ptr = ptx_sys::gpu_hot_init(0, std::ptr::null());
        if runtime_ptr.is_null() {
            panic!("Failed to initialize PTX-OS runtime");
        }
        println!("✓ PTX-OS runtime initialized\n");

        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);
        let ctx = KernelContext::new(runtime_ptr, stream)?;

        // Small test size for easy verification
        const N: usize = 10;
        const BYTES: usize = N * std::mem::size_of::<f32>();

        // Allocate via TLSF
        let d_input = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_output = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_temp = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);

        if d_input.is_null() || d_output.is_null() || d_temp.is_null() {
            panic!("TLSF allocation failed");
        }

        let ig = GuardedBuffer::new(d_input, BYTES, runtime_ptr)?;
        let og = GuardedBuffer::new(d_output, BYTES, runtime_ptr)?;
        let tg = GuardedBuffer::new(d_temp, BYTES, runtime_ptr)?;

        let mut passed = 0;
        let mut failed = 0;

        // ========================================================================
        // Test 1: ReLU - max(0, x)
        // ========================================================================
        println!("🧪 Test 1: ReLU (max(0, x))");
        let input_relu = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -3.0, 3.0, 0.1];
        let expected_relu = vec![0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5, 0.0, 3.0, 0.1];

        ptx_sys::cudaMemcpy(d_input, input_relu.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::relu(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_relu = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_relu.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_relu);
        println!("   Expected: {:?}", expected_relu);
        println!("   Got:      {:?}", output_relu);

        let relu_match = output_relu.iter()
            .zip(expected_relu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if relu_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 2: Absolute Value - |x|
        // ========================================================================
        println!("🧪 Test 2: Absolute Value (|x|)");
        let input_abs = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -3.0, 3.0, -0.1];
        let expected_abs = vec![2.0f32, 1.0, 0.0, 1.0, 2.0, 0.5, 0.5, 3.0, 3.0, 0.1];

        ptx_sys::cudaMemcpy(d_input, input_abs.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::abs(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_abs = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_abs.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_abs);
        println!("   Expected: {:?}", expected_abs);
        println!("   Got:      {:?}", output_abs);

        let abs_match = output_abs.iter()
            .zip(expected_abs.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if abs_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 3: Square Root - √x
        // ========================================================================
        println!("🧪 Test 3: Square Root (√x)");
        let input_sqrt = vec![0.0f32, 1.0, 4.0, 9.0, 16.0, 25.0, 0.25, 0.01, 100.0, 2.0];
        let expected_sqrt = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 0.1, 10.0, 1.414213];

        ptx_sys::cudaMemcpy(d_input, input_sqrt.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::sqrt(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_sqrt = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_sqrt.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_sqrt);
        println!("   Expected: {:?}", expected_sqrt);
        println!("   Got:      {:?}", output_sqrt);

        let sqrt_match = output_sqrt.iter()
            .zip(expected_sqrt.iter())
            .all(|(a, b)| (a - b).abs() < 1e-4);

        if sqrt_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 4: Exponential - e^x (small values to avoid overflow)
        // ========================================================================
        println!("🧪 Test 4: Exponential (e^x)");
        let input_exp = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 0.0];
        let expected_exp: Vec<f32> = input_exp.iter().map(|x| x.exp()).collect();

        ptx_sys::cudaMemcpy(d_input, input_exp.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::exp(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_exp = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_exp.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_exp);
        println!("   Expected: {:?}", expected_exp);
        println!("   Got:      {:?}", output_exp);

        let exp_match = output_exp.iter()
            .zip(expected_exp.iter())
            .all(|(a, b)| (a - b).abs() < 1e-4);

        if exp_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 5: Tanh - hyperbolic tangent
        // ========================================================================
        println!("🧪 Test 5: Tanh (hyperbolic tangent)");
        let input_tanh = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.1, -0.1, 3.0];
        let expected_tanh: Vec<f32> = input_tanh.iter().map(|x| x.tanh()).collect();

        ptx_sys::cudaMemcpy(d_input, input_tanh.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::tanh(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_tanh = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_tanh.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_tanh);
        println!("   Expected: {:?}", expected_tanh);
        println!("   Got:      {:?}", output_tanh);

        let tanh_match = output_tanh.iter()
            .zip(expected_tanh.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if tanh_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 6: Binary Addition - a + b
        // ========================================================================
        println!("🧪 Test 6: Binary Addition (a + b)");
        let input_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, 0.0, 0.5, 10.0];
        let input_b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 0.0, 0.5, -5.0];
        let expected_add = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 0.0, 0.0, 0.0, 1.0, 5.0];

        ptx_sys::cudaMemcpy(d_input, input_a.as_ptr() as *const _, BYTES, 1);
        ptx_sys::cudaMemcpy(d_temp, input_b.as_ptr() as *const _, BYTES, 1);

        safe_api::binary::add(&ig, &tg, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_add = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_add.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input A:  {:?}", input_a);
        println!("   Input B:  {:?}", input_b);
        println!("   Expected: {:?}", expected_add);
        println!("   Got:      {:?}", output_add);

        let add_match = output_add.iter()
            .zip(expected_add.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if add_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 7: Binary Multiplication - a * b
        // ========================================================================
        println!("🧪 Test 7: Binary Multiplication (a * b)");
        let input_mul_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, 0.0, 0.5, 10.0];
        let input_mul_b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 2.0, -3.0, 5.0, 4.0, 0.1];
        let expected_mul = vec![2.0f32, 6.0, 12.0, 20.0, 30.0, -2.0, 6.0, 0.0, 2.0, 1.0];

        ptx_sys::cudaMemcpy(d_input, input_mul_a.as_ptr() as *const _, BYTES, 1);
        ptx_sys::cudaMemcpy(d_temp, input_mul_b.as_ptr() as *const _, BYTES, 1);

        safe_api::binary::mul(&ig, &tg, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_mul = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_mul.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input A:  {:?}", input_mul_a);
        println!("   Input B:  {:?}", input_mul_b);
        println!("   Expected: {:?}", expected_mul);
        println!("   Got:      {:?}", output_mul);

        let mul_match = output_mul.iter()
            .zip(expected_mul.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if mul_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Test 8: Sigmoid - 1 / (1 + e^(-x))
        // ========================================================================
        println!("🧪 Test 8: Sigmoid (1 / (1 + e^(-x)))");
        let input_sigmoid = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -3.0, 3.0, 0.0];
        let expected_sigmoid: Vec<f32> = input_sigmoid.iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        ptx_sys::cudaMemcpy(d_input, input_sigmoid.as_ptr() as *const _, BYTES, 1);
        safe_api::unary::sigmoid(&ig, &og, N, &ctx)?;
        ctx.sync()?;

        let mut output_sigmoid = vec![0.0f32; N];
        ptx_sys::cudaMemcpy(output_sigmoid.as_mut_ptr() as *mut _, d_output, BYTES, 2);

        println!("   Input:    {:?}", input_sigmoid);
        println!("   Expected: {:?}", expected_sigmoid);
        println!("   Got:      {:?}", output_sigmoid);

        let sigmoid_match = output_sigmoid.iter()
            .zip(expected_sigmoid.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if sigmoid_match {
            println!("   ✅ PASS\n");
            passed += 1;
        } else {
            println!("   ❌ FAIL\n");
            failed += 1;
        }

        // ========================================================================
        // Final Summary
        // ========================================================================
        println!("═══════════════════════════════════════════════════");
        println!("📊 Final Results");
        println!("═══════════════════════════════════════════════════");
        println!("   ✅ Passed: {}/8", passed);
        println!("   ❌ Failed: {}/8", failed);
        println!("═══════════════════════════════════════════════════\n");

        // Cleanup
        ptx_sys::gpu_hot_free(runtime_ptr, d_input);
        ptx_sys::gpu_hot_free(runtime_ptr, d_output);
        ptx_sys::gpu_hot_free(runtime_ptr, d_temp);
        ptx_sys::gpu_hot_shutdown(runtime_ptr);

        if failed == 0 {
            println!("🎉 SUCCESS: All kernels compute correct values via safe API!");
            println!("   Safe API + PTX-OS TLSF = Fully Validated! ✅");
        } else {
            println!("⚠️  FAILURE: Some kernels produced incorrect results");
            std::process::exit(1);
        }
    }

    Ok(())
}
