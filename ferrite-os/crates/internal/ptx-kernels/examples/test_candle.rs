//! Test Candle kernels with PTX-OS runtime via safe API
use ptx_kernels::{GuardedBuffer, KernelContext, check_cuda, sync_stream, safe_api};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Candle F32 Kernels with PTX-OS\n");

    unsafe {
        // Initialize PTX-OS runtime with TLSF allocator
        let runtime_ptr = ptx_sys::gpu_hot_init(0, std::ptr::null());
        if runtime_ptr.is_null() {
            panic!("Failed to initialize PTX-OS runtime");
        }
        println!("PTX-OS runtime initialized");

        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);
        let ctx = KernelContext::new(runtime_ptr, stream)?;

        // Test parameters
        const N: usize = 1024 * 1024; // 1M elements
        const BYTES: usize = N * std::mem::size_of::<f32>();

        println!("Configuration:");
        println!("   Elements: {}", N);
        println!("   Size: {} KB\n", BYTES / 1024);

        // Allocate GPU memory via TLSF
        let d_input = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_output = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_temp = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);

        if d_input.is_null() || d_output.is_null() || d_temp.is_null() {
            panic!("TLSF allocation failed");
        }
        println!("Allocated GPU memory via TLSF");

        let ig = GuardedBuffer::new(d_input, BYTES, runtime_ptr)?;
        let og = GuardedBuffer::new(d_output, BYTES, runtime_ptr)?;
        let tg = GuardedBuffer::new(d_temp, BYTES, runtime_ptr)?;

        // Initialize input data
        let mut h_input = vec![0.0f32; N];
        for (i, val) in h_input.iter_mut().enumerate() {
            *val = (i as f32) / 1000.0 - 512.0; // Range: -512 to +512
        }

        check_cuda(
            ptx_sys::cudaMemcpy(
                d_input,
                h_input.as_ptr() as *const _,
                BYTES,
                1, // cudaMemcpyHostToDevice
            ),
            "Failed to copy input to device",
        )?;

        println!("Initialized input data\n");

        // Test 1: GELU
        println!("Test 1: GELU Activation");
        safe_api::unary::gelu(&ig, &og, N, &ctx)?;
        sync_stream(stream)?;
        println!("   GELU completed");

        // Test 2: ReLU
        println!("\nTest 2: ReLU Activation");
        safe_api::unary::relu(&ig, &tg, N, &ctx)?;
        sync_stream(stream)?;
        println!("   ReLU completed");

        // Test 3: Element-wise multiply
        println!("\nTest 3: Element-wise Multiplication");
        safe_api::binary::mul(&og, &tg, &og, N, &ctx)?;
        sync_stream(stream)?;
        println!("   Multiplication completed");

        // Test 4: Tanh
        println!("\nTest 4: Tanh Activation");
        safe_api::unary::tanh(&ig, &tg, N, &ctx)?;
        sync_stream(stream)?;
        println!("   Tanh completed");

        // Test 5: Binary add
        println!("\nTest 5: Element-wise Addition");
        safe_api::binary::add(&og, &tg, &og, N, &ctx)?;
        sync_stream(stream)?;
        println!("   Addition completed");

        // Test 6: Exp
        println!("\nTest 6: Exponential");
        let mut h_small_input = vec![0.0f32; 1024];
        for (i, val) in h_small_input.iter_mut().enumerate() {
            *val = (i as f32) / 1000.0; // Small values to avoid overflow
        }
        check_cuda(
            ptx_sys::cudaMemcpy(
                d_input,
                h_small_input.as_ptr() as *const _,
                1024 * std::mem::size_of::<f32>(),
                1, // cudaMemcpyHostToDevice
            ),
            "Failed to copy small input",
        )?;
        safe_api::unary::exp(&ig, &tg, 1024, &ctx)?;
        sync_stream(stream)?;
        println!("   Exp completed");

        // Copy result back and verify
        let mut h_output = vec![0.0f32; N];
        check_cuda(
            ptx_sys::cudaMemcpy(
                h_output.as_mut_ptr() as *mut _,
                d_output,
                BYTES,
                2, // cudaMemcpyDeviceToHost
            ),
            "Failed to copy output to host",
        )?;

        println!("\nSample Results:");
        println!("   First 5 values: {:?}", &h_output[0..5]);
        println!("   Middle 5 values: {:?}", &h_output[N / 2..N / 2 + 5]);
        println!("   Last 5 values: {:?}", &h_output[N - 5..N]);

        // Verify no NaNs or Infs
        let nan_count = h_output.iter().filter(|x| x.is_nan()).count();
        let inf_count = h_output.iter().filter(|x| x.is_infinite()).count();
        println!("\nValidation:");
        println!("   NaN count: {}", nan_count);
        println!("   Inf count: {}", inf_count);

        if nan_count == 0 && inf_count == 0 {
            println!("\nAll Candle kernels working correctly via safe API!");
        } else {
            println!("\nWarning: Found NaN or Inf values");
        }

        // Cleanup via TLSF
        ptx_sys::gpu_hot_free(runtime_ptr, d_input);
        ptx_sys::gpu_hot_free(runtime_ptr, d_output);
        ptx_sys::gpu_hot_free(runtime_ptr, d_temp);
        ptx_sys::gpu_hot_shutdown(runtime_ptr);

        println!("\nCleaned up resources");
    }

    Ok(())
}
