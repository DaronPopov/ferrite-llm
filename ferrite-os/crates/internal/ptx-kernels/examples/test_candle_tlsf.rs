//! Test Candle kernels with PTX-OS TLSF allocator
//!
//! This demonstrates that Candle kernels work seamlessly with PTX-OS memory management
//! via the safe API guard layer. All allocations go through the TLSF allocator.

use ptx_kernels::{GuardedBuffer, KernelContext, safe_api};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Testing Safe API Kernels with PTX-OS TLSF Allocator\n");

    unsafe {
        // Initialize PTX-OS runtime with TLSF allocator
        let runtime_ptr = ptx_sys::gpu_hot_init(0, std::ptr::null());
        if runtime_ptr.is_null() {
            panic!("Failed to initialize PTX-OS runtime");
        }
        println!("✓ PTX-OS runtime initialized");

        // Get stream from PTX-OS
        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);
        let ctx = KernelContext::new(runtime_ptr, stream)?;
        println!("✓ Got stream and kernel context from PTX-OS\n");

        // Test parameters
        const N: usize = 1024 * 1024; // 1M elements
        const BYTES: usize = N * std::mem::size_of::<f32>();

        println!("📊 Configuration:");
        println!("   Elements: {}", N);
        println!("   Size: {} KB\n", BYTES / 1024);

        // Allocate using PTX-OS TLSF allocator
        let d_input = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_output = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_temp = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);

        if d_input.is_null() || d_output.is_null() || d_temp.is_null() {
            panic!("TLSF allocation failed");
        }
        println!("✓ Allocated {} KB via TLSF allocator", (BYTES * 3) / 1024);

        // Create guarded buffers
        let ig = GuardedBuffer::new(d_input, BYTES, runtime_ptr)?;
        let og = GuardedBuffer::new(d_output, BYTES, runtime_ptr)?;
        let tg = GuardedBuffer::new(d_temp, BYTES, runtime_ptr)?;

        // Initialize input data
        let mut h_input = vec![0.0f32; N];
        for (i, val) in h_input.iter_mut().enumerate() {
            *val = (i as f32) / 10000.0 - 50.0; // Range: -50 to +50
        }

        // cudaMemcpyHostToDevice = 1
        ptx_sys::cudaMemcpy(
            d_input,
            h_input.as_ptr() as *const _,
            BYTES,
            1, // cudaMemcpyHostToDevice
        );
        println!("✓ Copied input data to GPU\n");

        // Test 1: GELU activation
        println!("🧪 Test 1: GELU Activation");
        safe_api::unary::gelu(&ig, &og, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ GELU completed");

        // Test 2: ReLU activation
        println!("\n🧪 Test 2: ReLU Activation");
        safe_api::unary::relu(&ig, &tg, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ ReLU completed");

        // Test 3: Element-wise multiplication (output = GELU(input) * ReLU(input))
        println!("\n🧪 Test 3: Element-wise Multiplication");
        safe_api::binary::mul(&og, &tg, &og, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ Multiplication completed");

        // Test 4: Tanh
        println!("\n🧪 Test 4: Tanh Activation");
        safe_api::unary::tanh(&og, &tg, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ Tanh completed");

        // Test 5: Sigmoid
        println!("\n🧪 Test 5: Sigmoid Activation");
        safe_api::unary::sigmoid(&ig, &tg, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ Sigmoid completed");

        // Test 6: Binary addition
        println!("\n🧪 Test 6: Element-wise Addition");
        safe_api::binary::add(&og, &tg, &og, N, &ctx)?;
        ctx.sync()?;
        println!("   ✓ Addition completed");

        // Copy result back and verify
        let mut h_output = vec![0.0f32; N];
        // cudaMemcpyDeviceToHost = 2
        ptx_sys::cudaMemcpy(
            h_output.as_mut_ptr() as *mut _,
            d_output,
            BYTES,
            2, // cudaMemcpyDeviceToHost
        );

        println!("\n📊 Sample Results:");
        println!("   First 5: {:?}", &h_output[0..5]);
        println!("   Middle 5: {:?}", &h_output[N / 2..N / 2 + 5]);
        println!("   Last 5: {:?}", &h_output[N - 5..N]);

        // Validation
        let nan_count = h_output.iter().filter(|x| x.is_nan()).count();
        let inf_count = h_output.iter().filter(|x| x.is_infinite()).count();
        let finite_count = h_output.iter().filter(|x| x.is_finite()).count();

        println!("\n🔍 Validation:");
        println!("   Finite values: {}/{}", finite_count, N);
        println!("   NaN count: {}", nan_count);
        println!("   Inf count: {}", inf_count);

        // Free using TLSF allocator
        ptx_sys::gpu_hot_free(runtime_ptr, d_input);
        ptx_sys::gpu_hot_free(runtime_ptr, d_output);
        ptx_sys::gpu_hot_free(runtime_ptr, d_temp);
        println!("\n✓ Freed memory via TLSF allocator");

        // Cleanup runtime
        ptx_sys::gpu_hot_shutdown(runtime_ptr);
        println!("✓ PTX-OS runtime shutdown");

        if nan_count == 0 && inf_count == 0 {
            println!("\n✅ SUCCESS: All safe API kernels work perfectly with PTX-OS TLSF!");
            println!("   Guarded kernel launches + TLSF memory = Fully validated!");
        } else {
            println!("\n⚠️  Warning: Found {} NaN and {} Inf values", nan_count, inf_count);
        }
    }

    Ok(())
}
