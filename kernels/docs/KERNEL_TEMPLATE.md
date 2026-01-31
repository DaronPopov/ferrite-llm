# Custom CUDA Kernel Template

This is a step-by-step template for adding your own custom CUDA kernels to Ferrite.

## Step 1: Write Your CUDA Kernel

Create `kernels/my_custom_kernel.cu`:

```cuda
#include <cuda_runtime.h>

// Your kernel implementation
__global__ void my_kernel_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Your custom operation
        output[idx] = input[idx] * 2.0f;  // Example: double the values
    }
}

// C interface for Rust FFI
extern "C" {
    void my_kernel_forward(
        const float* input,
        float* output,
        int n,
        cudaStream_t stream
    ) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;

        my_kernel_impl<<<blocks, threads, 0, stream>>>(input, output, n);
    }
}
```

## Step 2: Build Automatically

The kernel will be compiled automatically:

```bash
cargo build --features cuda
```

The build system finds all `.cu` files and compiles them to PTX automatically!

## Step 3: Create Rust Wrapper

Create `src/my_kernel.rs` (follow this exact pattern):

```rust
// My Custom Kernel - Rust integration
//
// This follows the standard Ferrite kernel pattern

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig, DevicePtr};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;

/// My custom operation
///
/// # Arguments
/// * `input` - Input tensor [any shape]
///
/// # Returns
/// Output tensor with same shape as input
pub fn my_custom_op(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(input.device(), Device::Cuda(_)) {
            return my_custom_op_cuda(input);
        }
    }

    // CPU fallback
    my_custom_op_cpu(input)
}

/// CUDA implementation - Follow this pattern!
#[cfg(feature = "cuda")]
fn my_custom_op_cuda(input: &Tensor) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaStorage;

    // STEP 1: Load compiled PTX
    const MY_KERNEL_PTX: &str = include_str!(concat!(
        env!("OUT_DIR"),
        "/kernels/my_custom_kernel.ptx"
    ));

    // STEP 2: Get CUDA device
    let cuda_device = match input.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("Expected CUDA device"),
    };

    let dev = cuda_device.cuda_device();

    // STEP 3: Load module and get kernel function
    let module = dev
        .load_ptx(
            MY_KERNEL_PTX.as_bytes().to_vec(),
            "my_kernel_module",
            &["my_kernel_forward"],
        )
        .w()
        .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;

    let func = module
        .get_func("my_kernel_forward")
        .ok_or_else(|| candle_core::Error::Msg("Kernel not found".into()))?;

    // STEP 4: Get input dimensions
    let n = input.elem_count();

    // STEP 5: Get CUDA pointer from input tensor
    let input_storage = input.storage_and_layout().0;

    let input_ptr = match input_storage {
        candle_core::Storage::Cuda(CudaStorage { slice, .. }) => {
            *slice.device_ptr() as *const f32
        }
        _ => candle_core::bail!("Expected CUDA storage"),
    };

    // STEP 6: Allocate output tensor
    let out_slice = unsafe {
        dev.alloc::<f32>(n)
            .w()
            .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
    };

    let out_ptr = *out_slice.device_ptr() as *mut f32;

    // STEP 7: Configure launch parameters
    let threads = 256;
    let blocks = (n as u32 + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // STEP 8: Prepare kernel parameters (must match C function signature!)
    let params = (
        input_ptr,
        out_ptr,
        n as i32,
    );

    // STEP 9: Get stream and launch kernel
    let stream = dev.fork_default_stream().w()?;

    unsafe {
        func.clone()
            .launch(&cfg, params)
            .w()
            .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
    }

    // STEP 10: Wait for completion
    stream.synchronize().w()?;

    // STEP 11: Create output tensor
    let shape = input.shape().clone();
    let cuda_storage = CudaStorage {
        slice: out_slice,
        device: cuda_device.clone(),
    };

    let storage = candle_core::Storage::Cuda(cuda_storage);
    Tensor::from_storage(storage, shape, false)
}

/// CPU fallback
fn my_custom_op_cpu(input: &Tensor) -> Result<Tensor> {
    // Implement CPU version here
    input.affine(2.0, 0.0)  // Example: 2x + 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_custom_op_cpu() {
        let device = Device::Cpu;
        let input = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let output = my_custom_op(&input).unwrap();

        let result = output.to_vec1::<f32>().unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }
}
```

## Step 4: Add Module to lib.rs

```rust
// In src/lib.rs
pub mod my_kernel;

// And export it
pub use my_kernel::my_custom_op;
```

## Step 5: Use Your Kernel!

```rust
use ferrite::my_custom_op;

let input = Tensor::randn(0f32, 1.0, (1000,), &device)?;
let output = my_custom_op(&input)?;
```

## Complete Example Structure

```
ferrite/
├── kernels/
│   ├── my_custom_kernel.cu          ← Your CUDA code
│   └── build_kernels.sh             ← Auto-compiles it
├── src/
│   ├── my_kernel.rs                 ← Your Rust wrapper
│   └── lib.rs                       ← Export it
└── Cargo.toml                       ← No changes needed!
```

## Key Points

1. **PTX Loading**: Always use `include_str!(concat!(env!("OUT_DIR"), "/kernels/...ptx"))`
2. **Module Name**: Use unique name for each kernel to avoid conflicts
3. **Kernel Parameters**: Must exactly match the C function signature
4. **Stream Sync**: Always call `stream.synchronize()` before returning
5. **CPU Fallback**: Always provide a CPU implementation
6. **Error Handling**: Use `.w()` to wrap cudarc errors into Candle errors

## Common Patterns

### Pattern 1: Element-wise Operations

```cuda
__global__ void elementwise_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Your operation
    }
}
```

### Pattern 2: Reduction

```cuda
__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
```

### Pattern 3: Matrix Operations

```cuda
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

## Debugging Tips

1. **Check PTX exists**:
   ```bash
   ls target/release/build/ferrite-*/out/kernels/
   ```

2. **Validate PTX**:
   ```bash
   cat target/release/build/ferrite-*/out/kernels/my_kernel.ptx
   ```

3. **Enable debug mode**:
   ```bash
   DEBUG=1 cargo build --features cuda
   ```

4. **Profile with Nsight**:
   ```bash
   nsys profile ./target/release/my_program
   ```

## See Also

- **Reference Implementation**: `src/flash_attention.rs`
- **Example Kernels**: `kernels/example_kernel.cu`
- **Build System**: `kernels/BUILD_SUMMARY.md`
- **Custom Kernels Guide**: `kernels/CUSTOM_KERNELS.md`

---

**This is the stable pattern for all Ferrite custom kernels! 🚀**

Follow these steps and your kernel will integrate seamlessly.
