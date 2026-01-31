# Adding Custom CUDA Kernels to Ferrite

Ferrite has a modular kernel build system that makes it easy to add your own CUDA kernels!

## Quick Start

1. **Write your CUDA kernel** in `kernels/my_kernel.cu`
2. **Build Ferrite** with `cargo build --features cuda`
3. **Your kernel is automatically compiled!** ✨

That's it! The build system finds all `.cu` files and compiles them automatically.

## Example: Adding a Custom Kernel

### Step 1: Create Your Kernel File

`kernels/vector_add.cu`:

```cuda
// Simple vector addition kernel - Example custom kernel

#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C interface for Rust FFI
extern "C" {
    void vector_add(
        const float* a,
        const float* b,
        float* c,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);
    }
}
```

### Step 2: Build

```bash
cargo build --release --features cuda
```

You'll see:
```
Ferrite CUDA Kernel Build System
✓ Found NVCC: /usr/local/cuda/bin/nvcc
✓ Auto-detected GPU architecture: sm_86
[1] Compiling: flash_attention.cu
    ✓ Success → flash_attention.ptx (42K)
[2] Compiling: vector_add.cu
    ✓ Success → vector_add.ptx (8K)
✓ Custom CUDA kernels compiled successfully
```

### Step 3: Use from Rust

`src/my_kernel.rs`:

```rust
use candle_core::{Device, Tensor};

#[cfg(feature = "cuda")]
pub fn vector_add_cuda(a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Load the compiled PTX
    let ptx = include_str!(concat!(env!("KERNEL_OUTPUT_DIR"), "/vector_add.ptx"));

    // Load module and get function
    // (Using cudarc or similar CUDA library)
    // ... kernel launch code here ...

    Ok(result)
}
```

## Kernel Build System

### Automatic Discovery

The build system automatically finds and compiles:
- All `*.cu` files in `kernels/`
- Generates PTX for each kernel
- Creates Rust constants for PTX paths

### Configuration

Edit `kernels/kernel_config.toml`:

```toml
[kernels.my_kernel]
enabled = true
description = "My custom operation"
file = "my_kernel.cu"
exports = ["my_kernel_forward"]
```

### Architecture-Specific Builds

The build system detects your GPU and compiles for the right architecture:

```bash
# Auto-detect (recommended)
cargo build --features cuda

# Specific architecture
CUDA_ARCH=sm_89 cargo build --features cuda
```

Supported architectures:
- `sm_86` - RTX 30xx (Ampere)
- `sm_89` - RTX 40xx (Ada Lovelace)
- `sm_80` - A100
- `sm_90` - H100

### Build Flags

The system uses optimal flags by default:

```bash
nvcc -O3 --use_fast_math -std=c++17 --gpu-architecture=sm_86 -ptx
```

Debug mode:
```bash
DEBUG=1 cargo build --features cuda
```

## Advanced: Kernel Template

Here's a complete template for a custom kernel:

`kernels/my_custom_kernel.cu`:

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel configuration
#define BLOCK_SIZE 256

// Your kernel implementation
__global__ void my_kernel_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Your custom operation here
        output[idx] = input[idx] * 2.0f;
    }
}

// C interface for Rust FFI
extern "C" {
    void my_kernel_launch(
        const float* input,
        float* output,
        int n,
        cudaStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        my_kernel_impl<<<grid, block, 0, stream>>>(
            input, output, n
        );
    }
}
```

## Rust Integration Pattern

```rust
// src/kernels/my_kernel.rs

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

pub fn my_kernel(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        // Load PTX
        let ptx_path = concat!(
            env!("KERNEL_OUTPUT_DIR"),
            "/my_custom_kernel.ptx"
        );
        let ptx = include_str!(ptx_path);

        // Get CUDA device
        if let Device::Cuda(cuda_dev) = input.device() {
            // Launch kernel
            // ... implementation ...
            todo!("Launch kernel")
        } else {
            candle_core::bail!("Kernel requires CUDA device");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // CPU fallback
        input.mul(&Tensor::new(2.0f32, input.device())?)
    }
}
```

## Common Patterns

### 1. Fused Operations

Combine multiple ops into one kernel:

```cuda
// Fused: x = (x + bias) * scale + residual
__global__ void fused_ops_kernel(
    const float* x,
    const float* bias,
    float scale,
    const float* residual,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (x[idx] + bias[idx]) * scale + residual[idx];
    }
}
```

### 2. Reduction Operations

Sum, max, etc. with shared memory:

```cuda
__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int n
) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load to shared memory
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}
```

### 3. Memory-Efficient Operations

Tiled computation for large tensors:

```cuda
#define TILE_SIZE 32

__global__ void tiled_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Testing Your Kernel

Create a test in Rust:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_my_kernel() {
        let device = Device::new_cuda(0).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1000,), &device).unwrap();

        let output = my_kernel(&input).unwrap();

        // Verify output shape
        assert_eq!(output.dims(), input.dims());

        // Verify values (example)
        let expected = input.mul(&Tensor::new(2.0f32, &device).unwrap()).unwrap();
        assert!(output.allclose(&expected, 1e-5, 1e-5).unwrap());
    }
}
```

## Debugging

### Enable Debug Mode

```bash
DEBUG=1 cargo build --features cuda
```

This adds `-g -G` flags for cuda-gdb debugging.

### Check PTX Output

```bash
cat target/debug/build/ferrite-*/out/kernels/my_kernel.ptx
```

### Profile with Nsight

```bash
nsys profile --stats=true ./target/release/my_program
```

## Best Practices

1. **Always use `__restrict__`** for pointer parameters (enables optimizations)
2. **Align data** to 128 bytes when possible
3. **Use shared memory** for data reuse
4. **Minimize divergent branches** (warp efficiency)
5. **Coalesce memory accesses** (access consecutive addresses)
6. **Test both CPU fallback and CUDA paths**

## Resources

- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Kernel build script**: `kernels/build_kernels.sh`
- **Config**: `kernels/kernel_config.toml`
- **Example kernel**: `kernels/flash_attention.cu`

## FAQ

**Q: Can I use C++17 features?**
A: Yes! The build system uses `-std=c++17`.

**Q: How do I use Tensor Cores?**
A: Use `wmma` API or `nvcuda::wmma` namespace for FP16 matrix operations.

**Q: Can I build for multiple architectures?**
A: Yes! Compile separately for each or use fatbin:
```bash
nvcc -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     ...
```

**Q: How do I add third-party CUDA libraries?**
A: Link them in `build.rs`:
```rust
println!("cargo:rustc-link-lib=my_cuda_lib");
```

---

**Happy kernel hacking! 🚀**

If you create a useful kernel, consider contributing it back to Ferrite!
