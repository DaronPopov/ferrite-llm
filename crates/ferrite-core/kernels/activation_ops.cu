// Example Custom Kernel - Template for Users
//
// This is a simple example showing how to write a custom CUDA kernel
// for Ferrite. Use this as a starting point for your own kernels!

#include <cuda_runtime.h>

// Kernel: Element-wise ReLU activation
// This demonstrates the basic pattern for a custom kernel
extern "C" __global__ void relu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (idx < n) {
        // ReLU: max(0, x)
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Kernel: Fused ReLU + Scale
// Example of fusing multiple operations into one kernel
__global__ void relu_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];
        // Fused: ReLU then scale
        output[idx] = fmaxf(0.0f, x) * scale;
    }
}

// Kernel: Vector reduction (sum)
// Example using shared memory for efficiency
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    // Shared memory for block-level reduction
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================================
// C Interface for Rust FFI
// ============================================================================

extern "C" {

// Optional host wrapper - not included in PTX
void relu_forward_launch(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_forward<<<blocks, threads, 0, stream>>>(input, output, n);
}

// Launch fused ReLU + Scale kernel
void relu_scale_forward(
    const float* input,
    float* output,
    float scale,
    int n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_scale_kernel<<<blocks, threads, 0, stream>>>(input, output, scale, n);
}

// Launch reduction kernel
void reduce_sum_forward(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Initialize output to zero
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    reduce_sum_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

} // extern "C"

// ============================================================================
// Usage from Rust:
// ============================================================================
//
// use candle_core::{Device, Tensor};
//
// pub fn relu_cuda(input: &Tensor) -> Result<Tensor> {
//     // 1. Load compiled PTX
//     let ptx = include_str!(concat!(
//         env!("KERNEL_OUTPUT_DIR"),
//         "/example_kernel.ptx"
//     ));
//
//     // 2. Get CUDA device and stream
//     if let Device::Cuda(cuda_dev) = input.device() {
//         // 3. Load module and get function
//         let module = CudaModule::load_ptx(ptx)?;
//         let kernel = module.get_function("relu_forward")?;
//
//         // 4. Allocate output
//         let n = input.elem_count();
//         let output = Tensor::zeros_like(input)?;
//
//         // 5. Launch kernel
//         kernel.launch(...)?;
//
//         Ok(output)
//     } else {
//         Err("CUDA device required")
//     }
// }
