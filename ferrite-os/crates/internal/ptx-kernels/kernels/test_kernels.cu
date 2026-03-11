// Simple test kernels for validating the guard layer
#include <cuda_runtime.h>

// Simple element-wise add kernel
extern "C" __global__ void test_add_f32(
    const float* a,
    const float* b,
    float* out,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Simple element-wise multiply kernel
extern "C" __global__ void test_mul_f32(
    const float* a,
    const float* b,
    float* out,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Simple GELU approximation (for testing)
extern "C" __global__ void test_gelu_f32(
    const float* input,
    float* out,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Simple tanh approximation of GELU
        float x_cubed = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);
        out[idx] = 0.5f * x * (1.0f + tanh_inner);
    }
}

// C wrapper launchers
extern "C" {

void test_launch_add_f32(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_add_f32<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

void test_launch_mul_f32(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_mul_f32<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

void test_launch_gelu_f32(
    const float* input,
    float* out,
    size_t n,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_gelu_f32<<<blocks, threads, 0, stream>>>(input, out, n);
}

} // extern "C"
