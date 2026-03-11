// Gather kernel — F32 only for initial testing
//
// gather(input, dim, indices) -> output
// where output has the same shape as indices and:
//   For dim=d:  output[i0][i1]...[id]...[in] = input[i0][i1]...[indices[i0][i1]...[id]...[in]]]...[in]
//
// Linearized: the tensor is viewed as [outer, input_dim_size, inner] where
//   outer = product of dims before d
//   inner = product of dims after d
// indices has shape [outer, idx_dim_size, inner]
// output has same shape as indices
//
// For each element in the output (total = outer * idx_dim_size * inner):
//   o = element / (idx_dim_size * inner)
//   k = (element / inner) % idx_dim_size
//   i = element % inner
//   idx = indices[o * idx_dim_size * inner + k * inner + i]  (cast to int)
//   output[element] = input[o * input_dim_size * inner + idx * inner + i]
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __global__ void gather_f32(
    const float *input,
    const int32_t *indices,
    float *output,
    const size_t outer,
    const size_t input_dim_size,
    const size_t idx_dim_size,
    const size_t inner
) {
    size_t total = outer * idx_dim_size * inner;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x;
         elem < total;
         elem += blockDim.x * gridDim.x)
    {
        size_t o = elem / (idx_dim_size * inner);
        size_t k = (elem / inner) % idx_dim_size;
        size_t i = elem % inner;

        int32_t idx = indices[o * idx_dim_size * inner + k * inner + i];
        // Clamp index to valid range
        if (idx < 0) idx += (int32_t)input_dim_size;
        if (idx < 0) idx = 0;
        if (idx >= (int32_t)input_dim_size) idx = (int32_t)input_dim_size - 1;

        output[elem] = input[o * input_dim_size * inner + (size_t)idx * inner + i];
    }
}

// ---------------------------------------------------------------------------
// C launcher
// ---------------------------------------------------------------------------
extern "C" {

void candle_launch_gather_f32(
    const float *input,
    const int32_t *indices,
    float *output,
    const size_t outer,
    const size_t input_dim_size,
    const size_t idx_dim_size,
    const size_t inner,
    cudaStream_t stream
) {
    size_t total = outer * idx_dim_size * inner;
    if (total == 0) return;

    const unsigned int threads = 256;
    unsigned int blocks = (total + threads - 1) / threads;
    gather_f32<<<blocks, threads, 0, stream>>>(
        input, indices, output, outer, input_dim_size, idx_dim_size, inner
    );
}

// ptx_sys FFI alias (so Rust ptx-tensor can find this symbol)
void ptx_tensor_gather_f32(
    float *input,
    int32_t *indices,
    float *output,
    const size_t outer,
    const size_t input_dim_size,
    const size_t idx_dim_size,
    const size_t inner,
    cudaStream_t stream
) {
    candle_launch_gather_f32(input, indices, output, outer, input_dim_size, idx_dim_size, inner, stream);
}

} // extern "C"
