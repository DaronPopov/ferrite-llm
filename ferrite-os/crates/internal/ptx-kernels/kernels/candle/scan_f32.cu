// Prefix scan (cumulative sum) kernel — F32 only for initial testing
//
// Layout: tensor is viewed as [outer, dim_size, inner] where we scan along
// the middle axis.  Each thread handles one (outer, inner) lane and walks
// sequentially along the scan dimension.  This is simple & correct; for
// very large dim_size a work-efficient Blelloch scan could be added later.
#include <cuda_runtime.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Cumulative sum kernel (inclusive prefix sum along middle axis)
// ---------------------------------------------------------------------------
// input  shape: [outer, dim_size, inner]  (contiguous, row-major)
// output shape: [outer, dim_size, inner]  (same shape as input)
//
// For each (o, i) pair:
//   output[o, 0, i]   = input[o, 0, i]
//   output[o, k, i]   = output[o, k-1, i] + input[o, k, i]   for k in 1..dim_size
//
extern "C" __global__ void cumsum_f32(
    const float *input,
    float *output,
    const size_t outer,
    const size_t dim_size,
    const size_t inner
) {
    // Each thread handles one (outer, inner) lane
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;

    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;

    // Base offset for this (outer, inner) lane
    // Layout: input[o * dim_size * inner + k * inner + i]
    size_t base = o * dim_size * inner + i;

    float acc = 0.0f;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        acc += input[idx];
        output[idx] = acc;
    }
}

// ---------------------------------------------------------------------------
// C launcher
// ---------------------------------------------------------------------------
extern "C" {

void candle_launch_cumsum_f32(
    const float *input,
    float *output,
    const size_t outer,
    const size_t dim_size,
    const size_t inner,
    cudaStream_t stream
) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;

    const unsigned int threads = 256;
    unsigned int blocks = (total_lanes + threads - 1) / threads;
    cumsum_f32<<<blocks, threads, 0, stream>>>(input, output, outer, dim_size, inner);
}

// ptx_sys FFI alias (so Rust ptx-tensor can find this symbol)
void ptx_tensor_cumsum_f32(
    float *input,
    float *output,
    const size_t outer,
    const size_t dim_size,
    const size_t inner,
    cudaStream_t stream
) {
    candle_launch_cumsum_f32(input, output, outer, dim_size, inner, stream);
}

} // extern "C"
