// Launcher wrappers for candle indexing.cu kernels (F32, u32 index variant)
//
// The raw kernels (is_u32_f32, sa_u32_f32, s_u32_f32, ia_u32_f32) are
// defined in indexing.cu via macro expansion.  With RDC enabled we can
// forward-declare them and provide host launcher functions.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Forward declarations of candle kernels (defined in indexing.cu)
// ---------------------------------------------------------------------------
extern "C" __global__ void is_u32_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint32_t *ids, const float *inp, float *out,
    const size_t left_size, const size_t src_dim_size,
    const size_t ids_dim_size, const size_t right_size
);

extern "C" __global__ void sa_u32_f32(
    const uint32_t *ids, const float *inp, float *out,
    const size_t left_size, const size_t src_dim_size,
    const size_t dst_dim_size, const size_t right_size
);

extern "C" __global__ void s_u32_f32(
    const uint32_t *ids, const float *inp, float *out,
    const size_t left_size, const size_t src_dim_size,
    const size_t dst_dim_size, const size_t right_size
);

extern "C" __global__ void ia_u32_f32(
    const uint32_t *ids, const size_t ids_dim_size,
    const float *inp, float *out,
    const size_t left_size, const size_t src_dim_size,
    const size_t dst_dim_size, const size_t right_size
);

// ---------------------------------------------------------------------------
// C Launchers
// ---------------------------------------------------------------------------
extern "C" {

// Index Select — select slices along a dimension by index.
// Tensor viewed as [left, src_dim, right]; indices select from src_dim.
// Output shape: [left, ids_dim, right].
void candle_launch_index_select_f32(
    const float *input,
    const uint32_t *ids,
    float *output,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size,
    cudaStream_t stream
) {
    size_t numel = left_size * ids_dim_size * right_size;
    if (numel == 0) return;
    const unsigned int threads = 256;
    unsigned int blocks = (numel + threads - 1) / threads;
    is_u32_f32<<<blocks, threads, 0, stream>>>(
        numel, 0, nullptr, ids, input, output,
        left_size, src_dim_size, ids_dim_size, right_size
    );
}

// Scatter Add — accumulate src values into output at positions given by ids.
// src viewed as [left, src_dim, right]; ids same shape as src.
// output viewed as [left, dst_dim, right] (must be pre-initialized, e.g. zeros).
void candle_launch_scatter_add_f32(
    const uint32_t *ids,
    const float *src,
    float *output,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size,
    cudaStream_t stream
) {
    size_t numel = left_size * right_size;
    if (numel == 0) return;
    const unsigned int threads = 256;
    unsigned int blocks = (numel + threads - 1) / threads;
    sa_u32_f32<<<blocks, threads, 0, stream>>>(
        ids, src, output, left_size, src_dim_size, dst_dim_size, right_size
    );
}

// Scatter — write src values into output at positions given by ids.
void candle_launch_scatter_f32(
    const uint32_t *ids,
    const float *src,
    float *output,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size,
    cudaStream_t stream
) {
    size_t numel = left_size * right_size;
    if (numel == 0) return;
    const unsigned int threads = 256;
    unsigned int blocks = (numel + threads - 1) / threads;
    s_u32_f32<<<blocks, threads, 0, stream>>>(
        ids, src, output, left_size, src_dim_size, dst_dim_size, right_size
    );
}

// Index Add — accumulate src into output using a 1D index vector.
void candle_launch_index_add_f32(
    const uint32_t *ids,
    const size_t ids_dim_size,
    const float *src,
    float *output,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size,
    cudaStream_t stream
) {
    size_t numel = left_size * right_size;
    if (numel == 0) return;
    const unsigned int threads = 256;
    unsigned int blocks = (numel + threads - 1) / threads;
    ia_u32_f32<<<blocks, threads, 0, stream>>>(
        ids, ids_dim_size, src, output,
        left_size, src_dim_size, dst_dim_size, right_size
    );
}

// ptx_tensor aliases
void ptx_tensor_index_select_f32(
    float *input, int32_t *ids, float *output,
    size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size,
    cudaStream_t stream
) {
    candle_launch_index_select_f32(input, (const uint32_t*)ids, output,
        left_size, src_dim_size, ids_dim_size, right_size, stream);
}

void ptx_tensor_scatter_add_f32(
    int32_t *ids, float *src, float *output,
    size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size,
    cudaStream_t stream
) {
    candle_launch_scatter_add_f32((const uint32_t*)ids, src, output,
        left_size, src_dim_size, dst_dim_size, right_size, stream);
}

} // extern "C"
