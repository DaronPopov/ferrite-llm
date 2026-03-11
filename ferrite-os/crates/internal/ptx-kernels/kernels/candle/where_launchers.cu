// Launcher wrapper for candle ternary.cu where kernel (F32, u8 condition)
//
// The where_u8_f32 kernel is defined in ternary.cu via WHERE_OP macro.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Forward declaration of candle kernel (defined in ternary.cu)
// ---------------------------------------------------------------------------
extern "C" __global__ void where_u8_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const uint8_t *ids,
    const float *t,
    const float *f,
    float *out
);

// ---------------------------------------------------------------------------
// C Launcher
// ---------------------------------------------------------------------------
extern "C" {

// Where — element-wise conditional: out[i] = cond[i] ? true_val[i] : false_val[i]
// All tensors must be contiguous and same shape.
void candle_launch_where_f32(
    const uint8_t *cond,
    const float *true_val,
    const float *false_val,
    float *output,
    const size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    const unsigned int threads = 256;
    unsigned int blocks = (numel + threads - 1) / threads;
    // For contiguous tensors: num_dims=0, info=NULL triggers fast path
    where_u8_f32<<<blocks, threads, 0, stream>>>(
        numel, 0, nullptr, cond, true_val, false_val, output
    );
}

// ptx_tensor alias
void ptx_tensor_where_f32(
    uint8_t *cond, float *true_val, float *false_val, float *output,
    size_t numel, cudaStream_t stream
) {
    candle_launch_where_f32(cond, true_val, false_val, output, numel, stream);
}

} // extern "C"
