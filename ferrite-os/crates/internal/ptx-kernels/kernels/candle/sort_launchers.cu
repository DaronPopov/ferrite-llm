// Launcher wrappers for candle sort.cu kernels (F32 argsort)
//
// The bitonic argsort kernels are defined in sort.cu via ASORT_OP macro.
// Layout: input is [nrows, ncols], one block per row, shared memory sort.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Forward declarations of candle kernels (defined in sort.cu)
// ---------------------------------------------------------------------------
extern "C" __global__ void asort_asc_f32(
    const float *x, uint32_t *dst, const int ncols, int ncols_pad
);

extern "C" __global__ void asort_desc_f32(
    const float *x, uint32_t *dst, const int ncols, int ncols_pad
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static unsigned int next_pow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

// ---------------------------------------------------------------------------
// C Launchers
// ---------------------------------------------------------------------------
extern "C" {

// Argsort — sort each row of a [nrows, ncols] matrix.
// Output is uint32 indices such that x[row, output[row,i]] is sorted.
// ascending=1 for ascending, 0 for descending.
void candle_launch_argsort_f32(
    const float *input,
    uint32_t *output,
    const size_t nrows,
    const size_t ncols,
    const int ascending,
    cudaStream_t stream
) {
    if (nrows == 0 || ncols == 0) return;

    unsigned int ncols_pad = next_pow2((unsigned int)ncols);
    // Block size: min(1024, ncols_pad) — bitonic sort needs threads >= ncols_pad/2
    unsigned int block_size = ncols_pad < 1024 ? ncols_pad : 1024;
    size_t shared_mem = ncols_pad * sizeof(int);

    if (ascending) {
        asort_asc_f32<<<(unsigned int)nrows, block_size, shared_mem, stream>>>(
            input, output, (int)ncols, (int)ncols_pad
        );
    } else {
        asort_desc_f32<<<(unsigned int)nrows, block_size, shared_mem, stream>>>(
            input, output, (int)ncols, (int)ncols_pad
        );
    }
}

// ptx_tensor alias
void ptx_tensor_argsort_f32(
    float *input, uint32_t *output,
    size_t nrows, size_t ncols, int ascending,
    cudaStream_t stream
) {
    candle_launch_argsort_f32(input, output, nrows, ncols, ascending, stream);
}

} // extern "C"
