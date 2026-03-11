// TopK kernel — F32 only for initial testing
//
// topk(input, k, dim, largest) -> (values, indices)
//
// Layout: tensor is viewed as [outer, dim_size, inner] where we select top-k
// along the middle axis.  Each thread handles one (outer, inner) lane and
// performs k passes of selection to find the k extreme values.
//
// values  shape: [outer, k, inner]
// indices shape: [outer, k, inner]  (int32)
//
// For each (o, i) lane:
//   Select k elements from input[o, 0..dim_size, i] that are the
//   largest (or smallest if !largest), breaking ties by smaller index.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __global__ void topk_f32(
    const float *input,
    float *values_out,
    int32_t *indices_out,
    const size_t outer,
    const size_t dim_size,
    const size_t inner,
    const size_t k,
    const int largest
) {
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t in_base  = o * dim_size * inner + i;
    size_t out_base = o * k * inner + i;

    size_t actual_k = k < dim_size ? k : dim_size;

    // For each of actual_k positions, find the best remaining element.
    // O(k * (dim_size + k)) per lane — fine for small k.
    for (size_t ki = 0; ki < actual_k; ki++) {
        float best_val;
        int32_t best_idx = -1;

        for (size_t d = 0; d < dim_size; d++) {
            float val = input[in_base + d * inner];

            // Skip indices already selected in previous passes
            bool skip = false;
            for (size_t prev = 0; prev < ki; prev++) {
                if (indices_out[out_base + prev * inner] == (int32_t)d) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;

            if (best_idx < 0) {
                best_val = val;
                best_idx = (int32_t)d;
            } else {
                bool better = largest ? (val > best_val) : (val < best_val);
                bool tie_smaller = (val == best_val && (int32_t)d < best_idx);
                if (better || tie_smaller) {
                    best_val = val;
                    best_idx = (int32_t)d;
                }
            }
        }

        values_out [out_base + ki * inner] = best_val;
        indices_out[out_base + ki * inner] = best_idx;
    }
}

// ---------------------------------------------------------------------------
// C launcher
// ---------------------------------------------------------------------------
extern "C" {

void candle_launch_topk_f32(
    const float *input,
    float *values_out,
    int32_t *indices_out,
    const size_t outer,
    const size_t dim_size,
    const size_t inner,
    const size_t k,
    const int largest,
    cudaStream_t stream
) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0 || k == 0) return;

    const unsigned int threads = 256;
    unsigned int blocks = (total_lanes + threads - 1) / threads;
    topk_f32<<<blocks, threads, 0, stream>>>(
        input, values_out, indices_out, outer, dim_size, inner, k, largest
    );
}

// ptx_sys FFI alias (so Rust ptx-tensor can find this symbol)
void ptx_tensor_topk_f32(
    float *input,
    float *values_out,
    int32_t *indices_out,
    const size_t outer,
    const size_t dim_size,
    const size_t inner,
    const size_t k,
    const int largest,
    cudaStream_t stream
) {
    candle_launch_topk_f32(input, values_out, indices_out, outer, dim_size, inner, k, largest, stream);
}

} // extern "C"
