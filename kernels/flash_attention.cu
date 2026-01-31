// Flash Attention - Custom CUDA Kernel
// Implements memory-efficient attention without materializing QK^T matrix
//
// Key idea:
// - Compute attention in tiles (e.g., 64x64 blocks)
// - Never store full (seq_len x seq_len) attention matrix
// - Use online softmax to incrementally compute attention weights
//
// Memory savings: O(N^2) → O(N) where N = sequence length

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Tile sizes - tune these for your GPU
#define TILE_SIZE_M 64  // Number of queries per tile
#define TILE_SIZE_N 64  // Number of keys per tile
#define TILE_SIZE_K 64  // Hidden dimension tile size

// Online softmax state
struct SoftmaxState {
    float max_val;
    float sum_exp;
};

// Device function: Online softmax update
// Allows incremental computation without storing full attention matrix
__device__ inline void online_softmax_update(
    SoftmaxState* state,
    float new_val
) {
    if (new_val > state->max_val) {
        // New max found - rescale previous sum
        float scale = expf(state->max_val - new_val);
        state->sum_exp = state->sum_exp * scale + expf(0.0f);
        state->max_val = new_val;
    } else {
        state->sum_exp += expf(new_val - state->max_val);
    }
}

// Flash Attention Kernel - Forward Pass
//
// Q: [batch, num_heads, seq_len, head_dim] - Queries
// K: [batch, num_heads, seq_len, head_dim] - Keys
// V: [batch, num_heads, seq_len, head_dim] - Values
// O: [batch, num_heads, seq_len, head_dim] - Output
//
// This kernel processes attention in tiles to minimize memory usage
extern "C" __global__ void flash_attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale  // 1/sqrt(head_dim)
) {
    // Thread block processes one query position
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_idx >= seq_len) return;

    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE_M];
    __shared__ float K_tile[TILE_SIZE_N];
    __shared__ float V_tile[TILE_SIZE_N];

    // Each thread computes attention for one query position
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;

    // Initialize output accumulator and softmax state
    float output_accum[64] = {0.0f};  // Assuming head_dim <= 64
    SoftmaxState softmax_state = {-INFINITY, 0.0f};

    // Tile over keys/values
    for (int k_tile_start = 0; k_tile_start < seq_len; k_tile_start += TILE_SIZE_N) {
        const int k_tile_end = min(k_tile_start + TILE_SIZE_N, seq_len);
        const int k_tile_size = k_tile_end - k_tile_start;

        // Process this tile of keys
        for (int k_idx = k_tile_start; k_idx < k_tile_end; k_idx++) {
            // Compute Q·K^T for this key (dot product)
            float qk_dot = 0.0f;

            const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * head_dim;

            for (int d = 0; d < head_dim; d++) {
                qk_dot += Q[q_offset + d] * K[k_offset + d];
            }

            qk_dot *= scale;  // Scale by 1/sqrt(head_dim)

            // Update online softmax
            float old_max = softmax_state.max_val;
            online_softmax_update(&softmax_state, qk_dot);

            // Compute attention weight for this key
            float attn_weight = expf(qk_dot - softmax_state.max_val);

            // If max changed, rescale previous output accumulator
            if (softmax_state.max_val != old_max) {
                float rescale = expf(old_max - softmax_state.max_val);
                for (int d = 0; d < head_dim; d++) {
                    output_accum[d] *= rescale;
                }
            }

            // Accumulate weighted value: O += attn_weight * V
            const int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                output_accum[d] += attn_weight * V[v_offset + d];
            }
        }
    }

    // Final normalization by sum of exponentials
    const float norm = 1.0f / softmax_state.sum_exp;
    const int o_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;

    for (int d = 0; d < head_dim; d++) {
        O[o_offset + d] = output_accum[d] * norm;
    }
}

// C interface for Rust FFI
// Note: When using PTX loading, call the __global__ kernels directly
// These host wrapper functions are optional and not included in PTX output
extern "C" {

// Optional host wrapper - not included in PTX
void flash_attention_forward_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Grid: one block per query position per head
    dim3 grid(
        (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M,
        num_heads,
        batch_size
    );

    dim3 block(TILE_SIZE_M, 1, 1);

    flash_attention_forward<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        batch_size, num_heads, seq_len, head_dim,
        scale
    );
}

// Causal mask version (for decoder-only models like Mistral)
extern "C" __global__ void flash_attention_causal_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_idx >= seq_len) return;

    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;

    float output_accum[64] = {0.0f};
    SoftmaxState softmax_state = {-INFINITY, 0.0f};

    // Causal masking: only attend to positions <= query_idx
    const int causal_limit = query_idx + 1;

    for (int k_idx = 0; k_idx < causal_limit; k_idx++) {
        float qk_dot = 0.0f;
        const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * head_dim;

        for (int d = 0; d < head_dim; d++) {
            qk_dot += Q[q_offset + d] * K[k_offset + d];
        }

        qk_dot *= scale;

        float old_max = softmax_state.max_val;
        online_softmax_update(&softmax_state, qk_dot);

        float attn_weight = expf(qk_dot - softmax_state.max_val);

        if (softmax_state.max_val != old_max) {
            float rescale = expf(old_max - softmax_state.max_val);
            for (int d = 0; d < head_dim; d++) {
                output_accum[d] *= rescale;
            }
        }

        const int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            output_accum[d] += attn_weight * V[v_offset + d];
        }
    }

    const float norm = 1.0f / softmax_state.sum_exp;
    const int o_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;

    for (int d = 0; d < head_dim; d++) {
        O[o_offset + d] = output_accum[d] * norm;
    }
}

// Optional host wrapper - not included in PTX
void flash_attention_causal_forward_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(
        (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M,
        num_heads,
        batch_size
    );

    dim3 block(TILE_SIZE_M, 1, 1);

    flash_attention_causal_forward<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        batch_size, num_heads, seq_len, head_dim,
        scale
    );
}

} // extern "C"
