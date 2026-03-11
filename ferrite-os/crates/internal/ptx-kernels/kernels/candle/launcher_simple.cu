// Simplified kernel launcher wrapper for Candle kernels (F32 only for initial testing)
#include <cuda_runtime.h>
#include <stddef.h>

// Forward declarations of F32 unary kernels
extern "C" __global__ void ugelu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void urelu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void usilu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void usigmoid_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void uabs_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void usqrt_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void uexp_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void ulog_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void utanh_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void usin_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

extern "C" __global__ void ucos_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out
);

// Forward declarations of F32 binary kernels
// Note: The BINARY_OP macro generates kernels with packed dims_and_strides
extern "C" __global__ void badd_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

extern "C" __global__ void bsub_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

extern "C" __global__ void bmul_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

extern "C" __global__ void bdiv_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

extern "C" __global__ void bminimum_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

extern "C" __global__ void bmaximum_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims_and_strides,
    const float *lhs,
    const float *rhs,
    float *out
);

// Launcher helper function
static inline dim3 get_grid_size(size_t numel, unsigned int threads_per_block = 256) {
    unsigned int num_blocks = (numel + threads_per_block - 1) / threads_per_block;
    return dim3(num_blocks);
}

// C wrapper functions that launch kernels
extern "C" {

// Unary F32 launchers
void candle_launch_ugelu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    ugelu_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_urelu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    urelu_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_usilu_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    usilu_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_usigmoid_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    usigmoid_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_uabs_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    uabs_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_usqrt_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    usqrt_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_uexp_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    uexp_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_ulog_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    ulog_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_utanh_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    utanh_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_usin_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    usin_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

void candle_launch_ucos_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const float *inp,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);
    ucos_f32<<<blocks, threads, 0, stream>>>(numel, num_dims, info, inp, out);
}

// Simple wrapper kernels for binary ops (contiguous only for now)
__global__ void badd_f32_wrapper(
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = lhs[i] + rhs[i];
    }
}

__global__ void bmul_f32_wrapper(
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = lhs[i] * rhs[i];
    }
}

__global__ void bsub_f32_wrapper(
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = lhs[i] - rhs[i];
    }
}

__global__ void bdiv_f32_wrapper(
    const size_t numel,
    const float *lhs,
    const float *rhs,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = lhs[i] / rhs[i];
    }
}

// Binary F32 launchers - using simple wrappers for contiguous case
void candle_launch_badd_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    // For contiguous (num_dims == 0 or nullptr strides), use simple wrapper
    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        badd_f32_wrapper<<<blocks, threads, 0, stream>>>(
            numel, left_ptr, right_ptr, out
        );
    } else {
        // For strided case, use the macro-generated kernel
        badd_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

void candle_launch_bmul_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        bmul_f32_wrapper<<<blocks, threads, 0, stream>>>(
            numel, left_ptr, right_ptr, out
        );
    } else {
        bmul_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

void candle_launch_bsub_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        bsub_f32_wrapper<<<blocks, threads, 0, stream>>>(
            numel, left_ptr, right_ptr, out
        );
    } else {
        bsub_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

void candle_launch_bdiv_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        bdiv_f32_wrapper<<<blocks, threads, 0, stream>>>(
            numel, left_ptr, right_ptr, out
        );
    } else {
        bdiv_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

void candle_launch_bminimum_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        // Simple min wrapper
        badd_f32_wrapper<<<blocks, threads, 0, stream>>>(numel, left_ptr, right_ptr, out); // TODO: Add min wrapper
    } else {
        bminimum_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

void candle_launch_bmaximum_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *left_strides,
    const float *left_ptr,
    const size_t *right_strides,
    const float *right_ptr,
    float *out,
    cudaStream_t stream
) {
    const unsigned int threads = 256;
    const dim3 blocks = get_grid_size(numel, threads);

    if (num_dims == 0 || (left_strides == nullptr && right_strides == nullptr)) {
        // Simple max wrapper
        badd_f32_wrapper<<<blocks, threads, 0, stream>>>(numel, left_ptr, right_ptr, out); // TODO: Add max wrapper
    } else {
        bmaximum_f32<<<blocks, threads, 0, stream>>>(
            numel, num_dims, dims, left_ptr, right_ptr, out
        );
    }
}

} // extern "C"
