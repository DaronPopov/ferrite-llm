// Kernel launcher wrapper for Candle kernels
// These C wrapper functions launch the actual CUDA kernels
#include <cuda_runtime.h>
#include <stddef.h>

// Forward declarations of kernel functions
#define DECLARE_UNARY_KERNEL(NAME) \
    extern "C" __global__ void NAME( \
        const size_t numel, \
        const size_t num_dims, \
        const size_t *info, \
        const float *inp, \
        float *out \
    );

#define DECLARE_UNARY_KERNEL_F64(NAME) \
    extern "C" __global__ void NAME( \
        const size_t numel, \
        const size_t num_dims, \
        const size_t *info, \
        const double *inp, \
        double *out \
    );

// Declare unary kernels
DECLARE_UNARY_KERNEL(ugelu_f32)
DECLARE_UNARY_KERNEL(urelu_f32)
DECLARE_UNARY_KERNEL(usilu_f32)
DECLARE_UNARY_KERNEL(usigmoid_f32)
DECLARE_UNARY_KERNEL(uabs_f32)
DECLARE_UNARY_KERNEL(usqrt_f32)
DECLARE_UNARY_KERNEL(usqr_f32)
DECLARE_UNARY_KERNEL(uexp_f32)
DECLARE_UNARY_KERNEL(ulog_f32)
DECLARE_UNARY_KERNEL(usin_f32)
DECLARE_UNARY_KERNEL(ucos_f32)
DECLARE_UNARY_KERNEL(utanh_f32)
DECLARE_UNARY_KERNEL(uneg_f32)

DECLARE_UNARY_KERNEL_F64(ugelu_f64)
DECLARE_UNARY_KERNEL_F64(urelu_f64)
DECLARE_UNARY_KERNEL_F64(usilu_f64)
DECLARE_UNARY_KERNEL_F64(usigmoid_f64)
DECLARE_UNARY_KERNEL_F64(uabs_f64)
DECLARE_UNARY_KERNEL_F64(usqrt_f64)

// Binary kernels
#define DECLARE_BINARY_KERNEL(NAME) \
    extern "C" __global__ void NAME( \
        const size_t numel, \
        const size_t num_dims, \
        const size_t *dims, \
        const size_t *left_strides, \
        const float *left_ptr, \
        const size_t *right_strides, \
        const float *right_ptr, \
        float *out \
    );

DECLARE_BINARY_KERNEL(badd_f32)
DECLARE_BINARY_KERNEL(bsub_f32)
DECLARE_BINARY_KERNEL(bmul_f32)
DECLARE_BINARY_KERNEL(bdiv_f32)
DECLARE_BINARY_KERNEL(bmaximum_f32)
DECLARE_BINARY_KERNEL(bminimum_f32)

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

// Binary F32 launchers
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
    badd_f32<<<blocks, threads, 0, stream>>>(
        numel, num_dims, dims, left_strides, left_ptr, right_strides, right_ptr, out
    );
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
    bmul_f32<<<blocks, threads, 0, stream>>>(
        numel, num_dims, dims, left_strides, left_ptr, right_strides, right_ptr, out
    );
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
    bsub_f32<<<blocks, threads, 0, stream>>>(
        numel, num_dims, dims, left_strides, left_ptr, right_strides, right_ptr, out
    );
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
    bdiv_f32<<<blocks, threads, 0, stream>>>(
        numel, num_dims, dims, left_strides, left_ptr, right_strides, right_ptr, out
    );
}

} // extern "C"
