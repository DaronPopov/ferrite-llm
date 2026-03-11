#define _USE_MATH_DEFINES
#include<math.h>
#include<stdint.h>
#include "cuda_utils.cuh"

// Unary operation macro for F32 only
#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
} \

// Helper functions
template<typename T>
__device__ __forceinline__ T gelu_erf_fwd(T x) {
  return x * normcdfg(x);
}

template<typename T>
__device__ __forceinline__ T gelu_fwd(T x) {
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanhg(static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha));
}

template<typename T>
__device__ __forceinline__ T elu_fwd(T x, T alpha) {
  if (x > static_cast<T>(0)) {
    return x;
  }
  return alpha * (expg(x) - static_cast<T>(1));
}

template<typename T>
__device__ __forceinline__ T relu_fwd(T x) {
    T zero = 0.;
    return maxg(x, zero);
}

template<typename T>
__device__ __forceinline__ T silu_fwd(T x) {
    return x / (static_cast<T>(1) + expg(-x));
}

template<typename T>
__device__ __forceinline__ T sigmoid_fwd(T x) {
    return recipg(static_cast<T>(1) + expg(-x));
}

template<typename T>
__device__ T sign_(T t) {
  return static_cast<T>(t > static_cast<T>(0)) - static_cast<T>(t < static_cast<T>(0));
}

// Generate all F32 unary operations
UNARY_OP(float, ucopy_f32, x)
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, urecip_f32, recipg(x))
UNARY_OP(float, uexp_f32, expg(x))
UNARY_OP(float, ulog_f32, logg(x))
UNARY_OP(float, usin_f32, sing(x))
UNARY_OP(float, ucos_f32, cosg(x))
UNARY_OP(float, utanh_f32, tanhg(x))
UNARY_OP(float, uerf_f32, erfg(x))
UNARY_OP(float, uceil_f32, ceilg(x))
UNARY_OP(float, ufloor_f32, floorg(x))
UNARY_OP(float, uround_f32, roundg(x))
UNARY_OP(float, unormcdf_f32, normcdfg(x))
UNARY_OP(float, uabs_f32, absg(x))
UNARY_OP(float, usqr_f32, x*x)
UNARY_OP(float, usqrt_f32, sqrtg(x))
UNARY_OP(float, ugelu_f32, gelu_fwd(x))
UNARY_OP(float, ugelu_erf_f32, gelu_erf_fwd(x))
UNARY_OP(float, urelu_f32, relu_fwd(x))
UNARY_OP(float, usilu_f32, silu_fwd(x))
UNARY_OP(float, usign_f32, sign_(x))
UNARY_OP(float, usigmoid_f32, sigmoid_fwd(x))
