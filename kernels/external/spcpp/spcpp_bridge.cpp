// SPCPP Bridge for Semantic Core RS
// Uses spcpp's JIT compilation system - no separate build step needed
// This module gets compiled by spcpp's IMPORT() system

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>

// ============================================================================
// GLOBAL STATE (persists across hot-reloads via spcpp STATE macros)
// ============================================================================

STATE(cublasHandle_t, g_blas, nullptr);
STATE(bool, g_initialized, false);

// Training state
STATE(int, adam_t, 0);
STATE(float, learning_rate, 0.001f);
STATE(float, beta1, 0.9f);
STATE(float, beta2, 0.999f);
STATE(float, epsilon, 1e-8f);
STATE(float, weight_decay, 0.0f);

// ============================================================================
// INITIALIZATION
// ============================================================================

EXPORT int spcpp_init() {
    if (g_initialized) return 0;

    cublasStatus_t status = cublasCreate(&g_blas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[spcpp_bridge] cuBLAS init failed: " << status << std::endl;
        return -1;
    }

    g_initialized = true;
    std::cout << "[spcpp_bridge] Initialized with cuBLAS" << std::endl;
    return 0;
}

EXPORT void spcpp_shutdown() {
    if (g_blas) {
        cublasDestroy(g_blas);
        g_blas = nullptr;
    }
    g_initialized = false;
}

// ============================================================================
// MATMUL OPERATIONS (cuBLAS)
// ============================================================================

// C = A @ B (row-major)
// A: [M, K], B: [K, N], C: [M, N]
EXPORT void spcpp_matmul(float* a, float* b, float* c, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS column-major: compute B^T @ A^T = (A @ B)^T in col-major = A @ B in row-major
    cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha, b, n, a, k,
                &beta, c, n);
}

// C = alpha * op(A) @ op(B) + beta * C
EXPORT void spcpp_gemm(float* a, float* b, float* c, int m, int k, int n,
                       float alpha, float beta, int trans_a, int trans_b) {
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = trans_a ? m : k;
    int ldb = trans_b ? k : n;

    cublasSgemm(g_blas, op_b, op_a,
                n, m, k,
                &alpha, b, ldb, a, lda,
                &beta, c, n);
}

// ============================================================================
// VECTOR OPERATIONS (cuBLAS)
// ============================================================================

EXPORT void spcpp_axpy(float* x, float* y, int n, float alpha) {
    cublasSaxpy(g_blas, n, &alpha, x, 1, y, 1);
}

EXPORT void spcpp_scal(float* x, int n, float alpha) {
    cublasSscal(g_blas, n, &alpha, x, 1);
}

EXPORT float spcpp_dot(float* x, float* y, int n) {
    float result;
    cublasSdot(g_blas, n, x, 1, y, 1, &result);
    return result;
}

EXPORT float spcpp_nrm2(float* x, int n) {
    float result;
    cublasSnrm2(g_blas, n, x, 1, &result);
    return result;
}

// ============================================================================
// LINEAR LAYER
// ============================================================================

EXPORT void spcpp_linear_forward(float* input, float* weights, float* bias, float* output,
                                  int batch, int in_features, int out_features, int has_bias) {
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                out_features, batch, in_features,
                &alpha, weights, out_features, input, in_features,
                &beta, output, out_features);

    if (has_bias && bias != nullptr) {
        for (int b = 0; b < batch; b++) {
            cublasSaxpy(g_blas, out_features, &alpha, bias, 1, output + b * out_features, 1);
        }
    }
}

EXPORT void spcpp_linear_backward(float* input, float* weights, float* grad_output,
                                   float* grad_input, float* grad_weights, float* grad_bias,
                                   int batch, int in_features, int out_features, int has_bias) {
    float alpha = 1.0f, beta = 0.0f;

    if (grad_input) {
        cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    in_features, batch, out_features,
                    &alpha, weights, out_features, grad_output, out_features,
                    &beta, grad_input, in_features);
    }

    if (grad_weights) {
        cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    out_features, in_features, batch,
                    &alpha, grad_output, out_features, input, in_features,
                    &beta, grad_weights, out_features);
    }
}

// ============================================================================
// WEIGHT INITIALIZATION
// ============================================================================

EXPORT void spcpp_init_weights(const char* name, float* data, int rows, int cols, const char* init_type) {
    int n = rows * cols;
    std::string init(init_type);
    std::vector<float> h_data(n);
    std::mt19937 gen(42);

    if (init == "xavier" || init == "glorot") {
        float limit = std::sqrt(6.0f / (rows + cols));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int i = 0; i < n; i++) h_data[i] = dist(gen);
    } else if (init == "kaiming" || init == "he") {
        float std_val = std::sqrt(2.0f / rows);
        std::normal_distribution<float> dist(0.0f, std_val);
        for (int i = 0; i < n; i++) h_data[i] = dist(gen);
    } else if (init == "normal") {
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (int i = 0; i < n; i++) h_data[i] = dist(gen);
    } else if (init == "zeros") {
        std::fill(h_data.begin(), h_data.end(), 0.0f);
    } else if (init == "ones") {
        std::fill(h_data.begin(), h_data.end(), 1.0f);
    }

    cudaMemcpy(data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

// ============================================================================
// OPTIMIZER
// ============================================================================

EXPORT void spcpp_set_optimizer_params(float lr, float b1, float b2, float eps, float wd) {
    learning_rate = lr;
    beta1 = b1;
    beta2 = b2;
    epsilon = eps;
    weight_decay = wd;
}

EXPORT void spcpp_zero_gradients(float* gradients, int n) {
    cudaMemset(gradients, 0, n * sizeof(float));
}

// ============================================================================
// MEMORY
// ============================================================================

EXPORT void* spcpp_alloc(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

EXPORT void spcpp_free(void* ptr) {
    cudaFree(ptr);
}

EXPORT void spcpp_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

EXPORT void spcpp_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

EXPORT void spcpp_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

EXPORT void spcpp_sync() {
    cudaDeviceSynchronize();
}

EXPORT int spcpp_get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

EXPORT float spcpp_mse_loss(float* pred, float* target, int n) {
    float* diff;
    cudaMalloc(&diff, n * sizeof(float));
    cudaMemcpy(diff, pred, n * sizeof(float), cudaMemcpyDeviceToDevice);

    float neg_one = -1.0f;
    cublasSaxpy(g_blas, n, &neg_one, target, 1, diff, 1);

    float norm;
    cublasSnrm2(g_blas, n, diff, 1, &norm);

    cudaFree(diff);
    return (norm * norm) / n;
}
