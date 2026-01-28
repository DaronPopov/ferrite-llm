// SPCPP Launcher - Entry point for Rust FFI
// This gets compiled once, then uses spcpp's JIT for kernels
//
// Build: g++ -shared -fPIC -O3 -std=c++17 \
//        -I/path/to/spcpp/include -I/usr/local/cuda/include \
//        spcpp_launcher.cpp -o libspcpp_launcher.so \
//        -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda -ldl

// SPC_SPCPP_DIR passed at compile time - points to kernels/external/spcpp
#ifndef SPC_SPCPP_DIR
#define SPC_SPCPP_DIR "."
#endif
#include <spcpp.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>
#include <unordered_map>

// ============================================================================
// GLOBAL STATE
// ============================================================================

static cublasHandle_t g_blas = nullptr;
static spc::CudaModule* g_ops = nullptr;
static bool g_initialized = false;
static std::string g_kernel_path;

// Optimizer state storage
static std::unordered_map<std::string, float*> g_momentum;
static std::unordered_map<std::string, float*> g_adam_m;
static std::unordered_map<std::string, float*> g_adam_v;
static int g_adam_t = 0;
static float g_lr = 0.001f, g_beta1 = 0.9f, g_beta2 = 0.999f, g_eps = 1e-8f, g_wd = 0.0f;

extern "C" {

// ============================================================================
// INITIALIZATION
// ============================================================================

int spcpp_init(const char* kernel_path) {
    if (g_initialized) return 0;

    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&g_blas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[spcpp] cuBLAS init failed: " << status << std::endl;
        return -1;
    }

    // JIT compile CUDA kernels via spcpp
    g_kernel_path = kernel_path;
    try {
        static spc::CudaModule ops = IMPORT_GPU(kernel_path);
        g_ops = &ops;
    } catch (const std::exception& e) {
        std::cerr << "[spcpp] Kernel compile failed: " << e.what() << std::endl;
        return -2;
    }

    g_initialized = true;
    std::cout << "[spcpp] Initialized - cuBLAS + JIT kernels from " << kernel_path << std::endl;
    return 0;
}

void spcpp_shutdown() {
    if (g_blas) {
        cublasDestroy(g_blas);
        g_blas = nullptr;
    }

    for (auto& [k, v] : g_momentum) cudaFree(v);
    for (auto& [k, v] : g_adam_m) cudaFree(v);
    for (auto& [k, v] : g_adam_v) cudaFree(v);
    g_momentum.clear();
    g_adam_m.clear();
    g_adam_v.clear();

    g_initialized = false;
}

// ============================================================================
// MATMUL (cuBLAS)
// ============================================================================

void spcpp_matmul(float* a, float* b, float* c, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, b, n, a, k, &beta, c, n);
}

void spcpp_gemm(float* a, float* b, float* c, int m, int k, int n,
                float alpha, float beta, int trans_a, int trans_b) {
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = trans_a ? m : k;
    int ldb = trans_b ? k : n;
    cublasSgemm(g_blas, op_b, op_a, n, m, k, &alpha, b, ldb, a, lda, &beta, c, n);
}

// ============================================================================
// UNARY OPS (JIT kernels)
// ============================================================================

void spcpp_unary(const char* op_name, float* data, int n) {
    if (!g_ops) return;

    int blocks = (n + 255) / 256;
    void* args[] = {&data, &n};

    std::string op(op_name);
    if (op == "relu") g_ops->launch("relu_forward", blocks, 256, args);
    else if (op == "sigmoid") g_ops->launch("sigmoid_forward", blocks, 256, args);
    else if (op == "tanh") g_ops->launch("tanh_forward", blocks, 256, args);
    else if (op == "gelu") g_ops->launch("gelu_forward", blocks, 256, args);
    else if (op == "silu") g_ops->launch("silu_forward", blocks, 256, args);
    else if (op == "exp") g_ops->launch("exp_kernel", blocks, 256, args);
    else if (op == "log") g_ops->launch("log_kernel", blocks, 256, args);
    else if (op == "sqrt") g_ops->launch("sqrt_kernel", blocks, 256, args);
    else if (op == "neg") g_ops->launch("neg_kernel", blocks, 256, args);
    else if (op == "abs") g_ops->launch("abs_kernel", blocks, 256, args);
    else std::cerr << "[spcpp] Unknown unary op: " << op << std::endl;
}

void spcpp_softmax(float* data, int rows, int cols) {
    if (!g_ops) return;
    void* args[] = {&data, &rows, &cols};
    g_ops->launch("softmax_forward", rows, 1, args);
}

void spcpp_leaky_relu(float* data, int n, float alpha) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&data, &n, &alpha};
    g_ops->launch("leaky_relu_forward", blocks, 256, args);
}

// ============================================================================
// BINARY OPS
// ============================================================================

void spcpp_binary(const char* op_name, float* a, float* b, float* c, int n) {
    if (!g_ops) return;

    int blocks = (n + 255) / 256;
    void* args[] = {&a, &b, &c, &n};

    std::string op(op_name);
    if (op == "add") g_ops->launch("add_kernel", blocks, 256, args);
    else if (op == "sub") g_ops->launch("sub_kernel", blocks, 256, args);
    else if (op == "mul") g_ops->launch("mul_kernel", blocks, 256, args);
    else if (op == "div") g_ops->launch("div_kernel", blocks, 256, args);
    else std::cerr << "[spcpp] Unknown binary op: " << op << std::endl;
}

void spcpp_scale(float* data, float scalar, int n) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&data, &scalar, &n};
    g_ops->launch("scale_kernel", blocks, 256, args);
}

void spcpp_fill(float* data, float val, int n) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&data, &val, &n};
    g_ops->launch("fill_kernel", blocks, 256, args);
}

// ============================================================================
// LINEAR LAYER
// ============================================================================

void spcpp_linear_forward(float* input, float* weights, float* bias, float* output,
                          int batch, int in_features, int out_features, int has_bias) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                out_features, batch, in_features,
                &alpha, weights, out_features, input, in_features,
                &beta, output, out_features);

    if (has_bias && bias && g_ops) {
        void* args[] = {&output, &bias, &batch, &out_features};
        g_ops->launch2d("add_bias", batch, 1, 1, out_features, args);
    }
}

void spcpp_linear_backward(float* input, float* weights, float* grad_out,
                           float* grad_input, float* grad_weights, float* grad_bias,
                           int batch, int in_f, int out_f, int has_bias) {
    float alpha = 1.0f, beta = 0.0f;

    if (grad_input) {
        cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    in_f, batch, out_f,
                    &alpha, weights, out_f, grad_out, out_f,
                    &beta, grad_input, in_f);
    }

    if (grad_weights) {
        cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    out_f, in_f, batch,
                    &alpha, grad_out, out_f, input, in_f,
                    &beta, grad_weights, out_f);
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void spcpp_init_weights(const char* name, float* data, int rows, int cols, const char* init_type) {
    int n = rows * cols;
    std::string init(init_type);
    std::vector<float> h(n);
    std::mt19937 gen(42);

    if (init == "xavier" || init == "glorot") {
        float lim = std::sqrt(6.0f / (rows + cols));
        std::uniform_real_distribution<float> d(-lim, lim);
        for (auto& x : h) x = d(gen);
    } else if (init == "kaiming" || init == "he") {
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / rows));
        for (auto& x : h) x = d(gen);
    } else if (init == "normal") {
        std::normal_distribution<float> d(0.0f, 0.02f);
        for (auto& x : h) x = d(gen);
    } else if (init == "zeros") {
        std::fill(h.begin(), h.end(), 0.0f);
    } else if (init == "ones") {
        std::fill(h.begin(), h.end(), 1.0f);
    }

    cudaMemcpy(data, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

// ============================================================================
// OPTIMIZER
// ============================================================================

void spcpp_set_optimizer_params(float lr, float b1, float b2, float eps, float wd) {
    g_lr = lr; g_beta1 = b1; g_beta2 = b2; g_eps = eps; g_wd = wd;
}

void spcpp_sgd_step(const char* name, float* weights, float* grad, int n,
                    float lr, float momentum, float wd) {
    if (!g_ops) return;

    std::string key(name);
    if (g_momentum.find(key) == g_momentum.end()) {
        float* m; cudaMalloc(&m, n * sizeof(float));
        cudaMemset(m, 0, n * sizeof(float));
        g_momentum[key] = m;
    }
    float* mom = g_momentum[key];

    int blocks = (n + 255) / 256;
    void* args[] = {&weights, &mom, &grad, &n, &lr, &momentum, &wd};
    g_ops->launch("sgd_step", blocks, 256, args);
}

void spcpp_adam_step(const char* name, float* weights, float* grad, int n) {
    if (!g_ops) return;

    std::string key(name);
    if (g_adam_m.find(key) == g_adam_m.end()) {
        float *m, *v;
        cudaMalloc(&m, n * sizeof(float));
        cudaMalloc(&v, n * sizeof(float));
        cudaMemset(m, 0, n * sizeof(float));
        cudaMemset(v, 0, n * sizeof(float));
        g_adam_m[key] = m;
        g_adam_v[key] = v;
    }

    g_adam_t++;
    float* m = g_adam_m[key];
    float* v = g_adam_v[key];

    int blocks = (n + 255) / 256;
    void* args[] = {&weights, &m, &v, &grad, &n, &g_lr, &g_beta1, &g_beta2, &g_eps, &g_wd, &g_adam_t};
    g_ops->launch("adam_step", blocks, 256, args);
}

void spcpp_zero_grad(float* grad, int n) {
    cudaMemset(grad, 0, n * sizeof(float));
}

// ============================================================================
// NORMALIZATION
// ============================================================================

void spcpp_layer_norm(float* x, float* gamma, float* beta, float* out,
                      int batch, int dim, float eps) {
    if (!g_ops) return;
    void* args[] = {&x, &gamma, &beta, &out, &batch, &dim, &eps};
    g_ops->launch("layer_norm_forward", batch, 1, args);
}

void spcpp_dropout(float* data, float* mask, int n, float p, unsigned int seed) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&data, &mask, &n, &p, &seed};
    g_ops->launch("dropout_forward", blocks, 256, args);
}

// ============================================================================
// EMBEDDING
// ============================================================================

void spcpp_embedding(float* table, int* indices, float* out, int batch, int dim) {
    if (!g_ops) return;
    void* args[] = {&table, &indices, &out, &batch, &dim};
    g_ops->launch2d("embedding_lookup", batch, 1, 1, dim, args);
}

// ============================================================================
// LOSS
// ============================================================================

float spcpp_cross_entropy(float* logits, int* labels, int batch, int classes) {
    if (!g_ops) return 0.0f;

    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    void* args[] = {&logits, &labels, &d_loss, &batch, &classes};
    g_ops->launch("cross_entropy_loss", 1, batch, args);

    float loss;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return loss;
}

float spcpp_mse_loss(float* pred, float* target, int n) {
    float* diff;
    cudaMalloc(&diff, n * sizeof(float));
    cudaMemcpy(diff, pred, n * sizeof(float), cudaMemcpyDeviceToDevice);

    float neg = -1.0f;
    cublasSaxpy(g_blas, n, &neg, target, 1, diff, 1);

    float norm;
    cublasSnrm2(g_blas, n, diff, 1, &norm);
    cudaFree(diff);

    return (norm * norm) / n;
}

// ============================================================================
// QUANTIZATION
// ============================================================================

void spcpp_quantize_int8(float* input, signed char* output, int n, float scale, int zp) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&input, &output, &n, &scale, &zp};
    g_ops->launch("quantize_f32_to_int8", blocks, 256, args);
}

void spcpp_dequantize_int8(signed char* input, float* output, int n, float scale, int zp) {
    if (!g_ops) return;
    int blocks = (n + 255) / 256;
    void* args[] = {&input, &output, &n, &scale, &zp};
    g_ops->launch("dequantize_int8_to_f32", blocks, 256, args);
}

// ============================================================================
// MEMORY
// ============================================================================

void* spcpp_alloc(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void spcpp_free(void* ptr) {
    cudaFree(ptr);
}

void spcpp_h2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void spcpp_d2h(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void spcpp_d2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

void spcpp_sync() {
    cudaDeviceSynchronize();
}

// ============================================================================
// cuBLAS VECTOR OPS
// ============================================================================

void spcpp_axpy(float* x, float* y, int n, float alpha) {
    cublasSaxpy(g_blas, n, &alpha, x, 1, y, 1);
}

void spcpp_scal(float* x, int n, float alpha) {
    cublasSscal(g_blas, n, &alpha, x, 1);
}

float spcpp_dot(float* x, float* y, int n) {
    float r;
    cublasSdot(g_blas, n, x, 1, y, 1, &r);
    return r;
}

float spcpp_nrm2(float* x, int n) {
    float r;
    cublasSnrm2(g_blas, n, x, 1, &r);
    return r;
}

} // extern "C"
