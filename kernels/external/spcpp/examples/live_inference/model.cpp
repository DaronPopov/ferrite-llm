// model.cpp - EDIT WHILE SERVER RUNS!
//
// ╔════════════════════════════════════════════════════════════════╗
// ║  Change any of these settings - they take effect immediately!  ║
// ╚════════════════════════════════════════════════════════════════╝

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <cstdio>

// ============================================================
// MODEL CONFIGURATION - Edit these live!
// ============================================================

// Model version (shows in logs when you change it)
#define MODEL_VERSION "v1.0-base"

// Architecture: true = 2 layers, false = 1 layer
constexpr bool USE_DEEP_MODEL = true;

// Activation: 0=relu, 1=tanh, 2=sigmoid, 3=none
constexpr int ACTIVATION = 0;

// Output transformation
constexpr bool APPLY_SOFTMAX = false;
constexpr float TEMPERATURE = 1.0f;  // For softmax scaling

// Confidence threshold (outputs below this show as "uncertain")
constexpr float CONFIDENCE_THRESHOLD = 0.3f;

// Debug mode - prints intermediate values
constexpr bool DEBUG_MODE = false;

// Post-processing
constexpr bool CLIP_OUTPUT = true;
constexpr float CLIP_MIN = -1.0f;
constexpr float CLIP_MAX = 1.0f;

// ============================================================
// Context structure (from main.cpp)
// ============================================================

struct ModelContext {
    float *d_W1, *d_W2, *d_input, *d_hidden, *d_output;
    cublasHandle_t blas;
    spc::CudaModule* gpu;
    int dim;
};

// ============================================================
// EXPORTED FUNCTIONS
// ============================================================

EXPORT void get_version(char* out) {
    strcpy(out, MODEL_VERSION);
}

EXPORT void predict(ModelContext* ctx, int request_id, char* result) {
    float alpha = 1.0f, beta = 0.0f;
    int dim = ctx->dim;

    // Layer 1: hidden = W1 @ input
    cublasSgemm(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, 1, dim,
                &alpha, ctx->d_W1, dim, ctx->d_input, dim,
                &beta, ctx->d_hidden, dim);

    // Activation after layer 1
    if (ACTIVATION == 0) {  // ReLU
        int n = dim;
        void* args[] = {&ctx->d_hidden, &n};
        ctx->gpu->launch("relu", (n + 255) / 256, 256, args);
    } else if (ACTIVATION == 1) {  // Tanh
        int n = dim;
        void* args[] = {&ctx->d_hidden, &n};
        ctx->gpu->launch("tanh_act", (n + 255) / 256, 256, args);
    } else if (ACTIVATION == 2) {  // Sigmoid
        int n = dim;
        void* args[] = {&ctx->d_hidden, &n};
        ctx->gpu->launch("sigmoid", (n + 255) / 256, 256, args);
    }

    if (DEBUG_MODE) {
        std::vector<float> h_hidden(dim);
        cudaMemcpy(h_hidden.data(), ctx->d_hidden, dim * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n  [DEBUG] Hidden[0:4] = [%.3f, %.3f, %.3f, %.3f]",
               h_hidden[0], h_hidden[1], h_hidden[2], h_hidden[3]);
    }

    // Layer 2 (optional)
    float* final_output;
    if (USE_DEEP_MODEL) {
        cublasSgemm(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    dim, 1, dim,
                    &alpha, ctx->d_W2, dim, ctx->d_hidden, dim,
                    &beta, ctx->d_output, dim);
        final_output = ctx->d_output;
    } else {
        final_output = ctx->d_hidden;
    }

    // Copy output to host
    std::vector<float> output(dim);
    cudaMemcpy(output.data(), final_output, dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-processing
    if (CLIP_OUTPUT) {
        for (auto& x : output) {
            x = std::max(CLIP_MIN, std::min(CLIP_MAX, x));
        }
    }

    if (APPLY_SOFTMAX) {
        float max_val = output[0];
        for (float x : output) max_val = std::max(max_val, x);

        float sum = 0;
        for (auto& x : output) {
            x = expf((x - max_val) / TEMPERATURE);
            sum += x;
        }
        for (auto& x : output) x /= sum;
    }

    // Compute statistics
    float sum = 0, max_val = output[0];
    int max_idx = 0;
    for (int i = 0; i < dim; i++) {
        sum += output[i];
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    float mean = sum / dim;

    // Format result
    const char* confidence_str = (max_val > CONFIDENCE_THRESHOLD) ? "HIGH" : "LOW ";

    if (DEBUG_MODE) {
        sprintf(result, "[%s] Class %2d (%.3f) | Mean: %+.3f | %s | out[0:3]=[%.2f,%.2f,%.2f]",
                MODEL_VERSION, max_idx, max_val, mean,
                USE_DEEP_MODEL ? "2-layer" : "1-layer",
                output[0], output[1], output[2]);
    } else {
        sprintf(result, "[%s] Pred: class %2d (conf: %.2f %s) | Mean: %+.3f",
                MODEL_VERSION, max_idx, max_val, confidence_str, mean);
    }
}
