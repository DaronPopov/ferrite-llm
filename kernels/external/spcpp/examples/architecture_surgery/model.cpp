// model.cpp - EDIT THIS FILE WHILE TRAINING RUNS!
//
// ╔═══════════════════════════════════════════════════════════════╗
// ║  CHANGE THESE VALUES AND SAVE - ARCHITECTURE CHANGES LIVE!    ║
// ╚═══════════════════════════════════════════════════════════════╝

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <string>

// ============================================================
// ARCHITECTURE DEFINITION - Edit while training!
// ============================================================

// Layer sizes: input -> hidden... -> output
// TRY: Add layers, change sizes! Examples:
//   {64}           - 1 hidden layer
//   {128, 64}      - 2 hidden layers
//   {128, 64, 128} - 3 hidden layers (bottleneck)
//   {256, 128, 64, 32, 64, 128, 256} - deep autoencoder style
constexpr int HIDDEN_DIMS[] = {64, 32, 64};
constexpr int NUM_HIDDEN = sizeof(HIDDEN_DIMS) / sizeof(HIDDEN_DIMS[0]);

constexpr int INPUT_DIM = 64;
constexpr int OUTPUT_DIM = 64;
constexpr int BATCH = 32;

// Learning rate - try: 0.01, 0.001, 0.0001, 0.1
constexpr float LEARNING_RATE = 0.001f;

// Activation: 0=relu, 1=tanh, 2=gelu, 3=leaky_relu, 4=swish
constexpr int ACTIVATION = 0;

// ============================================================
// Weight pool structure (from main.cpp)
// ============================================================

constexpr int MAX_DIM = 512;
constexpr int MAX_LAYERS = 8;

struct WeightPool {
    float* W[MAX_LAYERS];
    float* grad[MAX_LAYERS];
    float* act[MAX_LAYERS+1];
    float* target;
};

// ============================================================
// EXPORTED FUNCTIONS
// ============================================================

EXPORT void get_architecture(char* out) {
    const char* act_names[] = {"relu", "tanh", "gelu", "leaky", "swish"};
    char buf[256];
    sprintf(buf, "%d", INPUT_DIM);
    for (int i = 0; i < NUM_HIDDEN; i++) {
        sprintf(buf + strlen(buf), "->%d", HIDDEN_DIMS[i]);
    }
    sprintf(buf + strlen(buf), "->%d [%s]", OUTPUT_DIM, act_names[ACTIVATION]);
    strcpy(out, buf);
}

EXPORT float get_learning_rate() {
    return LEARNING_RATE;
}

// ============================================================
// TRAINING STEP
// ============================================================

EXPORT float train_step(cublasHandle_t blas, spc::CudaModule* gpu, WeightPool* pool, int step) {
    // Build dimension array
    int dims[MAX_LAYERS + 2];
    dims[0] = INPUT_DIM;
    for (int i = 0; i < NUM_HIDDEN; i++) dims[i + 1] = HIDDEN_DIMS[i];
    dims[NUM_HIDDEN + 1] = OUTPUT_DIM;
    int num_layers = NUM_HIDDEN + 1;

    float alpha = 1.0f, beta = 0.0f;

    // Kernel names for activations
    const char* act_kernels[] = {"relu", "tanh_act", "gelu", "leaky_relu", "swish"};

    // ========================================
    // FORWARD PASS
    // ========================================
    for (int l = 0; l < num_layers; l++) {
        int in_dim = dims[l];
        int out_dim = dims[l + 1];

        // Linear: act[l+1] = act[l] @ W[l]
        cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    out_dim, BATCH, in_dim,
                    &alpha, pool->W[l], MAX_DIM,
                    pool->act[l], MAX_DIM,
                    &beta, pool->act[l + 1], MAX_DIM);

        // Activation (except last layer)
        if (l < num_layers - 1) {
            int n = BATCH * out_dim;
            int blocks = (n + 255) / 256;
            void* args[] = {&pool->act[l + 1], &n};
            gpu->launch(act_kernels[ACTIVATION], blocks, 256, args);
        }
    }

    // ========================================
    // COMPUTE LOSS (MSE)
    // ========================================
    float loss = 0.0f;
    {
        int n = BATCH * OUTPUT_DIM;
        int blocks = (n + 255) / 256;
        float* d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));

        void* args[] = {&d_loss, &pool->act[num_layers], &pool->target, &n};
        gpu->launch("mse_loss", blocks, 256, args);

        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_loss);
    }

    // ========================================
    // BACKWARD PASS (simplified)
    // ========================================
    for (int l = num_layers - 1; l >= 0; l--) {
        int in_dim = dims[l];
        int out_dim = dims[l + 1];
        int n = MAX_DIM * MAX_DIM;  // Update full weight matrix
        int blocks = (n + 255) / 256;

        float lr = LEARNING_RATE;
        void* args[] = {&pool->W[l], &lr, &in_dim, &out_dim};
        gpu->launch("sgd_update", blocks, 256, args);
    }

    return loss;
}
