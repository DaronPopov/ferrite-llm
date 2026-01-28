// trainer.cpp - Hot-reloadable training module
//
// ╔═══════════════════════════════════════════════════════════╗
// ║  EDIT THIS FILE WHILE TRAINING RUNS!                      ║
// ║  Change the values below and save - instant effect!       ║
// ╚═══════════════════════════════════════════════════════════╝

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <vector>
#include <iomanip>

// ============================================================
// HYPERPARAMETERS - Edit these while running!
// ============================================================

constexpr float LEARNING_RATE = 0.001f;    // Try: 0.01, 0.001, 0.0001
constexpr float MOMENTUM = 0.9f;           // Try: 0.0, 0.5, 0.9, 0.99

// ============================================================
// Network architecture (must match main.cpp)
// ============================================================

constexpr int INPUT_DIM = 784;
constexpr int HIDDEN1 = 256;
constexpr int HIDDEN2 = 128;
constexpr int OUTPUT_DIM = 10;
constexpr int BATCH_SIZE = 64;

// GPU memory context (passed from main)
struct GPUContext {
    float* d_W1;
    float* d_W2;
    float* d_W3;
    float* d_m1;
    float* d_m2;
    float* d_m3;
    float* d_input;
    float* d_h1;
    float* d_h2;
    float* d_output;
    spc::CudaModule* gpu;
};

// Training state (persists in main's memory via STATE macro)
STATE(float, smoothed_loss, 1.0f);
STATE(float, best_loss, 999.0f);

// ============================================================
// FUNCTIONS
// ============================================================

EXPORT void init_weights(GPUContext* ctx) {
    std::cout << "\033[1;36m[Init]\033[0m Initializing weights with Xavier...\n";

    auto init = [](float* d_ptr, int fan_in, int fan_out) {
        int size = fan_in * fan_out;
        std::vector<float> h_data(size);
        std::mt19937 gen(42);
        float std = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, std);
        for (int i = 0; i < size; i++) h_data[i] = dist(gen);
        cudaMemcpy(d_ptr, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    };

    init(ctx->d_W1, INPUT_DIM, HIDDEN1);
    init(ctx->d_W2, HIDDEN1, HIDDEN2);
    init(ctx->d_W3, HIDDEN2, OUTPUT_DIM);

    std::cout << "\033[1;36m[Init]\033[0m Network ready: "
              << INPUT_DIM << " -> " << HIDDEN1 << " -> "
              << HIDDEN2 << " -> " << OUTPUT_DIM << "\n\n";
}

EXPORT void train_step(cublasHandle_t blas, GPUContext* ctx, int step) {
    // Generate random input batch
    {
        std::vector<float> h_input(BATCH_SIZE * INPUT_DIM);
        std::mt19937 gen(step * 17 + 1);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_input) x = dist(gen);
        cudaMemcpy(ctx->d_input, h_input.data(),
                   BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    }

    float alpha = 1.0f, beta = 0.0f;
    auto& gpu = *ctx->gpu;

    // Layer 1: h1 = W1^T @ input + ReLU
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                HIDDEN1, BATCH_SIZE, INPUT_DIM,
                &alpha, ctx->d_W1, HIDDEN1, ctx->d_input, INPUT_DIM,
                &beta, ctx->d_h1, HIDDEN1);

    int n1 = BATCH_SIZE * HIDDEN1;
    void* relu1_args[] = {&ctx->d_h1, &n1};
    gpu.launch("relu_forward", (n1 + 255) / 256, 256, relu1_args);

    // Layer 2: h2 = W2^T @ h1 + ReLU
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                HIDDEN2, BATCH_SIZE, HIDDEN1,
                &alpha, ctx->d_W2, HIDDEN2, ctx->d_h1, HIDDEN1,
                &beta, ctx->d_h2, HIDDEN2);

    int n2 = BATCH_SIZE * HIDDEN2;
    void* relu2_args[] = {&ctx->d_h2, &n2};
    gpu.launch("relu_forward", (n2 + 255) / 256, 256, relu2_args);

    // Layer 3: output = W3^T @ h2
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                OUTPUT_DIM, BATCH_SIZE, HIDDEN2,
                &alpha, ctx->d_W3, OUTPUT_DIM, ctx->d_h2, HIDDEN2,
                &beta, ctx->d_output, OUTPUT_DIM);

    // Compute loss (L2 norm as simple metric)
    float norm;
    cublasSnrm2(blas, BATCH_SIZE * OUTPUT_DIM, ctx->d_output, 1, &norm);
    float loss = norm / std::sqrt((float)(BATCH_SIZE * OUTPUT_DIM));

    smoothed_loss = 0.95f * smoothed_loss + 0.05f * loss;
    if (loss < best_loss) best_loss = loss;

    // SGD with momentum weight update
    float lr = LEARNING_RATE;
    float mom = MOMENTUM;

    int nw1 = INPUT_DIM * HIDDEN1;
    int nw2 = HIDDEN1 * HIDDEN2;
    int nw3 = HIDDEN2 * OUTPUT_DIM;

    void* sgd1[] = {&ctx->d_W1, &ctx->d_m1, &nw1, &lr, &mom};
    void* sgd2[] = {&ctx->d_W2, &ctx->d_m2, &nw2, &lr, &mom};
    void* sgd3[] = {&ctx->d_W3, &ctx->d_m3, &nw3, &lr, &mom};

    gpu.launch("sgd_step", (nw1 + 255) / 256, 256, sgd1);
    gpu.launch("sgd_step", (nw2 + 255) / 256, 256, sgd2);
    gpu.launch("sgd_step", (nw3 + 255) / 256, 256, sgd3);

    // Print progress
    if (step % 20 == 0) {
        std::cout << "\r\033[K"
                  << "\033[1;32m[" << std::setw(6) << step << "]\033[0m"
                  << " Loss: \033[1;33m" << std::fixed << std::setprecision(4) << smoothed_loss << "\033[0m"
                  << " Best: " << best_loss
                  << " | \033[1;36mLR=" << LEARNING_RATE << " Mom=" << MOMENTUM << "\033[0m"
                  << std::flush;
    }
}

EXPORT void final_report(int total_steps) {
    std::cout << "\n\n";
    std::cout << "\033[1;33m╔═══════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;33m║         TRAINING COMPLETE             ║\033[0m\n";
    std::cout << "\033[1;33m╚═══════════════════════════════════════╝\033[0m\n";
    std::cout << "  Total steps: " << total_steps << "\n";
    std::cout << "  Final loss:  " << smoothed_loss << "\n";
    std::cout << "  Best loss:   " << best_loss << "\n";
}
