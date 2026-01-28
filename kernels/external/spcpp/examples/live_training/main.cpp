// main.cpp - Live Neural Network Training with Hot-Reload
//
// Run: spcpp main.cpp
// Then edit trainer.cpp while running to change hyperparameters!

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <chrono>
#include <thread>
#include <csignal>

volatile bool running = true;
void signal_handler(int) { running = false; }

// Network dimensions
constexpr int INPUT_DIM = 784;
constexpr int HIDDEN1 = 256;
constexpr int HIDDEN2 = 128;
constexpr int OUTPUT_DIM = 10;
constexpr int BATCH_SIZE = 64;

// GPU memory context - owned by main, passed to module
struct GPUContext {
    // Weights
    float* d_W1;
    float* d_W2;
    float* d_W3;
    // Momentum
    float* d_m1;
    float* d_m2;
    float* d_m3;
    // Activations
    float* d_input;
    float* d_h1;
    float* d_h2;
    float* d_output;
    // Kernel launcher
    spc::CudaModule* gpu;
};

int main() {
    signal(SIGINT, signal_handler);

    std::cout << "\n\033[1;35m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;35m║   LIVE TRAINING - Hot-Reload Demo          ║\033[0m\n";
    std::cout << "\033[1;35m╚════════════════════════════════════════════╝\033[0m\n\n";
    std::cout << "Edit \033[1;33mtrainer.cpp\033[0m while running to change:\n";
    std::cout << "  • LEARNING_RATE (try 0.01, 0.001, 0.0001)\n";
    std::cout << "  • MOMENTUM (try 0.0, 0.9, 0.99)\n\n";
    std::cout << "Weights persist - only hyperparameters change!\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // Initialize cuBLAS
    cublasHandle_t blas;
    cublasCreate(&blas);

    // Load GPU kernels
    auto gpu = IMPORT_GPU("kernels.cu");

    // Allocate GPU memory (owned by main - persists across module reloads)
    GPUContext ctx;
    ctx.gpu = &gpu;

    cudaMalloc(&ctx.d_W1, INPUT_DIM * HIDDEN1 * sizeof(float));
    cudaMalloc(&ctx.d_W2, HIDDEN1 * HIDDEN2 * sizeof(float));
    cudaMalloc(&ctx.d_W3, HIDDEN2 * OUTPUT_DIM * sizeof(float));

    cudaMalloc(&ctx.d_m1, INPUT_DIM * HIDDEN1 * sizeof(float));
    cudaMalloc(&ctx.d_m2, HIDDEN1 * HIDDEN2 * sizeof(float));
    cudaMalloc(&ctx.d_m3, HIDDEN2 * OUTPUT_DIM * sizeof(float));

    cudaMalloc(&ctx.d_input, BATCH_SIZE * INPUT_DIM * sizeof(float));
    cudaMalloc(&ctx.d_h1, BATCH_SIZE * HIDDEN1 * sizeof(float));
    cudaMalloc(&ctx.d_h2, BATCH_SIZE * HIDDEN2 * sizeof(float));
    cudaMalloc(&ctx.d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float));

    // Initialize momentum to zero
    cudaMemset(ctx.d_m1, 0, INPUT_DIM * HIDDEN1 * sizeof(float));
    cudaMemset(ctx.d_m2, 0, HIDDEN1 * HIDDEN2 * sizeof(float));
    cudaMemset(ctx.d_m3, 0, HIDDEN2 * OUTPUT_DIM * sizeof(float));

    std::cout << "\033[1;36m[GPU]\033[0m Allocated "
              << (INPUT_DIM * HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 * OUTPUT_DIM) * 2 * sizeof(float) / 1024 / 1024.0f
              << " MB for weights + momentum\n\n";

    // Initialize weights via trainer
    auto trainer = IMPORT("trainer.cpp");
    trainer->call<void, GPUContext*>("init_weights", &ctx);

    // Training loop
    int step = 0;
    while (running) {
        // IMPORT_HOT reloads if trainer.cpp changed
        trainer = IMPORT_HOT("trainer.cpp");

        // Run one training step
        trainer->call<void, cublasHandle_t, GPUContext*, int>("train_step", blas, &ctx, step);
        step++;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Final report
    trainer->call<void, int>("final_report", step);

    // Cleanup
    cudaFree(ctx.d_W1); cudaFree(ctx.d_W2); cudaFree(ctx.d_W3);
    cudaFree(ctx.d_m1); cudaFree(ctx.d_m2); cudaFree(ctx.d_m3);
    cudaFree(ctx.d_input); cudaFree(ctx.d_h1); cudaFree(ctx.d_h2); cudaFree(ctx.d_output);
    cublasDestroy(blas);

    std::cout << "\n\033[1;32mTraining complete!\033[0m\n";
    return 0;
}
