// main.cpp - LIVE NEURAL ARCHITECTURE SURGERY
//
// This does something IMPOSSIBLE in PyTorch/JAX/TensorFlow:
// Edit model.cpp while training to:
//   - Add/remove layers
//   - Change layer sizes
//   - Swap activation functions
//   - Modify loss functions
//   - Change learning rate
// ALL WHILE TRAINING CONTINUES AND WEIGHTS PERSIST!
//
// Run: spcpp main.cpp
// Then edit model.cpp - watch the architecture change live!

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <chrono>
#include <thread>
#include <csignal>
#include <iostream>
#include <iomanip>

volatile bool running = true;
void signal_handler(int) { running = false; }

// Maximum network size (pre-allocated)
constexpr int MAX_DIM = 512;
constexpr int MAX_LAYERS = 8;
constexpr int BATCH = 32;

// Pre-allocated weight pool - survives architecture changes!
struct WeightPool {
    float* W[MAX_LAYERS];      // Weight matrices
    float* grad[MAX_LAYERS];   // Gradients
    float* act[MAX_LAYERS+1];  // Activations (including input)
    float* target;             // Target output

    void allocate() {
        for (int i = 0; i < MAX_LAYERS; i++) {
            cudaMalloc(&W[i], MAX_DIM * MAX_DIM * sizeof(float));
            cudaMalloc(&grad[i], MAX_DIM * MAX_DIM * sizeof(float));
            cudaMalloc(&act[i], BATCH * MAX_DIM * sizeof(float));

            // Initialize weights with small random values
            std::vector<float> h(MAX_DIM * MAX_DIM);
            for (auto& x : h) x = ((rand() / (float)RAND_MAX) - 0.5f) * 0.1f;
            cudaMemcpy(W[i], h.data(), MAX_DIM * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(grad[i], 0, MAX_DIM * MAX_DIM * sizeof(float));
        }
        cudaMalloc(&act[MAX_LAYERS], BATCH * MAX_DIM * sizeof(float));
        cudaMalloc(&target, BATCH * MAX_DIM * sizeof(float));
    }

    void free() {
        for (int i = 0; i < MAX_LAYERS; i++) {
            cudaFree(W[i]);
            cudaFree(grad[i]);
            cudaFree(act[i]);
        }
        cudaFree(act[MAX_LAYERS]);
        cudaFree(target);
    }
};

int main() {
    signal(SIGINT, signal_handler);
    srand(42);

    std::cout << "\n";
    std::cout << "\033[1;35m╔══════════════════════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;35m║     LIVE NEURAL ARCHITECTURE SURGERY                         ║\033[0m\n";
    std::cout << "\033[1;35m║     Something IMPOSSIBLE in PyTorch/JAX/TensorFlow           ║\033[0m\n";
    std::cout << "\033[1;35m╚══════════════════════════════════════════════════════════════╝\033[0m\n\n";

    std::cout << "Edit \033[1;33mmodel.cpp\033[0m while training to:\n";
    std::cout << "  • Change HIDDEN_DIMS[] array (add/remove layers!)\n";
    std::cout << "  • Change activation functions (relu/tanh/gelu)\n";
    std::cout << "  • Change LEARNING_RATE\n";
    std::cout << "  • Modify the loss function\n\n";
    std::cout << "\033[1;32mWeights persist across architecture changes!\033[0m\n\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // Initialize cuBLAS
    cublasHandle_t blas;
    cublasCreate(&blas);

    // Load kernels
    auto gpu = IMPORT_GPU("kernels.cu");

    // Pre-allocate all possible weights
    WeightPool pool;
    pool.allocate();

    std::cout << "\033[1;36m[Memory]\033[0m Pre-allocated "
              << (MAX_LAYERS * MAX_DIM * MAX_DIM * 2 * sizeof(float)) / 1024 / 1024
              << " MB weight pool for up to " << MAX_LAYERS << " layers\n\n";

    // Generate training data
    std::vector<float> h_input(BATCH * MAX_DIM);
    std::vector<float> h_target(BATCH * MAX_DIM);
    for (int i = 0; i < BATCH * MAX_DIM; i++) {
        h_input[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;
        h_target[i] = sinf(h_input[i] * 3.14159f);  // Learn sine function
    }
    cudaMemcpy(pool.act[0], h_input.data(), BATCH * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pool.target, h_target.data(), BATCH * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice);

    int step = 0;
    std::string last_arch = "";

    while (running) {
        // Hot-reload model definition
        auto model = IMPORT_HOT("model.cpp");

        // Get current architecture description
        char arch_desc[256];
        model->call<void, char*>("get_architecture", arch_desc);
        std::string current_arch(arch_desc);

        // Detect architecture change
        if (current_arch != last_arch) {
            std::cout << "\n\033[1;33m[ARCHITECTURE CHANGE]\033[0m " << current_arch << "\n\n";
            last_arch = current_arch;
        }

        // Run training step
        float loss = model->call<float, cublasHandle_t, spc::CudaModule*, WeightPool*, int>(
            "train_step", blas, &gpu, &pool, step);

        // Print progress
        if (step % 10 == 0) {
            float lr = model->call<float>("get_learning_rate");
            std::cout << "\r\033[K"
                      << "\033[1;32m[" << std::setw(6) << step << "]\033[0m"
                      << " Loss: \033[1;33m" << std::fixed << std::setprecision(6) << loss << "\033[0m"
                      << " | LR=" << std::setprecision(4) << lr
                      << " | " << current_arch
                      << std::flush;
        }

        step++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    pool.free();
    cublasDestroy(blas);

    std::cout << "\n\n\033[1;32mTraining complete!\033[0m\n";
    std::cout << "You just did live neural architecture surgery - impossible elsewhere!\n\n";

    return 0;
}
