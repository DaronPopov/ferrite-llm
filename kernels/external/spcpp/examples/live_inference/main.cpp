// main.cpp - LIVE INFERENCE: Edit Model While Serving
//
// An inference server where you can:
//   - Edit the model architecture
//   - Change post-processing logic
//   - Add debug outputs
//   - Swap entire model implementations
// ALL WITHOUT STOPPING THE SERVER!

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <thread>
#include <chrono>
#include <csignal>
#include <iostream>
#include <iomanip>
#include <cmath>

volatile bool running = true;
void signal_handler(int) { running = false; }

constexpr int DIM = 64;

int main() {
    signal(SIGINT, signal_handler);
    srand(42);

    std::cout << "\n";
    std::cout << "\033[1;35m╔══════════════════════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;35m║     LIVE INFERENCE SERVER                                    ║\033[0m\n";
    std::cout << "\033[1;35m║     Edit model.cpp while serving predictions!               ║\033[0m\n";
    std::cout << "\033[1;35m╚══════════════════════════════════════════════════════════════╝\033[0m\n\n";

    std::cout << "Edit \033[1;33mmodel.cpp\033[0m to:\n";
    std::cout << "  • Change model architecture\n";
    std::cout << "  • Modify output transformations\n";
    std::cout << "  • Add confidence thresholds\n";
    std::cout << "  • Toggle debug mode\n";
    std::cout << "  • Completely swap model logic\n\n";
    std::cout << "\033[1;32mChanges take effect on next prediction - zero downtime!\033[0m\n\n";

    // Initialize
    cublasHandle_t blas;
    cublasCreate(&blas);
    auto gpu = IMPORT_GPU("kernels.cu");

    // Allocate model weights (persist across hot-reloads)
    float *d_W1, *d_W2, *d_input, *d_hidden, *d_output;
    cudaMalloc(&d_W1, DIM * DIM * sizeof(float));
    cudaMalloc(&d_W2, DIM * DIM * sizeof(float));
    cudaMalloc(&d_input, DIM * sizeof(float));
    cudaMalloc(&d_hidden, DIM * sizeof(float));
    cudaMalloc(&d_output, DIM * sizeof(float));

    // Initialize weights
    std::vector<float> w1(DIM * DIM), w2(DIM * DIM);
    for (int i = 0; i < DIM * DIM; i++) {
        w1[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 0.2f;
        w2[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 0.2f;
    }
    cudaMemcpy(d_W1, w1.data(), DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, w2.data(), DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Bundle pointers for model
    struct ModelContext {
        float *d_W1, *d_W2, *d_input, *d_hidden, *d_output;
        cublasHandle_t blas;
        spc::CudaModule* gpu;
        int dim;
    };
    ModelContext ctx = {d_W1, d_W2, d_input, d_hidden, d_output, blas, &gpu, DIM};

    std::cout << "\033[1;36m[Server]\033[0m Starting inference loop...\n\n";

    int request_id = 0;
    std::string last_version = "";

    while (running) {
        // Simulate incoming request (random input)
        std::vector<float> input(DIM);
        for (int i = 0; i < DIM; i++) {
            input[i] = sinf(request_id * 0.1f + i * 0.2f);
        }
        cudaMemcpy(d_input, input.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);

        // Hot-reload model if changed
        auto model = IMPORT_HOT("model.cpp");

        // Get model version
        char version[128];
        model->call<void, char*>("get_version", version);
        std::string current_version(version);

        if (current_version != last_version) {
            std::cout << "\n\033[1;33m[HOT-RELOAD]\033[0m Model updated: " << current_version << "\n\n";
            last_version = current_version;
        }

        // Run inference
        char result[512];
        model->call<void, ModelContext*, int, char*>("predict", &ctx, request_id, result);

        // Display result
        std::cout << "\r\033[K"
                  << "\033[1;32m[Req " << std::setw(5) << request_id << "]\033[0m "
                  << result
                  << std::flush;

        request_id++;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Cleanup
    cudaFree(d_W1); cudaFree(d_W2);
    cudaFree(d_input); cudaFree(d_hidden); cudaFree(d_output);
    cublasDestroy(blas);

    std::cout << "\n\n\033[1;32m[Server]\033[0m Shutdown complete.\n";
    return 0;
}
