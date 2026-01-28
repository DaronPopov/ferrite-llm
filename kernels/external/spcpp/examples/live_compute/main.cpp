// main.cpp - Live GPU Compute Sandbox
//
// A GPU buffer runs in a loop. Edit compute.cu to change what happens to it!
// Like a REPL for CUDA.

#include <spcpp.hpp>
#include <thread>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <cmath>

volatile bool running = true;
void signal_handler(int) { running = false; }

constexpr int N = 1024;

int main() {
    signal(SIGINT, signal_handler);

    std::cout << "\n\033[1;35m╔═════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;35m║   LIVE COMPUTE - Edit GPU Code at Runtime   ║\033[0m\n";
    std::cout << "\033[1;35m╚═════════════════════════════════════════════╝\033[0m\n\n";

    std::cout << "Edit \033[1;33mcompute.cu\033[0m while running!\n";
    std::cout << "Try changing the transform() kernel.\n\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // GPU buffer - persists across hot-reloads
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize with pattern
    std::vector<float> h_init(N);
    for (int i = 0; i < N; i++) h_init[i] = (float)i / N;
    cudaMemcpy(d_data, h_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_result(N);
    int frame = 0;

    while (running) {
        // Hot-reload compute.cu if changed
        auto gpu = IMPORT_GPU("compute.cu");

        // Apply transformation
        float t = frame * 0.01f;
        int n = N;
        void* args[] = {&d_data, &n, &t};
        gpu.launch("transform", (N + 255) / 256, 256, args);

        // Read back and display
        cudaMemcpy(h_result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute stats
        float sum = 0, min_v = h_result[0], max_v = h_result[0];
        for (int i = 0; i < N; i++) {
            sum += h_result[i];
            min_v = std::min(min_v, h_result[i]);
            max_v = std::max(max_v, h_result[i]);
        }
        float mean = sum / N;

        // ASCII visualization (first 64 values)
        std::cout << "\r\033[K";
        std::cout << "\033[1;36mFrame " << std::setw(5) << frame << "\033[0m | ";
        std::cout << "Mean: " << std::fixed << std::setprecision(3) << mean << " | ";
        std::cout << "[" << min_v << ", " << max_v << "] | ";

        // Mini sparkline
        for (int i = 0; i < 32; i++) {
            int idx = i * (N / 32);
            float v = (h_result[idx] - min_v) / (max_v - min_v + 0.001f);
            v = std::max(0.0f, std::min(1.0f, v));
            const char* blocks = " ▁▂▃▄▅▆▇█";
            int level = (int)(v * 8);
            std::cout << &blocks[level * 3];  // UTF-8 chars are 3 bytes
        }
        std::cout << std::flush;

        frame++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cudaFree(d_data);
    std::cout << "\n\n\033[1;32mDone!\033[0m\n";
    return 0;
}
