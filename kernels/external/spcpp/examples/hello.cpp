// hello.cpp - Simple spcpp example
#include <spcpp.hpp>

int main() {
    std::cout << "=== spcpp Hello World ===" << std::endl;

    // Load GPU module
    auto gpu = IMPORT_GPU("hello.cu");

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, 256 * sizeof(float));

    // Launch kernel
    int n = 256;
    void* args[] = {&d_data, &n};
    gpu.launch("init_kernel", 1, 256, args);

    // Copy back and verify
    float h_data[256];
    cudaMemcpy(h_data, d_data, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    std::cout << "Success!" << std::endl;
    return 0;
}
