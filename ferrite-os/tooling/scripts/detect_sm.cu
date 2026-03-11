#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        }
        return 1;
    }

    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Ensure the runtime is initialized
    err = cudaFree(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree(0) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("%d%d\n", prop.major, prop.minor);
    return 0;
}
