// hello.cu - Simple CUDA kernel
extern "C" __global__ void init_kernel(float* data, int n) {
    int i = threadIdx.x;
    if (i < n) {
        data[i] = i * 1.0f;
    }
}
