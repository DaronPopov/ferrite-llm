extern "C" __global__ void relu_f32(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}
