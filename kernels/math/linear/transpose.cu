extern "C" __global__ void transpose_f32(const float* __restrict__ input, float* __restrict__ output, int width, int height) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < width && y_idx < height) {
        int in_idx = y_idx * width + x_idx;
        int out_idx = x_idx * height + y_idx;
        output[out_idx] = input[in_idx];
    }
}
