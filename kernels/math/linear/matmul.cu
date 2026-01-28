extern "C" __global__ void matmul_f32(
    const float* a,
    const float* b,
    float* c,
    int M, int K, int N
) {
    // Basic fractal-ready kernel implementation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}
