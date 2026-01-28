// kernels.cu - GPU kernels for inference

extern "C" __global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}

extern "C" __global__ void tanh_act(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

extern "C" __global__ void sigmoid(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

extern "C" __global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

extern "C" __global__ void softmax(float* x, int n) {
    // Single-block softmax for small vectors
    __shared__ float smax, ssum;

    if (threadIdx.x == 0) {
        smax = x[0];
        for (int i = 1; i < n; i++) smax = fmaxf(smax, x[i]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        ssum = 0;
        for (int i = 0; i < n; i++) {
            x[i] = expf(x[i] - smax);
            ssum += x[i];
        }
    }
    __syncthreads();

    int i = threadIdx.x;
    if (i < n) x[i] /= ssum;
}
