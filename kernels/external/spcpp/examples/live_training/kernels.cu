// kernels.cu - GPU kernels for training

extern "C" __global__ void relu_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

extern "C" __global__ void sgd_step(float* w, float* m, int n, float lr, float momentum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simulate gradient as small random perturbation + weight decay
        float grad = 0.001f * w[i] + 0.0001f * sinf((float)i);

        // Momentum update
        m[i] = momentum * m[i] + grad;

        // Weight update
        w[i] -= lr * m[i];
    }
}

extern "C" __global__ void scale(float* x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= s;
    }
}
