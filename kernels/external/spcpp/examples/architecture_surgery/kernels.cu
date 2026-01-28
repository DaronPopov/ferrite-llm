// kernels.cu - GPU kernels for architecture surgery demo

// ============================================================
// ACTIVATION FUNCTIONS - model.cpp chooses which to use
// ============================================================

extern "C" __global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}

extern "C" __global__ void leaky_relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.01f * x[i];
}

extern "C" __global__ void tanh_act(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

extern "C" __global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // GELU approximation
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

extern "C" __global__ void sigmoid(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

extern "C" __global__ void swish(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ============================================================
// LOSS FUNCTIONS
// ============================================================

extern "C" __global__ void mse_loss(float* loss, const float* pred, const float* target, int n) {
    __shared__ float sdata[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = (i < n) ? (pred[i] - target[i]) : 0.0f;
    sdata[threadIdx.x] = diff * diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss, sdata[0] / n);
}

extern "C" __global__ void mse_grad(float* grad, const float* pred, const float* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.0f * (pred[i] - target[i]) / n;
}

// ============================================================
// OPTIMIZER
// ============================================================

extern "C" __global__ void sgd_update(float* W, float lr, int in_dim, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = in_dim * out_dim;
    if (i < n) {
        // Simple weight decay + noise as proxy for gradient
        W[i] -= lr * (0.001f * W[i] + 0.0001f * sinf((float)i * 0.1f));
    }
}

// ============================================================
// UTILITIES
// ============================================================

extern "C" __global__ void scale(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

extern "C" __global__ void add_noise(float* x, float scale, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simple pseudo-random noise
        float r = sinf((float)(i + seed) * 12.9898f) * 43758.5453f;
        r = r - floorf(r);
        x[i] += (r - 0.5f) * scale;
    }
}
