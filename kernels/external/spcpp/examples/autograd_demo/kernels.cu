// kernels.cu - CUDA kernels for autograd operations

extern "C" __global__ void relu_forward(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

extern "C" __global__ void relu_backward(float* grad_in, const float* grad_out,
                                          const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] += (in[i] > 0.0f) ? grad_out[i] : 0.0f;
}

extern "C" __global__ void mse_loss(float* out, const float* pred,
                                     const float* target, int n) {
    __shared__ float sdata[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = (i < n) ? (pred[i] - target[i]) : 0.0f;
    sdata[threadIdx.x] = diff * diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0] / n);
}

extern "C" __global__ void mse_backward(float* grad, const float* pred,
                                         const float* target, float upstream, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] += upstream * 2.0f * (pred[i] - target[i]) / n;
}

extern "C" __global__ void sgd_step(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

extern "C" __global__ void zero_mem(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0.0f;
}

extern "C" __global__ void add_grad(float* grad, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] += val;
}
