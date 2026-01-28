// CUDA kernels for Semantic Core RS via spcpp JIT
// Compiled automatically by IMPORT_GPU()

extern "C" __global__ void relu_forward(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = fmaxf(0.0f, data[i]);
}

extern "C" __global__ void relu_backward(float* grad, float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = x[i] > 0.0f ? grad[i] : 0.0f;
}

extern "C" __global__ void sigmoid_forward(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 1.0f / (1.0f + expf(-data[i]));
}

extern "C" __global__ void tanh_forward(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = tanhf(data[i]);
}

extern "C" __global__ void gelu_forward(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

extern "C" __global__ void silu_forward(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void leaky_relu_forward(float* data, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x > 0.0f ? x : alpha * x;
    }
}

extern "C" __global__ void softmax_forward(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float* row_data = data + row * cols;

        float max_val = row_data[0];
        for (int i = 1; i < cols; i++) max_val = fmaxf(max_val, row_data[i]);

        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            row_data[i] = expf(row_data[i] - max_val);
            sum += row_data[i];
        }

        for (int i = 0; i < cols; i++) row_data[i] /= sum;
    }
}

extern "C" __global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void sub_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

extern "C" __global__ void mul_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

extern "C" __global__ void div_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] / b[i];
}

extern "C" __global__ void scale_kernel(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scalar;
}

extern "C" __global__ void exp_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = expf(data[i]);
}

extern "C" __global__ void log_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = logf(data[i]);
}

extern "C" __global__ void sqrt_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = sqrtf(data[i]);
}

extern "C" __global__ void neg_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = -data[i];
}

extern "C" __global__ void abs_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = fabsf(data[i]);
}

extern "C" __global__ void fill_kernel(float* data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

extern "C" __global__ void sgd_step(float* weights, float* momentum, float* grad, int n,
                                     float lr, float mom, float weight_decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i] + weight_decay * weights[i];
        momentum[i] = mom * momentum[i] + g;
        weights[i] -= lr * momentum[i];
    }
}

extern "C" __global__ void adam_step(float* weights, float* m, float* v, float* grad, int n,
                                      float lr, float beta1, float beta2, float epsilon,
                                      float weight_decay, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i] + weight_decay * weights[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[i] / (1.0f - powf(beta2, (float)t));
        weights[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

extern "C" __global__ void layer_norm_forward(float* x, float* gamma, float* beta, float* out,
                                               int batch, int dim, float eps) {
    int b = blockIdx.x;
    if (b < batch) {
        float* x_row = x + b * dim;
        float* out_row = out + b * dim;

        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x_row[i];
        mean /= dim;

        float var = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= dim;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) {
            float norm = (x_row[i] - mean) * inv_std;
            out_row[i] = gamma[i] * norm + beta[i];
        }
    }
}

extern "C" __global__ void dropout_forward(float* data, float* mask, int n, float p, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int state = seed + i * 1103515245u;
        state = state * 1103515245u + 12345u;
        float r = (float)(state & 0x7FFFFFFF) / 2147483647.0f;

        if (r < p) {
            mask[i] = 0.0f;
            data[i] = 0.0f;
        } else {
            mask[i] = 1.0f / (1.0f - p);
            data[i] *= mask[i];
        }
    }
}

extern "C" __global__ void embedding_lookup(float* table, int* indices, float* out,
                                             int batch, int embed_dim) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b < batch && d < embed_dim) {
        int idx = indices[b];
        out[b * embed_dim + d] = table[idx * embed_dim + d];
    }
}

extern "C" __global__ void transpose_2d(float* src, float* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        dst[j * rows + i] = src[i * cols + j];
    }
}

extern "C" __global__ void add_bias(float* data, float* bias, int batch, int features) {
    int b = blockIdx.x;
    int f = threadIdx.x;
    if (b < batch && f < features) {
        data[b * features + f] += bias[f];
    }
}

extern "C" __global__ void cross_entropy_loss(float* logits, int* labels, float* loss,
                                               int batch_size, int num_classes) {
    __shared__ float s_loss;
    if (threadIdx.x == 0) s_loss = 0.0f;
    __syncthreads();

    int i = threadIdx.x;
    if (i < batch_size) {
        float* row = logits + i * num_classes;
        int label = labels[i];

        float max_val = row[0];
        for (int j = 1; j < num_classes; j++) max_val = fmaxf(max_val, row[j]);

        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) sum += expf(row[j] - max_val);

        float log_prob = (row[label] - max_val) - logf(sum);
        atomicAdd(&s_loss, -log_prob);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *loss = s_loss / (float)batch_size;
    }
}

// Quantization kernels (from your custom implementation)
extern "C" __global__ void quantize_f32_to_int8(
    const float* input, signed char* output, int n, float scale, int zero_point
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = input[i] / scale + (float)zero_point;
        val = fminf(fmaxf(val, -128.0f), 127.0f);
        output[i] = (signed char)roundf(val);
    }
}

extern "C" __global__ void dequantize_int8_to_f32(
    const signed char* input, float* output, int n, float scale, int zero_point
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = ((float)input[i] - (float)zero_point) * scale;
    }
}
