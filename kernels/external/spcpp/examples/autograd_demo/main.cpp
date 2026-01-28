// main.cpp - Autograd Demo: Neural Network with Automatic Differentiation
//
// Shows spcpp as an AI foundation:
// - Tensor with GPU storage
// - cuBLAS matmul with autograd
// - Custom CUDA kernels via IMPORT_GPU
// - SGD optimizer

#include <spcpp.hpp>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>

// ============================================================
// TENSOR CLASS
// ============================================================

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

struct Tensor {
    float* data = nullptr;
    float* grad = nullptr;
    std::vector<int> shape;
    int size = 0;
    bool requires_grad = false;

    std::function<void()> backward_fn;
    std::vector<TensorPtr> parents;

    Tensor(const std::vector<int>& shape_, bool requires_grad_ = false)
        : shape(shape_), requires_grad(requires_grad_) {
        size = 1;
        for (int d : shape) size *= d;
        cudaMalloc(&data, size * sizeof(float));
        cudaMemset(data, 0, size * sizeof(float));
        if (requires_grad) {
            cudaMalloc(&grad, size * sizeof(float));
            cudaMemset(grad, 0, size * sizeof(float));
        }
    }

    ~Tensor() {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);
    }

    void copy_from(const std::vector<float>& h) {
        cudaMemcpy(data, h.data(), std::min((int)h.size(), size) * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    std::vector<float> to_host() const {
        std::vector<float> h(size);
        cudaMemcpy(h.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost);
        return h;
    }

    void zero_grad() {
        if (grad) cudaMemset(grad, 0, size * sizeof(float));
    }

    int dim(int i) const { return shape[i]; }
};

// ============================================================
// AUTOGRAD OPERATIONS
// ============================================================

class Autograd {
    cublasHandle_t blas;
    spc::CudaModule* gpu;

public:
    Autograd(spc::CudaModule* gpu_) : gpu(gpu_) {
        cublasCreate(&blas);
    }

    ~Autograd() {
        cublasDestroy(blas);
    }

    // Matmul: C = A @ B with autograd
    TensorPtr matmul(TensorPtr A, TensorPtr B) {
        int M = A->dim(0), K = A->dim(1), N = B->dim(1);
        auto C = std::make_shared<Tensor>(std::vector<int>{M, N},
                                          A->requires_grad || B->requires_grad);

        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B->data, N, A->data, K, &beta, C->data, N);

        if (C->requires_grad) {
            C->parents = {A, B};
            cublasHandle_t blas_copy = blas;
            C->backward_fn = [A, B, C, M, K, N, blas_copy]() {
                float alpha = 1.0f, beta = 1.0f;
                if (A->requires_grad) {
                    cublasSgemm(blas_copy, CUBLAS_OP_T, CUBLAS_OP_N,
                                K, M, N, &alpha, B->data, N, C->grad, N, &beta, A->grad, K);
                }
                if (B->requires_grad) {
                    cublasSgemm(blas_copy, CUBLAS_OP_N, CUBLAS_OP_T,
                                N, K, M, &alpha, C->grad, N, A->data, K, &beta, B->grad, N);
                }
            };
        }
        return C;
    }

    // ReLU with autograd
    TensorPtr relu(TensorPtr x) {
        auto out = std::make_shared<Tensor>(x->shape, x->requires_grad);
        int n = x->size;
        int blocks = (n + 255) / 256;

        void* args[] = {&out->data, &x->data, &n};
        gpu->launch("relu_forward", blocks, 256, args);

        if (out->requires_grad) {
            out->parents = {x};
            auto gpu_ptr = gpu;
            out->backward_fn = [x, out, n, gpu_ptr]() {
                int blocks = (n + 255) / 256;
                int size = n;
                void* args[] = {&x->grad, &out->grad, &x->data, &size};
                gpu_ptr->launch("relu_backward", blocks, 256, args);
            };
        }
        return out;
    }

    // MSE Loss
    TensorPtr mse_loss(TensorPtr pred, TensorPtr target) {
        auto loss = std::make_shared<Tensor>(std::vector<int>{1}, pred->requires_grad);
        int n = pred->size;
        int blocks = (n + 255) / 256;

        void* args[] = {&loss->data, &pred->data, &target->data, &n};
        gpu->launch("mse_loss", blocks, 256, args);

        if (loss->requires_grad) {
            loss->parents = {pred};
            auto gpu_ptr = gpu;
            loss->backward_fn = [pred, target, loss, n, gpu_ptr]() {
                int blocks = (n + 255) / 256;
                float upstream = 1.0f;
                int size = n;
                void* args[] = {&pred->grad, &pred->data, &target->data, &upstream, &size};
                gpu_ptr->launch("mse_backward", blocks, 256, args);
            };
        }
        return loss;
    }

    // Backward pass
    void backward(TensorPtr loss) {
        if (!loss->requires_grad) return;

        // Set loss gradient to 1
        float one = 1.0f;
        cudaMemcpy(loss->grad, &one, sizeof(float), cudaMemcpyHostToDevice);

        // Topological sort and backprop
        std::function<void(TensorPtr)> backprop = [&](TensorPtr t) {
            if (t->backward_fn) t->backward_fn();
            for (auto& p : t->parents) {
                if (p) backprop(p);
            }
        };
        backprop(loss);
    }

    // SGD step
    void sgd_step(std::vector<TensorPtr>& params, float lr) {
        for (auto& p : params) {
            if (p->grad) {
                int n = p->size;
                int blocks = (n + 255) / 256;
                void* args[] = {&p->data, &p->grad, &lr, &n};
                gpu->launch("sgd_step", blocks, 256, args);
            }
        }
        cudaDeviceSynchronize();
    }

    void zero_grad(std::vector<TensorPtr>& params) {
        for (auto& p : params) p->zero_grad();
    }
};

// ============================================================
// LINEAR LAYER
// ============================================================

struct Linear {
    TensorPtr weight;
    int in_dim, out_dim;

    Linear(int in_, int out_) : in_dim(in_), out_dim(out_) {
        weight = std::make_shared<Tensor>(std::vector<int>{in_, out_}, true);

        // Xavier init
        std::vector<float> w(in_ * out_);
        float std = std::sqrt(2.0f / (in_ + out_));
        for (auto& x : w) {
            float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            x = std * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        }
        weight->copy_from(w);
    }

    TensorPtr forward(Autograd& ag, TensorPtr x) {
        return ag.matmul(x, weight);
    }
};

// ============================================================
// MAIN
// ============================================================

int main() {
    srand(42);

    std::cout << "\n\033[1;35m╔═══════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;35m║   AUTOGRAD DEMO - Neural Net with Backprop    ║\033[0m\n";
    std::cout << "\033[1;35m╚═══════════════════════════════════════════════╝\033[0m\n\n";

    // Load kernels
    auto gpu = IMPORT_GPU("kernels.cu");
    Autograd ag(&gpu);

    // Create network: 4 -> 8 -> 2
    Linear layer1(4, 8);
    Linear layer2(8, 2);

    std::vector<TensorPtr> params = {layer1.weight, layer2.weight};

    // Training data (XOR-like)
    auto X = std::make_shared<Tensor>(std::vector<int>{4, 4}, false);
    X->copy_from({0,0,0,0, 0,1,1,0, 1,0,0,1, 1,1,1,1});

    auto Y = std::make_shared<Tensor>(std::vector<int>{4, 2}, false);
    Y->copy_from({0,1, 1,0, 1,0, 0,1});

    std::cout << "\033[1;36m[Training]\033[0m Network 4->8->2 on XOR pattern\n\n";

    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        ag.zero_grad(params);

        // Forward
        auto h1 = layer1.forward(ag, X);
        auto a1 = ag.relu(h1);
        auto h2 = layer2.forward(ag, a1);
        auto loss = ag.mse_loss(h2, Y);

        // Backward
        ag.backward(loss);

        // Update
        ag.sgd_step(params, 0.05f);

        if (epoch % 100 == 0) {
            float loss_val;
            cudaMemcpy(&loss_val, loss->data, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "  Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss_val
                      << std::endl;
        }
    }

    // Test
    std::cout << "\n\033[1;36m[Results]\033[0m Final predictions:\n\n";
    auto h1 = layer1.forward(ag, X);
    auto a1 = ag.relu(h1);
    auto pred = layer2.forward(ag, a1);

    auto p = pred->to_host();
    auto t = Y->to_host();

    for (int i = 0; i < 4; i++) {
        std::cout << "  Sample " << i << ": pred=[" << std::fixed << std::setprecision(2)
                  << p[i*2] << ", " << p[i*2+1] << "]  target=["
                  << t[i*2] << ", " << t[i*2+1] << "]\n";
    }

    std::cout << "\n\033[1;32m[Done]\033[0m Autograd working with cuBLAS + custom CUDA kernels!\n\n";

    return 0;
}
