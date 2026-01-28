# spcpp - Spatial C++

**C++ with Python-like imports, hot-reload, and zero build system.**

Write GPU code like this:

```cpp
#include <spcpp.hpp>

int main() {
    auto gpu = IMPORT_GPU("kernels.cu");   // JIT compiles to PTX
    auto math = IMPORT("math_ops.cpp");    // Hot-reload .so/.dll

    gpu.launch("matmul_kernel", grid, block, args);
    math->call<float>("dot_product", a, b, n);
}
```

Run it:
```bash
spcpp main.cpp
```

That's it. No CMake. No Makefile. No build system.

---

## Performance

Tested on RTX 3070:

| Precision | Throughput | vs Baseline |
|-----------|------------|-------------|
| FP32 | 18 TFLOPS | 1.0x |
| FP16 (tensor cores) | **70 TFLOPS** | 3.9x |
| INT8 | 37 TOPS | 2.1x |

Full transformer block at **57 TFLOPS** with readable code.

---

## Install

### Linux (30 seconds)

```bash
git clone https://github.com/DaronPopov/spcpp.git
cd spcpp

# Option 1: Use directly
export PATH="$PWD/bin:$PATH"
spcpp examples/hello.cpp

# Option 2: Install globally
sudo bash scripts/install.sh
```

### Windows

```batch
git clone https://github.com/DaronPopov/spcpp.git
cd spcpp

:: Build the runner
scripts\build_runner.bat

:: Use it
bin\spcpp.exe examples\hello.cpp
```

### Requirements

- **CUDA Toolkit** (nvcc)
- **C++ compiler** (g++ on Linux, cl.exe or g++ on Windows)

---

## Quick Start

### 1. Hello GPU

```cpp
// hello.cpp
#include <spcpp.hpp>

int main() {
    auto gpu = IMPORT_GPU("hello.cu");

    float* d_data;
    cudaMalloc(&d_data, 256 * sizeof(float));

    int n = 256;
    void* args[] = {&d_data, &n};
    gpu.launch("init_kernel", 1, 256, args);

    cudaFree(d_data);
    return 0;
}
```

```cuda
// hello.cu
extern "C" __global__ void init_kernel(float* data, int n) {
    int i = threadIdx.x;
    if (i < n) data[i] = i * 1.0f;
}
```

```bash
spcpp hello.cpp
```

### 2. Importable Modules

```cpp
// math_ops.cpp - A reusable module
#include <spcpp.hpp>
#include <cmath>

EXPORT float dot_product(float* a, float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

EXPORT void normalize(float* v, int n) {
    float norm = sqrt(dot_product(v, v, n));
    for (int i = 0; i < n; i++) v[i] /= norm;
}
```

```cpp
// main.cpp
#include <spcpp.hpp>

int main() {
    auto math = IMPORT("math_ops.cpp");  // Auto-compiles to .so

    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};

    float dot = math->call<float, float*, float*, int>("dot_product", a, b, 3);
    std::cout << "Dot product: " << dot << std::endl;
}
```

### 3. Mix cuBLAS + Custom Kernels

```cpp
#include <spcpp.hpp>
#include <cublas_v2.h>

int main() {
    cublasHandle_t blas;
    cublasCreate(&blas);

    auto fused = IMPORT_GPU("fused_ops.cu");

    // cuBLAS for heavy GEMM (tensor cores)
    cublasSgemm(blas, ..., d_A, d_B, d_C, ...);

    // Your fused kernel for post-processing
    fused.launch("bias_gelu", blocks, threads, args);

    cublasDestroy(blas);
}
```

---

## API Reference

### IMPORT(path)

Load a C++ module (compiles to .so/.dll automatically):

```cpp
auto mod = IMPORT("module.cpp");

// Call functions
float result = mod->call<float, int, int>("add", 1, 2);

// Get function pointer
auto fn = mod->get<float(int, int)>("add");
```

### IMPORT_GPU(path)

Load a CUDA module (compiles to .ptx automatically):

```cpp
auto gpu = IMPORT_GPU("kernels.cu");

// 1D launch
gpu.launch("kernel_name", grid, block, args);

// 2D launch
gpu.launch2d("kernel_name", gx, gy, bx, by, args);

// With shared memory
gpu.launch("kernel_name", grid, block, args, shared_bytes);
```

### EXPORT

Mark functions for export in modules:

```cpp
EXPORT int my_function(float x) { return (int)x; }
```

### IMPORT_HOT(path)

Auto-reloads module if source changed:

```cpp
while (training) {
    auto model = IMPORT_HOT("model.cpp");  // Reloads if file changed
    model->call<void>("train_step");
}
```

### STATE(type, name, default)

Declare state that persists across hot-reloads:

```cpp
// In your module (model.cpp)
STATE(int, epoch, 0);           // Persists!
STATE(float, learning_rate, 0.01f);

EXPORT void train_step() {
    epoch++;  // Survives reload
    // Edit this file, save - epoch keeps counting!
}
```

### GPU_STATE(type, name, size)

GPU memory that persists across hot-reloads:

```cpp
// Allocated once, survives reloads
GPU_STATE(float*, d_weights, 1024 * sizeof(float));

EXPORT void forward(float* input) {
    // d_weights persists even when you reload this module!
}
```

### Hot-Reload Workflow

1. Run your main program
2. Edit a module file
3. Save - module auto-reloads
4. State preserved, new code runs

```bash
# Terminal 1: Run
spcpp main.cpp

# Terminal 2: Edit while running
vim model.cpp   # Change code, save
                # See changes instantly!
```

---

## Project Structure

```
spcpp/
├── bin/spcpp              # Runner script (Linux)
├── include/
│   ├── spcpp.hpp          # Main header (Linux)
│   └── spcpp_portable.hpp # Cross-platform header
├── scripts/
│   ├── install.sh         # Linux installer
│   ├── build_runner.sh    # Build C++ runner (Linux)
│   └── build_runner.bat   # Build C++ runner (Windows)
└── src/
    └── spcpp_runner.cpp   # Cross-platform runner source
```

When you run `spcpp main.cpp`, it:

1. Creates `build/` next to your source
2. Compiles `main.cpp` → `build/bin/main`
3. Runs the binary
4. On `IMPORT()`: compiles modules → `build/lib/*.so`
5. On `IMPORT_GPU()`: compiles CUDA → `build/ptx/*.ptx`

---

## Why spcpp?

### The Problem

C++ GPU development is painful:
- CMake configs for CUDA
- Separate compilation rules
- Rebuild everything on small changes
- Complex linking

### The Solution

spcpp gives you:
- **Python-like imports** for C++ modules
- **Hot-reload** - edit, run, see results
- **JIT CUDA** - .cu files compile on demand
- **Zero config** - no build system needed
- **Full C++ ecosystem** - use any library

### What It's NOT

- Not a new language (it's C++)
- Not a framework (no lock-in)
- Not a VM (native compilation)

---

## Compatibility

### Works With

- All C++ standard library
- All NVIDIA libraries (cuBLAS, cuDNN, cuFFT, etc.)
- All header-only libraries (Eigen, nlohmann/json, etc.)
- All system libraries (OpenCV, LibTorch, etc.)

### Platforms

- Linux (tested)
- Windows (supported via portable header)
- macOS (should work, untested)

---

## Examples

See `examples/` for getting started:

- `hello.cpp` - Simple GPU kernel
- `hot_reload/` - Live coding with state preservation

Try the hot-reload demo:

```bash
spcpp examples/hot_reload/main.cpp &
# Edit examples/hot_reload/trainer.cpp while running
# Watch: epoch keeps counting, new code runs instantly!
```

---

## License

MIT

---

## Credits

Built by experimentation. Inspired by the simplicity of Python imports and the power of C++ performance.
