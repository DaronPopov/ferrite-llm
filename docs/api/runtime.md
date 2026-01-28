# Runtime API

The `Runtime` type provides a unified interface for CPU and GPU execution.

## Import

```rust
use ferrite::compute::{Runtime, DeviceType};
```

## Overview

Runtime abstracts over execution backends, allowing the same code to run on CPU or GPU:

```rust
let mut runtime = Runtime::auto();  // Automatically select best device

runtime.alloc("weights", &[784, 256]);
runtime.matmul("input", "weights", "output");
```

## Device Selection

### auto

```rust
pub fn auto() -> Self
```

Automatically selects the best available device. Prefers GPU if available, falls back to CPU.

```rust
let mut runtime = Runtime::auto();
println!("Using: {:?}", runtime.device_type());
```

### cpu

```rust
pub fn cpu() -> Self
```

Forces CPU execution.

```rust
let mut runtime = Runtime::cpu();
```

### gpu

```rust
pub fn gpu() -> Result<Self, String>
```

Forces GPU execution. Returns an error if no GPU is available.

```rust
let mut runtime = Runtime::gpu().expect("No GPU available");
```

## Device Information

### device_type

```rust
pub fn device_type(&self) -> DeviceType
```

Returns the current device type.

```rust
pub enum DeviceType {
    Cpu,
    Gpu,
}
```

```rust
match runtime.device_type() {
    DeviceType::Cpu => println!("Running on CPU"),
    DeviceType::Gpu => println!("Running on GPU"),
}
```

## Tensor Operations

### alloc

```rust
pub fn alloc(&mut self, name: &str, shape: &[usize])
```

Allocates a tensor with the given shape.

```rust
runtime.alloc("input", &[32, 784]);
runtime.alloc("weights", &[784, 256]);
runtime.alloc("output", &[32, 256]);
```

### fill

```rust
pub fn fill(&mut self, name: &str, value: f32)
```

Fills a tensor with a constant value.

```rust
runtime.fill("bias", 0.0);
```

### upload

```rust
pub fn upload(&mut self, name: &str, data: &[f32])
```

Uploads host data to a tensor.

```rust
let input_data: Vec<f32> = vec![0.5; 32 * 784];
runtime.upload("input", &input_data);
```

### download

```rust
pub fn download(&self, name: &str) -> Vec<f32>
```

Downloads tensor data to host.

```rust
let result = runtime.download("output");
```

## Compute Operations

### matmul

```rust
pub fn matmul(&self, a: &str, b: &str, c: &str)
```

Matrix multiplication: C = A * B

Dimensions are inferred from allocated tensor shapes.

```rust
runtime.alloc("a", &[32, 128]);
runtime.alloc("b", &[128, 64]);
runtime.alloc("c", &[32, 64]);
runtime.matmul("a", "b", "c");
```

### add

```rust
pub fn add(&self, a: &str, b: &str, c: &str)
```

Element-wise addition: C = A + B

```rust
runtime.add("x", "residual", "output");
```

### relu

```rust
pub fn relu(&mut self, x: &str)
```

In-place ReLU activation.

```rust
runtime.relu("hidden");
```

### sigmoid

```rust
pub fn sigmoid(&mut self, x: &str)
```

In-place sigmoid activation.

### softmax

```rust
pub fn softmax(&mut self, x: &str)
```

In-place softmax over the last dimension.

```rust
runtime.softmax("logits");
```

## Example: Device-Agnostic Model

```rust
use ferrite::compute::{Runtime, DeviceType};

struct SimpleModel {
    runtime: Runtime,
}

impl SimpleModel {
    fn new() -> Self {
        let mut runtime = Runtime::auto();

        // Allocate model parameters
        runtime.alloc("w1", &[784, 256]);
        runtime.alloc("b1", &[256]);
        runtime.alloc("w2", &[256, 10]);
        runtime.alloc("b2", &[10]);

        // Initialize weights (simplified)
        runtime.fill("w1", 0.01);
        runtime.fill("b1", 0.0);
        runtime.fill("w2", 0.01);
        runtime.fill("b2", 0.0);

        // Allocate intermediate buffers
        runtime.alloc("h1", &[32, 256]);
        runtime.alloc("output", &[32, 10]);

        SimpleModel { runtime }
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.runtime.alloc("input", &[32, 784]);
        self.runtime.upload("input", input);

        // Layer 1
        self.runtime.matmul("input", "w1", "h1");
        self.runtime.add("h1", "b1", "h1");
        self.runtime.relu("h1");

        // Layer 2
        self.runtime.matmul("h1", "w2", "output");
        self.runtime.add("output", "b2", "output");
        self.runtime.softmax("output");

        self.runtime.download("output")
    }
}

fn main() {
    let mut model = SimpleModel::new();
    println!("Device: {:?}", model.runtime.device_type());

    let input = vec![0.5f32; 32 * 784];
    let output = model.forward(&input);

    println!("Output shape: {}", output.len());
    println!("First prediction: {:?}", &output[0..10]);
}
```

## CPU Backend Details

When running on CPU, the runtime uses optimized implementations:

- **MatMul**: Blocked algorithm with cache optimization
- **Activations**: SIMD-vectorized where available
- **Memory**: Standard heap allocation

Performance is significantly lower than GPU but useful for:

- Development without GPU access
- Small models where GPU overhead dominates
- Testing and validation

## GPU Backend Details

When running on GPU, the runtime uses:

- **MatMul**: cuBLAS SGEMM
- **Activations**: JIT-compiled CUDA kernels
- **Memory**: TLSF allocator with pre-allocated pool

## Performance Comparison

Typical performance for a 784 -> 256 -> 10 MLP with batch size 32:

| Device | Forward Pass |
|--------|--------------|
| CPU (single thread) | ~500 us |
| CPU (multi-thread) | ~100 us |
| GPU (RTX 3090) | ~50 us |

GPU advantage increases with model size and batch size.
