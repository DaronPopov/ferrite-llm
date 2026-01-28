# Stream API

The `Stream` type provides zero-copy GPU execution. All operations execute directly on GPU memory without intermediate host transfers.

## Import

```rust
use ferrite::compute::Stream;
```

## Construction

### new

```rust
pub fn new(allocator: Arc<TlsfAllocator>) -> Self
```

Creates a new execution stream backed by the given allocator.

```rust
let device = CudaDevice::new(0).unwrap();
let allocator = Arc::new(TlsfAllocator::new(device, 128 * 1024 * 1024));
let mut stream = Stream::new(allocator);
```

### init

```rust
pub fn init(&mut self) -> Result<(), String>
```

Initializes the SPCPP backend. Must be called before any GPU operations.

```rust
stream.init().expect("Failed to initialize");
```

## Memory Operations

### alloc

```rust
pub fn alloc(&mut self, name: &str, num_floats: usize) -> GpuHandle
```

Allocates GPU memory for a tensor. Returns a handle containing the offset and size.

| Parameter | Type | Description |
|-----------|------|-------------|
| name | &str | Tensor identifier |
| num_floats | usize | Number of f32 elements |

```rust
stream.alloc("weights", 784 * 256);
stream.alloc("bias", 256);
```

### download

```rust
pub fn download(&mut self, name: &str) -> Vec<f32>
```

Copies tensor data from GPU to host. This is the only operation that performs a device-to-host transfer.

```rust
let result = stream.download("output");
println!("First element: {}", result[0]);
```

### upload

```rust
pub fn upload(&mut self, name: &str)
```

Copies staged host data to GPU. Must call `stage_host` first.

```rust
stream.stage_host("input", vec![1.0, 2.0, 3.0, 4.0]);
stream.upload("input");
```

### stage_host

```rust
pub fn stage_host(&mut self, name: &str, data: Vec<f32>)
```

Stages host data for later upload. Does not perform any GPU transfer.

```rust
stream.stage_host("input", input_data);
```

### copy

```rust
pub fn copy(&self, src: &str, dst: &str, n: i32)
```

Copies data between GPU tensors. No host involvement.

```rust
stream.copy("weights_v1", "weights_v2", num_elements);
```

## Initialization Operations

### fill

```rust
pub fn fill(&self, x: &str, value: f32, n: i32)
```

Fills a tensor with a constant value.

```rust
stream.fill("bias", 0.0, 256);
```

### fill_at

```rust
pub fn fill_at(&self, x: &str, offset: i32, value: f32, n: i32)
```

Fills a portion of a tensor starting at the given offset.

```rust
stream.fill_at("buffer", 100, 1.0, 50);  // Fill elements 100-149
```

### init_weights

```rust
pub fn init_weights(&self, name: &str, rows: i32, cols: i32, init_type: &str)
```

Initializes weight tensor using the specified strategy.

| init_type | Description |
|-----------|-------------|
| "xavier" | Xavier/Glorot initialization |
| "he" | He initialization for ReLU networks |
| "normal" | Standard normal distribution |
| "uniform" | Uniform distribution [-1, 1] |

```rust
stream.init_weights("w1", 784, 256, "xavier");
stream.init_weights("w2", 256, 10, "he");
```

## Matrix Operations

### matmul

```rust
pub fn matmul(&self, a: &str, b: &str, c: &str, m: i32, k: i32, n: i32)
```

Matrix multiplication: C = A * B

| Parameter | Description |
|-----------|-------------|
| a | Input matrix A [m, k] |
| b | Input matrix B [k, n] |
| c | Output matrix C [m, n] |
| m | Rows of A and C |
| k | Columns of A, rows of B |
| n | Columns of B and C |

```rust
stream.matmul("a", "b", "c", 64, 128, 32);
```

### gemm

```rust
pub fn gemm(&self, a: &str, b: &str, c: &str, m: i32, k: i32, n: i32,
            alpha: f32, beta: f32, trans_a: bool, trans_b: bool)
```

General matrix multiplication: C = alpha * op(A) * op(B) + beta * C

```rust
// C = 1.0 * A^T * B + 0.0 * C
stream.gemm("a", "b", "c", m, k, n, 1.0, 0.0, true, false);
```

### linear

```rust
pub fn linear(&self, input: &str, weights: &str, bias: Option<&str>, output: &str,
              batch: i32, in_features: i32, out_features: i32)
```

Linear layer: output = input * weights + bias

```rust
stream.linear("input", "w1", Some("b1"), "hidden", 32, 784, 256);
stream.linear("hidden", "w2", None, "output", 32, 256, 10);  // No bias
```

## Activation Functions

All activation functions operate in-place.

### relu

```rust
pub fn relu(&self, x: &str, n: i32)
```

Rectified Linear Unit: x = max(0, x)

```rust
stream.relu("hidden", batch * hidden_dim);
```

### sigmoid

```rust
pub fn sigmoid(&self, x: &str, n: i32)
```

Sigmoid activation: x = 1 / (1 + exp(-x))

```rust
stream.sigmoid("gate", batch * dim);
```

### tanh

```rust
pub fn tanh(&self, x: &str, n: i32)
```

Hyperbolic tangent: x = tanh(x)

```rust
stream.tanh("state", batch * dim);
```

### gelu

```rust
pub fn gelu(&self, x: &str, n: i32)
```

Gaussian Error Linear Unit.

```rust
stream.gelu("hidden", batch * dim);
```

### silu

```rust
pub fn silu(&self, x: &str, n: i32)
```

Sigmoid Linear Unit (Swish): x = x * sigmoid(x)

```rust
stream.silu("hidden", batch * dim);
```

### softmax

```rust
pub fn softmax(&self, x: &str, rows: i32, cols: i32)
```

Softmax over the last dimension.

```rust
stream.softmax("logits", batch, num_classes);
```

## Element-wise Operations

### scale

```rust
pub fn scale(&self, x: &str, scalar: f32, n: i32)
```

Scalar multiplication: x = x * scalar

```rust
stream.scale("gradients", 0.01, num_params);
```

### add

```rust
pub fn add(&self, a: &str, b: &str, c: &str, n: i32)
```

Element-wise addition: c = a + b

```rust
stream.add("x", "residual", "output", batch * dim);
```

### mul

```rust
pub fn mul(&self, a: &str, b: &str, c: &str, n: i32)
```

Element-wise multiplication: c = a * b

```rust
stream.mul("gate", "hidden", "output", batch * dim);
```

### axpy

```rust
pub fn axpy(&self, x: &str, y: &str, n: i32, alpha: f32)
```

BLAS axpy: y = alpha * x + y

```rust
stream.axpy("gradient", "accumulator", num_params, 1.0);
```

## Reduction Operations

### dot

```rust
pub fn dot(&self, x: &str, y: &str, n: i32) -> f32
```

Dot product of two vectors.

```rust
let similarity = stream.dot("vec_a", "vec_b", dim);
```

### nrm2

```rust
pub fn nrm2(&self, x: &str, n: i32) -> f32
```

L2 norm of a vector.

```rust
let norm = stream.nrm2("weights", num_weights);
```

## Loss Functions

### mse_loss

```rust
pub fn mse_loss(&self, pred: &str, target: &str, n: i32) -> f32
```

Mean squared error loss.

```rust
let loss = stream.mse_loss("predictions", "targets", batch * output_dim);
```

### cross_entropy

```rust
pub fn cross_entropy(&self, logits: &str, labels: *mut i32, batch: i32, classes: i32) -> f32
```

Cross entropy loss with integer labels.

```rust
let loss = stream.cross_entropy("logits", labels_ptr, batch, num_classes);
```

## Optimizer Operations

### set_optimizer

```rust
pub fn set_optimizer(&self, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32)
```

Configure optimizer hyperparameters for Adam.

```rust
stream.set_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0);
```

### adam_step

```rust
pub fn adam_step(&self, name: &str, weights: &str, grad: &str, n: i32)
```

Perform one Adam optimizer step.

```rust
stream.adam_step("layer1", "w1", "grad_w1", num_weights);
```

### sgd_step

```rust
pub fn sgd_step(&self, name: &str, weights: &str, grad: &str, n: i32,
                lr: f32, momentum: f32, wd: f32)
```

Perform one SGD step with momentum and weight decay.

```rust
stream.sgd_step("layer1", "w1", "grad_w1", num_weights, 0.01, 0.9, 0.0001);
```

### zero_grad

```rust
pub fn zero_grad(&self, grad: &str, n: i32)
```

Zero out gradient tensor.

```rust
stream.zero_grad("grad_w1", num_weights);
```

## Synchronization

### sync

```rust
pub fn sync(&self)
```

Wait for all queued GPU operations to complete.

```rust
stream.linear("input", "weights", None, "output", ...);
stream.relu("output", n);
stream.sync();  // Wait here
let result = stream.download("output");
```

## Pipeline Execution

### execute

```rust
pub fn execute(&mut self, ops: &[StreamOp])
```

Execute a sequence of operations built with the `Pipeline` builder.

```rust
let ops = Pipeline::new()
    .linear("input", "w1", Some("b1"), "h1", ...)
    .relu("h1", n)
    .build();

stream.execute(&ops);
```

See [Pipeline API](pipeline.md) for the builder interface.
