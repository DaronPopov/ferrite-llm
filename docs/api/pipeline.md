# Pipeline API

The `Pipeline` type provides a declarative builder for constructing operation sequences.

## Import

```rust
use ferrite::compute::Pipeline;
```

## Overview

Pipelines allow you to define a sequence of operations that can be:

- Built once and executed many times
- Inspected before execution
- Composed from smaller pipelines

## Construction

### new

```rust
pub fn new() -> Self
```

Creates an empty pipeline.

```rust
let pipeline = Pipeline::new();
```

## Builder Methods

All builder methods consume and return `self`, enabling method chaining.

### alloc

```rust
pub fn alloc(self, name: &'static str, size: usize) -> Self
```

Adds an allocation operation.

```rust
Pipeline::new()
    .alloc("hidden", 256)
```

### matmul

```rust
pub fn matmul(self, a: &'static str, b: &'static str, c: &'static str,
              m: i32, k: i32, n: i32) -> Self
```

Adds a matrix multiplication: C = A * B

```rust
Pipeline::new()
    .matmul("input", "weights", "output", 32, 784, 256)
```

### linear

```rust
pub fn linear(self, input: &'static str, weights: &'static str,
              bias: Option<&'static str>, output: &'static str,
              batch: i32, in_f: i32, out_f: i32) -> Self
```

Adds a linear layer operation.

```rust
Pipeline::new()
    .linear("x", "w1", Some("b1"), "h1", 32, 784, 256)
    .linear("h1", "w2", Some("b2"), "out", 32, 256, 10)
```

### relu

```rust
pub fn relu(self, x: &'static str, n: i32) -> Self
```

Adds an in-place ReLU activation.

```rust
Pipeline::new()
    .linear("x", "w", None, "h", 32, 784, 256)
    .relu("h", 32 * 256)
```

### sigmoid

```rust
pub fn sigmoid(self, x: &'static str, n: i32) -> Self
```

Adds an in-place sigmoid activation.

### softmax

```rust
pub fn softmax(self, x: &'static str, rows: i32, cols: i32) -> Self
```

Adds an in-place softmax operation.

```rust
Pipeline::new()
    .linear("h", "w_out", None, "logits", 32, 256, 10)
    .softmax("logits", 32, 10)
```

### adam_step

```rust
pub fn adam_step(self, name: &'static str, weights: &'static str,
                 grad: &'static str, n: i32) -> Self
```

Adds an Adam optimizer step.

### sync

```rust
pub fn sync(self) -> Self
```

Adds a synchronization point.

```rust
Pipeline::new()
    .linear("x", "w", None, "y", 32, 784, 256)
    .relu("y", 32 * 256)
    .sync()
```

## Execution

### build

```rust
pub fn build(self) -> Vec<StreamOp>
```

Consumes the pipeline and returns the operation list.

```rust
let ops = Pipeline::new()
    .linear("x", "w", None, "y", 32, 784, 256)
    .relu("y", 32 * 256)
    .build();

println!("Pipeline has {} operations", ops.len());
```

### run

```rust
pub fn run(self, stream: &mut Stream)
```

Builds and executes the pipeline on the given stream.

```rust
Pipeline::new()
    .linear("x", "w", None, "y", 32, 784, 256)
    .relu("y", 32 * 256)
    .run(&mut stream);
```

## StreamOp Enum

The `build()` method returns a vector of `StreamOp` values:

```rust
pub enum StreamOp {
    Alloc { size: usize, out: &'static str },
    MatMul { a: &'static str, b: &'static str, c: &'static str, m: i32, k: i32, n: i32 },
    Linear { input: &'static str, weights: &'static str, bias: Option<&'static str>,
             output: &'static str, batch: i32, in_f: i32, out_f: i32 },
    ReLU { x: &'static str, n: i32 },
    Sigmoid { x: &'static str, n: i32 },
    Softmax { x: &'static str, rows: i32, cols: i32 },
    AdamStep { name: &'static str, weights: &'static str, grad: &'static str, n: i32 },
    Sync,
    // ... additional variants
}
```

## Examples

### Forward Pass Pipeline

```rust
let forward = Pipeline::new()
    .linear("input", "w1", Some("b1"), "h1", batch, 784, 256)
    .relu("h1", batch * 256)
    .linear("h1", "w2", Some("b2"), "h2", batch, 256, 128)
    .relu("h2", batch * 128)
    .linear("h2", "w3", Some("b3"), "output", batch, 128, 10)
    .softmax("output", batch, 10)
    .sync();

// Execute multiple times
let ops = forward.build();
for _ in 0..1000 {
    stream.execute(&ops);
}
```

### Training Step Pipeline

```rust
let train_step = Pipeline::new()
    // Forward
    .linear("input", "w1", Some("b1"), "h1", batch, 784, 256)
    .relu("h1", batch * 256)
    .linear("h1", "w2", Some("b2"), "output", batch, 256, 10)
    .sync()
    // Backward (assuming gradients computed elsewhere)
    .adam_step("layer1", "w1", "grad_w1", 784 * 256)
    .adam_step("layer2", "w2", "grad_w2", 256 * 10)
    .sync();

train_step.run(&mut stream);
```

### Reusable Pipeline Function

```rust
fn create_mlp_pipeline(
    input: &'static str,
    output: &'static str,
    batch: i32,
    dims: &[i32]
) -> Pipeline {
    let mut p = Pipeline::new();

    // This is a simplified example - real implementation would
    // need to handle weight/bias names dynamically
    p = p.linear(input, "w1", Some("b1"), "h1", batch, dims[0], dims[1]);
    p = p.relu("h1", batch * dims[1]);

    if dims.len() > 2 {
        p = p.linear("h1", "w2", Some("b2"), output, batch, dims[1], dims[2]);
    }

    p.sync()
}
```

## Performance Notes

1. **Build once, run many**: The `build()` operation allocates memory. For repeated execution, build once and call `stream.execute()` in a loop.

2. **Static strings required**: Tensor names must be `&'static str` to avoid lifetime issues in the operation list.

3. **No validation**: The pipeline builder does not validate tensor dimensions or existence. Errors will occur at runtime during `execute()`.
