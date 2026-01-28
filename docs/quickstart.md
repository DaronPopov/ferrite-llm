# Quick Start

This guide walks through a complete example: a two-layer neural network with zero-copy execution.

## Project Setup

Create a new Rust project:

```bash
cargo new my_ferrite_app
cd my_ferrite_app
```

Add Ferrite to `Cargo.toml`:

```toml
[dependencies]
ferrite = { git = "https://github.com/DaronPopov/ferrite" }
cudarc = { version = "0.12", features = ["cuda-version-from-build-system", "driver"] }
```

## Basic Example

Replace `src/main.rs` with:

```rust
use ferrite::compute::Stream;
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    // Initialize GPU with pre-allocated memory pool
    let device = CudaDevice::new(0).expect("No CUDA device");
    let pool_size = 128 * 1024 * 1024; // 128 MB
    let allocator = Arc::new(TlsfAllocator::new(device, pool_size));

    // Create execution stream
    let mut stream = Stream::new(allocator.clone());
    stream.init().expect("Failed to initialize");

    // Define network dimensions
    let batch = 32;
    let input_dim = 784;
    let hidden_dim = 256;
    let output_dim = 10;

    // Allocate tensors (O(1) operations, no memory transfer)
    stream.alloc("input", batch * input_dim);
    stream.alloc("w1", input_dim * hidden_dim);
    stream.alloc("b1", hidden_dim);
    stream.alloc("h1", batch * hidden_dim);
    stream.alloc("w2", hidden_dim * output_dim);
    stream.alloc("b2", output_dim);
    stream.alloc("output", batch * output_dim);

    // Initialize weights on GPU (no host-device copy)
    stream.init_weights("w1", input_dim as i32, hidden_dim as i32, "xavier");
    stream.init_weights("w2", hidden_dim as i32, output_dim as i32, "xavier");
    stream.fill("b1", 0.0, hidden_dim as i32);
    stream.fill("b2", 0.0, output_dim as i32);
    stream.fill("input", 0.5, (batch * input_dim) as i32);
    stream.sync();

    // Warm up
    for _ in 0..10 {
        stream.linear("input", "w1", Some("b1"), "h1",
                      batch as i32, input_dim as i32, hidden_dim as i32);
        stream.relu("h1", (batch * hidden_dim) as i32);
        stream.linear("h1", "w2", Some("b2"), "output",
                      batch as i32, hidden_dim as i32, output_dim as i32);
        stream.softmax("output", batch as i32, output_dim as i32);
    }
    stream.sync();

    // Benchmark
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        stream.linear("input", "w1", Some("b1"), "h1",
                      batch as i32, input_dim as i32, hidden_dim as i32);
        stream.relu("h1", (batch * hidden_dim) as i32);
        stream.linear("h1", "w2", Some("b2"), "output",
                      batch as i32, hidden_dim as i32, output_dim as i32);
        stream.softmax("output", batch as i32, output_dim as i32);
    }
    stream.sync();

    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let throughput = (iterations * batch) as f64 / elapsed.as_secs_f64();

    println!("Forward pass: {:.2} us", per_iter_us);
    println!("Throughput: {:.0} samples/sec", throughput);

    // Download result (first host-device copy)
    let output = stream.download("output");
    let sum: f32 = output[0..output_dim].iter().sum();
    println!("Softmax sum (should be 1.0): {:.4}", sum);
}
```

Build and run:

```bash
cargo run --release
```

Expected output:

```
Forward pass: 45.23 us
Throughput: 22105 samples/sec
Softmax sum (should be 1.0): 1.0000
```

## Key Concepts

### Memory Pool

All GPU memory comes from a pre-allocated pool:

```rust
let allocator = Arc::new(TlsfAllocator::new(device, pool_size));
```

The TLSF allocator provides O(1) allocation and deallocation. No system calls occur during inference.

### Named Tensors

Tensors are referenced by string names:

```rust
stream.alloc("weights", 1024);
stream.fill("weights", 0.0, 1024);
stream.matmul("a", "b", "weights", m, k, n);
```

This enables the zero-copy execution model where operations reference GPU memory directly.

### Zero-Copy Execution

Operations execute directly on GPU memory without intermediate copies:

```rust
// All operations stay on GPU
stream.linear("input", "weights", Some("bias"), "output", ...);
stream.relu("output", n);
stream.softmax("output", rows, cols);
stream.sync();

// Only copy when you need the result
let result = stream.download("output");
```

### Synchronization

GPU operations are asynchronous. Call `sync()` to wait for completion:

```rust
stream.matmul("a", "b", "c", m, k, n);
stream.sync(); // Wait for matmul to complete
```

## Next Steps

- [Stream API Reference](api/stream.md) - Complete API documentation
- [Zero-Copy Guide](guides/zero-copy.md) - Optimize memory usage
- [Real-Time Guide](guides/real-time.md) - Achieve sub-100us latency
