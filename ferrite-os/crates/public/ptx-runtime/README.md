# ptx-runtime

Safe Rust runtime for PTX-OS GPU operating system with TLSF memory allocation and massive stream parallelism.

## Overview

ptx-runtime provides safe Rust abstractions over the PTX-OS GPU kernel, enabling applications to leverage constant-time memory allocation, stream-ordered operations, and concurrent execution of thousands of GPU kernels. The runtime is designed for high-throughput machine learning workloads requiring efficient memory management and fine-grained parallelism control.

## Features

### Memory Management
- **TLSF Allocator Integration**: O(1) allocation and deallocation via Two-Level Segregated Fit algorithm
- **Stream-Ordered Operations**: Asynchronous allocation and deallocation ordered by CUDA streams
- **Automatic Leak Detection**: Timestamp-based tracking with health monitoring
- **Zero Fragmentation**: Proven performance under sustained allocation patterns

### Stream Parallelism
- **Massive Concurrency**: Support for up to 100,000 concurrent CUDA streams
- **Priority Scheduling**: Six priority levels for real-time task management
- **Round-Robin Selection**: Automatic load balancing across stream pool
- **Independent Queues**: Per-stream allocation queues for isolation

### Safe API
- **RAII Memory Management**: Automatic cleanup via Drop trait
- **Type Safety**: Generic bounds checking and validation
- **Error Propagation**: Result-based error handling throughout
- **FFI Safety**: Validated ownership across language boundaries

## Performance Characteristics

### Allocation Performance

| Size  | Throughput | Latency |
|-------|-----------|---------|
| 64B   | 6.5M ops/sec | <200ns |
| 4KB   | 6.5M ops/sec | <200ns |
| 1MB   | 6.5M ops/sec | <1μs |

### Stream Scaling

| Streams | Total Time | Throughput |
|---------|-----------|-----------|
| 100     | 0.12ms    | 833K/sec |
| 1,000   | 2.09ms    | 478K/sec |
| 5,000   | 10.4ms    | 480K/sec |
| 10,000  | ~21ms     | ~476K/sec |

Performance measured on NVIDIA GeForce RTX 3070 (8GB VRAM, Compute Capability 8.6).

## Architecture

```
┌─────────────────────────────────────┐
│  Application                        │
│  (ML Training/Inference)            │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  ptx-kernels                        │
│  Guard Layer & Safety Wrappers     │
│  (GuardedBuffer, KernelContext)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Kernel Libraries (Pluggable)      │
│  (Candle, PyTorch, Custom)         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  ptx-runtime (This Crate)          │
│  Memory Management & Streams       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  PTX-OS Core                       │
│  TLSF Allocator Implementation     │
└─────────────────────────────────────┘
```

## API Reference

### Runtime Initialization

```rust
use ptx_runtime::PtxRuntime;

// Initialize with default configuration
let runtime = PtxRuntime::new(0)?;

// Initialize with custom configuration
use ptx_runtime::GPUHotConfig;

let mut config = GPUHotConfig::default();
config.pool_fraction = 0.7;           // Use 70% of VRAM
config.max_streams = 1024;            // Create 1024 streams
config.enable_leak_detection = true;  // Enable leak tracking
config.quiet_init = false;            // Show initialization details

let runtime = PtxRuntime::with_config(0, Some(config))?;
```

### Memory Allocation

**Synchronous Allocation**:
```rust
// Allocate memory (automatically freed on drop)
let ptr = runtime.alloc(1024 * 1024)?; // 1MB

// Memory is freed when ptr goes out of scope
```

**Asynchronous Stream-Ordered Allocation**:
```rust
let stream = runtime.stream(0);

// Allocate on specific stream
let ptr = runtime.alloc_async(size, &stream)?;

// Use memory in stream-ordered operations
// ...

// Free on same stream (ordering preserved)
unsafe {
    runtime.free_async(ptr, &stream);
}
```

### Stream Management

```rust
// Get stream by index
let stream = runtime.stream(5);

// Get next stream (round-robin)
let stream = runtime.next_stream();

// Synchronize specific stream
stream.sync()?;

// Synchronize all streams
runtime.sync_all();
```

### Memory Statistics

```rust
let stats = runtime.tlsf_stats();

println!("Pool size: {} bytes", stats.total_pool_size);
println!("Allocated: {} bytes", stats.allocated_bytes);
println!("Free: {} bytes", stats.free_bytes);
println!("Utilization: {:.1}%", stats.utilization_percent);
println!("Fragmentation: {:.6}%", stats.fragmentation_ratio * 100.0);
println!("Largest free: {} bytes", stats.largest_free_block);
println!("Health: {}", if stats.is_healthy { "OK" } else { "WARNING" });
```

### CUDA Graph Operations

```rust
// Begin graph capture
let capture = runtime.begin_capture(0, "my_graph")?;

// Perform operations to capture
// ...

// End capture and get graph
let graph = capture.end()?;

// Launch graph
runtime.launch_graph(&graph, &stream)?;
```

### cuBLAS Integration

```rust
use ptx_runtime::Gemm;

// Get cuBLAS handle (created on first access)
let cublas = runtime.cublas()?;

// Matrix multiplication
let m = 128;
let n = 128;
let k = 128;

let gemm = Gemm {
    m, n, k,
    alpha: 1.0,
    beta: 0.0,
    trans_a: false,
    trans_b: false,
};

gemm.execute(&*cublas, a_ptr, b_ptr, c_ptr, &stream)?;
```

## Configuration

### GPUHotConfig Parameters

```rust
pub struct GPUHotConfig {
    pub pool_fraction: f64,           // Fraction of VRAM to use (0.0-1.0)
    pub max_streams: u32,             // Number of streams to create
    pub min_pool_size: usize,         // Minimum pool size in bytes
    pub enable_leak_detection: bool,  // Enable leak tracking
    pub enable_pool_health: bool,     // Enable health monitoring
    pub warning_threshold: f32,       // Utilization warning threshold
    pub quiet_init: bool,             // Suppress initialization output
}
```

**Recommended Configurations**:

Development:
```rust
config.pool_fraction = 0.6;
config.max_streams = 32;
config.enable_leak_detection = true;
config.quiet_init = false;
```

Production:
```rust
config.pool_fraction = 0.75;
config.max_streams = 1024;
config.enable_leak_detection = false;
config.quiet_init = true;
```

High-Throughput:
```rust
config.pool_fraction = 0.8;
config.max_streams = 10000;
config.enable_leak_detection = false;
config.quiet_init = true;
```

## Examples

### Basic Neural Network Layer

```rust
use ptx_runtime::PtxRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = PtxRuntime::new(0)?;
    let stream = runtime.stream(0);

    // Allocate input and output buffers
    let input = runtime.alloc_async(batch * input_dim * 4, &stream)?;
    let output = runtime.alloc_async(batch * hidden_dim * 4, &stream)?;

    // Launch kernel (implementation-specific)
    // unsafe { neural_layer_kernel(...) }

    // Synchronize
    stream.sync()?;

    // Memory automatically freed when pointers drop
    Ok(())
}
```

### Massive Parallel Batch Processing

```rust
use ptx_runtime::{PtxRuntime, GPUHotConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure for massive parallelism
    let mut config = GPUHotConfig::default();
    config.max_streams = 5000;
    config.pool_fraction = 0.7;

    let runtime = PtxRuntime::with_config(0, Some(config))?;

    // Process 5000 independent batches in parallel
    for i in 0..5000 {
        let stream = runtime.stream(i);
        let buffer = runtime.alloc_async(batch_size, &stream)?;

        // Launch kernel on stream i
        // unsafe { process_batch(buffer, stream.raw()) }
    }

    // Synchronize all streams
    runtime.sync_all();

    Ok(())
}
```

### Stream-Ordered Pipeline

```rust
use ptx_runtime::PtxRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = PtxRuntime::new(0)?;
    let stream = runtime.stream(0);

    // All operations ordered on same stream
    let input = runtime.alloc_async(size, &stream)?;
    let temp = runtime.alloc_async(size, &stream)?;
    let output = runtime.alloc_async(size, &stream)?;

    // Launch kernels (automatically ordered)
    // unsafe { kernel_1(input, temp, stream.raw()) }
    // unsafe { kernel_2(temp, output, stream.raw()) }

    // Single synchronization point
    stream.sync()?;

    Ok(())
}
```

## Demonstration Programs

The `examples/` directory contains complete demonstration programs:

**massive_candle_parallelism.rs**: 5000 concurrent Candle kernels
**transformer_attention_layer.rs**: Multi-head attention implementation
**neural_layer_inference.rs**: Batch neural network inference
**parallel_batch_processing.rs**: Independent batch processing
**memory_efficient_pipelines.rs**: Stream-ordered pipeline operations
**llm_inference_demo.rs**: Large language model inference simulation

Run with:
```bash
cargo run --release --example <example_name>
```

## Safety Considerations

### Memory Safety

All memory allocations return RAII-wrapped pointers that automatically free memory on drop. For stream-ordered operations, use `free_async` to maintain ordering:

```rust
let ptr = runtime.alloc_async(size, &stream)?;

// Use ptr in stream-ordered operations

// Manual free (stream-ordered)
unsafe {
    runtime.free_async(ptr, &stream);
}
```

### FFI Safety

When passing pointers across FFI boundaries, validate ownership:

```rust
use ptx_kernels::GuardedBuffer;

// Validate TLSF ownership
let buffer = unsafe {
    GuardedBuffer::new(ptr.as_ptr(), size, runtime.raw())?
};

// Buffer validates ownership and bounds
```

### Thread Safety

PtxRuntime is Send and Sync. All operations use interior mutability with appropriate synchronization:

```rust
use std::sync::Arc;
use std::thread;

let runtime = Arc::new(PtxRuntime::new(0)?);

let handles: Vec<_> = (0..8)
    .map(|i| {
        let rt = Arc::clone(&runtime);
        thread::spawn(move || {
            let ptr = rt.alloc(1024)?;
            // Thread-safe operations
            Ok::<_, Error>(())
        })
    })
    .collect();

for handle in handles {
    handle.join().unwrap()?;
}
```

## Error Handling

All operations return `Result<T, Error>`:

```rust
pub enum Error {
    AllocationFailed { size: usize },
    InvalidPointer { ptr: usize },
    StreamError { message: String },
    GraphError { message: String },
    CublasError { message: String },
    Internal { message: String },
}
```

Handle errors appropriately:

```rust
match runtime.alloc(size) {
    Ok(ptr) => {
        // Use pointer
    }
    Err(Error::AllocationFailed { size }) => {
        // Handle OOM
        eprintln!("Failed to allocate {} bytes", size);
    }
    Err(e) => {
        // Handle other errors
        eprintln!("Error: {:?}", e);
    }
}
```

## Testing

Run test suite:
```bash
# Unit tests
cargo test

# Integration tests (requires GPU)
cargo test --test integration_tests -- --ignored

# GPU-specific tests
cargo test --features gpu-tests
```

## Benchmarks

Use `../../tooling/scripts/ptx_bench_all.sh` for benchmark orchestration, plus these core examples:

- `jitter_benchmark`: runtime jitter and latency stability
- `fused_kernel_benchmark`: fused-kernel execution path
- `candle_performance_benchmark`: candle op throughput
- `telemetry_demo`: end-to-end runtime telemetry visibility

## Performance Tuning

### Memory Pool Configuration

Adjust pool size based on workload:

```rust
// Workload with many small allocations
config.pool_fraction = 0.6;  // Conservative

// Workload with few large allocations
config.pool_fraction = 0.8;  // Aggressive
```

### Stream Count Optimization

Balance parallelism vs overhead:

```rust
// Low parallelism (< 100 kernels)
config.max_streams = 32;

// Medium parallelism (100-1000 kernels)
config.max_streams = 512;

// High parallelism (1000+ kernels)
config.max_streams = 10000;
```

### Monitoring Pool Health

Check pool statistics periodically:

```rust
let stats = runtime.tlsf_stats();

if stats.utilization_percent > 90.0 {
    eprintln!("Warning: Pool utilization high");
}

if stats.fragmentation_ratio > 0.01 {
    eprintln!("Warning: Fragmentation detected");
}
```

## Limitations

- **Linux Only**: Currently supports Linux operating systems only
- **NVIDIA GPUs**: Requires NVIDIA hardware with CUDA support
- **Single Device**: Multi-GPU support planned but not yet implemented
- **Leak Detection Overhead**: Enable only in development/debugging

## Dependencies

- `ptx-sys`: FFI bindings to PTX-OS C/CUDA core
- `parking_lot`: Efficient synchronization primitives
- `thiserror`: Error type derivation
- `tracing`: Structured logging and instrumentation

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.

## See Also

- [Runtime Architecture Guide](../../docs/02-runtime-architecture/README.md)
- [Build and Portability Guide](../../docs/03-build-and-portability/README.md)
- [PTX-Compute API](../../crates/internal/ptx-compute/README.md)
- [Benchmark Runner](../../tooling/scripts/ptx_bench_all.sh)
