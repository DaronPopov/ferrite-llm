# ptx-compute

High-level compute operations API for PTX-OS GPU runtime.

This crate provides ergonomic, high-level APIs for common GPU compute patterns, built on top of the low-level `ptx-runtime` wrapper.

## Features

- **Matrix Operations** (`gemm`) - High-level matrix multiplication APIs
- **Neural Network Ops** (`neural`) - Forward pass primitives and activation functions
- **Parallel Reductions** (`reduction`) - Sum, mean, max, min operations
- **Monte Carlo** (`monte_carlo`) - Monte Carlo simulation utilities
- **Custom Tiling** (`tiling`) - CuTe-like tiling for optimal GPU performance

## Architecture

```
┌─────────────────────────────────────┐
│         ptx-compute                 │  ← High-level compute APIs
│  (gemm, neural, reduction, etc.)   │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         ptx-runtime                 │  ← Safe runtime wrappers
│  (memory, streams, graphs)          │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         ptx-sys                     │  ← Raw FFI bindings
│  (CUDA, cuBLAS, PTX-OS)             │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         PTX-OS Runtime              │  ← Native GPU runtime
│  (libptx_os.so)                     │
└─────────────────────────────────────┘
```

## Usage

### Matrix Multiplication

```rust
use std::sync::Arc;
use ptx_runtime::PtxRuntime;
use ptx_compute::gemm::Matmul;

let runtime = Arc::new(PtxRuntime::new(0)?);
let matmul = Matmul::new(&runtime)?;

// Allocate matrices
let a = runtime.alloc(m * k * 4)?;
let b = runtime.alloc(k * n * 4)?;
let c = runtime.alloc(m * n * 4)?;

// C = A @ B
unsafe {
    matmul.multiply_f32(
        a.as_ptr() as *const f32,
        b.as_ptr() as *const f32,
        c.as_mut_ptr() as *mut f32,
        m, n, k,
    )?;
}
```

### Parallel Reductions

```rust
use ptx_compute::reduction::Reducer;

let reducer = Reducer::new(&runtime);
let stream = runtime.stream(0);

// Sum all elements
unsafe {
    reducer.sum_f32(input_ptr, output_ptr, size, &stream)?;
}
```

### Neural Network Forward Pass

```rust
use ptx_compute::neural::{Network, Layer, Activation};

let mut network = Network::new(&runtime, hidden_size, num_layers)?;

// Add layers
network.add_layer(Layer::new(weights_ptr, input_size, hidden_size, Activation::GELU));
network.add_layer(Layer::new(weights2_ptr, hidden_size, output_size, Activation::Softmax));

// Forward pass
unsafe {
    network.forward(input, output, temp_buffer, batch_size, &stream)?;
}
```

## Examples

See `examples/` for complete examples:

- `simple_matmul.rs` - Basic matrix multiplication
- `tiled_matmul.rs` - Custom tiling configuration (like CuTe)

For benchmarks, see `ptx-runtime/examples/*_benchmark.rs`.

## Design Philosophy

This crate follows a layered design:

1. **ptx-sys**: Raw, unsafe FFI bindings (like `cuda-sys`)
2. **ptx-runtime**: Safe RAII wrappers (like `cudarc`)
3. **ptx-compute**: High-level compute APIs (like PyTorch's `nn.functional`)

Users can choose their abstraction level:
- Use `ptx-sys` for custom kernels and maximum control
- Use `ptx-runtime` for safe memory/stream management
- Use `ptx-compute` for common compute patterns

## Performance

This crate is designed for performance:

- Zero-cost abstractions - compiles to same code as raw cuBLAS calls
- Multi-stream parallelism built-in
- Supports async operations and overlapping compute
- Integrates with PTX-OS's custom TLSF allocator

## License

MIT OR Apache-2.0
