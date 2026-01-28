# Ferrite

High-performance GPU ML runtime for Rust. Zero-copy execution, O(1) memory allocation, sub-100µs inference.

```
Rust Brain + GPU Muscle
```

## Quick Install

```bash
curl -sSf https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | sh
```

The installer will:
- Detect your GPU and compute capability automatically
- Download pre-built binaries for your specific GPU architecture
- Set up your PATH

## What is Ferrite?

Ferrite is a GPU compute runtime that combines:

- **Rust's safety** - Memory-safe orchestration, no segfaults
- **CUDA's speed** - cuBLAS matmul, JIT-compiled kernels
- **Zero-copy execution** - Operations stay on GPU, no hidden transfers
- **O(1) allocation** - TLSF allocator, no GC pauses, predictable latency

## Why Ferrite?

| Problem with Python/PyTorch | Ferrite Solution |
|---------------------------|------------------|
| GC pauses during inference | O(1) TLSF allocation, no GC |
| Memory fragmentation | Shared pool, zero fragmentation |
| Hidden host-device copies | Zero-copy streaming pipeline |
| Unpredictable latency | Bounded, deterministic latency |
| Service restart for model updates | Atomic hot-swap |

## Performance

- **Matmul**: 11+ TFLOPS via cuBLAS
- **Inference latency**: Sub-100µs for small models
- **Memory overhead**: Minimal - entire trading system in <1MB GPU memory

## Quick Start

```rust
use ferrite::compute::Stream;
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() {
    // Initialize with pre-allocated pool (no runtime allocation)
    let device = CudaDevice::new(0).unwrap();
    let allocator = Arc::new(TlsfAllocator::new(device, 128 * 1024 * 1024));

    let mut stream = Stream::new(allocator);
    stream.init().unwrap();

    // Allocate tensors (O(1) - just pointer arithmetic)
    stream.alloc("input", 784);
    stream.alloc("weights", 784 * 256);
    stream.alloc("output", 256);

    // Initialize on GPU (no host-device copy)
    stream.fill("input", 0.5, 784);
    stream.init_weights("weights", 784, 256, "xavier");

    // Forward pass (zero-copy, stays on GPU)
    stream.linear("input", "weights", None, "output", 1, 784, 256);
    stream.relu("output", 256);
    stream.sync();

    // Only copy to host when you need the result
    let result = stream.download("output");
    println!("Output: {:?}", &result[0..10]);
}
```

## Examples

See the `examples/` directory for complete use cases:

- **realtime_inference** - Sub-100µs latency with frame budget analysis
- **model_ensemble** - Multiple models sharing one memory pool
- **streaming_pipeline** - Double-buffered continuous processing
- **hot_swap** - Zero-downtime model updates
- **trading_system** - Microsecond-scale trading signals

## Requirements

- **OS**: Linux (x86_64)
- **GPU**: NVIDIA with Compute Capability 6.0+ (Pascal or newer)
- **CUDA**: Toolkit 11.0+
- **Driver**: NVIDIA driver compatible with your CUDA version

### Supported GPUs

| Generation | Compute Capability | Examples |
|-----------|-------------------|----------|
| Pascal | sm_60, sm_61 | GTX 1080, Tesla P100 |
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100 |

## Building from Source

```bash
# Clone
git clone https://github.com/DaronPopov/ferrite.git
cd ferrite

# Build spcpp backend
./build_spcpp.sh

# Build Rust
cargo build --release
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Code (Rust)               │
├─────────────────────────────────────────────────┤
│              Ferrite Runtime                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Stream  │  │ Pipeline │  │   Runtime    │  │
│  │ Zero-copy│  │ Builder  │  │  CPU/GPU     │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────┤
│                 SPCPP Backend                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  cuBLAS  │  │ JIT CUDA │  │    TLSF      │  │
│  │  MatMul  │  │ Kernels  │  │  Allocator   │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────┤
│                 CUDA / GPU                       │
└─────────────────────────────────────────────────┘
```

## License

MIT OR Apache-2.0
