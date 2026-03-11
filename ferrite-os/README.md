# Ferrite-OS

**Persistent GPU Runtime with Operating System Capabilities**

Ferrite-OS is a custom GPU operating system for NVIDIA GPUs featuring aerospace-grade memory management, massive stream parallelism, and multi-process GPU context sharing.

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

Ferrite-OS provides a persistent kernel runtime on NVIDIA GPUs with OS-level services including memory management, virtual filesystems, and inter-process communication. The system is designed for high-throughput, low-latency GPU computing workloads requiring efficient memory allocation and massive parallelism.

## Key Features

### Memory Management
- **TLSF Allocator**: Two-Level Segregated Fit allocation with O(1) time complexity
- **Zero Fragmentation**: Proven performance over 10,000+ allocation cycles
- **Stream-Ordered Operations**: Asynchronous allocation with stream awareness
- **Leak Detection**: Timestamp-based tracking with health monitoring

### Parallelism
- **Concurrent Streams**: Support for up to 100,000 concurrent CUDA streams
- **Priority Scheduling**: Six priority levels for real-time task management
- **Round-Robin Load Balancing**: Automatic distribution across stream pool

### Multi-Process Support
- **Shared Memory IPC**: System V shared memory for GPU context sharing
- **Daemon Architecture**: Central daemon with multiple client processes
- **Global State Management**: Coordinated access to GPU resources

### Virtual Memory System
- **Virtual Filesystem**: POSIX-like filesystem operations on GPU memory
- **Page-Based Management**: Virtual memory with CPU swap capability
- **Zero-Copy Access**: cudaHostRegister for direct GPU access

## Performance Characteristics

| Metric | Standard CUDA | Ferrite-OS | Improvement |
|--------|---------------|------------|-------------|
| Allocation Latency | ~1000 μs | 0.238 μs | 4,200x |
| Fragmentation | 15-35% | <0.001% | Negligible |
| Maximum Streams | 10-32 | 100,000 | 3,000x+ |
| GPU Utilization | 40-70% | 95-96% | 1.4x |

Performance measured on NVIDIA GeForce RTX 3070 (Ampere, 8GB VRAM, Compute Capability 8.6).

For a detailed guide to the repository structure, see [NAVIGATION.md](NAVIGATION.md).

## Architecture

```
┌─────────────────────────────────────┐
│  Application Layer                  │
│  (User Code)                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  ptx-compute                        │
│  High-Level Compute Operations      │
│  (GEMM, Neural, Reduction, MC)      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  ptx-runtime                        │
│  Safe Rust Wrappers                 │
│  (Memory, Streams, Graphs, cuBLAS)  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  PTX-OS Core                        │
│  C/CUDA Implementation              │
│  (TLSF, VFS, VMM, IPC)             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  CUDA Runtime & Driver              │
└─────────────────────────────────────┘
              ↓
        GPU Hardware
```

## System Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher
  - Turing (RTX 20xx, GTX 16xx) — SM 75
  - Ampere (RTX 30xx, A100, A6000) — SM 80/86
  - Ada Lovelace (RTX 40xx) — SM 89
  - Hopper (H100) — SM 90
  - Blackwell (B100, B200, GB200) — SM 100/120
- Minimum 6GB VRAM (8GB+ recommended for large workloads)

### Software
- CUDA Toolkit 11.0 or higher
- Rust 1.70 or higher
- Linux operating system (tested on Ubuntu 22.04 LTS; x86_64 and aarch64)
- GCC/Clang compiler with C++11 support

## Installation

### Quick Start (One-Liner)

```bash
# Clone and build with auto-detection
git clone https://github.com/DaronPopov/ferrite-os.git && cd ferrite-os && ./setup.sh

# Or specify CUDA version and GPU compute capability
git clone https://github.com/DaronPopov/ferrite-os.git && cd ferrite-os && ./setup.sh --cuda-version 12.6 --sm 89
```

Common compute capabilities:
- **RTX 40xx (Ada Lovelace)**: `--sm 89` (4090, 4080)
- **RTX 30xx (Ampere)**: `--sm 86` (3090, 3080, 3070)
- **A100 (Ampere)**: `--sm 80`
- **RTX 20xx (Turing)**: `--sm 75` (2080 Ti, 2070)
- **Jetson Orin**: `--sm 87`
- **H100 (Hopper)**: `--sm 90`
- **B100/B200 (Blackwell)**: `--sm 100`
- **GB200 (Blackwell Ultra)**: `--sm 120`

Current kernel profile requires `sm_75+`. Legacy Jetson SKUs
(`sm_72`/`sm_62`/`sm_53`) require a separate legacy kernel profile.

### Manual Build

```bash
# Clone repository
git clone https://github.com/DaronPopov/ferrite-os.git
cd ferrite-os

# Set CUDA path and compute capability (optional, will auto-detect)
export CUDA_PATH=/usr/local/cuda-12.6
export PTX_GPU_SM=sm_89  # For RTX 4090
# Optional embedded tuning toggles
export PTX_ORIN_UM=1      # Use Orin unified-memory scheduler branch
export PTX_MANAGED_POOL=1 # Use cudaMallocManaged for TLSF backing pool

# Build C/CUDA libraries
make

# Build Rust crates
cargo build --release
```

### Verification

```bash
# Run test suite (requires GPU)
./test.sh

# Run integration tests
cargo test --package ptx-runtime --test integration_tests -- --ignored

# Jetson-specific smoke check
tooling/scripts/jetson_doctor.sh

# Run benchmarks
tooling/scripts/ptx_bench_all.sh --no-build
```

## Usage

### Basic Example

```rust
use ptx_runtime::PtxRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize runtime on device 0
    let runtime = PtxRuntime::new(0)?;

    // Allocate 1MB of GPU memory
    let ptr = runtime.alloc(1024 * 1024)?;

    // Get stream handle
    let stream = runtime.stream(0);

    // Synchronize all streams
    runtime.sync_all();

    // Memory automatically freed when ptr is dropped
    Ok(())
}
```

### Advanced Configuration

```rust
use ptx_runtime::{PtxRuntime, GPUHotConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = GPUHotConfig::default();
    config.pool_fraction = 0.7;           // Use 70% of available VRAM
    config.max_streams = 128;             // Create 128 stream pool
    config.enable_leak_detection = true;  // Enable leak tracking
    config.quiet_init = false;            // Show initialization details

    let runtime = PtxRuntime::with_config(0, Some(config))?;

    // Runtime operations...
    Ok(())
}
```

## Documentation

### Core Documentation
- [Runtime Architecture](../docs/02-runtime-architecture/README.md) - Runtime internals and boundaries
- [Build and Portability](../docs/03-build-and-portability/README.md) - Installer and compatibility flow
- [System Overview](../docs/01-system-overview/README.md) - Repository and subsystem map

### Integration Guides
- [PyTorch Integration](crates/public/ptx-runtime/PYTORCH_INTEGRATION.md) - Using PyTorch kernels with Ferrite-OS
- [Runtime Operations Runbook](../docs/02-runtime-architecture/runbooks/runtime-operations.md) - Daemon and script execution workflow

### API Reference
- [ptx-runtime](crates/public/ptx-runtime/README.md) - Core runtime and memory management
- [ptx-compute](crates/internal/ptx-compute/README.md) - High-level compute operations
- [ptx-kernels](crates/internal/ptx-kernels/README.md) - Kernel library and guards

## Use Cases

### Suitable Applications

**Batch Inference**
- Processing thousands of independent inference requests
- High-throughput model serving
- Parallel batch processing

**Multi-Stream Training**
- Multi-GPU distributed training
- Pipeline parallelism
- Data parallel training with fine-grained stream control

**Low-Latency Serving**
- Real-time inference with sub-microsecond allocation
- High-frequency trading or real-time analytics
- Interactive applications requiring fast GPU response

**Research and Prototyping**
- Custom kernel development
- Algorithm experimentation
- Integration with existing CUDA codebases

### Limitations

**Not Recommended For**
- Single-threaded CPU-bound workloads
- Applications using fewer than 10 streams
- Workloads requiring only standard CUDA memory APIs
- Windows operating systems (Linux only)

## Technical Components

### TLSF Memory Allocator

The Two-Level Segregated Fit allocator provides constant-time memory allocation with minimal fragmentation. Originally designed for real-time embedded systems and used in aerospace applications (Mars rovers), TLSF maintains performance guarantees under all allocation patterns.

**Properties**
- O(1) allocation and deallocation
- O(1) block coalescing
- Segregated free lists by size class
- First-level index: power-of-two size classes
- Second-level index: linear subdivision within size class

**Implementation Details**
- Hash table for O(1) block lookup
- Automatic defragmentation on free
- Health monitoring and warning thresholds
- Memory leak detection via timestamps

### Stream Management System

Ferrite-OS extends CUDA's stream model to support massive parallelism:

**Configuration**
- Default pool: 10,000 streams
- Maximum pool: 100,000 streams (configurable at compile time)
- Priority levels: 6 levels (-5 to 0) for scheduling control

**Features**
- Per-stream allocation queues
- Round-robin stream selection
- Priority-based scheduling
- Independent synchronization

### Multi-Process IPC

System V shared memory enables multiple processes to share a single GPU context:

**Architecture**
- Daemon process: Creates and manages shared memory segment
- Client processes: Attach to existing shared memory
- Global state: Process registry, allocation tables, synchronization primitives

**Capabilities**
- Zero-copy memory sharing between processes
- Coordinated GPU access
- Shared allocation pool across processes

## Benchmarks

Benchmark and stress runs are driven through `tooling/scripts/ptx_bench_all.sh`:

- **jitter_benchmark** (`crates/public/ptx-runtime/examples`) - runtime jitter and latency stability
- **fused_kernel_benchmark** (`crates/public/ptx-runtime/examples`) - fused-kernel execution path
- **candle_performance_benchmark** (`crates/public/ptx-runtime/examples`) - candle op throughput
- **bench_dynamic_shapes** (`crates/internal/ptx-tensor/examples`) - shape variability impact
- **bench_ops** (`crates/internal/ptx-tensor/examples`) - elementwise op throughput
- **bench_matmul** (`crates/internal/ptx-tensor/examples`) - matrix multiply baseline

Outputs are written under `tooling/benchmarks/`.

## Development Roadmap

### Planned Features
- Multi-GPU support with peer-to-peer memory access
- Native cuBLAS and cuDNN integration
- Windows/WSL2 compatibility
- CUDA graph capture optimization
- Quantization kernel support (INT8, FP16, INT4)

### Future Targets
- ROCm/AMD GPU support
- Vulkan compute backend
- Distributed memory management across nodes

## Contributing

Contributions are welcome. Please ensure:
- All tests pass before submitting pull requests
- Code follows existing style conventions
- New features include documentation and tests
- Unsafe code includes safety documentation

See `../docs/04-llm-programming-guides/README.md` for change and validation playbooks.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## References

- Masmano, M., Ripoll, I., Crespo, A., & Real, J. (2004). TLSF: A New Dynamic Memory Allocator for Real-Time Systems. ECRTS 2004.
- NVIDIA CUDA Programming Guide
- RAPIDS Memory Manager (RMM) architecture
- PyTorch Caching Allocator design

## Acknowledgments

- TLSF algorithm implementation based on Masmano et al. (2004)
- Inspired by RAPIDS Memory Manager and PyTorch caching allocator
- Built on NVIDIA CUDA ecosystem

---

**Ferrite-OS**: Production-grade GPU operating system for high-performance computing.
