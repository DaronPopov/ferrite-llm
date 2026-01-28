# Ferrite Documentation

Ferrite is a high-performance GPU compute runtime for Rust. It provides zero-copy execution, O(1) memory allocation, and sub-100 microsecond inference latency.

## Overview

Ferrite combines Rust's memory safety with CUDA's computational power. The runtime eliminates common performance bottlenecks found in traditional ML frameworks:

- No garbage collection pauses
- No hidden memory transfers
- No dynamic allocation during inference
- Predictable, bounded latency

## Documentation

### Getting Started

- [Installation](getting-started.md) - System requirements and installation
- [Quick Start](quickstart.md) - Your first Ferrite program
- [Building from Source](building.md) - Compile Ferrite yourself

### Architecture

- [System Overview](architecture.md) - How Ferrite works
- [Memory Model](memory-model.md) - TLSF allocation and memory pools
- [Execution Model](execution-model.md) - Zero-copy streaming pipeline

### API Reference

- [Stream](api/stream.md) - Zero-copy execution context
- [Runtime](api/runtime.md) - Unified CPU/GPU interface
- [TlsfAllocator](api/allocator.md) - O(1) GPU memory allocator
- [Pipeline](api/pipeline.md) - Declarative operation builder

### Guides

- [Zero-Copy Execution](guides/zero-copy.md) - Eliminating memory transfers
- [Real-Time Inference](guides/real-time.md) - Sub-100 microsecond latency
- [Memory Management](guides/memory.md) - Pool sizing and optimization
- [Model Hot-Swap](guides/hot-swap.md) - Zero-downtime model updates

### Examples

- [Real-Time Inference](examples/realtime.md) - Latency-critical applications
- [Model Ensemble](examples/ensemble.md) - Multiple models, shared memory
- [Streaming Pipeline](examples/streaming.md) - Continuous data processing
- [Trading System](examples/trading.md) - Microsecond decision making

## Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Linux x86_64 |
| GPU | NVIDIA Compute Capability 6.0+ |
| CUDA Toolkit | 11.0 or later |
| Rust | 1.70 or later |

## License

Ferrite is dual-licensed under MIT and Apache 2.0.
