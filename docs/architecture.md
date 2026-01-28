# Architecture

This document describes Ferrite's internal architecture and design decisions.

## System Overview

Ferrite is structured in three layers:

```
+--------------------------------------------------+
|                  Application Code                 |
+--------------------------------------------------+
|                  Ferrite Runtime                  |
|  +------------+  +------------+  +-------------+ |
|  |   Stream   |  |  Runtime   |  |   Pipeline  | |
|  | Zero-copy  |  |  CPU/GPU   |  |   Builder   | |
|  +------------+  +------------+  +-------------+ |
+--------------------------------------------------+
|                  SPCPP Backend                    |
|  +------------+  +------------+  +-------------+ |
|  |   cuBLAS   |  |  JIT CUDA  |  |    TLSF     | |
|  |   MatMul   |  |  Kernels   |  |  Allocator  | |
|  +------------+  +------------+  +-------------+ |
+--------------------------------------------------+
|                    CUDA / GPU                     |
+--------------------------------------------------+
```

## Components

### Stream

The `Stream` type provides zero-copy GPU execution. It maintains a mapping from tensor names to GPU memory handles:

```rust
pub struct Stream {
    allocator: Arc<TlsfAllocator>,
    handles: HashMap<String, GpuHandle>,
    pool_base: u64,
}
```

Operations receive tensor names and resolve them to raw GPU pointers:

```rust
fn ptr(&self, name: &str) -> *mut f32 {
    let handle = self.handles.get(name).expect("Unknown tensor");
    (self.pool_base + handle.offset as u64) as *mut f32
}
```

### TlsfAllocator

The Two-Level Segregated Fit allocator provides O(1) allocation from a pre-allocated GPU memory pool:

```rust
pub struct TlsfAllocator {
    device: Arc<CudaDevice>,
    pool: CudaSlice<u8>,
    free_offset: AtomicUsize,
    capacity: usize,
}
```

Key properties:

- Constant-time allocation and deallocation
- Zero fragmentation for typical ML workloads
- No system calls during operation
- Deterministic performance

### SPCPP Backend

SPCPP provides GPU compute through two mechanisms:

1. **cuBLAS**: Optimized matrix operations (GEMM, GEMV)
2. **JIT Kernels**: Runtime-compiled CUDA for element-wise operations

The backend auto-detects GPU compute capability and compiles kernels accordingly:

```cpp
string detect_sm_version() {
    FILE* pipe = popen("nvidia-smi --query-gpu=compute_cap ...", "r");
    // Parse "8.6" -> "86"
    return sm;
}
```

### Runtime

The `Runtime` type provides a unified interface for CPU and GPU execution:

```rust
pub enum Runtime {
    Cpu(CpuBackend),
    Gpu(Backend),
}
```

Use `Runtime::auto()` to automatically select the best available device.

## Memory Model

### Pool Allocation

All GPU memory is allocated from a single contiguous pool:

```
+----------------------------------------------------------+
|                     GPU Memory Pool                       |
+----------------------------------------------------------+
| tensor_a | tensor_b |  free  | tensor_c | tensor_d | ... |
+----------------------------------------------------------+
     ^          ^                    ^          ^
     |          |                    |          |
   offset_a   offset_b            offset_c   offset_d
```

Benefits:

- No fragmentation from repeated alloc/free cycles
- Predictable memory bounds
- Cache-friendly access patterns

### Handle System

Tensors are represented as offset/size pairs:

```rust
pub struct GpuHandle {
    pub offset: usize,
    pub size: usize,
}
```

Converting a handle to a pointer is a single addition:

```rust
let ptr = pool_base + handle.offset;
```

## Execution Model

### Zero-Copy Pipeline

Operations execute directly on GPU memory:

```
  alloc("a")     alloc("b")     alloc("c")
      |              |              |
      v              v              v
+----------------------------------------------------------+
|     a      |      b      |      c      |    free         |
+----------------------------------------------------------+
      |              |              |
      +------+-------+              |
             |                      |
         matmul(a, b, c)            |
             |                      |
             +----------------------+
                      |
                  result in c
```

No intermediate buffers. No host-device copies. Operations write directly to their output location.

### Asynchronous Execution

GPU operations are queued and execute asynchronously:

```rust
stream.matmul("a", "b", "c", m, k, n);  // Queued
stream.relu("c", n);                     // Queued
stream.softmax("c", rows, cols);         // Queued
stream.sync();                           // Wait for all
```

The `sync()` call blocks until all queued operations complete.

## Backend Integration

### cuBLAS

Matrix operations use cuBLAS for maximum performance:

```cpp
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &alpha,
    b, n,
    a, k,
    &beta,
    c, n);
```

cuBLAS uses column-major ordering. Ferrite handles the row-major to column-major conversion transparently.

### JIT Compilation

Element-wise operations use runtime-compiled CUDA:

```cpp
// ops.cu - compiled at first use
__global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}
```

Kernels are compiled once and cached. The first invocation incurs compilation overhead; subsequent calls are instant.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Allocation | O(1) | TLSF from pool |
| Pointer lookup | O(1) | HashMap + addition |
| MatMul | O(n^3) | cuBLAS SGEMM |
| Element-wise | O(n) | JIT CUDA kernel |
| Sync | O(1) | CUDA stream sync |

### Latency Sources

Typical forward pass breakdown:

| Component | Time |
|-----------|------|
| Pointer resolution | < 1 us |
| Kernel launch | 2-5 us |
| MatMul execution | 10-50 us |
| Activation | 2-5 us |
| Sync overhead | 1-2 us |

Total: 15-65 us for a small network.

## Thread Safety

The `Stream` type is not thread-safe. For multi-threaded applications:

- Create one `Stream` per thread
- Share the `TlsfAllocator` across threads (it uses atomic operations internally)
- Synchronize access to shared tensors manually

## Comparison with Other Frameworks

| Aspect | PyTorch | TensorFlow | Ferrite |
|--------|---------|------------|---------|
| Memory allocation | Dynamic | Dynamic | Static pool |
| GC pauses | Yes (Python) | Yes (Python) | No |
| Host-device copies | Implicit | Implicit | Explicit |
| Latency variance | High | High | Low |
| Minimum latency | ~1 ms | ~1 ms | ~50 us |
