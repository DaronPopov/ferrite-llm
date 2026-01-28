# TlsfAllocator API

The `TlsfAllocator` provides O(1) GPU memory allocation using the Two-Level Segregated Fit algorithm.

## Import

```rust
use ferrite::dynamics::allocator::TlsfAllocator;
```

## Overview

Traditional GPU memory allocation involves system calls that can take milliseconds and have unpredictable latency. TLSF eliminates this by:

1. Pre-allocating a contiguous memory pool at startup
2. Managing allocations within the pool using O(1) operations
3. Avoiding fragmentation through segregated free lists

## Construction

### new

```rust
pub fn new(device: Arc<CudaDevice>, capacity: usize) -> Self
```

Creates a new allocator with a pre-allocated GPU memory pool.

| Parameter | Type | Description |
|-----------|------|-------------|
| device | Arc<CudaDevice> | CUDA device handle |
| capacity | usize | Pool size in bytes |

```rust
use cudarc::driver::CudaDevice;
use std::sync::Arc;

let device = CudaDevice::new(0).expect("No CUDA device");
let allocator = Arc::new(TlsfAllocator::new(device, 256 * 1024 * 1024));
```

## Pool Sizing

Choose pool size based on your workload:

| Workload | Recommended Size |
|----------|------------------|
| Small models (< 1M params) | 64 MB |
| Medium models (1-10M params) | 256 MB |
| Large models (10-100M params) | 1 GB |
| Very large models | 4+ GB |

The pool must fit in available GPU memory. Check availability with:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

## Methods

### alloc

```rust
pub fn alloc(&self, size: usize) -> Result<usize, String>
```

Allocates memory from the pool. Returns the byte offset within the pool.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | usize | Allocation size in bytes |

Returns: `Ok(offset)` on success, `Err(message)` if pool exhausted.

```rust
let offset = allocator.alloc(1024 * 4).expect("Allocation failed");
```

### capacity

```rust
pub fn capacity(&self) -> usize
```

Returns the total pool size in bytes.

```rust
println!("Pool size: {} MB", allocator.capacity() / (1024 * 1024));
```

### consumed

```rust
pub fn consumed(&self) -> usize
```

Returns the number of bytes currently allocated.

```rust
let used = allocator.consumed();
let free = allocator.capacity() - used;
println!("Used: {} MB, Free: {} MB", used / (1024 * 1024), free / (1024 * 1024));
```

### pool_ptr

```rust
pub fn pool_ptr(&self) -> u64
```

Returns the base GPU pointer of the memory pool.

```rust
let base = allocator.pool_ptr();
let tensor_ptr = (base + offset as u64) as *mut f32;
```

### copy_to_offset

```rust
pub fn copy_to_offset(&self, offset: usize, data: &[f32])
```

Copies host data to a location within the pool.

```rust
let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
allocator.copy_to_offset(offset, &weights);
```

### copy_from_offset

```rust
pub fn copy_from_offset(&self, offset: usize, data: &mut [f32])
```

Copies data from the pool to host memory.

```rust
let mut result = vec![0.0f32; 1024];
allocator.copy_from_offset(offset, &mut result);
```

## Thread Safety

The `TlsfAllocator` is thread-safe. Internal operations use atomic instructions for the allocation pointer. Multiple threads can share a single allocator:

```rust
let allocator = Arc::new(TlsfAllocator::new(device, pool_size));

let alloc_clone = allocator.clone();
std::thread::spawn(move || {
    let offset = alloc_clone.alloc(1024).unwrap();
    // Use offset...
});
```

## Memory Layout

Allocations are contiguous within the pool:

```
Pool Base Address
|
v
+--------+--------+--------+--------+--------+
| Alloc1 | Alloc2 | Alloc3 |  Free  |  ...   |
+--------+--------+--------+--------+--------+
         ^        ^        ^
         |        |        |
     offset1  offset2  offset3
```

To convert an offset to a usable pointer:

```rust
let ptr = (allocator.pool_ptr() + offset as u64) as *mut f32;
```

## Limitations

Current implementation limitations:

1. **No individual free**: Memory is reclaimed only when the allocator is dropped
2. **Fixed pool size**: Cannot grow after creation
3. **Single pool**: One contiguous allocation per allocator

For workloads requiring dynamic memory management, consider:

- Creating multiple allocators for different model components
- Sizing the pool for peak memory usage
- Reusing tensor slots instead of allocating new ones

## Performance Characteristics

| Operation | Time Complexity | Typical Latency |
|-----------|-----------------|-----------------|
| alloc | O(1) | < 100 ns |
| pool_ptr | O(1) | < 10 ns |
| copy_to_offset | O(n) | Depends on size |
| copy_from_offset | O(n) | Depends on size |

## Example: Multi-Model Setup

```rust
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() {
    let device = CudaDevice::new(0).unwrap();

    // One pool for all models
    let pool_size = 512 * 1024 * 1024; // 512 MB
    let allocator = Arc::new(TlsfAllocator::new(device, pool_size));

    // Allocate space for multiple models
    let model_a_weights = allocator.alloc(10 * 1024 * 1024).unwrap();
    let model_b_weights = allocator.alloc(20 * 1024 * 1024).unwrap();
    let shared_embeddings = allocator.alloc(50 * 1024 * 1024).unwrap();

    println!("Model A at offset: {}", model_a_weights);
    println!("Model B at offset: {}", model_b_weights);
    println!("Embeddings at offset: {}", shared_embeddings);
    println!("Total used: {} MB", allocator.consumed() / (1024 * 1024));
}
```
