# Memory Management Guide

This guide covers GPU memory allocation, pool sizing, and optimization strategies.

## Memory Architecture

Ferrite uses a single pre-allocated memory pool:

```
GPU Memory
+------------------------------------------------------------------+
|                        Memory Pool                                |
| +----------+----------+----------+----------+----------+--------+ |
| | Tensor A | Tensor B | Tensor C | Tensor D |   ...    |  Free  | |
| +----------+----------+----------+----------+----------+--------+ |
+------------------------------------------------------------------+
      ^
      |
   Pool Base Pointer
```

Benefits of this approach:

- O(1) allocation time
- No fragmentation
- Predictable memory bounds
- No GPU memory allocation during inference

## Pool Sizing

### Calculating Required Size

Sum the memory requirements for all tensors:

```rust
fn calculate_pool_size(batch: usize) -> usize {
    let mut total = 0;

    // Model weights
    total += 784 * 256 * 4;  // w1: f32
    total += 256 * 4;        // b1
    total += 256 * 10 * 4;   // w2
    total += 10 * 4;         // b2

    // Activations (batch-dependent)
    total += batch * 784 * 4;   // input
    total += batch * 256 * 4;   // hidden
    total += batch * 10 * 4;    // output

    // Gradients (for training)
    total += 784 * 256 * 4;  // grad_w1
    total += 256 * 4;        // grad_b1
    total += 256 * 10 * 4;   // grad_w2
    total += 10 * 4;         // grad_b2

    // Add 20% headroom
    (total as f64 * 1.2) as usize
}
```

### Common Model Sizes

Reference memory requirements:

| Model Type | Parameters | Weights (FP32) | Activations (batch=32) | Total |
|------------|------------|----------------|------------------------|-------|
| Small MLP | 200K | 800 KB | 100 KB | ~1 MB |
| Medium MLP | 2M | 8 MB | 500 KB | ~10 MB |
| Small CNN | 500K | 2 MB | 10 MB | ~15 MB |
| ResNet-18 | 11M | 44 MB | 50 MB | ~100 MB |
| BERT-base | 110M | 440 MB | 100 MB | ~600 MB |

### Checking Available Memory

Query GPU memory before allocation:

```bash
nvidia-smi --query-gpu=memory.free,memory.total --format=csv
```

In code:

```rust
// Using cudarc to get device properties
let device = CudaDevice::new(0).unwrap();
// Note: Memory queries depend on cudarc API
```

Leave headroom for:

- CUDA runtime overhead (~100-500 MB)
- cuBLAS workspace (~100-500 MB)
- Other processes

## Allocation Strategies

### Strategy 1: Fixed Pool

Allocate a fixed-size pool at startup:

```rust
let pool_size = 256 * 1024 * 1024; // 256 MB
let allocator = Arc::new(TlsfAllocator::new(device, pool_size));
```

Best for:
- Production deployments
- Known model sizes
- Predictable workloads

### Strategy 2: Adaptive Pool

Choose pool size based on model requirements:

```rust
fn create_allocator_for_model(device: Arc<CudaDevice>, model_config: &Config) -> Arc<TlsfAllocator> {
    let weights_size = model_config.total_parameters() * 4;
    let activation_size = model_config.max_activation_size() * 4;
    let gradient_size = if model_config.training { weights_size } else { 0 };

    let required = weights_size + activation_size + gradient_size;
    let pool_size = (required as f64 * 1.5) as usize; // 50% headroom

    Arc::new(TlsfAllocator::new(device, pool_size))
}
```

### Strategy 3: Multiple Pools

Use separate pools for different components:

```rust
let weights_allocator = Arc::new(TlsfAllocator::new(device.clone(), 100 * 1024 * 1024));
let activation_allocator = Arc::new(TlsfAllocator::new(device.clone(), 50 * 1024 * 1024));

let mut weights_stream = Stream::new(weights_allocator);
let mut activation_stream = Stream::new(activation_allocator);
```

Useful for:
- Isolating model weights from activations
- Different lifecycle management
- Multi-model systems

## Memory Optimization

### Activation Reuse

Reuse activation buffers across layers:

```rust
// Instead of:
stream.alloc("h1", batch * 256);
stream.alloc("h2", batch * 256);
stream.alloc("h3", batch * 256);

// Reuse buffers:
stream.alloc("hidden_a", batch * 256);
stream.alloc("hidden_b", batch * 256);

// Alternate between buffers
stream.linear("input", "w1", None, "hidden_a", ...);
stream.relu("hidden_a", n);
stream.linear("hidden_a", "w2", None, "hidden_b", ...);
stream.relu("hidden_b", n);
stream.linear("hidden_b", "w3", None, "hidden_a", ...);  // Reuse hidden_a
```

Memory savings: 33% for a 3-layer network.

### In-Place Operations

Many operations can be performed in-place:

```rust
// Activation functions are in-place
stream.relu("hidden", n);      // Modifies hidden directly
stream.sigmoid("gate", n);     // Modifies gate directly
stream.softmax("logits", b, c); // Modifies logits directly

// Some operations support output == input
stream.scale("gradients", 0.01, n);  // In-place scaling
```

### Weight Sharing

Share weights between model components:

```rust
// Shared embedding table
stream.alloc("shared_embedding", vocab_size * embed_dim);
stream.init_weights("shared_embedding", vocab_size as i32, embed_dim as i32, "normal");

// Use for both encoder and decoder
stream.embedding("shared_embedding", encoder_input, encoder_out, batch, embed_dim);
stream.embedding("shared_embedding", decoder_input, decoder_out, batch, embed_dim);
```

### Gradient Checkpointing

For training, trade computation for memory:

```rust
// Standard: Keep all activations
stream.alloc("h1", batch * 256);
stream.alloc("h2", batch * 256);
stream.alloc("h3", batch * 256);
// Memory: 3 * batch * 256

// Checkpointing: Only keep checkpoints
stream.alloc("checkpoint_1", batch * 256);
// Memory: 1 * batch * 256
// Recompute h2, h3 during backward pass
```

## Monitoring Memory Usage

### Runtime Monitoring

Track allocation during execution:

```rust
fn log_memory_usage(allocator: &TlsfAllocator, label: &str) {
    let used = allocator.consumed();
    let total = allocator.capacity();
    let pct = 100.0 * used as f64 / total as f64;

    println!("[{}] Memory: {} / {} MB ({:.1}%)",
             label,
             used / (1024 * 1024),
             total / (1024 * 1024),
             pct);
}

// Usage
stream.alloc("weights", size);
log_memory_usage(&allocator, "After weights");

stream.alloc("activations", size);
log_memory_usage(&allocator, "After activations");
```

### Peak Memory Tracking

Track maximum memory usage:

```rust
struct MemoryTracker {
    allocator: Arc<TlsfAllocator>,
    peak: usize,
}

impl MemoryTracker {
    fn update(&mut self) {
        let current = self.allocator.consumed();
        if current > self.peak {
            self.peak = current;
        }
    }

    fn report(&self) {
        println!("Peak memory usage: {} MB", self.peak / (1024 * 1024));
        println!("Pool utilization: {:.1}%",
                 100.0 * self.peak as f64 / self.allocator.capacity() as f64);
    }
}
```

## Troubleshooting

### Out of Memory

Symptoms:
- Allocation returns error
- Panic with "TLSF alloc failed"

Solutions:
1. Increase pool size
2. Reduce batch size
3. Use activation reuse
4. Enable gradient checkpointing

### Fragmentation

The TLSF allocator is designed to minimize fragmentation, but issues can occur with:
- Many small allocations followed by frees
- Highly variable allocation sizes

Current limitation: Individual tensor deallocation is not supported. Memory is reclaimed only when the allocator is dropped.

Workaround: Reuse tensor slots instead of allocating new ones.

### Memory Leaks

Symptoms:
- `consumed()` grows over time
- Eventually runs out of pool space

Cause: Allocating tensors in a loop without reusing names.

```rust
// Bad: Leaks memory
for i in 0..1000 {
    stream.alloc(&format!("temp_{}", i), size);  // 1000 allocations!
}

// Good: Reuse single buffer
stream.alloc("temp", size);
for i in 0..1000 {
    stream.fill("temp", i as f32, size as i32);  // Reuse same buffer
}
```

## Best Practices

1. **Size pools for peak usage** - Account for maximum batch size and all model components

2. **Allocate early** - All `alloc()` calls should happen during initialization

3. **Reuse buffers** - Design tensor layouts to maximize reuse

4. **Monitor usage** - Track memory consumption during development

5. **Test limits** - Verify behavior at maximum batch size before deployment

6. **Leave headroom** - Allocate 20-50% more than calculated minimum
