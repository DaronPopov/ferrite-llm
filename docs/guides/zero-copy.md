# Zero-Copy Execution Guide

This guide explains how to eliminate memory transfers and achieve maximum GPU utilization.

## Understanding Memory Transfers

In traditional ML frameworks, data frequently moves between host (CPU) and device (GPU):

```
Host Memory          GPU Memory
+-----------+        +-----------+
|  input    | -----> |  input    |
+-----------+        +-----------+
                           |
                     [computation]
                           |
                     +-----------+
                     |  output   |
+-----------+ <----- +-----------+
|  output   |
+-----------+
```

Each transfer has overhead:

| Transfer Type | Typical Latency |
|---------------|-----------------|
| Host to Device (small) | 5-20 us |
| Host to Device (large) | 100+ us |
| Device to Host (small) | 5-20 us |
| Device to Host (large) | 100+ us |
| Kernel launch | 2-10 us |

For latency-critical applications, these transfers dominate execution time.

## Zero-Copy Architecture

Ferrite keeps data on the GPU throughout the computation pipeline:

```
GPU Memory Pool
+----------------------------------------------------------+
| input | weights | hidden | output | gradients | ...      |
+----------------------------------------------------------+
    |        |         |        |
    +--------+---------+--------+
             |
    [all computation stays on GPU]
             |
             v
    Only copy when result needed
```

## Implementing Zero-Copy

### Step 1: Pre-allocate Everything

Allocate all tensors before the computation loop:

```rust
// Allocate at startup
stream.alloc("input", batch * input_dim);
stream.alloc("w1", input_dim * hidden_dim);
stream.alloc("b1", hidden_dim);
stream.alloc("h1", batch * hidden_dim);
stream.alloc("w2", hidden_dim * output_dim);
stream.alloc("b2", output_dim);
stream.alloc("output", batch * output_dim);

// Initialize weights on GPU
stream.init_weights("w1", input_dim as i32, hidden_dim as i32, "xavier");
stream.init_weights("w2", hidden_dim as i32, output_dim as i32, "xavier");
stream.fill("b1", 0.0, hidden_dim as i32);
stream.fill("b2", 0.0, output_dim as i32);
stream.sync();
```

### Step 2: Use GPU-Side Data Generation

When possible, generate or transform data on the GPU:

```rust
// Bad: Upload from host each iteration
for i in 0..num_batches {
    let host_data = generate_batch(i);
    stream.stage_host("input", host_data);
    stream.upload("input");  // Transfer!
    // ... inference
}

// Good: Generate on GPU
for i in 0..num_batches {
    stream.fill("input", (i as f32).sin(), (batch * input_dim) as i32);
    // ... inference (no transfer)
}
```

### Step 3: Chain Operations Without Sync

Queue multiple operations before synchronizing:

```rust
// Bad: Sync after each operation
stream.linear("input", "w1", Some("b1"), "h1", ...);
stream.sync();  // Unnecessary wait
stream.relu("h1", n);
stream.sync();  // Unnecessary wait
stream.linear("h1", "w2", Some("b2"), "output", ...);
stream.sync();

// Good: Single sync at the end
stream.linear("input", "w1", Some("b1"), "h1", ...);
stream.relu("h1", n);
stream.linear("h1", "w2", Some("b2"), "output", ...);
stream.sync();  // One sync for entire pipeline
```

### Step 4: Defer Downloads

Only download results when absolutely necessary:

```rust
// Bad: Download every iteration
for _ in 0..1000 {
    // ... inference
    let output = stream.download("output");  // 1000 transfers!
    if output[0] > threshold {
        // ...
    }
}

// Good: Download only when needed
for i in 0..1000 {
    // ... inference
    stream.sync();

    // Only download occasionally
    if i % 100 == 0 {
        let output = stream.download("output");
        println!("Checkpoint: {:?}", &output[0..10]);
    }
}

// Or: Download only the final result
for _ in 0..1000 {
    // ... inference
}
stream.sync();
let final_output = stream.download("output");  // Single transfer
```

## Measuring Transfer Overhead

Profile your application to identify transfer bottlenecks:

```rust
use std::time::Instant;

// Measure inference without download
let start = Instant::now();
for _ in 0..1000 {
    stream.linear("input", "w1", Some("b1"), "h1", ...);
    stream.relu("h1", n);
    stream.linear("h1", "w2", Some("b2"), "output", ...);
}
stream.sync();
let compute_time = start.elapsed();

// Measure with download
let start = Instant::now();
for _ in 0..1000 {
    stream.linear("input", "w1", Some("b1"), "h1", ...);
    stream.relu("h1", n);
    stream.linear("h1", "w2", Some("b2"), "output", ...);
    stream.sync();
    let _ = stream.download("output");  // Forces transfer
}
let total_time = start.elapsed();

let transfer_overhead = total_time - compute_time;
println!("Compute: {:?}", compute_time);
println!("Transfer overhead: {:?}", transfer_overhead);
println!("Overhead ratio: {:.1}%",
         100.0 * transfer_overhead.as_secs_f64() / total_time.as_secs_f64());
```

## Double Buffering

For streaming applications, use double buffering to overlap data loading with computation:

```rust
// Allocate two sets of buffers
stream.alloc("input_a", batch * dim);
stream.alloc("input_b", batch * dim);
stream.alloc("output_a", batch * out_dim);
stream.alloc("output_b", batch * out_dim);

let mut use_a = true;

for frame in frames {
    if use_a {
        // Load into B while processing A
        stream.fill("input_b", frame as f32, (batch * dim) as i32);
        forward_pass(&stream, "input_a", "output_a");
    } else {
        // Load into A while processing B
        stream.fill("input_a", frame as f32, (batch * dim) as i32);
        forward_pass(&stream, "input_b", "output_b");
    }
    use_a = !use_a;
}
```

## Common Pitfalls

### Hidden Allocations

Avoid operations that allocate internally:

```rust
// Bad: Creates temporary vector on each iteration
for _ in 0..1000 {
    let result = stream.download("output");  // Allocates Vec<f32>
    process(result);
}

// Good: Reuse buffer
let mut result_buffer = vec![0.0f32; output_size];
for _ in 0..1000 {
    stream.copy_to_host("output", &mut result_buffer);
    process(&result_buffer);
}
```

### Implicit Synchronization

Some operations force synchronization:

```rust
// These operations block until complete:
stream.download("tensor");       // Blocks
stream.dot("a", "b", n);         // Returns scalar, must sync
stream.nrm2("x", n);             // Returns scalar, must sync
stream.mse_loss("pred", "tgt", n); // Returns scalar, must sync
```

### Tensor Name Allocation

String operations can allocate:

```rust
// Bad: Allocates new String each iteration
for i in 0..1000 {
    let name = format!("tensor_{}", i);  // Allocation!
    stream.alloc(&name, size);
}

// Good: Use static names or pre-computed
const TENSOR_NAMES: [&str; 4] = ["t0", "t1", "t2", "t3"];
for name in TENSOR_NAMES.iter().cycle().take(1000) {
    stream.alloc(name, size);
}
```

## Performance Checklist

Use this checklist to verify zero-copy execution:

- [ ] All tensors pre-allocated before main loop
- [ ] No `upload()` calls in hot path
- [ ] No `download()` calls in hot path (or minimized)
- [ ] Single `sync()` at end of operation sequence
- [ ] No `format!()` or string allocation for tensor names
- [ ] Double buffering for streaming workloads
- [ ] Profile confirms < 5% transfer overhead
