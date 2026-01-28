# Real-Time Inference Guide

This guide covers techniques for achieving sub-100 microsecond inference latency.

## Latency Requirements

Different applications have different latency budgets:

| Application | Latency Budget | Frame/Sample Rate |
|-------------|----------------|-------------------|
| Audio processing | 5 ms | 48 kHz / 256 samples |
| Game AI | 16 ms | 60 fps |
| Robotics control | 1 ms | 1000 Hz |
| High-frequency trading | 100 us | - |
| Video analytics | 33 ms | 30 fps |

Ferrite targets the 50-100 microsecond range.

## Latency Sources

| Source | Typical Range | Reducible? |
|--------|---------------|------------|
| Memory allocation | 1-100 ms | Yes - pre-allocate |
| Host-device transfer | 10-1000 us | Yes - zero-copy |
| Kernel launch overhead | 2-10 us | Partially |
| GPU computation | 10-1000 us | Limited by math |
| Device-host transfer | 10-1000 us | Yes - defer |

## Achieving Low Latency

### 1. Pre-allocate All Memory

```rust
// Allocate during initialization
stream.alloc("input", BATCH * INPUT_DIM);
stream.alloc("w1", INPUT_DIM * HIDDEN_DIM);
stream.alloc("h1", BATCH * HIDDEN_DIM);
stream.alloc("output", BATCH * OUTPUT_DIM);

// Initialize weights once
stream.init_weights("w1", INPUT_DIM as i32, HIDDEN_DIM as i32, "xavier");
stream.sync();
```

### 2. Warm Up the GPU

```rust
// First inference is slower due to JIT compilation
for _ in 0..100 {
    forward_pass(&stream);
}
stream.sync();
```

### 3. Use Batch Size 1

Smaller batches have lower latency:

```rust
let batch = 1;  // Minimum latency
stream.linear("input", "w1", None, "h1", 1, input_dim, hidden_dim);
```

### 4. Minimize Sync Points

```rust
// Bad: Multiple syncs
stream.linear(...);
stream.sync();
stream.relu(...);
stream.sync();

// Good: Single sync
stream.linear(...);
stream.relu(...);
stream.sync();
```

### 5. Keep Model Small

| Model Size | Typical Latency |
|------------|-----------------|
| 10K params | 20-30 us |
| 100K params | 30-50 us |
| 1M params | 50-100 us |

## Latency Measurement

```rust
fn benchmark_latency(stream: &Stream, iterations: usize) {
    let mut latencies = Vec::new();

    // Warm up
    for _ in 0..100 {
        forward_pass(stream);
    }
    stream.sync();

    // Measure
    for _ in 0..iterations {
        let start = Instant::now();
        forward_pass(stream);
        stream.sync();
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies.len();

    println!("P50: {:.2} us", latencies[n / 2]);
    println!("P99: {:.2} us", latencies[n * 99 / 100]);
    println!("Max: {:.2} us", latencies[n - 1]);
}
```

## System Configuration

### GPU Settings

```bash
# Disable power management
sudo nvidia-smi -pm 1

# Lock clock speeds
sudo nvidia-smi -lgc 1500,1500
```

### CPU Affinity

Pin inference thread to dedicated core:

```rust
use core_affinity;

let cores = core_affinity::get_core_ids().unwrap();
core_affinity::set_for_current(cores[0]);
```

## Performance Targets

| Metric | Target |
|--------|--------|
| P50 | < 50 us |
| P99 | < 100 us |
| P99.9 | < 200 us |
| Max | < 500 us |
