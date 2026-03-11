# Integrating External Crates with Ferrite-OS

Any Rust crate that does GPU work can be wired into a FerApp application. The runtime exposes standard CUDA interop primitives — context, streams, and device pointers — so external libraries run on the same GPU, same memory pool, and same scheduler as native Ferrite operations.

## Built-in Crate Integrations

`ptx-app` ships with six external crates already wired into the SDK. These are available out of the box through `Ctx` methods — no extra dependencies needed in your application.

| Crate | Version | What it provides |
|-------|---------|-----------------|
| **bytemuck** | 1.16 | `Pod` trait for safe GPU memory transfers — all tensor `from_slice` / `to_vec` use `Pod` instead of raw `Copy` |
| **indicatif** | 0.17 | `ctx.progress(total, name)` returns a progress bar; `ctx.spinner(msg)` returns a spinner. `ProgressBar` is re-exported from `ptx_app` |
| **directories** | 5 | Platform-correct checkpoint paths via `ProjectDirs` (XDG on Linux, `~/Library/Application Support` on macOS) |
| **memmap2** | 0.9 | `ctx.mmap_load(path, shape, dtype)` — zero-copy file-to-GPU tensor upload via memory mapping |
| **tempfile** | 3 | Atomic checkpoint writes via `NamedTempFile::persist()` — crash-safe, never leaves partial files |
| **humansize** | 2 | Human-readable memory stats in `PoolStats::Display` (e.g. `"384 MiB / 1.5 GiB (25.6% used)"`) and error messages |

### Quick reference

```rust
use ptx_app::{FerApp, DType, ProgressBar};

FerApp::new("training-job")
    .pool_fraction(0.5)
    .run(|ctx| {
        // Memory stats (humansize)
        println!("{}", ctx.pool_stats());  // "1.99 GiB / 1.99 GiB (0.0% used, 1.99 GiB free)"

        // Progress bar (indicatif)
        let pb = ctx.progress(100, "training");
        for epoch in 0..100 {
            // ... GPU work ...
            pb.inc(1);
        }
        pb.finish_with_message("done");

        // Spinner for unknown-duration ops (indicatif)
        let sp = ctx.spinner("loading weights");
        // ... long operation ...
        sp.finish_with_message("loaded");

        // Zero-copy file loading (memmap2)
        let weights = ctx.mmap_load("model.bin", &[1024, 1024], DType::F32)?;

        // Atomic checkpoints (tempfile + directories)
        ctx.checkpoint("epoch-99", &serde_json::json!({"loss": 0.01}))?;
        let state: Option<serde_json::Value> = ctx.restore("epoch-99")?;

        // Safe tensor data transfer (bytemuck)
        let data: Vec<f32> = vec![1.0; 1024];
        let tensor = ctx.from_slice(&data, &[1024], DType::F32)?;

        Ok(())
    })
    .expect("app failed");
```

## TUI Event Rendering

Events emitted via `ctx.emit()` are delivered end-to-end to the daemon TUI:

```
ptx-app ctx.emit("result", &payload)
  → Unix socket → handle_app_event()
  → SchedulerEventStream.emit(SchedulerEvent::AppEvent)
  → bridge thread → DaemonEvent::AppEvent
  → TUI event loop → log panel (LogCategory::App)
```

App events appear in the TUI log panel with the `App` category. This means `ctx.emit()` and `ctx.log()` calls are visible in real time while the daemon is running.

## The Three Interop Primitives

Every GPU library in the Rust ecosystem ultimately needs one or more of these:

| Primitive | How to get it | What it is |
|-----------|--------------|------------|
| `runtime.context()` | `ctx.runtime().context().unwrap()` | Raw `CUcontext` — the CUDA driver-level context handle |
| `stream.raw()` | `ctx.stream().raw()` | Raw `cudaStream_t` — for ordering GPU work |
| `tensor.data_ptr()` | `tensor.data_ptr()` | Raw `*mut c_void` — a device memory pointer |

If a crate accepts any of these, it works inside Ferrite.

## Step-by-Step

### 1. Add the crate to your Cargo.toml

```toml
[dependencies]
ptx-app = { path = "../ferrite-os/crates/public/ptx-app" }
my-cuda-lib = "0.5"
```

### 2. Pass the runtime context to the external crate

Most external GPU libraries need a CUDA context or device handle during initialization. Get it from the runtime inside your `FerApp::run` closure:

```rust
use ptx_app::{FerApp, DType, AppError};

fn main() -> Result<(), AppError> {
    FerApp::new("my-pipeline")
        .pool_fraction(0.5)
        .streams(8)
        .run(|ctx| {
            let runtime = ctx.runtime();

            // Option A: Pass the CUcontext (driver API libraries)
            let cuda_ctx = runtime.context().unwrap();
            let session = my_cuda_lib::Session::from_context(cuda_ctx)?;

            // Option B: Pass a stream for ordered execution
            let stream = ctx.stream();
            session.set_stream(stream.raw())?;

            // Option C: Pass raw device pointers for data exchange
            let tensor = ctx.tensor(&[1024, 1024], DType::F32)?.randn()?;
            session.process(tensor.data_ptr(), tensor.size_bytes())?;

            Ok(())
        })
}
```

### 3. Share streams for correct ordering

GPU operations are asynchronous. If you launch work through an external crate and then read the result with Ferrite tensors, you need both to use the same stream, or synchronize between them:

```rust
FerApp::new("stream-example")
    .run(|ctx| {
        let stream = ctx.stream();

        // External lib launches a kernel on the stream
        external_lib::launch_kernel(data_ptr, stream.raw())?;

        // Synchronize before reading the result
        stream.synchronize()?;

        // Now safe to read
        let result = tensor.to_vec::<f32>()?;
        Ok(())
    })
```

If the external crate creates its own streams internally, synchronize before crossing the boundary:

```rust
// External lib did work on its own internal stream
external_lib::run(data_ptr)?;
external_lib::synchronize()?;  // wait for its work to finish

// Now safe to use the data with Ferrite streams
let output = ctx.from_slice(&host_data, &[1024], DType::F32)?;
```

### 4. Exchange data via device pointers

Tensors created through `ctx.tensor()` live in the TLSF memory pool. Their raw device pointers are standard CUDA pointers that any library can read from or write to:

```rust
// Ferrite tensor -> external lib
let input = ctx.tensor(&[batch, channels, h, w], DType::F32)?.randn()?;
external_lib::infer(input.data_ptr() as *const f32, batch)?;

// External lib -> Ferrite tensor
// If the external lib returns a device pointer:
let ext_ptr = external_lib::get_output_ptr();
let output = ctx.tensor(&[batch, num_classes], DType::F32)?.zeros()?;
unsafe {
    ptx_sys::gpu_hot_memcpy_d2d(
        output.data_ptr(),
        ext_ptr as *mut libc::c_void,
        output.size_bytes(),
    );
}

// Or if it returns host data, use from_slice:
let host_result = external_lib::get_result_host();
let output = ctx.from_slice(&host_result, &[1024], DType::F32)?;
```

## Common Integration Patterns

### Pattern: cudarc / rustacuda

Libraries that use the CUDA driver API need a `CUcontext`:

```rust
FerApp::new("cudarc-app")
    .pool_fraction(0.4)
    .run(|ctx| {
        // cudarc can adopt the existing context
        // so it shares the same GPU state
        let cuda_ctx = ctx.runtime().context().unwrap();

        // Launch custom PTX kernels through cudarc
        // while Ferrite manages the memory pool and lifecycle
        Ok(())
    })
```

### Pattern: ndarray + GPU roundtrip

CPU-side crates like `ndarray` work by moving data to/from the GPU:

```rust
use ndarray::Array2;

FerApp::new("hybrid-compute")
    .run(|ctx| {
        // GPU computation
        let gpu_result = ctx.tensor(&[256, 256], DType::F32)?.randn()?;

        // Pull to CPU for ndarray processing
        let host_data = gpu_result.to_vec::<f32>()?;
        let matrix = Array2::from_shape_vec((256, 256), host_data).unwrap();
        let processed = &matrix * 2.0;

        // Push back to GPU
        let back_on_gpu = ctx.from_slice(
            processed.as_slice().unwrap(),
            &[256, 256],
            DType::F32,
        )?;

        Ok(())
    })
```

### Pattern: Custom CUDA kernels via ptx-compute

If you write your own kernels in the `ptx-compute` crate or load PTX from a file:

```rust
FerApp::new("custom-kernels")
    .streams(16)
    .run(|ctx| {
        let runtime = ctx.runtime();
        let stream = ctx.stream();

        let data = ctx.tensor(&[1_000_000], DType::F32)?.zeros()?;

        // Launch a custom kernel from ptx-compute
        ptx_compute::my_kernel::launch(
            data.data_ptr_typed::<f32>(),
            data.elem_count(),
            stream.raw(),
        )?;

        stream.synchronize()?;
        Ok(())
    })
```

### Pattern: Wrapping an external crate as a Ctx extension

For repeated use, wrap the external crate in a helper that takes a `&Ctx`:

```rust
// my_extensions.rs
use ptx_app::Ctx;

pub fn run_external_inference(
    ctx: &Ctx,
    input: &ptx_tensor::Tensor,
) -> Result<Vec<f32>, ptx_app::AppError> {
    let stream = ctx.stream();
    let result = external_lib::infer(input.data_ptr(), stream.raw())
        .map_err(|e| ptx_app::AppError::App { message: e.to_string() })?;
    stream.synchronize()?;
    Ok(result)
}

// main.rs
FerApp::new("inference-server")
    .restart(Restart::on_failure(3))
    .run(|ctx| {
        let input = ctx.tensor(&[1, 3, 224, 224], DType::F32)?.randn()?;
        let classes = my_extensions::run_external_inference(ctx, &input)?;
        ctx.emit("prediction", &classes);
        Ok(())
    })
```

### Pattern: Zero-copy file loading with mmap

Load large binary weight files directly into GPU tensors without double-copying through heap memory:

```rust
FerApp::new("model-loader")
    .pool_fraction(0.6)
    .run(|ctx| {
        let sp = ctx.spinner("loading model weights");

        // Memory-map the file, then upload to GPU in one transfer
        let weights = ctx.mmap_load("weights.bin", &[4096, 4096], DType::F32)?;
        let biases = ctx.mmap_load("biases.bin", &[4096], DType::F32)?;

        sp.finish_with_message("loaded");
        ctx.emit("model_loaded", &"ok");
        Ok(())
    })
```

### Pattern: Checkpoint and resume across restarts

Save training state atomically so the daemon can restart the job without losing progress:

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct TrainState { epoch: u32, loss: f64 }

FerApp::new("training")
    .restart(Restart::on_failure(5))
    .run(|ctx| {
        // Resume from last checkpoint if it exists
        let mut state: TrainState = ctx.restore("train")?
            .unwrap_or(TrainState { epoch: 0, loss: f64::MAX });

        let pb = ctx.progress(1000, "training");
        pb.set_position(state.epoch as u64);

        for epoch in state.epoch..1000 {
            // ... training step ...
            state.epoch = epoch;
            state.loss = 1.0 / (epoch as f64 + 1.0);

            if epoch % 10 == 0 {
                ctx.checkpoint("train", &state)?;
            }
            pb.inc(1);
        }
        pb.finish_with_message("done");
        Ok(())
    })
```

## What the external crate gets for free

By running inside `FerApp::run`, the external crate's GPU work automatically benefits from:

- **TLSF memory pool** — if the crate uses the shared context, allocations come from the managed pool with fragmentation resistance and health monitoring
- **Daemon supervision** — crashes trigger the restart policy; the daemon respawns the process with checkpoint state preserved
- **Event stream** — `ctx.emit()` sends results and progress to the daemon's event stream, visible in the TUI log panel in real time
- **Checkpointing** — save and restore state across restarts with `ctx.checkpoint()` / `ctx.restore()`, using atomic writes that never leave partial files
- **Progress bars** — `ctx.progress()` and `ctx.spinner()` for user-facing feedback during long operations
- **Zero-copy file loading** — `ctx.mmap_load()` maps files directly into GPU memory without intermediate heap copies
- **Stream scheduling** — round-robin stream access prevents stream contention across libraries
- **Resource quotas** — the daemon enforces pool fraction and stream limits, so the external crate can't starve other tenants
- **Safe memory transfer** — `bytemuck::Pod` bounds ensure GPU data transfers are byte-safe, preventing undefined behavior from `Copy`-only types

## Troubleshooting

**External crate creates its own CUDA context:** Some libraries call `cuCtxCreate` internally. This creates a second context on the same device, which wastes memory and can cause synchronization issues. Look for an initialization function that accepts an existing context. If the library doesn't support context adoption, it will still work — it just won't share the TLSF pool.

**Memory allocated by external crate isn't tracked:** Only memory allocated through `runtime.alloc()` or tensor creation goes through TLSF. If the external crate calls `cudaMalloc` directly, those allocations bypass the pool. Call `runtime.enable_hooks(false)` before initializing the external crate to intercept `cudaMalloc` calls and route them through TLSF.

**Stream ordering issues:** If you see corrupted data, the likely cause is two libraries writing to the same device pointer on different streams without synchronization. Always synchronize a stream after external work before reading the data from Ferrite, or pass the same `stream.raw()` to both.

**Checkpoint not found after restart:** Ensure you're using the same app name in `FerApp::new()`. Checkpoints are stored under `~/.local/share/ferrite/checkpoints/<app-name>/` (Linux) or `~/Library/Application Support/ferrite/checkpoints/<app-name>/` (macOS). You can override the path with `.checkpoint_dir()`.

**mmap_load size mismatch:** The raw binary file must contain exactly `shape.product() * dtype.size_bytes()` bytes with no header. If you get a size mismatch error, check that the file doesn't have a header (e.g. NumPy `.npy` files have a header — use raw `.bin` exports instead).
