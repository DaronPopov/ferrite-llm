# ferrite-gpu-lang

Rust-first GPU language runtime on Ferrite OS.

## What this crate provides

- A small graph DSL (`Program`) with compile-time shape checks.
- Execution on `ptx-runtime` with all allocations through TLSF.
- Candle kernel-backed ops (`relu`, `tanh`, `sigmoid`, `add`, `mul`).
- Optional Torch bridge (`--features torch`) for scriptable PyTorch checks on your custom stack.
- Optional CV scripting graph (`cv::CvProgram`) for Torch-backed CV ops:
  - `conv2d`, `upsample_nearest2d`, `concat`
  - `yolo_decode`, `nms`
- Explicit CPU/GPU execution contexts:
  - `cpu(|c| { ... })`
  - `gpu(0, |g| { ... })`
  - typed transfer traits: `ToGpu` / `ToCpu`

## Scripting Examples

```bash
cd ferrite-gpu-lang
export LD_LIBRARY_PATH=../ferrite-os/lib:$LD_LIBRARY_PATH

cargo run --release --example script_runtime
cargo run --release --example script_handoff
cargo run --release --features torch --example script_cv_depth
cargo run --release --features torch --example script_cv_detect
```

## JIT Tile Annotations

The JIT DSL supports nested tiles and explicit scheduling annotations:

```text
x = input([1024])
tile y over (x) with (
  tile_m=128,
  tile_n=64,
  tile_k=32,
  unroll=4,
  pipeline_stages=2,
  precision=bf16,
  quant=nf4,
  dist=shard,
  replicas=2,
  mesh_axis=0,
  layout=blocked_32x8,
  accum=bf16,
  collective=all_reduce
):
  y = x * 2.0 + 1.0
end
return y
```

Supported enums:

- `precision`: `f32|f16|bf16`
- `quant`: `none|int8|nf4`
- `dist`: `none|replicate|shard|reduce_scatter`
- `layout`: `row_major|col_major|blocked_32x8|blocked_64x4`
- `accum`: `f32|bf16`
- `collective`: `none|all_reduce|reduce_scatter|all_gather`

## Symbolic Shape Contracts

Input shapes can use compile-time symbols resolved by named arguments:

```text
x = input([B, T, H], B=2, T=128, H=256) where B > 0, H >= 64
y = relu(x)
return y
```

Rules:

- Symbols in shape literals must be bound with positive integer values.
- Unbound shape symbols are compile-time errors.
- Extra/unused symbol bindings are compile-time errors.
- `where` clauses are compile-time boolean constraints on input symbols.

## CPU -> GPU -> CPU handoff

```rust
use ferrite_gpu_lang::{cpu, gpu, CpuTensor, ToCpu, ToGpu};

let cpu_in = cpu(|_| CpuTensor::new(vec![1,1,1,4096], data))?;
let cpu_out = gpu(0, |g| {
    let gpu_in = cpu_in.to_gpu(g)?;
    let gpu_out = gpu_in.relu(g)?; // GPU stage
    gpu_out.to_cpu()               // back to CPU
})?;
```

## Runtime Script Runner

```bash
./scripts/run_foundation_stack.sh
```

This runs only runtime scripting examples from `ferrite-gpu-lang`.
