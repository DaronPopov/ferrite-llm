# Running Ferrite

This guide is the shortest path to getting the Ferrite runtime working locally.

Use one of these two flows:

- repo workflow: run from a cloned checkout with `cargo run -p ferrite-cli -- ...`
- installed workflow: run the installed `ferrite-rt` binary after `install.sh`

## What This Runs

Ferrite has two layers:

- `ferrite-rt`: the host runtime binary
- a guest WASM component such as `mistral-inference`

The host loads models on the machine and exposes inference to the guest through the WIT interface.

## Prerequisites

- Rust toolchain installed
- CUDA installed and working if you want GPU inference
- `wasm-tools` installed
- `wasm32-wasip1` Rust target installed
- optional: `HF_TOKEN` for gated Hugging Face models

You can bootstrap the WASM prerequisites with:

```bash
cargo run -p ferrite-cli -- setup
```

By default this also runs `cargo update`. Use `--skip-deps-update` if you want to keep the current lockfile resolution.

## Build The Guest Component

From the repo root:

```bash
cd /home/daron/llm_engine/fer_llm/ferrite
```

Build the example guest:

```bash
cargo build -p mistral-inference --target wasm32-wasip1 --release
```

Convert it into a component:

```bash
wasm-tools component embed wit \
  target/wasm32-wasip1/release/mistral_inference.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.embed.wasm

wasm-tools component new \
  target/wasm32-wasip1/release/mistral_inference.embed.wasm \
  --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.component.wasm
```

## Check Runtime Policy

Before loading a model, inspect the resolved runtime policy:

```bash
cargo run -p ferrite-cli -- info
```

That reports CUDA availability, backend policy, and whether CUDA is required.

## Pick A Model

The example guest reads the model name from `FERRITE_MODEL` at runtime:

[examples/wasm/mistral-inference/src/lib.rs](/home/daron/llm_engine/fer_llm/ferrite/examples/wasm/mistral-inference/src/lib.rs)

Examples:

- `mistral-7b-q4`
- `qwen2.5-7b-q4`
- `qwen3-8b-q4`

## Run The Runtime

### Repo workflow

Use this while developing locally. It avoids stale installed binaries when the WIT interface changes.

```bash
HF_TOKEN=your_token_here \
FERRITE_MODEL=mistral-7b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

### Installed workflow

Use this after running the one-line installer:

```bash
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN=your_token_here \
FERRITE_MODEL=mistral-7b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
ferrite-rt run \
  "$HOME/.local/share/ferrite-llm/src/ferrite-llm/target/wasm32-wasip1/release/mistral_inference.component.wasm"
```

The installer also verifies Ferrite's owned CUDA kernel path by running:

```bash
cargo run -p ferrite-core --features cuda --example custom-kernel-smoke
```

### Qwen3 example

Set the model name at runtime:

```bash
HF_TOKEN=your_token_here \
FERRITE_MODEL=qwen3-8b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

## Useful Environment Variables

- `HF_TOKEN`: Hugging Face token
- `FERRITE_MODEL=qwen3-8b-q4`: select the model registry entry to load
- `FERRITE_REQUIRE_CUDA=1`: fail instead of silently falling back to CPU
- `FERRITE_BACKEND=mistralrs`: force the `mistralrs` backend
- `FERRITE_BACKEND=candle`: force the Candle backend
- `FERRITE_MODEL_CACHE=/path/to/models`: override model cache directory

Legacy compatibility:

- `FERRITE_INFERENCE_BACKEND` is still accepted as an alias for `FERRITE_BACKEND`

## Inspect Available Models

Repo workflow:

```bash
cargo run -p ferrite-cli -- models
```

Installed workflow:

```bash
ferrite-rt models
```

## Custom CUDA Kernels

If a model needs custom CUDA operators, Ferrite's owned path is in:

- [kernels](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/kernels)
- [kernel_config.toml](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/kernels/kernel_config.toml)
- [custom-kernel-smoke.rs](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/examples/custom-kernel-smoke.rs)

Smoke-test the inline CUDA path with:

```bash
cargo run -p ferrite-core --features cuda --example custom-kernel-smoke
```

That verifies Rust-side NVRTC compilation, PTX loading, and kernel launch through Ferrite's `cudarc` path.

## Common Failures

### `No such file or directory (os error 2)`

You passed the wrong WASM component path or did not build the component yet.

For a repo checkout, the expected component path is:

```bash
target/wasm32-wasip1/release/mistral_inference.component.wasm
```

For an installed setup, the expected component path is:

```bash
$HOME/.local/share/ferrite-llm/src/ferrite-llm/target/wasm32-wasip1/release/mistral_inference.component.wasm
```

### `resource implementation is missing`

Your guest component and host runtime are out of sync.

Use `cargo run -p ferrite-cli -- ...` in a repo checkout, or reinstall the host binary with:

```bash
cargo install --path crates/ferrite-cli --force
```

### CPU fallback

If CUDA is missing or the runtime was not built with GPU support, Candle can fall back to CPU. Set:

```bash
FERRITE_REQUIRE_CUDA=1
```

to turn that into a hard error.

### Hugging Face auth failures

Set:

```bash
export HF_TOKEN=your_token_here
```

before running the runtime.

## Fast Rebuild Loop

After editing the guest:

```bash
cargo build -p mistral-inference --target wasm32-wasip1 --release
wasm-tools component embed wit \
  target/wasm32-wasip1/release/mistral_inference.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
wasm-tools component new \
  target/wasm32-wasip1/release/mistral_inference.embed.wasm \
  --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.component.wasm
HF_TOKEN=your_token_here FERRITE_MODEL=qwen3-8b-q4 FERRITE_REQUIRE_CUDA=1 cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```
