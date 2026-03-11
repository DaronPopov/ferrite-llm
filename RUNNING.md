# Running Ferrite

This guide is the shortest path to getting the Ferrite runtime working locally.

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

## Pick A Model

The example guest decides which model to load in:

[examples/wasm/mistral-inference/src/lib.rs](/home/daron/llm_engine/fer_llm/ferrite/examples/wasm/mistral-inference/src/lib.rs)

Examples:

- `mistral-7b-q4`
- `qwen2.5-7b-q4`
- `qwen3-8b-q4`

Change the `load_model(...)` string, then rebuild the guest component.

## Run The Runtime

### CUDA path, fail if GPU is not available

```bash
HF_TOKEN=your_token_here \
FERRITE_REQUIRE_CUDA=1 \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

### Force the `mistralrs` backend

```bash
HF_TOKEN=your_token_here \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

### Qwen3 example

After changing the guest to:

```rust
let model = load_model("qwen3-8b-q4", hf_token.as_deref())?;
```

run:

```bash
HF_TOKEN=your_token_here \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

## Useful Environment Variables

- `HF_TOKEN`: Hugging Face token
- `FERRITE_REQUIRE_CUDA=1`: fail instead of silently falling back to CPU
- `FERRITE_BACKEND=mistralrs`: force the `mistralrs` backend
- `FERRITE_BACKEND=candle`: force the Candle backend
- `FERRITE_MODEL_CACHE=/path/to/models`: override model cache directory

Legacy compatibility:

- `FERRITE_INFERENCE_BACKEND` is still accepted as an alias for `FERRITE_BACKEND`

## Inspect Available Models

```bash
cargo run -p ferrite-cli -- models
```

## Common Failures

### `No such file or directory (os error 2)`

You passed the wrong WASM component path or did not build the component yet.

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
HF_TOKEN=your_token_here FERRITE_REQUIRE_CUDA=1 cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```
