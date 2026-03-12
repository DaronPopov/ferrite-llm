# Running Ferrite

This guide is the shortest engine-level runbook for Ferrite.

For the high-level explanation of what the repo is, read
[Architecture](ARCHITECTURE.md) first.

## CLI Shape

Ferrite now has a split CLI:

- `ferrite-rt`: lightweight front-end for admin, validation, setup, cache, and model listing
- `ferrite-rt-runner`: heavy runtime execution binary

`ferrite-rt run ...` dispatches into the runner when needed.

## Installer Profiles

- `runtime`: smallest supported runtime slice
- `platform`: runtime + `ferrite-os`
- `full`: platform + `ferrite-gpu-lang`
- `mega`: full engine + `ferrite-graphics`

The plain one-line installer defaults to `mega`.

## Quick Commands

### Validate the engine

```bash
cargo run -p ferrite-cli -- doctor --profile mega
```

Installed workflow:

```bash
ferrite-rt doctor --profile mega
```

### Check runtime policy

```bash
cargo run -p ferrite-cli -- info
```

### Check WASM prerequisites

```bash
cargo run -p ferrite-cli -- setup --check
```

### List built-in models

```bash
cargo run -p ferrite-cli -- models
```

## Repo Workflow

Build the guest:

```bash
cargo build -p mistral-inference --target wasm32-wasip1 --release
wasm-tools component new \
  target/wasm32-wasip1/release/mistral_inference.wasm \
  --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.component.wasm
```

Run the runtime:

```bash
HF_TOKEN=your_token_here \
FERRITE_MODEL=qwen3-8b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

## Installed Workflow

```bash
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN=your_token_here \
FERRITE_MODEL=qwen3-8b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
ferrite-rt run "$HOME/.local/share/ferrite/src/ferrite/target/wasm32-wasip1/release/mistral_inference.component.wasm"
```

## Hook Safety

Script hooks can be controlled with:

- `FERRITE_SCRIPT_HOOK_ON_ERROR=strict|warn`
- `FERRITE_SCRIPT_HOOK_TIMEOUT_MS=<ms>`
- `FERRITE_SCRIPT_HOOK_METRICS=1`

`strict` preserves the existing loaded-hook behavior.

## Useful Variables

- `HF_TOKEN`
- `FERRITE_MODEL`
- `FERRITE_BACKEND`
- `FERRITE_REQUIRE_CUDA=1`
- `FERRITE_MODEL_CACHE=/path/to/models`
- `FERRITE_TLSF_ALLOC=1`

## Common Failure Shapes

### Missing component path

Expected repo path:

```bash
target/wasm32-wasip1/release/mistral_inference.component.wasm
```

Expected installed path:

```bash
$HOME/.local/share/ferrite-llm/src/ferrite-llm/target/wasm32-wasip1/release/mistral_inference.component.wasm
```

### Host/guest mismatch

Use the repo workflow or reinstall the runtime binaries.

### CUDA fallback

Set:

```bash
FERRITE_REQUIRE_CUDA=1
```

to turn CPU fallback into a hard failure.
