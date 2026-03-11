# ferrite-llm

GPU-first LLM runtime with a WASM guest boundary.

Ferrite splits model execution from guest logic:

- `ferrite-cli`: host runtime binary
- `ferrite-wasm-host`: WIT/WASM bridge
- `ferrite-core`: native inference/session code
- `ferrite-sdk`: guest-side bindings

It now supports two explicit host backends:

- `candle`: Ferrite's custom/native backend
- `mistralrs`: broader GGUF model backend with true polling-based streaming

Both backends can run on CUDA. They are backend implementations, not device modes.

## Install

The intended bootstrap path is a curl one-liner:

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/ferrite-llm/main/install.sh | bash
```

What it does:

- clones or updates `ferrite-llm` into `~/.local/share/ferrite-llm/src/ferrite-llm`
- refreshes Cargo dependencies
- installs `ferrite-rt` into `~/.local/bin`
- provisions WASM prerequisites with `ferrite-rt setup`
- builds the sample guest component
- verifies the owned custom CUDA kernel path with `custom-kernel-smoke`

Requirements:

- `curl` or `wget`
- CUDA installed already if you want GPU inference

What it bootstraps automatically in user space:

- Rust via `rustup` if Rust is missing
- repo download via `git` if available, otherwise a source tarball
- `ferrite-rt`
- WASM prerequisites and sample component build

Installer notes:

- on `aarch64` Jetsons, the installer now detects the platform and prefers `/usr/local/cuda-arm64` when present
- the install path is intended to work on desktop Linux and Jetson-class ARM Linux with CUDA already installed

The installer stays in user space. It does not attempt `apt`, `dnf`, or system package changes.

## Quick Run

After install:

```bash
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN=your_token_here \
FERRITE_MODEL=qwen3-8b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
ferrite-rt run \
  ~/.local/share/ferrite-llm/src/ferrite-llm/target/wasm32-wasip1/release/mistral_inference.component.wasm
```

For local development from a repo checkout, prefer:

```bash
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

That keeps the host binary in sync with WIT changes and avoids stale installed binaries.

## Docker

Ferrite can also run as a GPU container. Build the image from the repo root:

```bash
docker build -t ferrite-llm .
```

Run it with NVIDIA GPU passthrough:

```bash
docker run --rm -it --gpus all \
  -e HF_TOKEN=your_token_here \
  -e FERRITE_MODEL=qwen2.5-7b-q4 \
  -e FERRITE_REQUIRE_CUDA=1 \
  -e FERRITE_BACKEND=mistralrs \
  -v ferrite-models:/models \
  ferrite-llm
```

The image entrypoint is `ferrite-rt`, and the default command runs the bundled sample component at `/opt/ferrite/mistral_inference.component.wasm`.

## Backend Model

Ferrite exposes backend selection explicitly.

- `FERRITE_BACKEND=candle`
- `FERRITE_BACKEND=mistralrs`
- `FERRITE_REQUIRE_CUDA=1` to fail instead of falling back to CPU

Legacy compatibility:

- `FERRITE_INFERENCE_BACKEND` is still accepted as an alias for `FERRITE_BACKEND`

To inspect the runtime policy before loading a model:

```bash
ferrite-rt info
```

That reports:

- CUDA availability
- backend policy
- whether CUDA is required

## Add A Model

For many GGUF models, you do not need custom kernels first.

Typical process:

1. Add a `ModelSpec` entry in [catalog.rs](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/src/registry/catalog.rs)
2. Choose the backend:
   `candle` if you want to extend Ferrite's native path
   `mistralrs` if you want broad GGUF support quickly
3. Rebuild and run

Current examples include:

- `mistral-7b-q4`
- `qwen2.5-7b-q4`
- `qwen3-8b-q4`

Set the model at runtime with `FERRITE_MODEL=<registry-name>`. Rebuilding the guest is not required just to switch models.

For custom architectures or model-specific fused operators, Ferrite also owns a custom CUDA kernel path in [crates/ferrite-core](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core). Kernel sources live under [kernels](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/kernels), and the owned smoke example is [custom-kernel-smoke.rs](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/examples/custom-kernel-smoke.rs).
The owned attention benchmark is [custom-attention-bench.rs](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/examples/custom-attention-bench.rs).

## Streaming

True live token streaming across the WASM boundary now uses a polling handle in [ferrite.wit](/home/daron/llm_engine/fer_llm/ferrite/wit/ferrite.wit):

- `start-generate-stream`
- `next-chunk`

The `mistralrs` backend is the reference live-streaming path.

## Docs

- [RUNNING.md](/home/daron/llm_engine/fer_llm/ferrite/RUNNING.md): runbook and commands
- [crates/ferrite-core/README.md](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/README.md)
- [crates/ferrite-wasm-host/README.md](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-wasm-host/README.md)
- [crates/ferrite-sdk/README.md](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-sdk/README.md)
- [crates/ferrite-cli/README.md](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-cli/README.md)

## Development

Manual workspace build:

```bash
cargo build --workspace
```

Manual guest rebuild:

```bash
cargo build -p mistral-inference --target wasm32-wasip1 --release
wasm-tools component embed wit \
  target/wasm32-wasip1/release/mistral_inference.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
wasm-tools component new \
  target/wasm32-wasip1/release/mistral_inference.embed.wasm \
  --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.component.wasm
```

## Repository

- Canonical repo: https://github.com/DaronPopov/ferrite-llm
