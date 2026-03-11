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

Requirements:

- `git`
- `cargo`
- `rustup`
- CUDA installed already if you want GPU inference

The installer stays in user space. It does not attempt `apt` or system package changes.

## Quick Run

After install:

```bash
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN=your_token_here \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
ferrite-rt run \
  ~/.local/share/ferrite-llm/src/ferrite-llm/target/wasm32-wasip1/release/mistral_inference.component.wasm
```

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
