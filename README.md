# ferrite

> **Copyright Daron Popov. All rights reserved.**  \
> This source is viewable for reference only.  \
> No license is granted for use, copying, modification, redistribution, sublicensing, or commercial use without prior written permission.

GPU-first LLM runtime with a WASM guest boundary.

Ferrite splits model execution from guest logic:

- `ferrite-cli`: host runtime binary
- `ferrite-wasm-host`: WIT/WASM bridge
- `ferrite-core`: native inference/session code
- `ferrite-sdk`: guest-side bindings

It now supports two explicit host backends:

- `candle`: Ferrite's custom/native backend
- `mistralrs`: broader GGUF model backend

Both backends can run on CUDA. They are backend implementations, not device modes.
Both backends now support live polling-based streaming through the WASM generation handle.

## Install

The intended bootstrap path is a curl one-liner:

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | bash
```

That default path now builds the full Ferrite engine stack with the `mega`
profile.

Available installer profiles:

- `runtime`: build the smallest supported Ferrite runtime slice
- `platform`: add the `ferrite-os` workspace build
- `full`: add `ferrite-gpu-lang`
- `mega`: build the full Ferrite engine, including `external/ferrite-graphics`

Examples:

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | \
  bash -s -- --profile full
```

```bash
curl -fsSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | \
  bash -s -- --profile mega
```

What the default one-line install now does:

- clones or updates `ferrite` into `~/.local/share/ferrite/src/ferrite`
- refreshes Cargo dependencies
- builds `ferrite-os/lib/libptx_os.so`
- builds the main `ferrite-llm` Rust workspace
- builds the `ferrite-os` Rust workspace
- builds `ferrite-gpu-lang`
- configures, builds, tests, and installs `external/ferrite-graphics`
- installs `ferrite-rt` into `~/.local/bin`
- provisions WASM prerequisites with `ferrite-rt setup`
- builds the sample guest component
- verifies the owned custom CUDA kernel path with `custom-kernel-smoke`
- verifies Ferrite's direct `ug-cuda` path with `ug-cuda-smoke`
- runs `ferrite-rt doctor --profile mega` to validate the installed stack

Additional profile behavior:

- `runtime` skips the larger platform/graphics layers
- `platform` skips `ferrite-gpu-lang` and `ferrite-graphics`
- `full` skips `ferrite-graphics`
- `mega` is the default full-engine path

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
- TLSF allocator is built by default; set `FERRITE_ENABLE_TLSF_ALLOC=0` to skip it

The installer stays in user space. It does not attempt `apt`, `dnf`, or system package changes.
`FERRITE_BUILD_EVERYTHING` is still accepted for compatibility:

- `FERRITE_BUILD_EVERYTHING=0` maps to `--profile runtime`
- `FERRITE_BUILD_EVERYTHING=1` maps to `--profile mega`

## Quick Run

After install:

```bash
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN=your_token_here \
FERRITE_MODEL=qwen3-8b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
ferrite-rt run \
  ~/.local/share/ferrite/src/ferrite/target/wasm32-wasip1/release/mistral_inference.component.wasm
```

For local development from a repo checkout, prefer:

```bash
HF_TOKEN=your_token_here \
FERRITE_MODEL=mistral-7b-q4 \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_BACKEND=mistralrs \
cargo run -p ferrite-cli -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

That keeps the host binary in sync with WIT changes and avoids stale installed binaries.
If you want the CLI TLSF allocator path, add `--features tlsf-alloc` and use the `FERRITE_TLSF_*` variables documented below.

The CLI is now split:

- `ferrite-rt`: lightweight front-end for `doctor`, `info`, `setup`, `cache`, and model listing
- `ferrite-rt-runner`: heavy runtime binary used for actual WASM execution

That keeps admin commands fast and avoids linking the full inference stack just
to inspect or validate the system.

## Doctor

Use the built-in doctor command to validate the local Ferrite install without
changing anything:

```bash
ferrite-rt doctor --profile runtime
```

From a repo checkout:

```bash
cargo run -p ferrite-cli -- doctor --profile full
```

What it checks:

- required tools like `cargo`, `rustup`, `wasm-tools`, and bootstrap download tools
- the `wasm32-wasip1` target
- model cache path validity
- CUDA/runtime availability for the current binary
- repo/install artifacts such as `libptx_os.so` and the sample component
- profile-specific modules for `platform`, `full`, and `mega`

Use `--strict` if you want warnings to fail the command.

## Script Hooks

The runtime now supports opt-in host-boundary script hooks without changing the
default inference flow. Pass a Rhai script that can define any of:

```rhai
fn pre_prompt(prompt) {
    prompt
}

fn post_response(response) {
    response
}

fn post_chunk(chunk) {
    chunk
}

fn post_logits(candidates) {
    candidates
}

fn gpu_logits_program() {
    "
        x = input([1, 1, 1, 16])
        return relu(x)
    "
}
```

Example:

```bash
ferrite-rt run \
  --script-hook examples/script-hooks/pre_prompt_hook.rhai \
  target/wasm32-wasip1/release/mistral_inference.component.wasm
```

Implemented hook behavior:

- `pre_prompt`: rewrites the prompt before generation
- `post_response`: rewrites the final response for blocking generation
- `post_chunk`: rewrites each streamed chunk
- `post_logits`: rewrites the top logits candidate list before sampling on the Candle chat-session path
- `gpu_logits_program`: provides a `ferrite-gpu-lang` JIT program that runs over the top-16 logits values on the Candle chat-session path

Missing hook functions are treated as no-ops. This keeps the integration seam
safe while the deeper `ferrite-gpu-lang` tensor/logit hook path is built out.

Hook safety controls:

- `FERRITE_SCRIPT_HOOK_ON_ERROR=strict|warn`
- `FERRITE_SCRIPT_HOOK_TIMEOUT_MS=<ms>` for a post-execution hook budget check
- `FERRITE_SCRIPT_HOOK_METRICS=1` to emit per-hook timing/error telemetry

The default policy is `strict`, which preserves the current behavior for loaded
hooks. Set `warn` if you want hook failures to degrade to pass-through behavior
instead of aborting generation.

Current backend scope:

- `pre_prompt`, `post_response`, `post_chunk`: apply across the runtime host path
- `post_logits`: currently applies only to the Candle backend path that runs through `ferrite-core::ChatSession`
- `gpu_logits_program`: currently applies only to the Candle backend path and requires CUDA because it runs through `ferrite-gpu-lang`

## Backend Model

Ferrite exposes backend selection explicitly.

- `FERRITE_BACKEND=candle`
- `FERRITE_BACKEND=mistralrs`
- `FERRITE_REQUIRE_CUDA=1` to fail instead of falling back to CPU
- `FERRITE_TLSF_ALLOC=1` to enable Ferrite's TLSF CUDA allocator at runtime when the binary was built with `tlsf-alloc`

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
- whether TLSF allocator support was compiled into the runtime

To build and run the runtime with TLSF allocator support locally:

```bash
LD_LIBRARY_PATH=ferrite-os/lib:$LD_LIBRARY_PATH \
cargo run -p ferrite-cli --features tlsf-alloc -- info
LD_LIBRARY_PATH=ferrite-os/lib:$LD_LIBRARY_PATH \
HF_TOKEN=your_token_here \
FERRITE_TLSF_ALLOC=1 \
FERRITE_TLSF_POOL_FRACTION=0.97 \
FERRITE_TLSF_RESERVE_VRAM_MB=128 \
FERRITE_BACKEND=mistralrs \
FERRITE_REQUIRE_CUDA=1 \
cargo run -p ferrite-cli --features tlsf-alloc -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

For the `ferrite-cli --features tlsf-alloc` path, use the `FERRITE_TLSF_*` variables above. The daemon-only `FERRITE_INFERENCE_POOL_FRACTION` setting does not affect this CLI allocator path.

This now patches both the native Ferrite/Candle CUDA path and the `mistralrs` CUDA stack. For installed binaries, build with `FERRITE_ENABLE_TLSF_ALLOC=1` during install. The installer copies `libptx_os.so` into `~/.local/lib`; add that directory to `LD_LIBRARY_PATH` before running the TLSF-enabled binary.

If you change native PTX-OS code under `ferrite-os/native/core`, rebuild the shared runtime library before rerunning the CLI:

```bash
cd ferrite-os
make lib/libptx_os.so
```

That target now tracks `.cu`, `.c`, `.h`, `.hpp`, and `.inl` inputs, so native allocator and runtime edits correctly invalidate `libptx_os.so`.

## Add A Model

For many GGUF models, you do not need custom kernels first.

Typical process:

1. Add a `ModelSpec` entry in [catalog.rs](crates/ferrite-core/src/registry/catalog.rs)
2. Choose the backend:
   `candle` if you want to extend Ferrite's native path
   `mistralrs` if you want broad GGUF support quickly
3. Rebuild and run

Current examples include:

- `mistral-7b-q4`
- `qwen2.5-7b-q4`
- `qwen3-8b-q4`

Set the model at runtime with `FERRITE_MODEL=<registry-name>`. Rebuilding the guest is not required just to switch models.

For custom architectures or model-specific fused operators, Ferrite also owns a custom CUDA kernel path in [crates/ferrite-core](crates/ferrite-core). Kernel sources live under [kernels](crates/ferrite-core/kernels), and the owned smoke example is [custom-kernel-smoke.rs](crates/ferrite-core/examples/custom-kernel-smoke.rs).
The owned attention benchmark is [custom-attention-bench.rs](crates/ferrite-core/examples/custom-attention-bench.rs).

## Streaming

True live token streaming across the WASM boundary now uses a polling handle in [ferrite.wit](wit/ferrite.wit):

- `start-generate-stream`
- `next-chunk`

The `mistralrs` backend is the reference live-streaming path.

## Docs

- [docs/README.md](docs/README.md): top-level docs index
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): diagrams and system explanation
- [docs/RUNNING.md](docs/RUNNING.md): runbook and commands
- [docs/ENGINE_IDENTITY.md](docs/ENGINE_IDENTITY.md): monorepo engine identity
- [docs/INSTALLER_ARCHITECTURE.md](docs/INSTALLER_ARCHITECTURE.md): installer design
- [crates/ferrite-core/README.md](crates/ferrite-core/README.md)
- [crates/ferrite-wasm-host/README.md](crates/ferrite-wasm-host/README.md)
- [crates/ferrite-sdk/README.md](crates/ferrite-sdk/README.md)
- [crates/ferrite-cli/README.md](crates/ferrite-cli/README.md)

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

- Canonical repo: https://github.com/DaronPopov/ferrite
