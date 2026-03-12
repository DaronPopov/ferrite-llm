# ferrite-cli

Production WASM runtime for neural inference.

## Installation

```bash
cargo install --path . && ferrite-rt setup
# or
cargo install ferrite-cli && ferrite-rt setup
```

For the full repo bootstrap flow, use the top-level `install.sh`. It now builds the native `libptx_os.so`, the main Rust workspace, the `ferrite-os` workspace, installs `ferrite-rt`, and builds the sample WASM component by default.

## Commands

### `ferrite-rt run`

Execute a WASM module:

```bash
ferrite-rt run module.wasm [OPTIONS]

Options:
  -v, --verbose...              Logging level (-v info, -vv debug, -vvv trace)
  --model-cache <PATH>          Model cache directory [default: ./models]
  --hf-token <TOKEN>            HuggingFace authentication token
  --metrics                     Show performance metrics after execution
```

Examples:
```bash
# Basic run
ferrite-rt run chatbot.wasm

# With metrics and verbose logging
ferrite-rt run chatbot.wasm --metrics -v

# Custom cache and auth
ferrite-rt run chatbot.wasm --model-cache ~/.cache/ferrite --hf-token $HF_TOKEN
```

### `ferrite-rt models`

List downloaded models:

```bash
ferrite-rt models [OPTIONS]

Options:
  -d, --detailed    Show detailed model information
```

### `ferrite-rt cache`

Manage model cache:

```bash
ferrite-rt cache [OPTIONS]

Options:
  --stats    Show cache statistics
  --clear    Clear local cache
```

### `ferrite-rt info`

Show system information:

```bash
ferrite-rt info
```

### `ferrite-rt setup`

Install the local WASM guest build prerequisites:

```bash
ferrite-rt setup

Options:
  --check                      Validate prerequisites without installing
  --skip-target                Skip `rustup target add wasm32-wasip1`
  --skip-wasm-tools            Skip `cargo install wasm-tools`
```

Output:
```
Ferrite Runtime - Neural Inference OS
   Version: 0.3.0
   Homepage: https://github.com/sperabality/ferrite

Components:
   - ferrite-core: Pure inference engine
   - ferrite-wasm-host: WASM orchestration
   - ferrite-sdk: Guest SDK for WASM modules

System:
   - OS: linux
   - Arch: x86_64
   - CUDA: Available / Not compiled

Model Support:
   - Mistral 7B (Q4 quantized)
   - Qwen2 0.5B
   - Custom GGUF models
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace authentication token |
| `FERRITE_MODEL_CACHE` | Default model cache directory |
| `FERRITE_MODEL` | Model registry entry to load at runtime |
| `FERRITE_BACKEND` | Backend selection such as `mistralrs` or `candle` |
| `FERRITE_REQUIRE_CUDA=1` | Fail instead of falling back to CPU |
| `FERRITE_TLSF_ALLOC=1` | Enable Ferrite TLSF allocator hooks when built with `--features tlsf-alloc` |
| `FERRITE_TLSF_POOL_FRACTION` | TLSF pool fraction for the CLI allocator path |
| `FERRITE_TLSF_RESERVE_VRAM_MB` | VRAM headroom reserved outside the TLSF pool |
| `FERRITE_TLSF_VERBOSE=1` | Re-enable TLSF attach and allocator health logging |

## TLSF Run Example

For local CUDA inference with the TLSF allocator:

```bash
LD_LIBRARY_PATH=/home/daron/llm_engine/fer_llm/ferrite/ferrite-os/lib:$LD_LIBRARY_PATH \
HF_TOKEN=your_token_here \
FERRITE_MODEL=mistral-7b-q4 \
FERRITE_BACKEND=mistralrs \
FERRITE_REQUIRE_CUDA=1 \
FERRITE_TLSF_ALLOC=1 \
FERRITE_TLSF_POOL_FRACTION=0.97 \
FERRITE_TLSF_RESERVE_VRAM_MB=128 \
cargo run -p ferrite-cli --features tlsf-alloc -- \
  run target/wasm32-wasip1/release/mistral_inference.component.wasm
```

The CLI TLSF path is quiet by default. Routine pool-health warnings and attach banners are suppressed unless `FERRITE_TLSF_VERBOSE=1` is set.

If you edit native PTX-OS code under `ferrite-os/native/core`, rebuild the shared runtime library before rerunning the CLI:

```bash
cd /home/daron/llm_engine/fer_llm/ferrite/ferrite-os
make lib/libptx_os.so
```

## License

MIT
