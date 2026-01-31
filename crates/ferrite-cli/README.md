# ferrite-cli

Production WASM runtime for neural inference.

## Installation

```bash
cargo install --path .
# or
cargo install ferrite-cli
```

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

## License

MIT
