# Ferrite Runtime

WASM-based neural inference runtime - boot neural inference as a programmable OS layer.

## Architecture

Ferrite Runtime is a host environment that executes WebAssembly Component Model modules with neural inference capabilities as first-class primitives.

```
┌─────────────────────────────────────┐
│  WASM Module (guest)                │
│  - Written in Rust                  │
│  - Uses ferrite-guest SDK           │
└───────────┬─────────────────────────┘
            │ WIT Interface
┌───────────▼─────────────────────────┐
│  Ferrite Runtime (host)             │
│  - Wasmtime engine                  │
│  - Ferrite inference backend        │
│  - GPU acceleration (CUDA/Metal)    │
└───────────┬─────────────────────────┘
            │
┌───────────▼─────────────────────────┐
│  Operating System (Linux/Windows)   │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Build the Runtime

```bash
cargo build --release -p ferrite-runtime
```

The runtime binary will be at `target/release/ferrite-rt`.

### 2. Build a WASM Module

```bash
cd wasm-examples/hello-inference
cargo build --target wasm32-wasip1 --release
```

### 3. Run the Module

```bash
../../target/release/ferrite-rt \
    ../../target/wasm32-wasip1/release/hello_inference.wasm
```

## Writing WASM Modules

WASM modules use the `ferrite-guest` SDK:

```rust
use ferrite_guest::*;

struct MyModule;

impl Guest for MyModule {
    fn run() -> Result<(), String> {
        // Load a model
        let model = Model::load("qwen2-0.5b")?;

        // Configure generation
        let config = GenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 100,
            seed: Some(42),
        };

        // Generate text
        let response = model.generate("Hello, world!", &config)?;
        println!("{}", response);

        Ok(())
    }
}

export!(MyModule);
```

## Available Models

The runtime supports these models (prototype uses mock inference):

- `qwen2-0.5b` - Qwen2 0.5B Instruct
- `tinyllama` - TinyLlama 1.1B Chat
- `mistral-7b` - Mistral 7B Instruct (FP16)
- `mistral-7b-q4` - Mistral 7B Instruct (4-bit quantized)
- `gemma-2b` - Gemma 2B Instruct
- `phi-2` - Phi-2 2.7B

## Runtime Options

```
ferrite-rt [OPTIONS] <MODULE>

OPTIONS:
    -v, --verbose           Enable verbose logging
    --model-cache <DIR>     Model cache directory (default: ./models)
```

## Development Status

**Current (v0.1.0 - Prototype)**:
- ✅ WASM Component Model runtime
- ✅ WIT interface definitions
- ✅ Guest SDK (Rust only)
- ✅ Mock inference (demonstrates architecture)
- ✅ Example modules

**Coming Soon**:
- 🚧 Full Candle integration (real inference)
- 🚧 Streaming interface via WASI streams
- 🚧 Chat session management
- 🚧 Multi-model support
- 🚧 KV-cache resource management
- 🚧 Custom CUDA kernel support

## How It's Different from Containers

| Aspect | Docker/Containers | Ferrite Runtime |
|--------|-------------------|-----------------|
| **Startup** | 1-5 seconds | <50ms |
| **Size** | ~100MB+ | ~5MB WASM |
| **Isolation** | Full OS | Memory sandbox only |
| **GPU** | Passthrough | Native access |
| **Portability** | Needs daemon | Single binary |

Ferrite Runtime is a **host for WASM programs**, not a container. WASM modules are portable bytecode that calls native ferrite functions through a well-defined interface.

## License

MIT
