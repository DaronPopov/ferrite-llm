# ferrite-sdk

Guest SDK for writing portable WASM AI modules.

## Features

- **WIT Bindings** - Pre-generated interface bindings
- **Simple API** - Load models, generate text
- **Portable** - Runs on any ferrite-compatible host

## Quick Start

```toml
[package]
name = "my-ai-module"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wit-bindgen = "0.37"
```

```rust
wit_bindgen::generate!({
    path: "path/to/wit",
    world: "ferrite-guest",
});

use ferrite::inference::inference::{load_model, GenerationConfig};

struct MyAI;

impl Guest for MyAI {
    fn run() -> Result<(), String> {
        // Load a model (downloads automatically if needed)
        let model = load_model("mistral-7b-q4", None)?;

        // Configure generation
        let config = GenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 200,
            seed: Some(42),
        };

        // Generate text
        let response = model.generate("What is WebAssembly?", config)?;
        println!("{}", response);

        Ok(())
    }
}

export!(MyAI);
```

## Build & Run

```bash
# Build WASM module
cargo build --target wasm32-wasip1 --release

# Componentize
wasm-tools component embed wit target/wasm32-wasip1/release/my_ai.wasm \
  -o my_ai.embed.wasm
wasm-tools component new my_ai.embed.wasm \
  --adapt wasi_snapshot_preview1.reactor.wasm \
  -o my_ai.component.wasm

# Run
ferrite-rt run my_ai.component.wasm
```

## API Reference

### `load_model(name, auth_token) -> Result<Model, String>`

Load a model by name. Supported models:
- `mistral-7b-q4` - Mistral 7B quantized (4-bit)

### `Model::generate(prompt, config) -> Result<String, String>`

Generate text from a prompt.

### `GenerationConfig`

```rust
GenerationConfig {
    temperature: f32,  // Randomness (0.0-1.0)
    top_p: f32,        // Nucleus sampling
    top_k: u32,        // Top-k sampling
    max_tokens: u32,   // Maximum tokens to generate
    seed: Option<u64>, // Reproducibility seed
}
```

## License

MIT
