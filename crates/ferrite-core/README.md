# ferrite-core

Pure Rust neural inference engine built on Candle.

## Features

- **Quantized Models** - GGUF 4-bit/8-bit support (run 7B models in 4GB VRAM)
- **Chat Templates** - Mistral, Llama, ChatML, Gemma formats
- **KV-Cache** - Efficient multi-turn conversations
- **Streaming** - Token-by-token generation
- **GPU Support** - CUDA acceleration (optional)

## Usage

```toml
[dependencies]
ferrite-core = "0.3"
```

```rust
use ferrite_core::{ChatSession, GenerationConfig, Tokenizer};
use ferrite_core::models::quantized::QuantizedMistral;
use candle_core::Device;

// Load model and tokenizer
let device = Device::cuda_if_available(0)?;
let model = QuantizedMistral::new("model.gguf", device)?;
let tokenizer = Tokenizer::from_dir("./tokenizer")?;

// Configure generation
let config = GenerationConfig {
    max_tokens: 200,
    temperature: 0.8,
    top_p: 0.9,
    top_k: 50,
    ..Default::default()
};

// Create chat session with KV-cache
let mut session = ChatSession::new(model, tokenizer, None, config)?;

// Multi-turn conversation
let response1 = session.user_turn("What is Rust?")?;
let response2 = session.user_turn("Tell me more about ownership.")?;
```

## Modules

| Module | Purpose |
|--------|---------|
| `tokenizer` | HuggingFace tokenizers + chat templates |
| `generation` | Generation config and statistics |
| `sampling` | Top-p, top-k, temperature, min-p |
| `chat_session` | Multi-turn with KV-cache |
| `models` | Model implementations |

## Optional Features

```toml
[dependencies]
ferrite-core = { version = "0.3", features = ["cuda"] }
```

## Custom CUDA Kernels

Ferrite owns a custom-kernel path in this crate. This is the place to wire model-specific CUDA code that should live with the inference engine.

- Kernel sources live in `crates/ferrite-core/kernels/`
- Kernel build settings live in `crates/ferrite-core/kernels/kernel_config.toml`
- The build script compiles enabled kernels during `cargo build --features cuda`
- A minimal end-to-end smoke example lives at `crates/ferrite-core/examples/custom-kernel-smoke.rs`

Run the smoke test with:

```bash
cargo run -p ferrite-core --features cuda --example custom-kernel-smoke
```

Use this path when a model needs custom CUDA operators beyond the standard Candle or `mistralrs` backends.

## License

MIT
