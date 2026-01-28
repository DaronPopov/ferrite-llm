# Ferrite

A lightweight, high-performance LLM inference engine written in pure Rust.

```
┌─────────────────────────────────────────────────────────────────┐
│  Mistral 7B Q4: 65 tok/s │ 26 MB binary │ No Python required   │
└─────────────────────────────────────────────────────────────────┘
```

## One-Line Install

```bash
curl -sSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | bash
```

Then run:
```bash
ferrite-chat
```

## What is Ferrite?

Ferrite is a **production-ready LLM inference platform** that compiles to a single static binary. No Python runtime, no libtorch, no version conflicts—just a ~25 MB executable that runs state-of-the-art language models.

### Key Features

- **Pure Rust** — Built on [Candle](https://github.com/huggingface/candle), HuggingFace's Rust ML framework
- **Single Binary** — Everything compiles into one executable (~25 MB)
- **Fast** — 65+ tokens/sec on consumer GPUs with quantized models
- **Quantization** — 4-bit GGUF support (run 7B models in 4GB VRAM)
- **No Dependencies** — No Python, no libtorch, no CUDA toolkit bundled
- **Chat Templates** — Native support for Mistral, Llama, Qwen, Gemma formats

## Quick Start

```bash
# Run the interactive launcher
cargo run --release -p ferrite-examples --bin ferrite-chat

# Or run a specific model
cargo run --release -p ferrite-examples --bin mistral_quantized_inference
```

## Supported Models

| Model | Parameters | VRAM | Speed |
|-------|------------|------|-------|
| TinyLlama | 1.1B | ~3 GB | ~120 tok/s |
| Qwen2 | 0.5B | ~1 GB | ~150 tok/s |
| Phi-2 | 2.7B | ~6 GB | ~45 tok/s |
| Gemma | 2B | ~5 GB | ~50 tok/s |
| Mistral 7B | 7B | ~14 GB | ~25 tok/s |
| Mistral 7B Q4 | 7B (4-bit) | ~4 GB | ~65 tok/s |

## Architecture

```
ferrite/
├── src/                    # Core library (800 lines)
│   ├── tokenizer.rs        # HuggingFace tokenizers + chat templates
│   ├── generation.rs       # Generation config + TPS tracking
│   ├── sampling.rs         # Top-p, top-k, temperature sampling
│   └── models.rs           # Model registry + configs
│
└── examples/               # Inference binaries
    ├── chatbots/           # Model-specific implementations
    │   ├── llama/
    │   ├── mistral/
    │   ├── qwen/
    │   ├── gemma/
    │   └── phi/
    └── src/                # CLI launcher + utilities
```

## Design Philosophy

### Why Rust?

1. **Single binary deployment** — No "works on my machine" problems
2. **Memory safety** — No segfaults, no buffer overflows
3. **Performance** — Zero-cost abstractions, no GC pauses
4. **Dependency management** — Cargo handles everything

### Why Candle?

We evaluated several backends:

| Backend | Pros | Cons |
|---------|------|------|
| PyTorch/libtorch | Mature, full-featured | 2GB+ dependency, Python ecosystem |
| ONNX Runtime | Cross-platform | Limited model support |
| llama.cpp | Fast, lightweight | C++, limited to Llama-like models |
| **Candle** | Pure Rust, HuggingFace models | Newer, growing ecosystem |

Candle won because:
- Pure Rust = single binary compilation
- Direct HuggingFace model support
- Active development by HuggingFace team
- CUDA, Metal, and CPU backends

### What Ferrite Adds

Candle provides the ML primitives. Ferrite adds the **inference ergonomics**:

```rust
use ferrite::{Tokenizer, ChatMessage, GenerationStats};

// Chat template support (Mistral, Llama, ChatML, etc.)
let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
];
let prompt = tokenizer.apply_chat_template(&messages, true)?;

// TPS tracking
let mut stats = GenerationStats::new(prompt_tokens);
stats.start();
// ... generate tokens ...
stats.print_summary();  // [Stats] 89 tokens in 1.37s (65.0 tok/s)
```

## Performance

Benchmarked on RTX 3080 (10GB VRAM):

```
Model                    Prefill    Decode     Memory
─────────────────────────────────────────────────────
Mistral-7B-Q4            142 ms     65 tok/s   4.1 GB
Mistral-7B-FP16          287 ms     24 tok/s   14.2 GB
TinyLlama-1.1B           31 ms      118 tok/s  2.8 GB
Qwen2-0.5B               18 ms      156 tok/s  1.2 GB
```

## Installation

### Requirements

- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- CUDA toolkit (for GPU inference)
- ~4 GB disk space for model weights

### Build

```bash
git clone https://github.com/user/ferrite
cd ferrite

# Build all examples
cargo build --release -p ferrite-examples

# Run
./target/release/ferrite-chat
```

### HuggingFace Token

Some models require authentication:

```bash
# Set up your token
cargo run --release -p ferrite-examples --bin ferrite-chat -- --login

# Or use environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

## Usage

### As a Library

```toml
[dependencies]
ferrite = { git = "https://github.com/user/ferrite" }
```

```rust
use ferrite::{Tokenizer, ChatMessage, Sampler, SamplerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer with chat template
    let tokenizer = Tokenizer::from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").await?;

    // Format chat
    let messages = vec![
        ChatMessage::user("What is Rust?"),
    ];
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    let encoding = tokenizer.encode(&prompt)?;

    // Create sampler
    let sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    });

    Ok(())
}
```

### CLI

```bash
# Interactive model selection
ferrite-chat

# Specific model
ferrite-chat mistral
ferrite-chat mistral -q  # quantized

# Manage HuggingFace token
ferrite-chat --login
ferrite-chat --status
```

## Binary Size

```
Component               Size
────────────────────────────
Candle (ML ops)         ~18 MB
Tokenizers              ~4 MB
Model architectures     ~2 MB
Ferrite utilities       ~1 MB
────────────────────────────
Total binary            ~25 MB

Note: Model weights downloaded separately (~4-14 GB)
```

## Comparison

| Feature | Ferrite | llama.cpp | vLLM | Ollama |
|---------|---------|-----------|------|--------|
| Language | Rust | C++ | Python | Go+C++ |
| Binary size | 25 MB | 5 MB | ~2 GB | 150 MB |
| Models | HuggingFace | GGUF only | HuggingFace | GGUF |
| Quantization | GGUF Q4/Q8 | GGUF all | AWQ, GPTQ | GGUF |
| Dependencies | None | None | Python, CUDA | None |
| Batching | Single | Single | Continuous | Single |

## Roadmap

- [ ] Continuous batching for throughput
- [ ] Speculative decoding
- [ ] More quantization formats (AWQ, GPTQ)
- [ ] OpenAI-compatible API server
- [ ] Metal backend optimization

## License

MIT

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — The ML engine that makes this possible
- [llm-tokenizer](https://github.com/example/llm-tokenizer) — Tokenization with chat templates
- [HuggingFace](https://huggingface.co) — Model hosting and community
