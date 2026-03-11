# ferrite-llm

Transformer inference runtime with WASM sandboxing.

> Production-ready neural inference runtime with WASM sandboxing

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`ferrite-llm` is a modular framework for running neural inference workloads in WebAssembly sandboxes. It separates the inference engine, WASM orchestration, and guest modules so deployments stay portable and contained.

## 🏗️ Architecture

Ferrite is built as a multi-crate framework with clear separation of concerns:

```
ferrite/
├── crates/
│   ├── ferrite-core/        # Pure inference engine (Candle/CUDA)
│   ├── ferrite-wasm-host/   # WASM host library
│   ├── ferrite-sdk/         # Guest SDK for WASM modules
│   └── ferrite-cli/         # Standalone runtime binary
├── wit/                     # WIT interface definitions
├── examples/                # Example modules
└── docs/                    # Detailed documentation
```

### Component Overview

| Crate | Purpose | Use Case |
|-------|---------|----------|
| **ferrite-core** | Pure inference engine using Candle | Embed inference in Rust apps |
| **ferrite-wasm-host** | WASM orchestration library | Add WASM-based AI to any app |
| **ferrite-sdk** | Guest bindings and helpers | Write portable AI modules |
| **ferrite-cli** | Production runtime | Run AI workloads from CLI |

## 🚀 Quick Start

### Install the Runtime

```bash
git clone https://github.com/DaronPopov/ferrite.git
cd ferrite
cargo install --path crates/ferrite-cli
ferrite-rt setup
```

### Run an AI Module

```bash
cargo build -p mistral-inference --target wasm32-wasip1 --release
  wasm-tools component embed wit \
    target/wasm32-wasip1/release/mistral_inference.wasm \
    -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
  wasm-tools component new \
    target/wasm32-wasip1/release/mistral_inference.embed.wasm \
    --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
    -o target/wasm32-wasip1/release/mistral_inference.component.wasm
  ferrite-rt run target/wasm32-wasip1/release/mistral_inference.component.wasm

```

`ferrite-rt setup` is idempotent and installs the local WASM build prerequisites used by the examples:
- `rustup target add wasm32-wasip1`
- `cargo install wasm-tools`

### Write Your First AI Module

```rust
// Cargo.toml
[dependencies]
ferrite-sdk = "0.3"
wit-bindgen = "0.37"

[lib]
crate-type = ["cdylib"]

// src/lib.rs
wit_bindgen::generate!({
    path: "path/to/wit",
    world: "ferrite-guest",
});

use ferrite::inference::inference::{load_model, GenerationConfig};

struct MyAI;

impl Guest for MyAI {
    fn run() -> Result<(), String> {
        let model = load_model("mistral-7b-q4", None)?;

        let config = GenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 200,
            seed: Some(42),
        };

        let response = model.generate("What is Rust?", config)?;
        println!("{}", response);
        Ok(())
    }
}

export!(MyAI);
```

Build it:
```bash
cargo build --target wasm32-wasip1 --release
wasm-tools component embed wit target/wasm32-wasip1/release/my_ai.wasm -o my_ai.embed.wasm
wasm-tools component new my_ai.embed.wasm \
  --adapt wasi_snapshot_preview1.reactor.wasm \
  -o my_ai.component.wasm
ferrite-rt run my_ai.component.wasm
```

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Deep dive into the design
- [Programming Guide](docs/PROGRAMMING.md) - Writing WASM modules
- [Host Integration](docs/HOST_INTEGRATION.md) - Embedding ferrite in apps
- [CLI Reference](docs/CLI_REFERENCE.md) - Runtime command documentation
- [WIT Reference](docs/WIT_REFERENCE.md) - Interface definitions

## 🎯 Use Cases

### 1. Standalone AI Runtime
Use `ferrite-cli` to run AI workloads with full sandboxing:
```bash
ferrite-rt run chatbot.wasm --metrics
```

### 2. Embedded in Applications
Use `ferrite-wasm-host` to add AI to your Rust app:
```rust
use ferrite_wasm_host::{HostState, bindings::FerriteGuest, create_engine};
use wasmtime::{Store, Component, component::Linker};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

let engine = create_engine()?;
let component = Component::from_file(&engine, "ai.wasm")?;

struct AppState {
    wasi: WasiCtx,
    host: HostState,
}

impl WasiView for AppState {
    fn ctx(&mut self) -> &mut WasiCtx { &mut self.wasi }
    fn table(&mut self) -> &mut ResourceTable { self.host.table() }
}

let wasi = WasiCtxBuilder::new().inherit_stdio().build();
let host = HostState::new("./models")?;
let mut store = Store::new(&engine, AppState { wasi, host });

let mut linker = Linker::new(&engine);
wasmtime_wasi::add_to_linker_sync(&mut linker)?;
FerriteGuest::add_to_linker(&mut linker, |s: &mut AppState| &mut s.host)?;

let guest = FerriteGuest::instantiate(&mut store, &component, &linker)?;
guest.call_run(&mut store)?;
```

### 3. Pure Inference
Use `ferrite-core` for direct inference without WASM:
```rust
use ferrite_core::{ChatSession, GenerationConfig, Tokenizer};

let tokenizer = Tokenizer::from_dir("./model")?;
let session = ChatSession::new(model, tokenizer, None, config)?;
let response = session.user_turn("Hello!")?;
```


## 🛠️ Development

### Prerequisites

- Rust 1.70+
- `wasm-tools` for componentization: `cargo install wasm-tools`
- (Optional) CUDA toolkit for GPU support

### Build All Crates

```bash
cargo build --workspace --release
```

### Build with CUDA

```bash
cargo build --workspace --release --features cuda
```

### Run Tests

```bash
cargo test --workspace
```

## 📦 Crate Documentation

Each crate has its own README with detailed information:

- [ferrite-core](crates/ferrite-core/README.md) - Pure inference engine
- [ferrite-wasm-host](crates/ferrite-wasm-host/README.md) - WASM host library
- [ferrite-sdk](crates/ferrite-sdk/README.md) - Guest SDK
- [ferrite-cli](crates/ferrite-cli/README.md) - Runtime CLI

## 🚢 Publishing to crates.io

Each crate is independently versioned and can be published:

```bash
# Publish in dependency order
cd crates/ferrite-core && cargo publish
cd ../ferrite-sdk && cargo publish
cd ../ferrite-wasm-host && cargo publish
cd ../ferrite-cli && cargo publish
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure `cargo build --workspace` passes with zero warnings
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with [Candle](https://github.com/huggingface/candle) for inference
- Uses [Wasmtime](https://github.com/bytecodealliance/wasmtime) for WASM runtime
- Model support via [HuggingFace Hub](https://huggingface.co)
- WASM Component Model and WIT specifications

## 🔗 Links

- [Documentation](docs/)
- [Examples](examples/)
- [Issue Tracker](https://github.com/sperabality/ferrite/issues)
- [Releases](https://github.com/sperabality/ferrite/releases)

---

**Status**: Production-ready (v0.3.0) - Zero compilation warnings, full feature set
