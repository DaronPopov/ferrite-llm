# ferrite-wasm-host

WASM host library for embedding neural inference in any Rust application.

## Features

- **Component Model** - Full WASM Component Model support
- **Type Adapter** - Automatic WIT to Rust type conversion
- **Resource Management** - Safe model lifecycle handling
- **HuggingFace Integration** - Automatic model downloading

## Usage

```toml
[dependencies]
ferrite-wasm-host = "0.3"
wasmtime = { version = "27", features = ["component-model"] }
wasmtime-wasi = "27"
```

```rust
use ferrite_wasm_host::{HostState, bindings::FerriteGuest, create_engine};
use wasmtime::{Store, Component, component::Linker};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView, ResourceTable};

// Create engine
let engine = create_engine()?;

// Load WASM component
let component = Component::from_file(&engine, "ai_module.wasm")?;

// Set up host state
struct AppState {
    wasi: WasiCtx,
    host: HostState,
}

impl WasiView for AppState {
    fn ctx(&mut self) -> &mut WasiCtx { &mut self.wasi }
    fn table(&mut self) -> &mut ResourceTable { self.host.table() }
}

let wasi = WasiCtxBuilder::new().inherit_stdio().inherit_env().build();
let host = HostState::new("./models")?;
let mut store = Store::new(&engine, AppState { wasi, host });

// Link host functions
let mut linker = Linker::new(&engine);
wasmtime_wasi::add_to_linker_sync(&mut linker)?;
FerriteGuest::add_to_linker(&mut linker, |s: &mut AppState| &mut s.host)?;

// Run guest module
let guest = FerriteGuest::instantiate(&mut store, &component, &linker)?;
guest.call_run(&mut store)?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
├─────────────────────────────────────────────────────────┤
│  ferrite-wasm-host                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  bindings   │  │   adapter   │  │      host       │  │
│  │ (WIT types) │→ │ (type conv) │→ │ (ferrite-core)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    WASM Guest Module                     │
└─────────────────────────────────────────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `bindings` | WIT-generated types and traits |
| `adapter` | Type conversion layer |
| `host` | Host trait implementations |

## License

MIT
