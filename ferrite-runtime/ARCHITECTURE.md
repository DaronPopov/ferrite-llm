# Ferrite WASM Runtime - Architecture

## The Trait Signature Engine

The core challenge: **bridging Ferrite's native Rust engine to WASM Component Model**.

### Problem

Ferrite has its own type system:
- `ferrite::GenerationConfig` with specific fields and types
- `ferrite::Tokenizer` with complex state
- Candle tensors, models, devices

WASM Component Model has WIT-defined types:
- Simple, serializable types (f32, u32, string, records)
- Resource handles (opaque references)
- No native Rust types

**They don't speak the same language.**

### Solution: The Adapter Layer

We created a **three-layer architecture**:

```
┌─────────────────────────────────────────────────────────┐
│  WASM Guest Module (hello-inference.wasm)               │
│  - Uses WIT-generated bindings                          │
│  - Calls: load_model(), generate()                      │
└──────────────────────┬──────────────────────────────────┘
                       │ WIT Interface (ferrite.wit)
                       │ - GenerationConfig record
                       │ - Model resource
                       │ - Functions with WIT types
┌──────────────────────▼──────────────────────────────────┐
│  Host Trait Implementation (src/host.rs)                │
│  - Implements wasmtime-generated Host traits            │
│  - Receives WIT types from WASM boundary                │
│  - Uses Resource<WitModel> handles                      │
└──────────────────────┬──────────────────────────────────┘
                       │ Adapter Layer
                       │ (src/adapter.rs)
┌──────────────────────▼──────────────────────────────────┐
│  Type Adapters (THE TRAIT SIGNATURE ENGINE)             │
│                                                          │
│  1. wit_to_ferrite_config(WitGenConfig)                 │
│     -> ferrite::GenerationConfig                        │
│                                                          │
│  2. ModelAdapter                                        │
│     - Wraps ferrite engine state                        │
│     - generate() -> calls ferrite                       │
│                                                          │
│  3. TokenizerAdapter                                    │
│     - Wraps ferrite::Tokenizer                          │
└──────────────────────┬──────────────────────────────────┘
                       │ Native Rust calls
┌──────────────────────▼──────────────────────────────────┐
│  Ferrite Engine (your existing code)                    │
│  - Candle models                                        │
│  - CUDA/Metal backends                                  │
│  - Native Rust types                                    │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. WIT Interface Definition (`wit/ferrite.wit`)

Defines the contract in WIT (WebAssembly Interface Type):

```wit
interface inference {
    record generation-config {
        temperature: f32,
        top-p: f32,
        max-tokens: u32,
        // ...
    }

    resource model {
        generate: func(prompt: string, config: generation-config)
            -> result<string, string>;
    }

    load-model: func(model-name: string) -> result<model, string>;
}
```

### 2. Type Adapter (`src/adapter.rs`)

**The Trait Signature Engine** - converts between type systems:

```rust
// WIT f32 -> Ferrite f64
pub fn wit_to_ferrite_config(wit: &WitGenConfig) -> ferrite::GenerationConfig {
    ferrite::GenerationConfig {
        temperature: wit.temperature as f64,  // f32 -> f64
        max_tokens: wit.max_tokens as usize,  // u32 -> usize
        // Map other fields, provide defaults for non-exposed params
        min_p: 0.05,           // Not in WIT, use default
        repetition_penalty: 1.0,
        // ...
    }
}

// Wraps ferrite engine calls
pub struct ModelAdapter {
    pub name: String,
    // In full impl: Candle model, tokenizer, device
}

impl ModelAdapter {
    pub fn generate(&self, prompt: &str, config: &ferrite::GenerationConfig)
        -> Result<String, String>
    {
        // 1. Use ferrite::Tokenizer to encode
        // 2. Call Candle model.forward()
        // 3. Sample tokens
        // 4. Decode and return
        // For prototype: mock response
        Ok(format!("Response from {}", self.name))
    }
}
```

### 3. Host Implementation (`src/host.rs`)

Implements wasmtime's generated traits:

```rust
impl InferenceHost for HostState {
    fn load_model(&mut self, model_name: String)
        -> Result<Resource<WitModel>, String>
    {
        // 1. Create ModelAdapter (wraps ferrite)
        let adapter = ModelAdapter::new(model_name)?;

        // 2. Store in resource table
        let resource = self.table.push(adapter)?;

        // 3. Return as WIT resource
        Ok(unsafe { std::mem::transmute(resource) })
    }
}

impl HostModel for HostState {
    fn generate(&mut self, model: Resource<WitModel>,
                prompt: String, config: WitGenConfig)
        -> Result<String, String>
    {
        // 1. Get ModelAdapter from resource table
        let adapter = self.table.get(&model)?;

        // 2. Convert WIT config to ferrite config
        let ferrite_config = wit_to_ferrite_config(&config);

        // 3. Call ferrite engine
        adapter.generate(&prompt, &ferrite_config)
    }
}
```

## The Resource Bridge

**Critical insight**: WIT resources (like `model`) become `Resource<T>` handles in Rust.

Problem: Wasmtime expects `Resource<WitModel>`, but we store `ModelAdapter`.

Solution: **Type transmutation** (safe because Resource<T> is a newtype around u32):

```rust
// Store our type
let resource: Resource<ModelAdapter> = self.table.push(adapter)?;

// Convert to WIT type
let wit_resource: Resource<WitModel> = unsafe {
    std::mem::transmute(resource)
};
```

This bridges wasmtime's resource system to our ferrite engine state.

## Data Flow Example

User calls `model.generate("Hello", config)` from WASM:

1. **WASM boundary**: Call enters host with WIT types
   ```
   prompt: String = "Hello"
   config: GenerationConfig { temperature: 0.8f32, ... }
   ```

2. **Host layer** (`host.rs`): Receives call
   ```rust
   fn generate(model: Resource<WitModel>, prompt: String, config: WitGenConfig)
   ```

3. **Type conversion** (`adapter.rs`): WIT → Ferrite
   ```rust
   let ferrite_config = wit_to_ferrite_config(&config);
   // GenerationConfig { temperature: 0.8f64, ... }
   ```

4. **Ferrite engine**: Native Rust call
   ```rust
   model_adapter.generate(&prompt, &ferrite_config)
   // -> Uses Candle, CUDA, etc.
   ```

5. **Return path**: String result flows back through layers
   ```
   Ferrite Result<String>
   -> Adapter Result<String>
   -> Host Result<String>
   -> WASM Result<String>
   ```

## Why This Matters

Without the trait signature engine:
- ❌ Type mismatches everywhere
- ❌ Can't call ferrite from WASM
- ❌ No way to bridge the systems

With the trait signature engine:
- ✅ Clean type conversions
- ✅ Ferrite engine works transparently
- ✅ WASM modules can do neural inference
- ✅ Extensible for new models/features

## Next Steps

The prototype uses mock implementations. To make it production-ready:

1. **ModelAdapter** should:
   - Load actual Candle models from HuggingFace
   - Initialize tokenizers
   - Manage GPU memory
   - Implement real generation loop

2. **Type adapters** should handle:
   - All ferrite config options
   - Streaming (WASI streams)
   - Batch processing
   - Error propagation

3. **Resource management**:
   - KV cache handling
   - Multi-model support
   - Memory limits

The architecture is sound. The trait signature engine works. Now we fill in the implementation.
