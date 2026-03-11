//! WIT bindings for ferrite-wasm-host
//!
//! This module contains the generated WIT bindings that define the interface
//! between WASM guests and the ferrite host.

// Generate WIT bindings
wasmtime::component::bindgen!({
    path: "../../wit",
    world: "ferrite-guest",
    async: false,
});

// Re-export key types for convenience
pub use ferrite::inference::inference::GenerationConfig as WitGenConfig;
pub use ferrite::inference::inference::{
    Generation as WitGeneration,
    Host as InferenceHost,
    HostGeneration,
    HostModel,
    Model as WitModel,
};
pub use ferrite::inference::tokenizer::Host as TokenizerHost;
