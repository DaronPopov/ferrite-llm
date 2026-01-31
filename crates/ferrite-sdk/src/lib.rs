//! Ferrite Guest SDK - Write neural inference WASM modules in Rust
//!
//! This crate provides bindings for writing WASM modules that run in the ferrite runtime.

// Re-export wit-bindgen for use in macros
pub use wit_bindgen;

// Generate WIT bindings
// This will generate imports (ferrite::inference::*) and exports (Guest trait)
wit_bindgen::generate!({
    path: "../../wit",
    world: "ferrite-guest",
});

// Re-export for convenience
pub mod prelude {
    pub use crate::ferrite::inference::inference::*;
    pub use crate::ferrite::inference::tokenizer::*;
}
