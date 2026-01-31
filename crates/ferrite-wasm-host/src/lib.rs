//! Ferrite WASM Host - Embed Neural Inference in Any Rust App
//!
//! This library provides everything you need to add WASM-based neural inference
//! to your Rust application. It handles:
//! - WIT interface implementation
//! - Model loading and caching
//! - Type conversion (WIT ↔ Ferrite)
//! - Resource management
//!
//! # Example
//!
//! ```no_run
//! use ferrite_wasm_host::{FerriteHost, create_runtime};
//! use wasmtime::{Engine, Store};
//!
//! // Create the runtime
//! let (engine, linker) = create_runtime()?;
//!
//! // Load your WASM component
//! let component = /* ... */;
//!
//! // Create store with ferrite host
//! let host = FerriteHost::new("./models")?;
//! let mut store = Store::new(&engine, host);
//!
//! // Instantiate and run
//! let guest = YourGuest::instantiate(&mut store, &component, &linker)?;
//! guest.call_run(&mut store)?;
//! ```

pub mod bindings;
pub mod adapter;
pub mod host;

// Re-export key types
pub use adapter::{ModelAdapter, wit_to_ferrite_config};
pub use host::HostState;

// Re-export ferrite-core for convenience
pub use ferrite_core;

use anyhow::Result;
use wasmtime::{Config, Engine};

/// Create a configured wasmtime engine for ferrite
pub fn create_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(false);

    Ok(Engine::new(&config)?)
}

/// Convenience type alias
pub type FerriteHost = HostState;
