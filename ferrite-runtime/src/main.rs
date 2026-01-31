use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn};
use wasmtime::{
    component::{Component, Linker},
    Config, Engine, Store,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

mod adapter;
mod host;

// Generate WIT bindings for the host
wasmtime::component::bindgen!({
    path: "../wit",
    world: "ferrite-guest",
    async: false,
});

/// Ferrite WASM Runtime - Neural Inference OS
#[derive(Parser, Debug)]
#[command(name = "ferrite-rt")]
#[command(about = "WASM runtime for neural inference", long_about = None)]
struct Args {
    /// Path to the WASM module to execute
    #[arg(value_name = "MODULE")]
    module: PathBuf,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Model cache directory
    #[arg(long, default_value = "./models")]
    model_cache: PathBuf,
}

/// Runtime state that implements the host interface
struct RuntimeState {
    wasi: WasiCtx,
    host: host::HostState,
}

impl WasiView for RuntimeState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable {
        self.host.table()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    info!("🔥 Ferrite Runtime v0.1.0");
    info!("Loading WASM module: {}", args.module.display());

    // Configure wasmtime engine with component model
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(false); // Synchronous for now

    let engine = Engine::new(&config)?;

    // Load the WASM component
    let component = Component::from_file(&engine, &args.module)
        .with_context(|| format!("Failed to load WASM module: {}", args.module.display()))?;

    info!("✓ Component loaded successfully");

    // Create linker and add host functions
    let mut linker = Linker::new(&engine);

    // Add WASI support
    wasmtime_wasi::add_to_linker_sync(&mut linker)?;

    // Add ferrite host functions
    FerriteGuest::add_to_linker(&mut linker, |state: &mut RuntimeState| &mut state.host)?;

    info!("✓ Host functions linked");

    // Create store with runtime state
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_env()
        .build();

    let host = host::HostState::new(args.model_cache)?;

    let mut store = Store::new(&engine, RuntimeState { wasi, host });

    // Instantiate the component
    let guest = FerriteGuest::instantiate(&mut store, &component, &linker)?;

    info!("✓ Module instantiated");
    info!("▶ Executing guest module...\n");

    // Call the guest's run() function
    match guest.call_run(&mut store) {
        Ok(Ok(())) => {
            info!("\n✓ Module executed successfully");
            Ok(())
        }
        Ok(Err(e)) => {
            warn!("Module returned error: {}", e);
            anyhow::bail!("Guest module failed: {}", e)
        }
        Err(e) => {
            warn!("Runtime error: {}", e);
            Err(e)
        }
    }
}
