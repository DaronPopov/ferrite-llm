//! Ferrite CLI - Standalone WASM Runtime for Neural Inference
//!
//! This is the command-line interface for running ferrite WASM modules.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ferrite_wasm_host::{HostState, bindings::FerriteGuest};
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use tracing::{info, warn, debug};
use wasmtime::{
    component::{Component, Linker},
    Config, Engine, Store,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

/// Ferrite WASM Runtime - Neural Inference OS
#[derive(Parser, Debug)]
#[command(name = "ferrite-rt")]
#[command(version, about = "WASM runtime for neural inference")]
#[command(long_about = "Production-ready neural inference runtime with WASM sandboxing")]
struct Args {
    /// Subcommand to execute
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a WASM module
    Run {
        /// Path to the WASM module
        #[arg(value_name = "MODULE")]
        module: PathBuf,

        /// Verbose logging (-v for info, -vv for debug, -vvv for trace)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,

        /// Model cache directory
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,

        /// HuggingFace authentication token
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,

        /// Show performance metrics after execution
        #[arg(long)]
        metrics: bool,
    },

    /// List downloaded models in cache
    Models {
        /// Show detailed model information
        #[arg(short, long)]
        detailed: bool,

        /// Model cache directory
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,

        /// Verbose logging
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Show model cache information
    Cache {
        /// Clear the model cache
        #[arg(long)]
        clear: bool,

        /// Show cache statistics
        #[arg(long)]
        stats: bool,

        /// Model cache directory
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,

        /// Verbose logging
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Show system information
    Info {
        /// Verbose logging
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Install local prerequisites for building guest WASM modules
    Setup {
        /// Check prerequisites without installing anything
        #[arg(long)]
        check: bool,

        /// Skip installing the wasm32-wasip1 Rust target
        #[arg(long)]
        skip_target: bool,

        /// Skip installing the wasm-tools CLI
        #[arg(long)]
        skip_wasm_tools: bool,

        /// Verbose logging
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
}

/// Runtime state that implements the host interface
struct RuntimeState {
    wasi: WasiCtx,
    host: HostState,
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

    // Execute the appropriate command
    match args.command {
        Command::Run { module, verbose, model_cache, hf_token, metrics } => {
            init_logging(verbose);
            if let Some(ref token) = hf_token {
                debug!("🔑 Using HuggingFace token from CLI/env (length: {})", token.len());
            }
            run_module(&module, &model_cache, metrics)
        }
        Command::Models { detailed, model_cache, verbose } => {
            init_logging(verbose);
            list_models(&model_cache, detailed)
        }
        Command::Cache { clear, stats, model_cache, verbose } => {
            init_logging(verbose);
            manage_cache(&model_cache, clear, stats)
        }
        Command::Info { verbose } => {
            init_logging(verbose);
            show_info()
        }
        Command::Setup { check, skip_target, skip_wasm_tools, verbose } => {
            init_logging(verbose);
            setup_wasm_toolchain(check, skip_target, skip_wasm_tools)
        }
    }
}

fn init_logging(verbose: u8) {
    let log_level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_target(verbose >= 2)
        .init();
}

fn run_module(module: &PathBuf, model_cache: &PathBuf, metrics: bool) -> Result<()> {
    info!("🔥 Ferrite Runtime v{}", env!("CARGO_PKG_VERSION"));
    info!("📦 Loading WASM module: {}", module.display());

    let start_time = std::time::Instant::now();

    // Configure wasmtime engine with component model
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(false);

    let engine = Engine::new(&config)?;

    // Load the WASM component
    let component = Component::from_file(&engine, module)
        .with_context(|| format!("Failed to load WASM module: {}", module.display()))?;

    info!("✅ Component loaded ({:.2}s)", start_time.elapsed().as_secs_f32());

    // Create linker and add host functions
    let mut linker = Linker::new(&engine);

    // Add WASI support
    wasmtime_wasi::add_to_linker_sync(&mut linker)?;

    // Add ferrite host functions
    FerriteGuest::add_to_linker(&mut linker, |state: &mut RuntimeState| &mut state.host)?;

    debug!("✅ Host functions linked");

    // Create store with runtime state
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_env()
        .build();

    let host = HostState::new(model_cache.clone())?;

    let mut store = Store::new(&engine, RuntimeState { wasi, host });

    // Instantiate the component
    let guest = FerriteGuest::instantiate(&mut store, &component, &linker)?;

    debug!("✅ Module instantiated");
    info!("▶️  Executing guest module...\n");

    let exec_start = std::time::Instant::now();

    // Call the guest's run() function
    let result = match guest.call_run(&mut store) {
        Ok(Ok(())) => {
            info!("\n✅ Module executed successfully");
            Ok(())
        }
        Ok(Err(e)) => {
            warn!("❌ Module returned error: {}", e);
            anyhow::bail!("Guest module failed: {}", e)
        }
        Err(e) => {
            warn!("❌ Runtime error: {}", e);
            Err(e)
        }
    };

    if metrics {
        let total_time = start_time.elapsed().as_secs_f32();
        let exec_time = exec_start.elapsed().as_secs_f32();
        println!("\n📊 Performance Metrics:");
        println!("   Total time:     {:.2}s", total_time);
        println!("   Execution time: {:.2}s", exec_time);
        println!("   Startup time:   {:.2}s", total_time - exec_time);
    }

    result
}

fn list_models(_model_cache: &PathBuf, detailed: bool) -> Result<()> {
    // Show available models from registry
    let catalog = ferrite_wasm_host::ferrite_core::Catalog::new();

    println!("📚 Available Models ({} in registry):\n", catalog.len());

    for spec in catalog.list() {
        if detailed {
            println!("  {} [{}]", spec.name, spec.size);
            println!("    Family: {:?}", spec.family);
            println!("    Format: {:?}", spec.format);
            println!("    Context: {} tokens", spec.context_length);
            println!("    Auth required: {}", if spec.requires_auth { "yes" } else { "no" });
            println!("    {}\n", spec.description);
        } else {
            let auth_marker = if spec.requires_auth { " 🔑" } else { "" };
            println!("  {:30} {:6}  {}{}", spec.name, spec.size, spec.description, auth_marker);
        }
    }

    println!();

    // Check HuggingFace cache for downloaded models
    if let Some(home) = dirs::home_dir() {
        let hf_cache = home.join(".cache/huggingface/hub");
        if hf_cache.exists() {
            println!("📥 Downloaded Models (in HF cache):\n");

            let entries = std::fs::read_dir(&hf_cache)?;
            let mut count = 0;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().unwrap().to_string_lossy();
                    if name.starts_with("models--") {
                        count += 1;
                        let model_name = name.replace("models--", "").replace("--", "/");
                        println!("  • {}", model_name);

                        if detailed {
                            println!("    Path: {}", path.display());
                            if let Ok(size) = get_dir_size(&path) {
                                println!("    Size: {}", format_size(size));
                            }
                        }
                    }
                }
            }

            if count == 0 {
                println!("  (no models found)");
            } else {
                println!("\n📊 Total models: {}", count);
            }
        }
    }

    Ok(())
}

fn manage_cache(model_cache: &PathBuf, clear: bool, stats: bool) -> Result<()> {
    if stats {
        println!("📊 Cache Statistics:");
        println!("   Cache directory: {}", model_cache.display());

        if model_cache.exists() {
            match get_dir_size(model_cache) {
                Ok(size) => println!("   Local cache size: {}", format_size(size)),
                Err(e) => warn!("   Could not calculate cache size: {}", e),
            }
        } else {
            println!("   Local cache: (empty)");
        }

        // Show HF cache stats
        if let Some(home) = dirs::home_dir() {
            let hf_cache = home.join(".cache/huggingface/hub");
            if hf_cache.exists() {
                match get_dir_size(&hf_cache) {
                    Ok(size) => println!("   HuggingFace cache: {}", format_size(size)),
                    Err(e) => warn!("   Could not calculate HF cache size: {}", e),
                }
            }
        }
    }

    if clear {
        println!("🗑️  Clearing cache...");

        if model_cache.exists() {
            std::fs::remove_dir_all(model_cache)?;
            println!("✅ Cleared local cache: {}", model_cache.display());
        }

        println!("\n⚠️  HuggingFace cache not cleared (managed by hf-hub)");
        println!("   To clear HF cache manually, delete: ~/.cache/huggingface/hub");
    }

    Ok(())
}

fn setup_wasm_toolchain(check: bool, skip_target: bool, skip_wasm_tools: bool) -> Result<()> {
    println!("Ferrite WASM setup");
    println!("==================\n");

    let rustup_available = command_exists("rustup");
    let cargo_available = command_exists("cargo");
    let target_installed = rustup_available && rustup_target_installed("wasm32-wasip1")?;
    let wasm_tools_installed = command_exists("wasm-tools");

    print_status(
        "rustup",
        rustup_available,
        "required to install the wasm32-wasip1 target",
    );
    print_status("cargo", cargo_available, "required to install wasm-tools");
    print_status("wasm32-wasip1 target", target_installed, "Rust stdlib for WASI Preview 1");
    print_status("wasm-tools", wasm_tools_installed, "used for component embed/new");

    if check {
        if (!skip_target && !target_installed) || (!skip_wasm_tools && !wasm_tools_installed) {
            anyhow::bail!("WASM prerequisites are missing");
        }
        println!("\nAll requested prerequisites are installed.");
        return Ok(());
    }

    if !skip_target && !target_installed {
        anyhow::ensure!(rustup_available, "rustup is not available in PATH");
        run_bootstrap_command(
            "Installing Rust target wasm32-wasip1",
            "rustup",
            &["target", "add", "wasm32-wasip1"],
        )?;
    }

    if !skip_wasm_tools && !wasm_tools_installed {
        anyhow::ensure!(cargo_available, "cargo is not available in PATH");
        run_bootstrap_command("Installing wasm-tools", "cargo", &["install", "wasm-tools"])?;
    }

    println!("\nWASM setup complete.");
    println!("You can now build guest modules with `cargo build --target wasm32-wasip1 --release`.");
    Ok(())
}

fn show_info() -> Result<()> {
    println!("🔥 Ferrite Runtime - Neural Inference OS");
    println!("   Version: {}", env!("CARGO_PKG_VERSION"));
    println!("   Homepage: {}", env!("CARGO_PKG_HOMEPAGE"));
    println!();
    println!("📦 Components:");
    println!("   • ferrite-core: Pure inference engine");
    println!("   • ferrite-wasm-host: WASM orchestration");
    println!("   • ferrite-sdk: Guest SDK for WASM modules");
    println!();
    println!("🔧 System:");
    println!("   • OS: {}", std::env::consts::OS);
    println!("   • Arch: {}", std::env::consts::ARCH);

    // Check CUDA availability
    #[cfg(feature = "cuda")]
    {
        use candle_core::Device;
        match Device::cuda_if_available(0) {
            Ok(Device::Cuda(_)) => println!("   • CUDA: ✅ Available"),
            _ => println!("   • CUDA: ❌ Not available"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("   • CUDA: ❌ Not compiled with CUDA support");
    }

    println!();

    // Show model registry info
    let catalog = ferrite_wasm_host::ferrite_core::Catalog::new();
    println!("📚 Model Registry: {} models available", catalog.len());
    println!("   Families: Llama, Mistral, Qwen, Phi, Gemma, CodeLlama");
    println!("   Formats: GGUF (quantized), SafeTensors");
    println!("   Run 'ferrite-rt models' to list all");

    println!();
    println!("🌐 Interfaces:");
    println!("   • WIT (WebAssembly Interface Types)");
    println!("   • WASI (WebAssembly System Interface)");

    Ok(())
}

// Helper functions
fn get_dir_size(path: &PathBuf) -> Result<u64> {
    let mut size = 0;
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            size += entry.metadata()?.len();
        }
    }
    Ok(size)
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

fn print_status(name: &str, installed: bool, description: &str) {
    let marker = if installed { "ok" } else { "missing" };
    println!("{name:20} {marker:8} {description}");
}

fn command_exists(program: &str) -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join(program))
        .any(|path| path.is_file())
}

fn rustup_target_installed(target: &str) -> Result<bool> {
    let output = ProcessCommand::new("rustup")
        .args(["target", "list", "--installed"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .context("failed to query installed Rust targets")?;

    anyhow::ensure!(
        output.status.success(),
        "rustup target list --installed exited with status {}",
        output.status
    );

    let stdout = String::from_utf8(output.stdout).context("rustup returned non-UTF8 output")?;
    Ok(stdout.lines().any(|line| line.trim() == target))
}

fn run_bootstrap_command(label: &str, program: &str, args: &[&str]) -> Result<()> {
    println!("\n{label}...");

    let status = ProcessCommand::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to launch `{program}`"))?;

    anyhow::ensure!(status.success(), "`{program}` exited with status {status}");
    Ok(())
}
