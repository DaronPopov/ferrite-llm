use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ferrite_wasm_host::{bindings::FerriteGuest, HostState, ScriptHooks};
use std::path::PathBuf;
use tracing::{debug, info, warn};
use wasmtime::{
    component::{Component, Linker},
    Config, Engine, Store,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

#[derive(Parser, Debug)]
#[command(name = "ferrite-rt-runner")]
#[command(version, about = "Ferrite runtime runner")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Run {
        #[arg(value_name = "MODULE")]
        module: PathBuf,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        #[arg(long)]
        metrics: bool,
        #[arg(long, env = "FERRITE_SCRIPT_HOOK")]
        script_hook: Option<PathBuf>,
    },
    Models {
        #[arg(short, long)]
        detailed: bool,
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
}

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
    match args.command {
        Command::Run {
            module,
            verbose,
            model_cache,
            hf_token,
            metrics,
            script_hook,
        } => {
            init_logging(verbose);
            if let Some(ref token) = hf_token {
                debug!("Using HuggingFace token from CLI/env (length: {})", token.len());
            }
            run_module(&module, &model_cache, metrics, script_hook.as_ref())
        }
        Command::Models {
            detailed,
            model_cache,
            verbose,
        } => {
            init_logging(verbose);
            list_models(&model_cache, detailed)
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

fn run_module(
    module: &PathBuf,
    model_cache: &PathBuf,
    metrics: bool,
    script_hook: Option<&PathBuf>,
) -> Result<()> {
    info!("Ferrite Runtime v{}", env!("CARGO_PKG_VERSION"));
    info!("Loading WASM module: {}", module.display());

    let start_time = std::time::Instant::now();
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(false);

    let engine = Engine::new(&config)?;
    let component = Component::from_file(&engine, module)
        .with_context(|| format!("Failed to load WASM module: {}", module.display()))?;

    info!("Component loaded ({:.2}s)", start_time.elapsed().as_secs_f32());

    let mut linker = Linker::new(&engine);
    wasmtime_wasi::add_to_linker_sync(&mut linker)?;
    FerriteGuest::add_to_linker(&mut linker, |state: &mut RuntimeState| &mut state.host)?;

    let wasi = WasiCtxBuilder::new().inherit_stdio().inherit_env().build();
    let hooks = if let Some(path) = script_hook {
        info!("Loading script hook: {}", path.display());
        Some(ScriptHooks::from_file(path).map_err(anyhow::Error::msg)?)
    } else {
        None
    };
    let host = HostState::with_hooks(model_cache.clone(), hooks)?;
    let mut store = Store::new(&engine, RuntimeState { wasi, host });
    let guest = FerriteGuest::instantiate(&mut store, &component, &linker)?;

    let exec_start = std::time::Instant::now();
    let result = match guest.call_run(&mut store) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            warn!("Module returned error: {}", e);
            anyhow::bail!("Guest module failed: {}", e)
        }
        Err(e) => {
            warn!("Runtime error: {}", e);
            Err(e)
        }
    };

    if metrics {
        let total_time = start_time.elapsed().as_secs_f32();
        let exec_time = exec_start.elapsed().as_secs_f32();
        println!("\nPerformance Metrics:");
        println!("  Total time:     {:.2}s", total_time);
        println!("  Execution time: {:.2}s", exec_time);
        println!("  Startup time:   {:.2}s", total_time - exec_time);
    }

    result
}

fn list_models(_model_cache: &PathBuf, detailed: bool) -> Result<()> {
    let catalog = ferrite_wasm_host::ferrite_core::Catalog::new();
    println!("Available Models ({} in registry):\n", catalog.len());

    for spec in catalog.list() {
        if detailed {
            println!("  {} [{}]", spec.name, spec.size);
            println!("    Family: {:?}", spec.family);
            println!("    Format: {:?}", spec.format);
            println!("    Context: {} tokens", spec.context_length);
            println!(
                "    Auth required: {}",
                if spec.requires_auth { "yes" } else { "no" }
            );
            println!("    {}\n", spec.description);
        } else {
            let auth_marker = if spec.requires_auth { " 🔑" } else { "" };
            println!(
                "  {:30} {:6}  {}{}",
                spec.name, spec.size, spec.description, auth_marker
            );
        }
    }

    println!();
    Ok(())
}
