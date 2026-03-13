use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use tracing::warn;

#[derive(Parser, Debug)]
#[command(name = "ferrite-rt")]
#[command(version, about = "Ferrite CLI")]
#[command(long_about = "Lightweight front-end for Ferrite runtime administration and dispatch")]
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
    Cache {
        #[arg(long)]
        clear: bool,
        #[arg(long)]
        stats: bool,
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
    Info {
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
    Doctor {
        #[arg(long, default_value = "runtime", env = "FERRITE_INSTALL_PROFILE")]
        profile: String,
        #[arg(long)]
        repo_root: Option<PathBuf>,
        #[arg(long, default_value = "./models", env = "FERRITE_MODEL_CACHE")]
        model_cache: PathBuf,
        #[arg(long)]
        strict: bool,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
    Setup {
        #[arg(long)]
        check: bool,
        #[arg(long)]
        skip_deps_update: bool,
        #[arg(long)]
        skip_target: bool,
        #[arg(long)]
        skip_wasm_tools: bool,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
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
            let mut runner_args = vec![OsString::from("run"), module.into_os_string()];
            push_count_flag(&mut runner_args, "--verbose", verbose);
            push_opt_flag(&mut runner_args, "--model-cache", Some(model_cache.into_os_string()));
            push_opt_flag(&mut runner_args, "--hf-token", hf_token.map(OsString::from));
            if metrics {
                runner_args.push(OsString::from("--metrics"));
            }
            push_opt_flag(
                &mut runner_args,
                "--script-hook",
                script_hook.map(|path| path.into_os_string()),
            );
            delegate_to_runner(&runner_args)
        }
        Command::Models {
            detailed,
            model_cache,
            verbose,
        } => {
            init_logging(verbose);
            list_models(&model_cache, detailed)
        }
        Command::Cache {
            clear,
            stats,
            model_cache,
            verbose,
        } => {
            init_logging(verbose);
            manage_cache(&model_cache, clear, stats)
        }
        Command::Info { verbose } => {
            init_logging(verbose);
            show_info()
        }
        Command::Doctor {
            profile,
            repo_root,
            model_cache,
            strict,
            verbose,
        } => {
            init_logging(verbose);
            run_doctor(&profile, repo_root.as_deref(), &model_cache, strict)
        }
        Command::Setup {
            check,
            skip_deps_update,
            skip_target,
            skip_wasm_tools,
            verbose,
        } => {
            init_logging(verbose);
            setup_wasm_toolchain(check, skip_deps_update, skip_target, skip_wasm_tools)
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

fn delegate_to_runner(args: &[OsString]) -> Result<()> {
    if let Some(runner) = find_runner_binary() {
        let mut command = ProcessCommand::new(&runner);
        command
            .args(args)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        apply_native_runtime_env(&mut command, runner.parent());
        let status = command
            .status()
            .with_context(|| format!("failed to launch runner {}", runner.display()))?;
        anyhow::ensure!(status.success(), "runner exited with status {status}");
        return Ok(());
    }

    let repo_root = find_repo_root()?
        .ok_or_else(|| anyhow::anyhow!("runner binary not found and repo root could not be inferred"))?;
    anyhow::ensure!(
        command_exists("cargo"),
        "runner binary not found and cargo is unavailable for repo fallback"
    );

    let features = std::env::var("FERRITE_RUNNER_FEATURES")
        .unwrap_or_else(|_| "runtime,cuda".to_string());
    let mut command = ProcessCommand::new("cargo");
    command
        .current_dir(repo_root)
        .arg("run")
        .arg("-p")
        .arg("ferrite-cli")
        .arg("--bin")
        .arg("ferrite-rt-runner")
        .arg("--features")
        .arg(features)
        .arg("--")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .context("failed to launch cargo fallback runner")?;
    anyhow::ensure!(status.success(), "runner fallback exited with status {status}");
    Ok(())
}

fn find_runner_binary() -> Option<PathBuf> {
    let current = std::env::current_exe().ok()?;
    let dir = current.parent()?;
    let direct = dir.join("ferrite-rt-runner");
    if direct.is_file() {
        return Some(direct);
    }
    None
}

fn apply_native_runtime_env(command: &mut ProcessCommand, runner_dir: Option<&Path>) {
    let native_dirs = native_runtime_library_dirs(runner_dir);
    if native_dirs.is_empty() {
        return;
    }

    if let Ok(value) = std::env::join_paths(prepend_env_paths("LD_LIBRARY_PATH", &native_dirs)) {
        command.env("LD_LIBRARY_PATH", value);
    }
}

fn native_runtime_library_dirs(runner_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(dir) = std::env::var_os("FERRITE_NATIVE_LIB_DIR").map(PathBuf::from) {
        candidates.push(dir);
    }

    if let Some(runner_dir) = runner_dir {
        candidates.push(runner_dir.join("lib"));
        candidates.push(runner_dir.join("../share/ferrite/src/ferrite/ferrite-os/lib"));
        candidates.push(runner_dir.join("../share/ferrite-llm/src/ferrite-llm/ferrite-os/lib"));
    }

    if let Ok(Some(repo_root)) = find_repo_root() {
        candidates.push(repo_root.join("ferrite-os/lib"));
    }

    if let Some(repo_root) = default_install_repo_root() {
        candidates.push(repo_root.join("ferrite-os/lib"));
    }

    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|path| canonicalize_if_exists(path))
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

fn canonicalize_if_exists(path: PathBuf) -> Option<PathBuf> {
    if !path.is_dir() {
        return None;
    }
    std::fs::canonicalize(path).ok()
}

fn prepend_env_paths(var: &str, extra_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = extra_paths.to_vec();
    if let Some(existing) = std::env::var_os(var) {
        paths.extend(std::env::split_paths(&existing));
    }
    paths
}

fn push_opt_flag(args: &mut Vec<OsString>, flag: &str, value: Option<OsString>) {
    if let Some(value) = value {
        args.push(OsString::from(flag));
        args.push(value);
    }
}

fn push_count_flag(args: &mut Vec<OsString>, flag: &str, count: u8) {
    for _ in 0..count {
        args.push(OsString::from(flag));
    }
}

fn manage_cache(model_cache: &PathBuf, clear: bool, stats: bool) -> Result<()> {
    if stats {
        println!("Cache Statistics:");
        println!("  Cache directory: {}", model_cache.display());

        if model_cache.exists() {
            match get_dir_size(model_cache) {
                Ok(size) => println!("  Local cache size: {}", format_size(size)),
                Err(e) => warn!("  Could not calculate cache size: {}", e),
            }
        } else {
            println!("  Local cache: (empty)");
        }

        if let Some(home) = dirs::home_dir() {
            let hf_cache = home.join(".cache/huggingface/hub");
            if hf_cache.exists() {
                match get_dir_size(&hf_cache) {
                    Ok(size) => println!("  HuggingFace cache: {}", format_size(size)),
                    Err(e) => warn!("  Could not calculate HF cache size: {}", e),
                }
            }
        }
    }

    if clear {
        println!("Clearing cache...");

        if model_cache.exists() {
            std::fs::remove_dir_all(model_cache)?;
            println!("Cleared local cache: {}", model_cache.display());
        }

        println!("\nHuggingFace cache not cleared (managed by hf-hub)");
        println!("To clear HF cache manually, delete: ~/.cache/huggingface/hub");
    }

    Ok(())
}

fn list_models(_model_cache: &PathBuf, detailed: bool) -> Result<()> {
    let models = built_in_model_catalog();
    println!("Available Models ({} in registry):\n", models.len());

    for model in &models {
        if detailed {
            println!("  {} [{}]", model.name, model.size);
            println!("    Family: {}", model.family);
            println!("    Format: {}", model.format);
            println!("    Context: {} tokens", model.context_length);
            println!(
                "    Auth required: {}",
                if model.requires_auth { "yes" } else { "no" }
            );
            println!("    {}\n", model.description);
        } else {
            let auth_marker = if model.requires_auth { " 🔑" } else { "" };
            println!(
                "  {:30} {:6}  {}{}",
                model.name, model.size, model.description, auth_marker
            );
        }
    }

    println!();
    Ok(())
}

fn setup_wasm_toolchain(
    check: bool,
    skip_deps_update: bool,
    skip_target: bool,
    skip_wasm_tools: bool,
) -> Result<()> {
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
    print_status(
        "wasm32-wasip1 target",
        target_installed,
        "Rust stdlib for WASI Preview 1",
    );
    print_status("wasm-tools", wasm_tools_installed, "used for component embed/new");
    print_status(
        "cargo dependency refresh",
        !skip_deps_update,
        "runs `cargo update` before rebuilding",
    );

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

    if !skip_deps_update {
        anyhow::ensure!(cargo_available, "cargo is not available in PATH");
        run_bootstrap_command("Refreshing Cargo dependencies", "cargo", &["update"])?;
    }

    println!("\nWASM setup complete.");
    println!("You can now build guest modules with `cargo build --target wasm32-wasip1 --release`.");
    Ok(())
}

fn show_info() -> Result<()> {
    println!("Ferrite Runtime");
    println!("  Version: {}", env!("CARGO_PKG_VERSION"));
    println!("  Homepage: {}", env!("CARGO_PKG_HOMEPAGE"));
    println!();
    println!("System:");
    println!("  OS: {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);
    println!(
        "  nvidia-smi: {}",
        if command_exists("nvidia-smi") {
            "available"
        } else {
            "missing"
        }
    );
    println!(
        "  runner binary: {}",
        find_runner_binary()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(not installed beside ferrite-rt)".to_string())
    );

    let backend = std::env::var("FERRITE_BACKEND")
        .ok()
        .or_else(|| std::env::var("FERRITE_INFERENCE_BACKEND").ok())
        .unwrap_or_else(|| "auto".to_string());
    println!("  Backend policy: {}", backend);
    println!(
        "  Require CUDA: {}",
        if std::env::var("FERRITE_REQUIRE_CUDA").ok().as_deref() == Some("1") {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  Runner fallback features: {}",
        std::env::var("FERRITE_RUNNER_FEATURES").unwrap_or_else(|_| "runtime,cuda".to_string())
    );

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DoctorStatus {
    Pass,
    Warn,
    Fail,
}

impl DoctorStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug)]
struct DoctorCheck {
    name: String,
    status: DoctorStatus,
    detail: String,
}

impl DoctorCheck {
    fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), status: DoctorStatus::Pass, detail: detail.into() }
    }

    fn warn(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), status: DoctorStatus::Warn, detail: detail.into() }
    }

    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), status: DoctorStatus::Fail, detail: detail.into() }
    }
}

#[derive(Debug)]
struct DoctorReport {
    checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    fn new() -> Self {
        Self { checks: Vec::new() }
    }

    fn push(&mut self, check: DoctorCheck) {
        self.checks.push(check);
    }

    fn failures(&self) -> usize {
        self.checks.iter().filter(|check| check.status == DoctorStatus::Fail).count()
    }

    fn warnings(&self) -> usize {
        self.checks.iter().filter(|check| check.status == DoctorStatus::Warn).count()
    }

    fn print(&self, profile: &str, repo_root: Option<&Path>) {
        println!("Ferrite doctor");
        println!("==============");
        println!("profile    {}", profile);
        println!(
            "repo root   {}",
            repo_root
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "(not found)".to_string())
        );
        println!();

        for check in &self.checks {
            println!("{:20} {:5} {}", check.name, check.status.label(), check.detail);
        }

        println!();
        println!(
            "summary     {} pass  {} warn  {} fail",
            self.checks.len().saturating_sub(self.warnings() + self.failures()),
            self.warnings(),
            self.failures()
        );
    }
}

fn run_doctor(profile: &str, repo_root: Option<&Path>, model_cache: &Path, strict: bool) -> Result<()> {
    let profile = normalize_install_profile(profile)?;
    let repo_root = repo_root
        .map(PathBuf::from)
        .or_else(|| find_repo_root().ok().flatten())
        .or_else(default_install_repo_root);

    let mut report = DoctorReport::new();
    report.push(check_command("cargo", true, "required for workspace builds and setup"));
    report.push(check_command("rustup", true, "required for wasm target installation"));
    report.push(check_command("wasm-tools", true, "required for component embedding"));
    report.push(check_command("git", false, "recommended for repo updates"));
    report.push(check_either_command(&["curl", "wget"], true, "required for bootstrap downloads"));

    let rustup_available = command_exists("rustup");
    match rustup_available {
        true => match rustup_target_installed("wasm32-wasip1") {
            Ok(true) => report.push(DoctorCheck::pass("wasm target", "wasm32-wasip1 installed")),
            Ok(false) => report.push(DoctorCheck::fail("wasm target", "missing wasm32-wasip1; run `ferrite-rt setup`")),
            Err(err) => report.push(DoctorCheck::fail("wasm target", format!("unable to query rustup targets: {err}"))),
        },
        false => report.push(DoctorCheck::fail("wasm target", "cannot verify without rustup in PATH")),
    }

    report.push(check_model_cache(model_cache));
    report.push(check_runtime_binary());
    report.push(check_runner_availability());
    report.push(check_cuda_runtime());

    if let Some(repo_root) = repo_root.as_deref() {
        add_repo_checks(&mut report, repo_root, profile);
    } else {
        report.push(DoctorCheck::warn(
            "repo layout",
            "repo root not found; skipped artifact validation",
        ));
    }

    report.print(profile, repo_root.as_deref());

    let failures = report.failures();
    let warnings = report.warnings();
    if failures > 0 {
        anyhow::bail!("doctor found {failures} failing check(s)");
    }
    if strict && warnings > 0 {
        anyhow::bail!("doctor found {warnings} warning(s) in strict mode");
    }

    Ok(())
}

fn normalize_install_profile(profile: &str) -> Result<&'static str> {
    match profile {
        "runtime" => Ok("runtime"),
        "platform" => Ok("platform"),
        "full" => Ok("full"),
        "mega" => Ok("mega"),
        other => anyhow::bail!("unknown install profile `{other}`"),
    }
}

fn find_repo_root() -> Result<Option<PathBuf>> {
    let current = std::env::current_dir().context("failed to read current directory")?;
    Ok(current
        .ancestors()
        .find(|path| path.join("install.sh").is_file() && path.join("Cargo.toml").is_file())
        .map(Path::to_path_buf))
}

fn default_install_repo_root() -> Option<PathBuf> {
    dirs::home_dir().and_then(|home| {
        let standalone = home.join(".local/share/ferrite-llm/src/ferrite-llm");
        if standalone.is_dir() {
            return Some(standalone);
        }
        let monorepo = home.join(".local/share/ferrite/src/ferrite");
        if monorepo.is_dir() {
            return Some(monorepo);
        }
        Some(standalone)
    })
}

fn add_repo_checks(report: &mut DoctorReport, repo_root: &Path, profile: &str) {
    report.push(check_path_exists(
        "install script",
        &repo_root.join("install.sh"),
        true,
        "installer entrypoint",
    ));
    report.push(check_path_exists(
        "workspace root",
        &repo_root.join("Cargo.toml"),
        true,
        "root Cargo workspace manifest",
    ));
    report.push(check_path_exists(
        "native runtime",
        &repo_root.join("ferrite-os/lib/libptx_os.so"),
        true,
        "native PTX runtime library",
    ));
    report.push(check_path_exists(
        "sample wasm",
        &repo_root.join("target/wasm32-wasip1/release/mistral_inference.component.wasm"),
        true,
        "sample guest component",
    ));

    if matches!(profile, "platform" | "full" | "mega") {
        report.push(check_path_exists(
            "platform src",
            &repo_root.join("ferrite-os/Cargo.toml"),
            true,
            "ferrite-os workspace manifest",
        ));
    }

    if matches!(profile, "full" | "mega") {
        report.push(check_path_exists(
            "gpu-lang src",
            &repo_root.join("ferrite-gpu-lang/Cargo.toml"),
            true,
            "ferrite-gpu-lang manifest",
        ));
    }

    if profile == "mega" {
        report.push(check_command("cmake", true, "required for ferrite-graphics"));
        report.push(check_command("ctest", true, "required for ferrite-graphics tests"));
        report.push(check_command("ffmpeg", true, "required for ferrite-graphics workflows"));
        report.push(check_path_exists(
            "graphics src",
            &repo_root.join("external/ferrite-graphics/CMakeLists.txt"),
            true,
            "ferrite-graphics CMake project",
        ));
    }
}

fn check_command(name: &str, required: bool, detail: &str) -> DoctorCheck {
    if command_exists(name) {
        DoctorCheck::pass(name, detail)
    } else if required {
        DoctorCheck::fail(name, format!("missing; {detail}"))
    } else {
        DoctorCheck::warn(name, format!("missing; {detail}"))
    }
}

fn check_either_command(names: &[&str], required: bool, detail: &str) -> DoctorCheck {
    if let Some(found) = names.iter().copied().find(|name| command_exists(name)) {
        DoctorCheck::pass(found, detail)
    } else if required {
        DoctorCheck::fail(names.join("/"), format!("missing; {detail}"))
    } else {
        DoctorCheck::warn(names.join("/"), format!("missing; {detail}"))
    }
}

fn check_path_exists(name: &str, path: &Path, required: bool, detail: &str) -> DoctorCheck {
    if path.exists() {
        DoctorCheck::pass(name, format!("{} ({detail})", path.display()))
    } else if required {
        DoctorCheck::fail(name, format!("missing {} ({detail})", path.display()))
    } else {
        DoctorCheck::warn(name, format!("missing {} ({detail})", path.display()))
    }
}

fn check_model_cache(model_cache: &Path) -> DoctorCheck {
    if model_cache.exists() {
        if model_cache.is_dir() {
            DoctorCheck::pass("model cache", format!("{} exists", model_cache.display()))
        } else {
            DoctorCheck::fail("model cache", format!("{} exists but is not a directory", model_cache.display()))
        }
    } else if let Some(parent) = model_cache.parent() {
        if parent.exists() {
            DoctorCheck::warn("model cache", format!("{} missing; parent exists so runtime can create it", model_cache.display()))
        } else {
            DoctorCheck::fail(
                "model cache",
                format!("{} missing and parent {} does not exist", model_cache.display(), parent.display()),
            )
        }
    } else {
        DoctorCheck::fail("model cache", format!("{} has no valid parent directory", model_cache.display()))
    }
}

fn check_runtime_binary() -> DoctorCheck {
    match std::env::current_exe() {
        Ok(path) => DoctorCheck::pass("runtime bin", format!("{}", path.display())),
        Err(err) => DoctorCheck::warn("runtime bin", format!("unable to resolve current executable: {err}")),
    }
}

fn check_runner_availability() -> DoctorCheck {
    if let Some(path) = find_runner_binary() {
        DoctorCheck::pass("runner bin", format!("{}", path.display()))
    } else if find_repo_root().ok().flatten().is_some() {
        DoctorCheck::warn("runner bin", "not installed beside ferrite-rt; repo cargo fallback will be used")
    } else {
        DoctorCheck::fail("runner bin", "ferrite-rt-runner not found beside ferrite-rt")
    }
}

fn check_cuda_runtime() -> DoctorCheck {
    if command_exists("nvidia-smi") {
        DoctorCheck::pass("cuda runtime", "`nvidia-smi` is available")
    } else {
        DoctorCheck::warn("cuda runtime", "`nvidia-smi` is missing; CUDA execution may be unavailable")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelEntry {
    name: String,
    family: String,
    format: String,
    context_length: usize,
    description: String,
    size: String,
    requires_auth: bool,
}

fn built_in_model_catalog() -> Vec<ModelEntry> {
    let source = include_str!("../../ferrite-core/src/registry/catalog.rs");
    let mut models = Vec::new();
    let mut current = Vec::new();
    let mut in_spec = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed == "self.register(ModelSpec {" {
            in_spec = true;
            current.clear();
            continue;
        }

        if in_spec && trimmed == "});" {
            if let Some(model) = parse_model_entry(&current) {
                models.push(model);
            }
            in_spec = false;
            continue;
        }

        if in_spec {
            current.push(trimmed.to_string());
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

fn parse_model_entry(lines: &[String]) -> Option<ModelEntry> {
    Some(ModelEntry {
        name: parse_string_field(lines, "name")?,
        family: parse_enum_field(lines, "family")?,
        format: parse_enum_field(lines, "format")?,
        context_length: parse_usize_field(lines, "context_length")?,
        description: parse_string_field(lines, "description")?,
        size: parse_string_field(lines, "size")?,
        requires_auth: parse_bool_field(lines, "requires_auth")?,
    })
}

fn parse_string_field(lines: &[String], field: &str) -> Option<String> {
    let prefix = format!("{field}: ");
    lines.iter().find_map(|line| {
        let rest = line.strip_prefix(&prefix)?;
        let value = rest.strip_suffix(".into(),")?;
        value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .map(ToString::to_string)
    })
}

fn parse_enum_field(lines: &[String], field: &str) -> Option<String> {
    let prefix = format!("{field}: ");
    lines.iter().find_map(|line| {
        let rest = line.strip_prefix(&prefix)?;
        let value = rest.strip_suffix(',')?;
        value.rsplit("::").next().map(ToString::to_string)
    })
}

fn parse_usize_field(lines: &[String], field: &str) -> Option<usize> {
    let prefix = format!("{field}: ");
    lines.iter().find_map(|line| {
        let rest = line.strip_prefix(&prefix)?;
        let value = rest.strip_suffix(',')?;
        value.parse::<usize>().ok()
    })
}

fn parse_bool_field(lines: &[String], field: &str) -> Option<bool> {
    let prefix = format!("{field}: ");
    lines.iter().find_map(|line| {
        let rest = line.strip_prefix(&prefix)?;
        let value = rest.strip_suffix(',')?;
        value.parse::<bool>().ok()
    })
}

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
        .flat_map(|unparsed| std::env::split_paths(&unparsed).collect::<Vec<_>>())
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

#[cfg(test)]
mod tests {
    use super::{
        built_in_model_catalog, check_model_cache, normalize_install_profile, prepend_env_paths,
        DoctorStatus,
    };
    use std::path::PathBuf;

    #[test]
    fn normalize_install_profile_accepts_known_profiles() {
        assert_eq!(normalize_install_profile("runtime").unwrap(), "runtime");
        assert_eq!(normalize_install_profile("platform").unwrap(), "platform");
        assert_eq!(normalize_install_profile("full").unwrap(), "full");
        assert_eq!(normalize_install_profile("mega").unwrap(), "mega");
    }

    #[test]
    fn normalize_install_profile_rejects_unknown_profiles() {
        assert!(normalize_install_profile("everything").is_err());
    }

    #[test]
    fn model_cache_missing_under_existing_parent_is_warning() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("models");
        let check = check_model_cache(cache.as_path());
        assert_eq!(check.status, DoctorStatus::Warn);
    }

    #[test]
    fn built_in_model_catalog_parses_known_entry() {
        let models = built_in_model_catalog();
        assert!(models.iter().any(|model| model.name == "mistral-7b-q4"));
    }

    #[test]
    fn prepend_env_paths_puts_new_entries_first() {
        let extra = vec![PathBuf::from("/tmp/ferrite-lib")];
        let joined = std::env::join_paths([PathBuf::from("/usr/lib"), PathBuf::from("/opt/lib")])
            .unwrap();
        std::env::set_var("FERRITE_TEST_PATHS", &joined);
        let paths = prepend_env_paths("FERRITE_TEST_PATHS", &extra);
        std::env::remove_var("FERRITE_TEST_PATHS");
        assert_eq!(
            paths,
            vec![
                PathBuf::from("/tmp/ferrite-lib"),
                PathBuf::from("/usr/lib"),
                PathBuf::from("/opt/lib")
            ]
        );
    }
}
