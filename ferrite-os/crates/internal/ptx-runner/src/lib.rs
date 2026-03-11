use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TargetKind {
    Example,
    Bin,
    Script,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EntryKind {
    Main,
    FerriteMain,
}

#[derive(Debug, Clone)]
struct Target {
    name: String,
    crate_name: Option<String>,
    kind: TargetKind,
    path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct CrateContext {
    crate_name: String,
    crate_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RunFileRequest {
    pub path: PathBuf,
    pub entry: Option<String>,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RunEntryRequest {
    pub id: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunEntry {
    pub id: String,
    pub crate_name: String,
    pub path: String,
    pub entry: String,
    pub is_default: bool,
}

struct TempExampleGuard {
    path: PathBuf,
}

pub struct PreparedRunCommand {
    command: Command,
    _guard: Option<TempExampleGuard>,
}

impl PreparedRunCommand {
    pub fn command_mut(&mut self) -> &mut Command {
        &mut self.command
    }

    pub fn run(self) -> Result<(), String> {
        run_command(self.command)
    }
}

impl Drop for TempExampleGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn run_cli() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();

    let workspace_root = find_workspace_root(env::current_dir().map_err(|e| e.to_string())?)
        .ok_or_else(|| "Unable to locate workspace root (Cargo.toml with [workspace])".to_string())?;
    let repo_root = workspace_root.parent().unwrap_or(&workspace_root).to_path_buf();

    if args.is_empty() || args[0] == "list" || args[0] == "--list" || args[0] == "-l" {
        let targets = discover_targets(&workspace_root, &repo_root)?;
        print_targets(&targets);
        return Ok(());
    }

    match args[0].as_str() {
        "run-file" => {
            let req = parse_run_file_request(&args[1..])?;
            run_file(&workspace_root, &repo_root, req)?;
            return Ok(());
        }
        "run-list" => {
            let entries = discover_run_entries(&workspace_root)?;
            let response = serde_json::json!({
                "ok": true,
                "count": entries.len(),
                "entries": entries,
            });
            println!("{}", serde_json::to_string(&response).map_err(|e| e.to_string())?);
            return Ok(());
        }
        "run-entry" => {
            let req = parse_run_entry_request(&args[1..])?;
            run_entry(&workspace_root, &repo_root, req)?;
            return Ok(());
        }
        _ => {}
    }

    let targets = discover_targets(&workspace_root, &repo_root)?;
    let (target_key, passthrough) = (args[0].as_str(), &args[1..]);
    let (crate_hint, name) = parse_target_hint(target_key);

    let matches = resolve_target(&targets, crate_hint.as_deref(), name);
    if matches.is_empty() {
        print_targets(&targets);
        return Err(format!("Unknown target '{target_key}'"));
    }
    if matches.len() > 1 {
        eprintln!("Ambiguous target '{target_key}'. Please disambiguate with crate:name.");
        for t in matches {
            eprintln!("  {}:{} ({:?})", t.crate_name.clone().unwrap_or_default(), t.name, t.kind);
        }
        return Err("Ambiguous target".to_string());
    }

    let target = &matches[0];
    match target.kind {
        TargetKind::Example => run_cargo_example(&workspace_root, &repo_root, target, passthrough)?,
        TargetKind::Bin => run_cargo_bin(&workspace_root, &repo_root, target, passthrough)?,
        TargetKind::Script => run_script(&workspace_root, &repo_root, target, passthrough)?,
    }

    Ok(())
}

pub fn parse_run_file_request(args: &[String]) -> Result<RunFileRequest, String> {
    if args.is_empty() {
        return Err("usage: run-file <path> [--entry <name>] [-- <args...>]".to_string());
    }

    let (head, tail) = split_passthrough(args);
    let mut entry: Option<String> = None;
    let mut idx = 1;

    while idx < head.len() {
        match head[idx].as_str() {
            "--entry" => {
                let value = head
                    .get(idx + 1)
                    .ok_or_else(|| "missing value for --entry".to_string())?;
                entry = Some(value.to_string());
                idx += 2;
            }
            flag => {
                return Err(format!("unknown flag '{flag}' (expected --entry or --)"));
            }
        }
    }

    Ok(RunFileRequest {
        path: PathBuf::from(&head[0]),
        entry,
        args: tail.to_vec(),
    })
}

pub fn parse_run_entry_request(args: &[String]) -> Result<RunEntryRequest, String> {
    if args.is_empty() {
        return Err("usage: run-entry <entry-id> [-- <args...>]".to_string());
    }
    let (head, tail) = split_passthrough(args);
    if head.len() != 1 {
        return Err("usage: run-entry <entry-id> [-- <args...>]".to_string());
    }
    Ok(RunEntryRequest {
        id: head[0].clone(),
        args: tail.to_vec(),
    })
}

fn split_passthrough(args: &[String]) -> (&[String], &[String]) {
    if let Some(idx) = args.iter().position(|arg| arg == "--") {
        (&args[..idx], &args[idx + 1..])
    } else {
        (args, &[])
    }
}

pub fn run_entry(workspace_root: &Path, repo_root: &Path, req: RunEntryRequest) -> Result<(), String> {
    prepare_run_entry_command(workspace_root, repo_root, req)?.run()
}

pub fn run_file(workspace_root: &Path, repo_root: &Path, req: RunFileRequest) -> Result<(), String> {
    prepare_run_file_command(workspace_root, repo_root, req)?.run()
}

pub fn prepare_run_entry_command(
    workspace_root: &Path,
    repo_root: &Path,
    req: RunEntryRequest,
) -> Result<PreparedRunCommand, String> {
    let (path_part, entry_part) = req
        .id
        .split_once('#')
        .ok_or_else(|| "entry-id must look like '<workspace-relative-path>#<entry>'".to_string())?;

    let run_req = RunFileRequest {
        path: workspace_root.join(path_part),
        entry: Some(entry_part.to_string()),
        args: req.args,
    };
    prepare_run_file_command(workspace_root, repo_root, run_req)
}

pub fn prepare_run_file_command(
    workspace_root: &Path,
    repo_root: &Path,
    req: RunFileRequest,
) -> Result<PreparedRunCommand, String> {
    let file_path = normalize_input_path(&req.path)?;
    if !file_path.exists() {
        return Err(format!("run-file target does not exist: {}", file_path.display()));
    }
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return Err(format!(
            "run-file expects a .rs file, got: {}",
            file_path.display()
        ));
    }

    let members = parse_workspace_members(workspace_root)?;
    let crate_ctx = match resolve_crate_context(workspace_root, &members, &file_path)? {
        Some(ctx) => ctx,
        None => fallback_runner_context(workspace_root, &members)?,
    };

    let entry_kind = resolve_entry_kind(&file_path, req.entry.as_deref())?;
    let temp_target = make_temp_example_name(&file_path, entry_kind);
    let guard = create_temp_example(&crate_ctx.crate_dir, &file_path, &temp_target, entry_kind)?;

    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("-p")
        .arg(&crate_ctx.crate_name)
        .arg("--example")
        .arg(&temp_target);
    if !req.args.is_empty() {
        cmd.arg("--");
        cmd.args(&req.args);
    }
    apply_ld_library_path(&mut cmd, repo_root);

    Ok(PreparedRunCommand {
        command: cmd,
        _guard: Some(guard),
    })
}

fn fallback_runner_context(workspace_root: &Path, members: &[String]) -> Result<CrateContext, String> {
    let preferred = members
        .iter()
        .find(|member| member.ends_with("internal/ptx-runner") || member.ends_with("crates/internal/ptx-runner"));

    if let Some(member) = preferred {
        let crate_dir = workspace_root.join(member);
        if crate_dir.exists() {
            return Ok(CrateContext {
                crate_name: read_package_name(workspace_root, member),
                crate_dir: crate_dir.canonicalize().map_err(|e| e.to_string())?,
            });
        }
    }

    for member in members {
        let crate_dir = workspace_root.join(member);
        if crate_dir.exists() {
            return Ok(CrateContext {
                crate_name: read_package_name(workspace_root, member),
                crate_dir: crate_dir.canonicalize().map_err(|e| e.to_string())?,
            });
        }
    }

    Err("No workspace crate available for run-file fallback context".to_string())
}

fn make_temp_example_name(file_path: &Path, entry_kind: EntryKind) -> String {
    let mut hasher = DefaultHasher::new();
    file_path.hash(&mut hasher);
    entry_kind.hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    nanos.hash(&mut hasher);
    format!("__ptx_runfile_{:016x}", hasher.finish())
}

/// Read a source file and convert inner doc comments (`//!`) to regular
/// comments (`//`).  `include!()` cannot expand files that contain `//!`
/// because inner doc comments are only valid at the crate root, not inside
/// a macro expansion.  Inlining the source with this conversion avoids
/// error E0753 while preserving the comment text.
fn read_source_sanitized(source_file: &Path) -> Result<String, String> {
    let source = fs::read_to_string(source_file).map_err(|e| e.to_string())?;
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//!") {
            // Convert `//! text` → `// text`, preserving leading whitespace.
            let prefix_len = line.len() - trimmed.len();
            out.push_str(&line[..prefix_len]);
            out.push_str("//");
            out.push_str(&trimmed[3..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    Ok(out)
}

fn create_temp_example(
    crate_dir: &Path,
    source_file: &Path,
    target_name: &str,
    entry_kind: EntryKind,
) -> Result<TempExampleGuard, String> {
    let examples_dir = crate_dir.join("examples");
    fs::create_dir_all(&examples_dir).map_err(|e| e.to_string())?;
    let temp_path = examples_dir.join(format!("{target_name}.rs"));

    let sanitized = read_source_sanitized(source_file)?;

    let contents = match entry_kind {
        EntryKind::Main => sanitized,
        EntryKind::FerriteMain => {
            format!(
                "{sanitized}\n\
                 trait __PtxRunnerFerriteEntryReturn {{\n\
                     fn __ptx_runner_finish(self) -> i32;\n\
                 }}\n\n\
                 impl __PtxRunnerFerriteEntryReturn for () {{\n\
                     fn __ptx_runner_finish(self) -> i32 {{\n\
                         0\n\
                     }}\n\
                 }}\n\n\
                 impl<E> __PtxRunnerFerriteEntryReturn for Result<(), E>\n\
                 where\n\
                     E: std::fmt::Display,\n\
                 {{\n\
                     fn __ptx_runner_finish(self) -> i32 {{\n\
                         match self {{\n\
                             Ok(()) => 0,\n\
                             Err(err) => {{\n\
                                 eprintln!(\"{{}}\", err);\n\
                                 1\n\
                             }}\n\
                         }}\n\
                     }}\n\
                 }}\n\n\
                 fn main() {{\n\
                     std::process::exit(__PtxRunnerFerriteEntryReturn::__ptx_runner_finish(ferrite_main()));\n\
                 }}\n",
            )
        }
    };

    fs::write(&temp_path, contents).map_err(|e| e.to_string())?;
    Ok(TempExampleGuard { path: temp_path })
}

fn normalize_input_path(path: &Path) -> Result<PathBuf, String> {
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir().map_err(|e| e.to_string())?.join(path)
    };
    full.canonicalize().map_err(|e| e.to_string())
}

fn resolve_entry_kind(path: &Path, requested: Option<&str>) -> Result<EntryKind, String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let has_main = has_function(&source, "main");
    let has_ferrite_main = has_function(&source, "ferrite_main");

    if let Some(name) = requested {
        return match name {
            "main" if has_main => Ok(EntryKind::Main),
            "ferrite_main" if has_ferrite_main => Ok(EntryKind::FerriteMain),
            "main" => Err(format!("entry 'main' not found in {}", path.display())),
            "ferrite_main" => Err(format!(
                "entry 'ferrite_main' not found in {}",
                path.display()
            )),
            _ => Err(format!(
                "unsupported entry '{name}' (supported: main, ferrite_main)"
            )),
        };
    }

    if has_main {
        Ok(EntryKind::Main)
    } else if has_ferrite_main {
        Ok(EntryKind::FerriteMain)
    } else {
        Err(format!(
            "No entrypoint found in {} (expected fn main() or fn ferrite_main(...))",
            path.display()
        ))
    }
}

fn has_function(source: &str, fn_name: &str) -> bool {
    let candidates = [
        format!("fn {fn_name}"),
        format!("pub fn {fn_name}"),
        format!("pub(crate) fn {fn_name}"),
        format!("pub(super) fn {fn_name}"),
        format!("pub(in crate) fn {fn_name}"),
        format!("async fn {fn_name}"),
        format!("pub async fn {fn_name}"),
        format!("pub(crate) async fn {fn_name}"),
        format!("pub(super) async fn {fn_name}"),
        format!("pub(in crate) async fn {fn_name}"),
    ];

    source.lines().any(|line| {
        let trimmed = line.trim_start();
        if trimmed.contains("\\n\\") {
            return false;
        }
        candidates.iter().any(|needle| trimmed.starts_with(needle))
            || (trimmed.starts_with("pub(in") && trimmed.contains(&format!("fn {fn_name}")))
    })
}

pub fn discover_run_entries(workspace_root: &Path) -> Result<Vec<RunEntry>, String> {
    let members = parse_workspace_members(workspace_root)?;
    let mut entries = Vec::new();

    for member in members {
        let crate_dir = workspace_root.join(&member);
        if !crate_dir.exists() {
            continue;
        }
        let crate_name = read_package_name(workspace_root, &member);
        let mut files = Vec::new();
        collect_rs_files(&crate_dir, &mut files)?;
        files.sort();

        for file in files {
            let source = match fs::read_to_string(&file) {
                Ok(src) => src,
                Err(_) => continue,
            };
            let has_main = has_function(&source, "main");
            let has_ferrite_main = has_function(&source, "ferrite_main");
            if !has_main && !has_ferrite_main {
                continue;
            }

            let rel_path = file
                .strip_prefix(workspace_root)
                .unwrap_or(&file)
                .to_string_lossy()
                .to_string();

            if has_main {
                entries.push(RunEntry {
                    id: format!("{rel_path}#main"),
                    crate_name: crate_name.clone(),
                    path: rel_path.clone(),
                    entry: "main".to_string(),
                    is_default: true,
                });
            }
            if has_ferrite_main {
                entries.push(RunEntry {
                    id: format!("{rel_path}#ferrite_main"),
                    crate_name: crate_name.clone(),
                    path: rel_path,
                    entry: "ferrite_main".to_string(),
                    is_default: !has_main,
                });
            }
        }
    }

    entries.sort_by_key(|entry| entry.id.clone());
    Ok(entries)
}

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut dir_entries: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .collect();
    dir_entries.sort();

    for path in dir_entries {
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if matches!(name, "target" | ".git" | ".idea") {
                continue;
            }
            collect_rs_files(&path, out)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn resolve_crate_context(
    workspace_root: &Path,
    members: &[String],
    file_path: &Path,
) -> Result<Option<CrateContext>, String> {
    let mut best: Option<CrateContext> = None;
    let mut best_depth = 0usize;
    let canonical_file = file_path.canonicalize().map_err(|e| e.to_string())?;

    for member in members {
        let crate_dir = workspace_root.join(member);
        if !crate_dir.exists() {
            continue;
        }
        let canonical_dir = match crate_dir.canonicalize() {
            Ok(path) => path,
            Err(_) => continue,
        };
        if !canonical_file.starts_with(&canonical_dir) {
            continue;
        }
        let depth = canonical_dir.components().count();
        if depth >= best_depth {
            best_depth = depth;
            best = Some(CrateContext {
                crate_name: read_package_name(workspace_root, member),
                crate_dir: canonical_dir,
            });
        }
    }

    Ok(best)
}

fn parse_target_hint(input: &str) -> (Option<String>, String) {
    if let Some((a, b)) = input.split_once("::") {
        return (Some(a.to_string()), b.to_string());
    }
    if let Some((a, b)) = input.split_once(':') {
        return (Some(a.to_string()), b.to_string());
    }
    if let Some((a, b)) = input.split_once('/') {
        return (Some(a.to_string()), b.to_string());
    }
    (None, input.to_string())
}

fn resolve_target<'a>(targets: &'a [Target], crate_hint: Option<&str>, name: String) -> Vec<&'a Target> {
    targets
        .iter()
        .filter(|t| t.name == name)
        .filter(|t| match crate_hint {
            Some(hint) => t.crate_name.as_deref() == Some(hint),
            None => true,
        })
        .collect()
}

fn print_targets(targets: &[Target]) {
    println!("Available targets:");
    for kind in [TargetKind::Example, TargetKind::Bin, TargetKind::Script] {
        let mut list: Vec<&Target> = targets.iter().filter(|t| t.kind == kind).collect();
        if !list.is_empty() {
            println!("\n{:?}:", kind);
            list.sort_by_key(|t| (t.crate_name.clone().unwrap_or_default(), t.name.clone()));
            for t in list {
                let crate_name = t.crate_name.clone().unwrap_or_default();
                if crate_name.is_empty() {
                    println!("  {}", t.name);
                } else {
                    println!("  {}:{}", crate_name, t.name);
                }
            }
        }
    }

    println!("\nUsage:");
    println!("  cargo run -- <target> [args...]");
    println!("  cargo run -- list");
    println!("  cargo run -- run-file <path> [--entry <name>] [-- <args...>]");
    println!("  cargo run -- run-list");
    println!("  cargo run -- run-entry <entry-id> [-- <args...>]");
}

fn run_cargo_example(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let crate_name = target.crate_name.clone().ok_or_else(|| "Missing crate for example".to_string())?;

    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("-p")
        .arg(crate_name)
        .arg("--example")
        .arg(&target.name);

    if !args.is_empty() {
        cmd.arg("--");
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_cargo_bin(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let crate_name = target.crate_name.clone().ok_or_else(|| "Missing crate for bin".to_string())?;

    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("-p")
        .arg(crate_name)
        .arg("--bin")
        .arg(&target.name);

    if !args.is_empty() {
        cmd.arg("--");
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_script(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let path = target.path.as_ref().ok_or_else(|| "Missing script path".to_string())?;

    let mut cmd = Command::new("bash");
    cmd.current_dir(workspace_root).arg(path);
    if !args.is_empty() {
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_command(mut cmd: Command) -> Result<(), String> {
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Command failed with status {status}"))
    }
}

fn apply_ld_library_path(cmd: &mut Command, repo_root: &Path) {
    let mut dirs: Vec<PathBuf> = Vec::new();

    for dir in runtime_lib_candidates(repo_root) {
        if dir.exists() {
            dirs.push(dir);
        }
    }

    for dir in libtorch_lib_candidates(repo_root) {
        if dir.exists() {
            dirs.push(dir);
        }
    }

    ferrite_platform::dylib_env::apply_dylib_path(cmd, &dirs);
}

fn runtime_lib_candidates(repo_root: &Path) -> Vec<PathBuf> {
    vec![
        repo_root.join("ferrite-os/lib"),
        repo_root.join("lib"),
    ]
}

fn libtorch_lib_candidates(repo_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Ok(root) = env::var("LIBTORCH") {
        let env_path = PathBuf::from(root).join("lib");
        out.push(env_path);
    }

    out.push(repo_root.join("external/libtorch/lib"));
    out.push(repo_root.join("../external/libtorch/lib"));
    out
}

/// Read the actual `[package] name` from a crate's Cargo.toml.
/// Falls back to the directory name if the file can't be read or parsed.
fn read_package_name(workspace_root: &Path, member: &str) -> String {
    let cargo_toml = workspace_root.join(member).join("Cargo.toml");
    if let Ok(contents) = fs::read_to_string(&cargo_toml) {
        // Simple TOML extraction: find `name = "..."` in [package] section.
        let mut in_package = false;
        for line in contents.lines() {
            let trimmed = line.split('#').next().unwrap_or("").trim();
            if trimmed.starts_with("[package]") {
                in_package = true;
                continue;
            }
            if trimmed.starts_with('[') {
                in_package = false;
                continue;
            }
            if in_package && trimmed.starts_with("name") {
                if let Some(val) = trimmed.split('=').nth(1) {
                    let name = val.trim().trim_matches('"').trim_matches('\'');
                    if !name.is_empty() {
                        return name.to_string();
                    }
                }
            }
        }
    }
    // Fallback: derive from directory path
    member.split('/').last().unwrap_or(member).to_string()
}

pub fn find_workspace_root(start: PathBuf) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(contents) = fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return Some(cur);
                }
            }
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

fn discover_targets(workspace_root: &Path, repo_root: &Path) -> Result<Vec<Target>, String> {
    let members = parse_workspace_members(workspace_root)?;
    let mut targets: Vec<Target> = Vec::new();

    for member in members {
        let crate_dir = workspace_root.join(&member);
        if !crate_dir.exists() {
            continue;
        }
        let crate_name = read_package_name(workspace_root, &member);

        // Examples
        let examples_dir = crate_dir.join("examples");
        if examples_dir.exists() {
            for entry in read_rs_files(&examples_dir)? {
                let name = entry.file_stem().unwrap_or_default().to_string_lossy().to_string();
                targets.push(Target {
                    name,
                    crate_name: Some(crate_name.clone()),
                    kind: TargetKind::Example,
                    path: Some(entry),
                });
            }
        }

        // Bins from src/bin
        let bins_dir = crate_dir.join("src/bin");
        if bins_dir.exists() {
            for entry in read_rs_files(&bins_dir)? {
                let name = entry.file_stem().unwrap_or_default().to_string_lossy().to_string();
                targets.push(Target {
                    name,
                    crate_name: Some(crate_name.clone()),
                    kind: TargetKind::Bin,
                    path: Some(entry),
                });
            }
        }

        // Default bin from src/main.rs
        if crate_dir.join("src/main.rs").exists() {
            targets.push(Target {
                name: crate_name.clone(),
                crate_name: Some(crate_name.clone()),
                kind: TargetKind::Bin,
                path: Some(crate_dir.join("src/main.rs")),
            });
        }
    }

    // Scripts from repo_root/scripts
    let scripts_dir = repo_root.join("scripts");
    if scripts_dir.exists() {
        for entry in fs::read_dir(&scripts_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                continue;
            }
            let file_name = match path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };
            let name = file_name.strip_suffix(".sh").unwrap_or(&file_name).to_string();
            targets.push(Target {
                name,
                crate_name: None,
                kind: TargetKind::Script,
                path: Some(path),
            });
        }
    }

    Ok(targets)
}

fn read_rs_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "rs" {
                    files.push(path);
                }
            }
        }
    }
    Ok(files)
}

fn parse_workspace_members(workspace_root: &Path) -> Result<Vec<String>, String> {
    let cargo_toml = workspace_root.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml).map_err(|e| e.to_string())?;

    let mut members: Vec<String> = Vec::new();
    let mut in_members = false;

    for line in contents.lines() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
        }
        if in_members {
            let mut chars = trimmed.chars().peekable();
            while let Some(ch) = chars.next() {
                if ch == '"' {
                    let mut value = String::new();
                    while let Some(c) = chars.next() {
                        if c == '"' {
                            break;
                        }
                        value.push(c);
                    }
                    if !value.is_empty() {
                        members.push(value);
                    }
                }
            }
            if trimmed.contains(']') {
                in_members = false;
            }
        }
    }

    if members.is_empty() {
        return Err("No workspace members found".to_string());
    }

    Ok(members)
}
