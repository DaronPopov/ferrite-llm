use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::Command;

type DynError = Box<dyn Error>;

#[derive(Debug, Deserialize)]
struct GraphManifest {
    graph: GraphMeta,
    #[serde(default)]
    runtime_contract: Vec<RuntimeContract>,
    #[serde(default)]
    source_dep: Vec<SourceDep>,
    #[serde(default)]
    binary_asset: Vec<BinaryAsset>,
    #[serde(default)]
    derived_artifact: Vec<DerivedArtifact>,
}

#[derive(Debug, Deserialize)]
struct GraphMeta {
    version: u32,
    state_root: String,
    materialization_root: String,
}

#[derive(Debug, Deserialize)]
struct RuntimeContract {
    id: String,
    kind: String,
    #[serde(default)]
    source_deps: Vec<String>,
    compat: RuntimeContractCompat,
}

#[derive(Debug, Deserialize)]
struct RuntimeContractCompat {
    tch: String,
    aten_ptx: String,
    ferrite_torch: String,
}

#[derive(Debug, Deserialize)]
struct SourceDep {
    id: String,
    repo: String,
    commit: String,
    subdir: String,
    target: String,
}

#[derive(Debug, Deserialize)]
struct SourceDepState {
    resolved_rev: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BinaryAsset {
    id: String,
    contract: String,
    provider: String,
    version: String,
    kind: String,
    url: String,
    sha256: String,
    target: String,
    layout: String,
    compat: BinaryAssetCompat,
    machine: BinaryAssetMachine,
}

#[derive(Debug, Deserialize)]
struct BinaryAssetCompat {
    torch_flavor: String,
    archive: String,
    cxx11_abi: bool,
    tch: String,
    aten_ptx: String,
    ferrite_torch: String,
}

#[derive(Debug, Deserialize)]
struct BinaryAssetMachine {
    os: String,
    arch: Vec<String>,
    cuda: String,
    #[serde(default)]
    requires: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DerivedArtifact {
    id: String,
    recipe: String,
    outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FeatureManifest {
    #[serde(default)]
    feature_bundle: Vec<FeatureBundle>,
}

#[derive(Debug, Deserialize)]
struct FeatureBundle {
    id: String,
    description: String,
    #[serde(default)]
    source_deps: Vec<String>,
    #[serde(default)]
    runtime_contracts: Vec<String>,
    #[serde(default)]
    derived_artifacts: Vec<String>,
    #[serde(default)]
    validations: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HostToolsManifest {
    #[serde(default)]
    tool: Vec<HostTool>,
}

#[derive(Debug, Deserialize)]
struct HostTool {
    name: String,
    version_req: String,
    bootstrap: BootstrapMode,
    #[serde(default)]
    providers: BTreeMap<String, Provider>,
}

#[derive(Debug, Deserialize)]
struct Provider {
    #[serde(default)]
    packages: Vec<String>,
    script_url: Option<String>,
    #[serde(default)]
    script_args: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BootstrapMode {
    System,
    User,
}

#[derive(Debug, Deserialize)]
struct ProfileManifest {
    profile: Profile,
}

#[derive(Debug, Deserialize)]
struct Profile {
    id: String,
    extends: String,
    host_family: String,
    features: Vec<String>,
    validations: Vec<String>,
    #[serde(default)]
    runtime_assets: Vec<String>,
    providers: Providers,
}

#[derive(Debug, Deserialize)]
struct Providers {
    system: String,
    binary_assets: String,
}

struct InstallerContext {
    root: PathBuf,
    graph: GraphManifest,
    features: FeatureManifest,
    host_tools: HostToolsManifest,
    profile: ProfileManifest,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ferrite-installer: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), DynError> {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("plan");
    let apply_flag = args.iter().any(|arg| arg == "--apply");
    let apply = apply_flag || matches!(command, "bootstrap-host" | "bootstrap-all");
    let profile = args
        .windows(2)
        .find(|window| window[0] == "--profile")
        .map(|window| window[1].as_str())
        .map(str::to_owned)
        .unwrap_or_else(detect_default_profile);

    let root = installer_root()?;
    let ctx = load_context(&root, &profile)?;

    match command {
        "plan" => print_plan(&ctx),
        "resolve" => print_resolve(&ctx)?,
        "materialize" => materialize_layout(&ctx)?,
        "bootstrap-host" => bootstrap_host(&ctx, apply)?,
        "fetch-sources" => fetch_sources(&ctx)?,
        "fetch-assets" => fetch_assets(&ctx)?,
        "generate-env" => generate_env(&ctx)?,
        "build-profile" => build_profile(&ctx)?,
        "validate-profile" => validate_profile(&ctx)?,
        "bootstrap-all" => bootstrap_all(&ctx, apply)?,
        "detect" => print_detect(&ctx),
        "help" | "--help" | "-h" => print_help(),
        other => {
            return Err(format!("unknown command: {other}").into());
        }
    }

    Ok(())
}

fn installer_root() -> Result<PathBuf, DynError> {
    let cwd = env::current_dir()?;
    if cwd.join("installer").is_dir() {
        return Ok(cwd.join("installer"));
    }
    if cwd.join("manifests").is_dir() {
        return Ok(cwd);
    }
    Err("run from repo root or installer/".into())
}

fn load_context(root: &Path, profile_name: &str) -> Result<InstallerContext, DynError> {
    Ok(InstallerContext {
        root: root.to_path_buf(),
        graph: read_toml(&root.join("manifests/graph.toml"))?,
        features: read_toml(&root.join("manifests/features.toml"))?,
        host_tools: read_toml(&root.join("manifests/host-tools.toml"))?,
        profile: read_toml(&root.join(format!("profiles/{profile_name}.toml")))?,
    })
}

fn read_toml<T>(path: &Path) -> Result<T, DynError>
where
    T: for<'de> Deserialize<'de>,
{
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}

fn print_plan(ctx: &InstallerContext) {
    println!("profile={}", ctx.profile.profile.id);
    println!("extends={}", ctx.profile.profile.extends);
    println!("host_family={}", ctx.profile.profile.host_family);
    println!("current_host={}/{}", env::consts::OS, env::consts::ARCH);
    println!("state_root={}", ctx.graph.graph.state_root);
    println!("materialization_root={}", ctx.graph.graph.materialization_root);
    println!(
        "providers=system:{},binary_assets:{}",
        ctx.profile.profile.providers.system, ctx.profile.profile.providers.binary_assets
    );
    println!("features={}", ctx.profile.profile.features.join(","));
    println!(
        "profile_validations={}",
        ctx.profile.profile.validations.join(",")
    );
    if !ctx.profile.profile.runtime_assets.is_empty() {
        println!(
            "runtime_assets={}",
            ctx.profile.profile.runtime_assets.join(",")
        );
    }

    let selected_features = ctx
        .features
        .feature_bundle
        .iter()
        .filter(|bundle| ctx.profile.profile.features.contains(&bundle.id));

    for bundle in selected_features {
        println!("feature.{}={}", bundle.id, bundle.description);
        if !bundle.source_deps.is_empty() {
            println!("  source_deps={}", bundle.source_deps.join(","));
        }
        if !bundle.runtime_contracts.is_empty() {
            println!(
                "  runtime_contracts={}",
                bundle.runtime_contracts.join(",")
            );
        }
        if !bundle.derived_artifacts.is_empty() {
            println!("  derived_artifacts={}", bundle.derived_artifacts.join(","));
        }
        if !bundle.validations.is_empty() {
            println!("  validations={}", bundle.validations.join(","));
        }
    }

    println!("host_tools={}", ctx.host_tools.tool.len());
    for tool in &ctx.host_tools.tool {
        println!(
            "tool.{}={} via {}",
            tool.name,
            tool.version_req,
            provider_summary(tool)
        );
    }

    println!("runtime_contracts={}", ctx.graph.runtime_contract.len());
    for contract in &ctx.graph.runtime_contract {
        println!("runtime_contract.{}={}", contract.id, contract.kind);
        if !contract.source_deps.is_empty() {
            println!("  source_deps={}", contract.source_deps.join(","));
        }
        println!(
            "  compat=tch:{},aten_ptx:{},ferrite_torch:{}",
            contract.compat.tch, contract.compat.aten_ptx, contract.compat.ferrite_torch
        );
    }

    println!("source_deps={}", ctx.graph.source_dep.len());
    for dep in &ctx.graph.source_dep {
        println!(
            "source_dep.{}={}#{}:{} -> {}",
            dep.id, dep.repo, dep.commit, dep.subdir, dep.target
        );
    }

    println!("binary_assets={}", ctx.graph.binary_asset.len());
    for asset in &ctx.graph.binary_asset {
        println!(
            "binary_asset.{}={} {} {} contract:{} -> {}",
            asset.id, asset.provider, asset.kind, asset.version, asset.contract, asset.target
        );
        println!(
            "  compat=torch_flavor:{},archive:{},cxx11_abi:{},tch:{},aten_ptx:{},ferrite_torch:{}",
            asset.compat.torch_flavor,
            asset.compat.archive,
            asset.compat.cxx11_abi,
            asset.compat.tch,
            asset.compat.aten_ptx,
            asset.compat.ferrite_torch
        );
        println!(
            "  machine=os:{},arch:{},cuda:{},requires:{}",
            asset.machine.os,
            asset.machine.arch.join("+"),
            asset.machine.cuda,
            asset.machine.requires.join("+")
        );
        println!("  layout={}", asset.layout);
        println!("  url={}", asset.url);
        println!("  sha256={}", asset.sha256);
    }

    println!("derived_artifacts={}", ctx.graph.derived_artifact.len());
    for artifact in &ctx.graph.derived_artifact {
        println!(
            "derived_artifact.{}={} -> {}",
            artifact.id,
            artifact.recipe,
            artifact.outputs.join(",")
        );
    }
}

fn print_resolve(ctx: &InstallerContext) -> Result<(), DynError> {
    let resolved_assets = selected_runtime_assets(ctx)?;

    println!("profile={}", ctx.profile.profile.id);
    println!("current_host={}/{}", env::consts::OS, env::consts::ARCH);
    println!("selected_runtime_assets={}", resolved_assets.len());

    for asset in resolved_assets {
        let contract = find_runtime_contract(ctx, &asset.contract)?;
        let host_match = asset_matches_current_host(asset);
        let contract_match = asset_satisfies_contract(asset, contract);
        println!("runtime_asset.{}={}", asset.id, asset.provider);
        println!("  contract={}", asset.contract);
        println!("  host_match={}", yes_no(host_match));
        println!("  contract_match={}", yes_no(contract_match));
        println!(
            "  machine=os:{},arch:{},cuda:{}",
            asset.machine.os,
            asset.machine.arch.join("+"),
            asset.machine.cuda
        );
        println!(
            "  compat=tch:{},aten_ptx:{},ferrite_torch:{},torch_flavor:{},archive:{},cxx11_abi:{}",
            asset.compat.tch,
            asset.compat.aten_ptx,
            asset.compat.ferrite_torch,
            asset.compat.torch_flavor,
            asset.compat.archive,
            asset.compat.cxx11_abi
        );
        println!("  url={}", asset.url);
        println!("  sha256={}", asset.sha256);
    }

    Ok(())
}

fn print_detect(ctx: &InstallerContext) {
    println!("installer_root={}", ctx.root.display());
    println!("graph_version={}", ctx.graph.graph.version);
    println!("profile={}", ctx.profile.profile.id);
    println!("current_host={}/{}", env::consts::OS, env::consts::ARCH);
    println!("jetson_detected={}", yes_no(is_jetson_host()));
    println!("bootstrap_tools={}", ctx.host_tools.tool.len());
}

fn materialize_layout(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    let state_root = repo_root.join(&ctx.graph.graph.state_root);
    let store_root = repo_root.join(&ctx.graph.graph.materialization_root);
    let store_src_root = store_root.join("src");
    let manifests_root = state_root.join("manifests");
    fs::create_dir_all(&store_src_root)?;
    fs::create_dir_all(&manifests_root)?;

    println!("repo_root={}", repo_root.display());
    println!("state_root={}", state_root.display());
    println!("store_root={}", store_root.display());

    for dep in &ctx.graph.source_dep {
        materialize_source_dep(repo_root, &store_root, &store_src_root, dep)?;
    }

    for asset in selected_runtime_assets(ctx)? {
        materialize_runtime_asset(repo_root, &state_root, asset)?;
    }

    Ok(())
}

fn bootstrap_host(ctx: &InstallerContext, apply: bool) -> Result<(), DynError> {
    let provider = &ctx.profile.profile.providers.system;
    let mut missing_system_packages = Vec::new();
    let mut apt_update_needed = false;

    println!("profile={}", ctx.profile.profile.id);
    println!("system_provider={provider}");
    println!("apply={}", yes_no(apply));

    for tool in &ctx.host_tools.tool {
        let installed = command_exists(&tool.name);
        println!("tool.{}={}", tool.name, if installed { "present" } else { "missing" });
        if installed {
            continue;
        }

        match tool.bootstrap {
            BootstrapMode::System => {
                let tool_provider = tool
                    .providers
                    .get(provider)
                    .ok_or_else(|| format!("tool {} missing provider {}", tool.name, provider))?;
                for package in &tool_provider.packages {
                    missing_system_packages.push(package.clone());
                }
                if provider == "apt" {
                    apt_update_needed = true;
                }
            }
            BootstrapMode::User => {
                install_user_tool(tool, apply)?;
            }
        }
    }

    if !missing_system_packages.is_empty() {
        missing_system_packages.sort();
        missing_system_packages.dedup();
        println!("missing_system_packages={}", missing_system_packages.join(","));
        if apply {
            install_system_packages(provider, &missing_system_packages, apt_update_needed)?;
        }
    } else {
        println!("missing_system_packages=");
    }

    Ok(())
}

fn fetch_sources(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    let state_root = repo_root.join(&ctx.graph.graph.state_root);
    let store_root = repo_root.join(&ctx.graph.graph.materialization_root);
    let repo_cache_root = store_root.join("repos");
    let materialized_root = store_root.join("materialized");
    fs::create_dir_all(&repo_cache_root)?;
    fs::create_dir_all(&materialized_root)?;

    println!("repo_root={}", repo_root.display());
    println!("repo_cache_root={}", repo_cache_root.display());
    println!("materialized_root={}", materialized_root.display());

    for dep in &ctx.graph.source_dep {
        let repo_cache = ensure_repo_cache(&repo_cache_root, &dep.repo)?;
        let resolved_rev = resolve_repo_revision(&repo_cache, &dep.commit)?;
        let dep_root = materialized_root.join(&dep.id).join(&resolved_rev);
        export_subdir(&repo_cache, &resolved_rev, &dep.subdir, &dep_root)?;
        update_source_manifest_revision(&state_root, dep, &resolved_rev)?;
        println!(
            "fetched.{}=repo:{} rev:{} subdir:{} -> {}",
            dep.id,
            repo_cache.display(),
            resolved_rev,
            dep.subdir,
            dep_root.display()
        );
    }

    Ok(())
}

fn fetch_assets(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    let state_root = repo_root.join(&ctx.graph.graph.state_root);
    let downloads_root = state_root.join("store").join("downloads");
    fs::create_dir_all(&downloads_root)?;

    println!("profile={}", ctx.profile.profile.id);
    println!("current_host={}/{}", env::consts::OS, env::consts::ARCH);
    println!("downloads_root={}", downloads_root.display());

    for asset in selected_runtime_assets(ctx)? {
        if !asset_matches_current_host(asset) {
            println!(
                "runtime_asset.{}=skipped host mismatch for {}/{}",
                asset.id,
                asset.machine.os,
                asset.machine.arch.join("+")
            );
            continue;
        }

        match asset.provider.as_str() {
            "pytorch-prebuilt" => fetch_pytorch_prebuilt_asset(repo_root, &state_root, &downloads_root, asset)?,
            "jetson-system" => detect_jetson_system_asset(&state_root, asset)?,
            other => return Err(format!("unsupported runtime asset provider: {other}").into()),
        }
    }

    Ok(())
}

fn generate_env(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    let state_root = repo_root.join(&ctx.graph.graph.state_root);
    let env_dir = state_root.join("env");
    let bin_dir = state_root.join("bin");
    fs::create_dir_all(&env_dir)?;
    fs::create_dir_all(&bin_dir)?;

    let env_template = fs::read_to_string(ctx.root.join("templates/env.sh.tpl"))?;
    let run_template = fs::read_to_string(ctx.root.join("templates/ferrite-run.tpl"))?;

    let libtorch_path = resolved_libtorch_path(&state_root, ctx)?;
    let runtime_lib_dir = repo_root.join("ferrite-os/lib");
    let libtorch_lib_dir = libtorch_path.join("lib");
    let ld_library_path = if libtorch_lib_dir.is_dir() {
        format!(
            "{}:{}",
            runtime_lib_dir.display(),
            libtorch_lib_dir.display()
        )
    } else {
        runtime_lib_dir.display().to_string()
    };

    let env_script_path = env_dir.join("profile.sh");
    let ferrite_run_path = bin_dir.join("ferrite-run");

    let env_content = env_template
        .replace("{{repo_root}}", &repo_root.display().to_string())
        .replace("{{state_root}}", &state_root.display().to_string())
        .replace("{{libtorch_path}}", &libtorch_path.display().to_string())
        .replace("{{runtime_lib_path}}", &ld_library_path);
    fs::write(&env_script_path, env_content)?;

    let run_content = run_template
        .replace("{{repo_root}}", &repo_root.display().to_string())
        .replace("{{env_script}}", &env_script_path.display().to_string());
    fs::write(&ferrite_run_path, run_content)?;
    set_executable(&ferrite_run_path)?;

    println!("env_script={}", env_script_path.display());
    println!("runner={}", ferrite_run_path.display());
    println!("libtorch_path={}", libtorch_path.display());
    println!("ld_library_path={ld_library_path}");

    Ok(())
}

fn build_profile(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    generate_env(ctx)?;
    let env_script = repo_root.join(&ctx.graph.graph.state_root).join("env/profile.sh");

    println!("profile={}", ctx.profile.profile.id);
    println!("env_script={}", env_script.display());

    run_shell_in_repo(
        repo_root,
        &env_script,
        "make -C ferrite-os lib/libptx_os.so",
        "build libptx_os.so",
    )?;
    println!("build.native-runtime=ok");

    run_shell_in_repo(
        repo_root,
        &env_script,
        "cargo build --workspace",
        "build workspace",
    )?;
    println!("build.workspace=ok");

    if profile_needs_platform(ctx) {
        run_shell_in_repo(
            repo_root,
            &env_script,
            "cargo build --manifest-path ferrite-os/Cargo.toml --workspace",
            "build ferrite-os workspace",
        )?;
        println!("build.platform=ok");
    }

    if profile_needs_torch(ctx) {
        run_shell_in_repo(
            repo_root,
            &env_script,
            "cargo build --manifest-path ferrite-gpu-lang/Cargo.toml --features torch",
            "build ferrite-gpu-lang torch",
        )?;
        println!("build.torch=ok");
    }

    if profile_has_feature(ctx, "graphics") {
        let graphics_build_dir = repo_root.join(".ferrite/build/ferrite-graphics");
        fs::create_dir_all(&graphics_build_dir)?;
        let cmake_cmd = "\
cmake -S external/ferrite-graphics -B .ferrite/build/ferrite-graphics \
  -DCMAKE_INSTALL_PREFIX=\"$PWD/.ferrite/prefix\" && \
cmake --build .ferrite/build/ferrite-graphics -j";
        run_shell_in_repo(repo_root, &env_script, cmake_cmd, "build ferrite-graphics")?;
        println!("build.graphics=ok");
    }

    Ok(())
}

fn validate_profile(ctx: &InstallerContext) -> Result<(), DynError> {
    let repo_root = ctx
        .root
        .parent()
        .ok_or("installer root has no repo parent")?;
    generate_env(ctx)?;
    let env_script = repo_root.join(&ctx.graph.graph.state_root).join("env/profile.sh");

    println!("profile={}", ctx.profile.profile.id);
    println!("env_script={}", env_script.display());

    run_shell_in_repo(
        repo_root,
        &env_script,
        "cargo check --workspace",
        "validate workspace check",
    )?;
    println!("validate.workspace-check=ok");

    if profile_needs_torch(ctx) {
        run_shell_in_repo(
            repo_root,
            &env_script,
            "cargo check --manifest-path ferrite-gpu-lang/Cargo.toml --features torch",
            "validate ferrite-gpu-lang torch check",
        )?;
        println!("validate.torch-check=ok");
    }

    Ok(())
}

fn bootstrap_all(ctx: &InstallerContext, apply: bool) -> Result<(), DynError> {
    bootstrap_host(ctx, apply)?;
    fetch_sources(ctx)?;
    materialize_layout(ctx)?;
    fetch_assets(ctx)?;
    generate_env(ctx)?;
    build_profile(ctx)?;
    validate_profile(ctx)?;
    Ok(())
}

fn selected_runtime_assets<'a>(ctx: &'a InstallerContext) -> Result<Vec<&'a BinaryAsset>, DynError> {
    let mut resolved = Vec::new();
    for asset_id in &ctx.profile.profile.runtime_assets {
        let asset = ctx
            .graph
            .binary_asset
            .iter()
            .find(|asset| asset.id == *asset_id)
            .ok_or_else(|| format!("profile references unknown runtime asset: {asset_id}"))?;
        resolved.push(asset);
    }
    Ok(resolved)
}

fn materialize_source_dep(
    repo_root: &Path,
    store_root: &Path,
    store_src_root: &Path,
    dep: &SourceDep,
) -> Result<(), DynError> {
    let target_path = repo_root.join(&dep.target);
    let manifest_path = store_src_root.join(format!("{}.toml", dep.id));

    if !manifest_path.exists() {
        write_source_manifest(&manifest_path, dep, None)?;
    }

    let materialized_view = read_source_manifest(&manifest_path)?
        .resolved_rev
        .map(|rev| store_root.join("materialized").join(&dep.id).join(rev))
        .filter(|path| path.exists())
        .unwrap_or_else(|| {
            let placeholder = store_src_root.join(&dep.id);
            let _ = fs::create_dir_all(&placeholder);
            placeholder
        });

    if target_path.exists() {
        let target_meta = fs::symlink_metadata(&target_path)?;
        if target_meta.file_type().is_symlink()
            && fs::read_link(&target_path)
                .map(|link| link == materialized_view)
                .unwrap_or(false)
        {
            println!(
                "source_dep.{}=ready store:{} target:{}",
                dep.id,
                materialized_view.display(),
                target_path.display()
            );
            return Ok(());
        }

        remove_existing_path(&target_path)?;
    }

    create_symlink(&materialized_view, &target_path)?;
    println!(
        "source_dep.{}=created-managed-view store:{} target:{}",
        dep.id,
        materialized_view.display(),
        target_path.display()
    );
    Ok(())
}

fn materialize_runtime_asset(
    repo_root: &Path,
    state_root: &Path,
    asset: &BinaryAsset,
) -> Result<(), DynError> {
    let target_path = repo_root.join(&asset.target);
    let asset_root = state_root.join("assets").join(&asset.id);
    let manifest_path = asset_root.join("asset.toml");
    fs::create_dir_all(&asset_root)?;
    write_runtime_asset_manifest(&manifest_path, asset)?;

    if target_path.exists() {
        println!(
            "runtime_asset.{}=target-present target:{} asset_root:{}",
            asset.id,
            target_path.display(),
            asset_root.display()
        );
        return Ok(());
    }

    create_symlink(&asset_root, &target_path)?;
    println!(
        "runtime_asset.{}=created-managed-view target:{} -> {}",
        asset.id,
        target_path.display(),
        asset_root.display()
    );
    Ok(())
}

fn fetch_pytorch_prebuilt_asset(
    repo_root: &Path,
    state_root: &Path,
    downloads_root: &Path,
    asset: &BinaryAsset,
) -> Result<(), DynError> {
    let asset_root = state_root.join("assets").join(&asset.id);
    let payload_path = asset_payload_path(state_root, asset);
    fs::create_dir_all(&asset_root)?;

    if payload_path.exists() {
        update_runtime_alias(state_root, asset, &payload_path)?;
        println!(
            "runtime_asset.{}=ready payload:{}",
            asset.id,
            payload_path.display()
        );
        return Ok(());
    }

    let legacy_libtorch = repo_root.join("external/libtorch");
    if legacy_libtorch.exists() {
        create_symlink(&legacy_libtorch, &payload_path)?;
        update_runtime_alias(state_root, asset, &payload_path)?;
        write_asset_state(asset_root.join("state.toml"), asset, "adopted-legacy", &legacy_libtorch)?;
        println!(
            "runtime_asset.{}=adopted-legacy payload:{} -> {}",
            asset.id,
            payload_path.display(),
            legacy_libtorch.display()
        );
        return Ok(());
    }

    let archive_path = downloads_root.join(format!("{}.zip", asset.id));
    if !archive_path.exists() {
        run_command(download_command(&asset.url, &archive_path), Some("download libtorch"))?;
    }
    let observed_sha = compute_sha256(&archive_path)?;
    if asset.sha256 != "declare-me" && asset.sha256 != observed_sha {
        return Err(format!(
            "sha256 mismatch for {}: expected {}, got {}",
            asset.id, asset.sha256, observed_sha
        )
        .into());
    }

    let unpack_root = asset_root.join("unpacked");
    if !unpack_root.exists() {
        fs::create_dir_all(&unpack_root)?;
        extract_zip_with_python(&archive_path, &unpack_root)?;
    }
    let extracted_libtorch = unpack_root.join("libtorch");
    let final_payload = if extracted_libtorch.is_dir() {
        extracted_libtorch
    } else {
        unpack_root.clone()
    };
    if payload_path.exists() {
        remove_existing_path(&payload_path)?;
    }
    create_symlink(&final_payload, &payload_path)?;
    update_runtime_alias(state_root, asset, &payload_path)?;
    write_asset_state(asset_root.join("state.toml"), asset, "downloaded", &archive_path)?;
    println!(
        "runtime_asset.{}=downloaded payload:{} archive:{} observed_sha:{}",
        asset.id,
        payload_path.display(),
        archive_path.display(),
        observed_sha
    );
    Ok(())
}

fn detect_jetson_system_asset(state_root: &Path, asset: &BinaryAsset) -> Result<(), DynError> {
    let asset_root = state_root.join("assets").join(&asset.id);
    fs::create_dir_all(&asset_root)?;
    let candidates = jetson_libtorch_candidates();
    for candidate in candidates {
        if looks_like_libtorch_root(&candidate) {
            let payload_path = asset_payload_path(state_root, asset);
            if payload_path.exists() {
                remove_existing_path(&payload_path)?;
            }
            create_symlink(&candidate, &payload_path)?;
            update_runtime_alias(state_root, asset, &payload_path)?;
            write_asset_state(asset_root.join("state.toml"), asset, "detected-system", &candidate)?;
            println!(
                "runtime_asset.{}=detected-system payload:{} -> {}",
                asset.id,
                payload_path.display(),
                candidate.display()
            );
            return Ok(());
        }
    }

    println!(
        "runtime_asset.{}=not-found expected one of:{}",
        asset.id,
        jetson_libtorch_candidates()
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

fn find_runtime_contract<'a>(
    ctx: &'a InstallerContext,
    contract_id: &str,
) -> Result<&'a RuntimeContract, DynError> {
    ctx.graph
        .runtime_contract
        .iter()
        .find(|contract| contract.id == contract_id)
        .ok_or_else(|| format!("runtime asset references unknown contract: {contract_id}").into())
}

fn asset_matches_current_host(asset: &BinaryAsset) -> bool {
    asset.machine.os == env::consts::OS && asset.machine.arch.iter().any(|arch| arch == env::consts::ARCH)
}

fn asset_satisfies_contract(asset: &BinaryAsset, contract: &RuntimeContract) -> bool {
    asset.compat.tch == contract.compat.tch
        && asset.compat.aten_ptx == contract.compat.aten_ptx
        && asset.compat.ferrite_torch == contract.compat.ferrite_torch
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn write_source_manifest(path: &Path, dep: &SourceDep, resolved_rev: Option<&str>) -> Result<(), DynError> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "id = {:?}", dep.id)?;
    writeln!(file, "repo = {:?}", dep.repo)?;
    writeln!(file, "commit = {:?}", dep.commit)?;
    writeln!(file, "subdir = {:?}", dep.subdir)?;
    writeln!(file, "target = {:?}", dep.target)?;
    if let Some(resolved_rev) = resolved_rev {
        writeln!(file, "resolved_rev = {:?}", resolved_rev)?;
    }
    Ok(())
}

fn read_source_manifest(path: &Path) -> Result<SourceDepState, DynError> {
    read_toml(path)
}

fn write_runtime_asset_manifest(path: &Path, asset: &BinaryAsset) -> Result<(), DynError> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "id = {:?}", asset.id)?;
    writeln!(file, "contract = {:?}", asset.contract)?;
    writeln!(file, "provider = {:?}", asset.provider)?;
    writeln!(file, "version = {:?}", asset.version)?;
    writeln!(file, "kind = {:?}", asset.kind)?;
    writeln!(file, "url = {:?}", asset.url)?;
    writeln!(file, "sha256 = {:?}", asset.sha256)?;
    writeln!(file, "target = {:?}", asset.target)?;
    writeln!(file, "layout = {:?}", asset.layout)?;
    Ok(())
}

fn write_asset_state(
    path: PathBuf,
    asset: &BinaryAsset,
    status: &str,
    source: &Path,
) -> Result<(), DynError> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "id = {:?}", asset.id)?;
    writeln!(file, "provider = {:?}", asset.provider)?;
    writeln!(file, "status = {:?}", status)?;
    writeln!(file, "source = {:?}", source.display().to_string())?;
    Ok(())
}

fn update_runtime_alias(state_root: &Path, asset: &BinaryAsset, payload_path: &Path) -> Result<(), DynError> {
    let alias_path = state_root.join(asset.target.trim_start_matches(".ferrite/"));
    if alias_path.exists() {
        remove_existing_path(&alias_path)?;
    }
    create_symlink(payload_path, &alias_path)?;
    Ok(())
}

fn looks_like_libtorch_root(path: &Path) -> bool {
    (path.join("include/torch/torch.h").exists()
        || path
            .join("include/torch/csrc/api/include/torch/torch.h")
            .exists())
        && (path.join("lib/libtorch.so").exists()
            || path.join("lib/libtorch_cuda.so").exists()
            || path.join("lib/libc10_cuda.so").exists())
}

fn resolve_libtorch_root(path: &Path) -> Option<PathBuf> {
    if looks_like_libtorch_root(path) {
        return Some(path.to_path_buf());
    }
    if let Ok(real_path) = path.canonicalize() {
        if looks_like_libtorch_root(&real_path) {
            return Some(real_path);
        }
    }
    None
}

fn create_symlink(src: &Path, dst: &Path) -> Result<(), DynError> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dst)?;
    }
    #[cfg(not(unix))]
    {
        return Err("symlink materialization is only implemented for unix hosts".into());
    }
    Ok(())
}

fn remove_existing_path(path: &Path) -> Result<(), DynError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path)?;
    } else if metadata.is_dir() {
        fs::remove_dir_all(path)?;
    }
    Ok(())
}

fn install_user_tool(tool: &HostTool, apply: bool) -> Result<(), DynError> {
    let provider = tool
        .providers
        .get("script")
        .ok_or_else(|| format!("user tool {} missing script provider", tool.name))?;
    let url = provider
        .script_url
        .as_ref()
        .ok_or_else(|| format!("user tool {} missing script_url", tool.name))?;
    println!("tool.{}=script {}", tool.name, url);
    if !apply {
        return Ok(());
    }

    let script_path = env::temp_dir().join(format!("ferrite-bootstrap-{}.sh", tool.name));
    run_command(download_command(url, &script_path), None)?;
    let mut args = vec![script_path.display().to_string()];
    args.extend(provider.script_args.clone());
    run_command(script_shell_command(&args), None)?;
    Ok(())
}

fn compute_sha256(path: &Path) -> Result<String, DynError> {
    let output = command_output(command_with_args(
        "python3",
        &[
            "-c",
            "import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256();\nwith p.open('rb') as f:\n    [h.update(chunk) for chunk in iter(lambda: f.read(1024*1024), b'')]\nprint(h.hexdigest())",
            path.to_str().ok_or("invalid archive path")?,
        ],
    ))?;
    if !output.status.success() {
        return Err(String::from_utf8(output.stderr)?.into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn extract_zip_with_python(archive_path: &Path, out_dir: &Path) -> Result<(), DynError> {
    run_command(
        command_with_args(
            "python3",
            &[
                "-c",
                "import pathlib, sys, zipfile; archive=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); zipfile.ZipFile(archive).extractall(out)",
                archive_path.to_str().ok_or("invalid archive path")?,
                out_dir.to_str().ok_or("invalid output dir")?,
            ],
        ),
        Some("extract libtorch zip"),
    )
}

fn install_system_packages(
    provider: &str,
    packages: &[String],
    update_first: bool,
) -> Result<(), DynError> {
    match provider {
        "apt" => {
            if update_first {
                run_command(
                    command_with_args("sudo", &["apt-get", "update"]),
                    Some("apt-get update"),
                )?;
            }
            let mut args = vec!["apt-get".to_string(), "install".to_string(), "-y".to_string()];
            args.extend(packages.iter().cloned());
            run_command(command_with_args("sudo", &string_refs(&args)), Some("apt-get install"))?;
        }
        "dnf" => {
            let mut args = vec!["dnf".to_string(), "install".to_string(), "-y".to_string()];
            args.extend(packages.iter().cloned());
            run_command(command_with_args("sudo", &string_refs(&args)), Some("dnf install"))?;
        }
        other => {
            return Err(format!("unsupported system provider: {other}").into());
        }
    }
    Ok(())
}

fn ensure_repo_cache(repo_cache_root: &Path, repo_url: &str) -> Result<PathBuf, DynError> {
    let repo_key = sanitize_repo_key(repo_url);
    let repo_cache = repo_cache_root.join(repo_key);
    if repo_cache.exists() {
        run_command(
            command_in_dir("git", &["fetch", "--all", "--tags", "--prune"], &repo_cache),
            Some("git fetch"),
        )?;
    } else {
        run_command(
            command_with_args(
                "git",
                &[
                    "clone",
                    "--mirror",
                    repo_url,
                    repo_cache.to_str().ok_or("invalid repo cache path")?,
                ],
            ),
            Some("git clone --mirror"),
        )?;
    }
    Ok(repo_cache)
}

fn resolved_libtorch_path(state_root: &Path, ctx: &InstallerContext) -> Result<PathBuf, DynError> {
    for asset in selected_runtime_assets(ctx)? {
        if !asset_matches_current_host(asset) {
            continue;
        }
        let payload = asset_payload_path(state_root, asset);
        if let Some(resolved) = resolve_libtorch_root(&payload) {
            return Ok(resolved);
        }

        let alias = state_root.join(asset.target.trim_start_matches(".ferrite/"));
        if let Some(resolved) = resolve_libtorch_root(&alias) {
            return Ok(resolved);
        }
    }
    for candidate in jetson_libtorch_candidates() {
        if let Some(resolved) = resolve_libtorch_root(&candidate) {
            return Ok(resolved);
        }
    }
    Err("unable to resolve a valid libtorch root".into())
}

fn asset_payload_path(state_root: &Path, asset: &BinaryAsset) -> PathBuf {
    state_root.join("assets").join(&asset.id).join("payload")
}

fn jetson_libtorch_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(env_path) = env::var_os("LIBTORCH") {
        candidates.push(PathBuf::from(env_path));
    }
    candidates.push(PathBuf::from("/usr/local/libtorch"));
    candidates.push(PathBuf::from("/opt/libtorch"));
    candidates.push(PathBuf::from("/usr/lib/aarch64-linux-gnu/libtorch"));
    candidates.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    candidates.push(PathBuf::from("/usr/local"));
    candidates.push(PathBuf::from("/usr"));
    candidates.extend(discover_libtorch_roots(
        &[
            PathBuf::from("/usr/local"),
            PathBuf::from("/opt"),
            PathBuf::from("/usr/lib/aarch64-linux-gnu"),
            PathBuf::from("/usr"),
        ],
        4,
    ));
    candidates.sort();
    candidates.dedup();
    candidates
}

fn discover_libtorch_roots(search_roots: &[PathBuf], max_depth: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for root in search_roots {
        discover_libtorch_roots_inner(root, 0, max_depth, &mut found);
    }
    found
}

fn discover_libtorch_roots_inner(path: &Path, depth: usize, max_depth: usize, found: &mut Vec<PathBuf>) {
    if depth > max_depth || !path.is_dir() {
        return;
    }
    if looks_like_libtorch_root(path) {
        found.push(path.to_path_buf());
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let child = entry.path();
        if child.is_dir() {
            discover_libtorch_roots_inner(&child, depth + 1, max_depth, found);
        }
    }
}

fn resolve_repo_revision(repo_cache: &Path, commit: &str) -> Result<String, DynError> {
    let candidates = [
        commit.to_string(),
        format!("refs/heads/{commit}"),
        format!("refs/tags/{commit}"),
        format!("origin/{commit}"),
    ];
    for candidate in candidates {
        let output = command_output(command_in_dir("git", &["rev-parse", &candidate], repo_cache))?;
        if output.status.success() {
            let rev = String::from_utf8(output.stdout)?.trim().to_string();
            if !rev.is_empty() {
                return Ok(rev);
            }
        }
    }
    Err(format!("unable to resolve revision {commit} in {}", repo_cache.display()).into())
}

fn export_subdir(
    repo_cache: &Path,
    rev: &str,
    subdir: &str,
    dep_root: &Path,
) -> Result<(), DynError> {
    if dep_root.exists() {
        return Ok(());
    }
    fs::create_dir_all(dep_root)?;
    let output = if subdir == "." || subdir.is_empty() {
        command_output(command_in_dir("git", &["archive", "--format=tar", rev], repo_cache))?
    } else {
        let archive_spec = format!("{rev}:{subdir}");
        command_output(command_in_dir("git", &["archive", "--format=tar", &archive_spec], repo_cache))?
    };
    if !output.status.success() {
        return Err(format!(
            "git archive failed for {} {}",
            repo_cache.display(),
            if subdir == "." || subdir.is_empty() {
                rev.to_string()
            } else {
                format!("{rev}:{subdir}")
            }
        )
        .into());
    }
    let mut tar = Command::new("tar");
    tar.arg("-xf").arg("-").arg("-C").arg(dep_root);
    tar.stdin(std::process::Stdio::piped());
    tar.stdout(std::process::Stdio::null());
    tar.stderr(std::process::Stdio::piped());
    let mut child = tar.spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or("tar stdin unavailable")?
        .write_all(&output.stdout)?;
    let tar_output = child.wait_with_output()?;
    if !tar_output.status.success() {
        return Err(String::from_utf8(tar_output.stderr)?.into());
    }
    Ok(())
}

fn update_source_manifest_revision(
    state_root: &Path,
    dep: &SourceDep,
    resolved_rev: &str,
) -> Result<(), DynError> {
    let path = state_root
        .join("store")
        .join("src")
        .join(format!("{}.toml", dep.id));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_source_manifest(&path, dep, Some(resolved_rev))
}

fn sanitize_repo_key(repo_url: &str) -> String {
    repo_url
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn command_exists(name: &str) -> bool {
    env::var_os("PATH")
        .map(|paths| {
            env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(name);
                candidate.exists()
                    && candidate
                        .components()
                        .all(|component| !matches!(component, Component::ParentDir))
            })
        })
        .unwrap_or(false)
}

fn command_with_args(program: &str, args: &[&str]) -> Command {
    let mut command = Command::new(program);
    command.args(args);
    command
}

fn command_in_dir(program: &str, args: &[&str], dir: &Path) -> Command {
    let mut command = command_with_args(program, args);
    command.current_dir(dir);
    command
}

fn script_shell_command(args: &[String]) -> Command {
    let mut command = Command::new("sh");
    command.args(args);
    command
}

fn download_command(url: &str, out: &Path) -> Command {
    if command_exists("curl") {
        let mut command = Command::new("curl");
        command.args(["-fsSL", url, "-o"]);
        command.arg(out);
        command
    } else {
        let mut command = Command::new("wget");
        command.arg("-qO");
        command.arg(out);
        command.arg(url);
        command
    }
}

fn command_output(mut command: Command) -> Result<std::process::Output, DynError> {
    Ok(command.output()?)
}

fn run_command(mut command: Command, label: Option<&str>) -> Result<(), DynError> {
    let output = command.output()?;
    if output.status.success() {
        return Ok(());
    }
    let name = label.unwrap_or("command");
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!("{name} failed: {}", stderr.trim()).into())
}

fn string_refs(items: &[String]) -> Vec<&str> {
    items.iter().map(String::as_str).collect()
}

fn profile_has_feature(ctx: &InstallerContext, name: &str) -> bool {
    ctx.profile.profile.features.iter().any(|feature| feature == name)
}

fn detect_default_profile() -> String {
    if is_jetson_host() {
        return "jetson".to_string();
    }
    match (env::consts::OS, env::consts::ARCH) {
        ("linux", "x86_64") => "workstation-nvidia".to_string(),
        _ => "cpu-only-dev".to_string(),
    }
}

fn is_jetson_host() -> bool {
    if env::consts::ARCH != "aarch64" {
        return false;
    }
    if let Ok(model) = fs::read_to_string("/proc/device-tree/model") {
        let normalized = model.replace('\0', "");
        if normalized.contains("Jetson") || normalized.contains("Orin") {
            return true;
        }
    }
    Path::new("/etc/nv_tegra_release").exists()
}

fn profile_needs_platform(ctx: &InstallerContext) -> bool {
    profile_needs_torch(ctx) || profile_has_feature(ctx, "graphics")
}

fn profile_needs_torch(ctx: &InstallerContext) -> bool {
    profile_has_feature(ctx, "torch")
}

fn run_shell_in_repo(
    repo_root: &Path,
    env_script: &Path,
    command: &str,
    label: &str,
) -> Result<(), DynError> {
    let script = format!(
        ". '{}' && {}",
        env_script.display(),
        command
    );
    run_command(
        command_in_dir("bash", &["-lc", &script], repo_root),
        Some(label),
    )
}

fn set_executable(path: &Path) -> Result<(), DynError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn provider_summary(tool: &HostTool) -> String {
    let mut entries = Vec::new();
    for (name, provider) in &tool.providers {
        if !provider.packages.is_empty() {
            entries.push(format!("{name}:{}", provider.packages.join("+")));
        } else if let Some(url) = &provider.script_url {
            if provider.script_args.is_empty() {
                entries.push(format!("{name}:{url}"));
            } else {
                entries.push(format!("{name}:{} {}", url, provider.script_args.join(" ")));
            }
        } else {
            entries.push(name.clone());
        }
    }

    let mode = match tool.bootstrap {
        BootstrapMode::System => "system",
        BootstrapMode::User => "user",
    };

    format!("{mode}[{}]", entries.join(","))
}

fn print_help() {
    println!("ferrite-installer");
    println!("usage: cargo run --manifest-path installer/Cargo.toml -- <command> [--profile NAME] [--apply]");
    println!("commands: plan, resolve, materialize, bootstrap-host, fetch-sources, fetch-assets, generate-env, build-profile, validate-profile, bootstrap-all, detect, help");
    println!("bootstrap-host and bootstrap-all provision missing host dependencies automatically.");
    println!("default profile auto-detects: jetson on aarch64 Jetson hosts, workstation-nvidia on linux/x86_64, cpu-only-dev otherwise");
}
