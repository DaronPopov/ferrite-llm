//! Build script for ptx-sys
//!
//! Automatically finds CUDA installation and links libraries.
//! Uses manual type definitions instead of parsing CUDA headers.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=FERRITE_PLATFORM_MANIFEST");

    // ── Platform manifest seam ──────────────────────────────────────
    // When FERRITE_PLATFORM_MANIFEST is set, parse the manifest for
    // pre-resolved include/lib/link artifacts instead of probing the
    // host.  This allows Windows (or cross-compile) builds to supply
    // paths discovered by `scripts/windows_builder/generate_manifest.py`.
    if let Ok(manifest_path) = env::var("FERRITE_PLATFORM_MANIFEST") {
        if apply_platform_manifest(&manifest_path) {
            return; // manifest handled all linking
        }
        // Fall through to normal discovery if manifest was invalid.
        println!("cargo:warning=FERRITE_PLATFORM_MANIFEST set but invalid; falling back to host discovery");
    }

    let script_env = load_env_from_script();
    let verbose = env::var("PTX_SYS_VERBOSE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    let cuda_path = env::var("CUDA_PATH")
        .ok()
        .or_else(|| env::var("CUDA_HOME").ok())
        .or_else(|| script_env.get("CUDA_PATH").cloned())
        .map(PathBuf::from)
        .or_else(find_cuda)
        .expect("Could not find CUDA installation");
    if verbose {
        println!("cargo:warning=Found CUDA at: {}", cuda_path.display());
    }

    let cuda_lib = [
        cuda_path.join("lib64"),
        cuda_path.join("lib"),
        cuda_path.join("targets/aarch64-linux/lib"),
        cuda_path.join("targets/x86_64-linux/lib"),
        cuda_path.join("targets/sbsa-linux/lib"),
        cuda_path.join("lib/aarch64-linux-gnu"),
        cuda_path.join("lib/x86_64-linux-gnu"),
    ]
    .into_iter()
    .find(|path| path.exists())
    .unwrap_or_else(|| cuda_path.join("lib"));

    // Project paths
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let lib_dir = project_root.join("lib");

    // Link paths
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    // Set rpath so binaries can find libptx_os.so without LD_LIBRARY_PATH
    // Multiple levels to handle different binary locations (examples/, target/release/, etc.)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

    // Always emit PTX-OS runtime linkage so stale Cargo build-script cache never
    // drops -lptx_os from downstream links. Missing files should fail loudly.
    let runtime_lib = lib_dir.join("libptx_os.so");
    if !runtime_lib.exists() {
        build_native_runtime(&project_root, &runtime_lib);
    }
    if !runtime_lib.exists() {
        panic!(
            "\n\n\
             ============================================================\n\
             ERROR: libptx_os.so not found at {}\n\n\
             Automatic build did not produce the Ferrite-OS runtime.\n\
             Ensure CUDA is installed and working, then rerun the build.\n\
             Manual fallback:\n\
               cd ferrite-os && make all\n\
             ============================================================\n",
            runtime_lib.display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=ptx_os");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libptx_os.so").display()
    );
    println!("cargo:rerun-if-changed=../../../tooling/scripts/ptx_env.sh");
    println!("cargo:rerun-if-changed=../../../tooling/scripts/detect_sm.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=PTX_SYS_VERBOSE");
    println!("cargo:rerun-if-env-changed=PTX_GPU_SM");
    println!("cargo:rerun-if-env-changed=GPU_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");

    // Detect GPU compute capability
    if let Some(sm) = detect_gpu_sm(&cuda_path, &project_root, &script_env) {
        if verbose {
            println!("cargo:warning=Detected GPU SM: {}", sm);
        }
        println!("cargo:rustc-env=PTX_GPU_SM={}", sm);
    }
}

/// Find CUDA installation path
fn find_cuda() -> Option<PathBuf> {
    // Check environment variables first
    for var in &["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(path) = env::var(var) {
            let p = PathBuf::from(&path);
            if p.join("include/cuda_runtime.h").exists() {
                return Some(p);
            }
        }
    }

    // Check common paths
    let common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.9",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-11",
        "/usr/local/cuda-11.8",
        "/opt/cuda",
        "/usr/lib/cuda",
        "/usr/cuda",
        // ARM / Jetson paths
        "/usr/local/cuda-arm64",
    ];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.join("include/cuda_runtime.h").exists() {
            return Some(p);
        }
    }

    // Try to find via nvcc
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Some(cuda_path) = PathBuf::from(&nvcc_path).parent().and_then(|p| p.parent()) {
                if cuda_path.join("include/cuda_runtime.h").exists() {
                    return Some(cuda_path.to_path_buf());
                }
            }
        }
    }

    None
}

/// Detect GPU compute capability using nvidia-smi
fn detect_gpu_sm(
    cuda_path: &PathBuf,
    project_root: &PathBuf,
    script_env: &HashMap<String, String>,
) -> Option<String> {
    if let Ok(val) = env::var("PTX_GPU_SM") {
        if let Some(sm) = normalize_sm(&val) {
            return Some(sm);
        }
    }
    if let Ok(val) = env::var("GPU_SM") {
        if let Some(sm) = normalize_sm(&val) {
            return Some(sm);
        }
    }
    if let Ok(val) = env::var("CUDA_SM") {
        if let Some(sm) = normalize_sm(&val) {
            return Some(sm);
        }
    }
    if let Some(val) = script_env.get("PTX_GPU_SM") {
        if let Some(sm) = normalize_sm(val) {
            return Some(sm);
        }
    }
    if let Some(val) = script_env.get("GPU_SM") {
        if let Some(sm) = normalize_sm(val) {
            return Some(sm);
        }
    }
    if let Some(val) = script_env.get("CUDA_SM") {
        if let Some(sm) = normalize_sm(val) {
            return Some(sm);
        }
    }

    if let Some(sm) = detect_sm_with_nvidia_smi() {
        return Some(sm);
    }

    detect_sm_with_nvcc(cuda_path, project_root)
}

fn normalize_sm(raw: &str) -> Option<String> {
    let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        Some(format!("sm_{}", digits))
    }
}

fn detect_sm_with_nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok();

    if let Some(out) = output {
        if out.status.success() {
            let cap = String::from_utf8_lossy(&out.stdout).to_string();
            let cap = cap.lines().next()?.trim();
            if let Some(sm) = normalize_sm(cap) {
                return Some(sm);
            }
        }
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_capability", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout).to_string();
        let cap = cap.lines().next()?.trim();
        normalize_sm(cap)
    } else {
        None
    }
}

fn detect_sm_with_nvcc(cuda_path: &PathBuf, project_root: &PathBuf) -> Option<String> {
    let detect_src = project_root.join("tooling/scripts").join("detect_sm.cu");
    if !detect_src.exists() {
        return None;
    }

    let nvcc_path = cuda_path.join("bin").join("nvcc");
    let nvcc = if nvcc_path.exists() {
        nvcc_path
    } else {
        PathBuf::from("nvcc")
    };

    let out_bin = env::temp_dir().join("ptx_detect_sm");
    let status = Command::new(nvcc)
        .args(["-O2", "-std=c++11"])
        .arg(&detect_src)
        .arg("-o")
        .arg(&out_bin)
        .status()
        .ok()?;

    if !status.success() {
        return None;
    }

    let output = Command::new(out_bin).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let cap = String::from_utf8_lossy(&output.stdout);
    normalize_sm(&cap)
}

/// Apply link directives from a platform manifest file (TOML).
///
/// Expected format:
/// ```toml
/// [ptx-sys]
/// lib_dirs = ["C:\\cuda\\lib\\x64", "C:\\ferrite\\lib"]
/// link_libs = ["ptx_os", "cudart", "cublas"]
/// link_kind = "dylib"     # optional, default "dylib"
/// include_dirs = ["C:\\cuda\\include"]
/// gpu_sm = "sm_86"        # optional
/// ```
fn apply_platform_manifest(path: &str) -> bool {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=Cannot read platform manifest {}: {}", path, e);
            return false;
        }
    };

    // Minimal TOML parsing — extract key = value and key = ["...", "..."]
    // lines under a [ptx-sys] section.
    let mut in_section = false;
    let mut lib_dirs: Vec<String> = Vec::new();
    let mut link_libs: Vec<String> = Vec::new();
    let mut link_kind = "dylib".to_string();
    let mut gpu_sm: Option<String> = None;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == "[ptx-sys]";
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some((key, val)) = trimmed.split_once('=') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "lib_dirs" => lib_dirs = parse_toml_string_array(val),
                "link_libs" => link_libs = parse_toml_string_array(val),
                "link_kind" => link_kind = val.trim_matches('"').to_string(),
                "gpu_sm" => gpu_sm = Some(val.trim_matches('"').to_string()),
                _ => {}
            }
        }
    }

    if lib_dirs.is_empty() && link_libs.is_empty() {
        return false;
    }

    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    for lib in &link_libs {
        println!("cargo:rustc-link-lib={}={}", link_kind, lib);
    }
    if let Some(sm) = gpu_sm {
        println!("cargo:rustc-env=PTX_GPU_SM={}", sm);
    }

    println!("cargo:rerun-if-changed={}", path);
    true
}

/// Parse a TOML-style string array: `["a", "b", "c"]`
fn parse_toml_string_array(raw: &str) -> Vec<String> {
    let inner = raw.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn load_env_from_script() -> HashMap<String, String> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let script = project_root.join("tooling/scripts").join("ptx_env.sh");
    if !script.exists() {
        return HashMap::new();
    }

    let output = Command::new(script)
        .args(["--format", "env", "--quiet"])
        .output();

    let out = match output {
        Ok(out) if out.status.success() => out,
        _ => return HashMap::new(),
    };

    let mut map = HashMap::new();
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    map
}

fn build_native_runtime(project_root: &PathBuf, runtime_lib: &PathBuf) {
    println!(
        "cargo:warning=libptx_os.so missing at {}; attempting `make all`",
        runtime_lib.display()
    );

    let status = Command::new("make")
        .arg("all")
        .current_dir(project_root)
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            println!(
                "cargo:warning=`make all` exited with status {}; PTX runtime build incomplete",
                status
            );
        }
        Err(err) => {
            println!(
                "cargo:warning=failed to invoke `make all` for Ferrite-OS runtime: {}",
                err
            );
        }
    }
}
