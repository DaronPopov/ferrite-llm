use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=FERRITE_PLATFORM_MANIFEST");

    // ── Platform manifest seam ──────────────────────────────────────
    // When FERRITE_PLATFORM_MANIFEST is set, use manifest-declared
    // artifacts for linking instead of probing the host.
    if let Ok(manifest_path) = env::var("FERRITE_PLATFORM_MANIFEST") {
        if apply_platform_manifest(&manifest_path) {
            return; // manifest handled everything
        }
        println!("cargo:warning=FERRITE_PLATFORM_MANIFEST set but invalid for ptx-kernels; falling back");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Prefer explicit CUDA_ARCH, then PTX/installer SM hints, then conservative default (sm_75).
    let cuda_arch = env::var("CUDA_ARCH")
        .ok()
        .and_then(|v| normalize_cuda_arch(&v))
        .or_else(|| env::var("PTX_GPU_SM").ok().and_then(|v| normalize_cuda_arch(&v)))
        .or_else(|| env::var("GPU_SM").ok().and_then(|v| normalize_cuda_arch(&v)))
        .or_else(|| env::var("CUDA_SM").ok().and_then(|v| normalize_cuda_arch(&v)))
        .unwrap_or_else(|| "sm_75".to_string());

    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PTX_GPU_SM");
    println!("cargo:rerun-if-env-changed=GPU_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");

    // Build Candle kernels
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-cudart=shared")
        .flag(&format!("-arch={}", cuda_arch))
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-Xcompiler=-fPIC")
        .flag("--relocatable-device-code=true")  // Enable RDC for multi-file CUDA projects
        .opt_level(3)
        .include("kernels/candle");

    // Add test kernels (simple, always work)
    let kernel_files = [
        "test_kernels.cu",
        "sha256_kernel.cu",
    ];

    // Candle kernels - now enabled (F32 only for initial testing)
    let candle_files = [
        "candle/launcher_simple.cu",  // Simplified launcher with F32 only
        "candle/binary_f32.cu",       // F32-only binary ops
        "candle/unary_f32.cu",        // F32-only unary ops
        "candle/scan_f32.cu",         // F32-only prefix scan (cumsum)
        "candle/gather_f32.cu",       // F32-only gather/scatter
        "candle/topk_f32.cu",         // F32-only top-k selection
        "candle/indexing.cu",         // Candle index_select, gather, scatter, scatter_add
        "candle/indexing_launchers.cu", // Launchers for indexing kernels
        "candle/sort.cu",             // Candle bitonic argsort
        "candle/sort_launchers.cu",   // Launchers for argsort kernels
        "candle/ternary.cu",          // Candle where conditional
        "candle/where_launchers.cu",  // Launchers for where kernel
    ];

    for file in &kernel_files {
        let path = format!("kernels/{}", file);
        build.file(&path);
    }

    for file in &candle_files {
        let path = format!("kernels/{}", file);
        build.file(&path);
    }

    build.compile("candle_kernels");

    // Link against CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Add CUDA library search paths
    if let Some(cuda_path) = env::var("CUDA_PATH")
        .ok()
        .or_else(|| env::var("CUDA_HOME").ok())
        .map(PathBuf::from)
        .or_else(find_cuda_kernels)
    {
        let cuda_lib = if cuda_path.join("lib64").exists() {
            cuda_path.join("lib64")
        } else {
            cuda_path.join("lib")
        };
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        #[cfg(target_arch = "x86_64")]
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        #[cfg(target_arch = "aarch64")]
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}

/// Apply link/compile directives from a platform manifest file (TOML).
///
/// Expected format:
/// ```toml
/// [ptx-kernels]
/// cuda_path = "C:\\cuda"
/// cuda_arch = "sm_86"
/// lib_dirs = ["C:\\cuda\\lib\\x64"]
/// link_libs = ["cudart", "cuda"]
/// skip_kernel_build = false  # optional, default false
/// ```
fn apply_platform_manifest(path: &str) -> bool {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=Cannot read platform manifest {}: {}", path, e);
            return false;
        }
    };

    let mut in_section = false;
    let mut lib_dirs: Vec<String> = Vec::new();
    let mut link_libs: Vec<String> = Vec::new();
    let mut skip_kernel_build = false;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == "[ptx-kernels]";
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
                "skip_kernel_build" => skip_kernel_build = val.trim() == "true",
                "cuda_arch" => {
                    // Override CUDA_ARCH from manifest
                    // This is consumed by the cc::Build above, but the manifest
                    // path runs before the cc::Build, so we set env.
                    std::env::set_var("CUDA_ARCH", val.trim_matches('"'));
                }
                _ => {}
            }
        }
    }

    if lib_dirs.is_empty() && link_libs.is_empty() && !skip_kernel_build {
        return false;
    }

    if skip_kernel_build {
        // On Windows without NVCC, skip the CUDA kernel compilation entirely.
        // A pre-built static library should be provided via lib_dirs.
        println!("cargo:warning=Skipping CUDA kernel build (manifest: skip_kernel_build=true)");
    }

    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    for lib in &link_libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    println!("cargo:rerun-if-changed={}", path);
    true
}

fn parse_toml_string_array(raw: &str) -> Vec<String> {
    let inner = raw.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn normalize_cuda_arch(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(stripped) = trimmed.strip_prefix("sm_") {
        let digits: String = stripped.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            None
        } else {
            Some(format!("sm_{}", digits))
        }
    } else {
        let digits: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            None
        } else {
            Some(format!("sm_{}", digits))
        }
    }
}

fn find_cuda_kernels() -> Option<PathBuf> {
    let common = [
        "/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6", "/usr/local/cuda-12",
        "/opt/cuda", "/usr/lib/cuda",
    ];
    for p in &common {
        let p = PathBuf::from(p);
        if p.join("include/cuda_runtime.h").exists() { return Some(p); }
    }
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Some(cuda) = PathBuf::from(&s).parent().and_then(|p| p.parent()) {
                if cuda.join("include/cuda_runtime.h").exists() {
                    return Some(cuda.to_path_buf());
                }
            }
        }
    }
    None
}
