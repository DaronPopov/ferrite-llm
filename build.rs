// Ferrite Build Script
// Compiles custom CUDA kernels using the modular kernel build system

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=kernels/build_kernels.sh");
    println!("cargo:rerun-if-changed=kernels/kernel_config.toml");

    // Only compile CUDA kernels if cuda feature is enabled
    let cuda_enabled = cfg!(feature = "cuda");

    if cuda_enabled {
        compile_cuda_kernels();
    } else {
        println!("cargo:warning=CUDA feature not enabled - custom kernels will not be compiled");
        println!("cargo:warning=Enable with: cargo build --features cuda");
    }
}

fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_output_dir = out_dir.join("kernels");

    std::fs::create_dir_all(&kernel_output_dir).expect("Failed to create kernel output dir");

    println!("cargo:warning=╔══════════════════════════════════════════════════════════════╗");
    println!("cargo:warning=║  Compiling Custom CUDA Kernels                              ║");
    println!("cargo:warning=╚══════════════════════════════════════════════════════════════╝");

    // Run the modular kernel build script
    let build_script = PathBuf::from("kernels/build_kernels.sh");

    if !build_script.exists() {
        println!("cargo:warning=Build script not found: {:?}", build_script);
        println!("cargo:warning=Skipping kernel compilation");
        return;
    }

    // Detect architecture
    let arch = detect_gpu_architecture().unwrap_or_else(|| "auto".to_string());

    let status = Command::new("bash")
        .arg(&build_script)
        .arg(&kernel_output_dir)
        .arg(&arch)
        .env("CUDA_HOME", get_cuda_home())
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:warning=✓ Custom CUDA kernels compiled successfully");

            // Set environment variables for runtime
            println!(
                "cargo:rustc-env=KERNEL_OUTPUT_DIR={}",
                kernel_output_dir.display()
            );

            // Link CUDA libraries
            let cuda_home = get_cuda_home();
            println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
        }
        Ok(status) => {
            println!(
                "cargo:warning=Kernel compilation failed with status: {}",
                status
            );
            println!("cargo:warning=Continuing build without custom kernels");
        }
        Err(e) => {
            println!("cargo:warning=Failed to run build script: {}", e);
            println!("cargo:warning=Continuing build without custom kernels");
        }
    }
}

fn detect_gpu_architecture() -> Option<String> {
    // Try to detect GPU using nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let compute_cap = String::from_utf8(output.stdout).ok()?;
    let compute_cap = compute_cap.trim();

    // Convert compute capability to sm_XX format
    let sm_arch = compute_cap.replace(".", "");
    Some(format!("sm_{}", sm_arch))
}

fn get_cuda_home() -> String {
    env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string())
}
