# Building from Source

This guide covers compiling Ferrite from source code.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Rust | 1.70+ | Compiler and package manager |
| CUDA Toolkit | 11.0+ | GPU compiler and libraries |
| g++ | 9.0+ | C++ compiler for spcpp |
| Git | Any | Source control |

### Verify Installation

```bash
# Rust
rustc --version
cargo --version

# CUDA
nvcc --version
nvidia-smi

# C++
g++ --version

# Git
git --version
```

## Clone Repository

```bash
git clone https://github.com/DaronPopov/ferrite.git
cd ferrite
```

## Build spcpp Backend

The spcpp backend provides GPU compute through cuBLAS and JIT-compiled CUDA kernels.

```bash
./build_spcpp.sh
```

This script:

1. Detects CUDA installation path
2. Compiles the C++ launcher library
3. Outputs `kernels/external/spcpp/build/lib/spcpp_launcher.so`

### Manual spcpp Build

If the script fails, build manually:

```bash
CUDA_PATH=/usr/local/cuda
SPCPP_DIR=kernels/external/spcpp

mkdir -p ${SPCPP_DIR}/build/lib

g++ -shared -fPIC -O3 -std=c++17 \
    -DSPC_SPCPP_DIR="\"${SPCPP_DIR}\"" \
    -I"${SPCPP_DIR}/include" \
    -I"${CUDA_PATH}/include" \
    "${SPCPP_DIR}/spcpp_launcher.cpp" \
    -o "${SPCPP_DIR}/build/lib/spcpp_launcher.so" \
    -L"${CUDA_PATH}/lib64" -lcudart -lcublas -lcuda -ldl
```

## Build Ferrite

### Debug Build

```bash
cargo build
```

Output: `target/debug/ferrite`

### Release Build

```bash
cargo build --release
```

Output: `target/release/ferrite`

Release builds are significantly faster due to optimizations.

### Build Options

```bash
# Build with all features
cargo build --release --all-features

# Build specific binary
cargo build --release --bin stream_demo

# Build library only
cargo build --release --lib
```

## Run Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture
```

GPU tests are ignored by default (require GPU). Run them with:

```bash
cargo test -- --ignored
```

## Verify Build

Run the demo binaries:

```bash
# Stream demo
cargo run --release --bin stream_demo

# CPU demo
cargo run --release --bin cpu_demo

# spcpp demo
cargo run --release --bin spcpp_demo
```

## Build Configuration

### Cargo.toml Options

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

These settings maximize performance at the cost of build time.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | CUDA toolkit path |
| `CUDA_PATH` | Alternative CUDA path variable |
| `LD_LIBRARY_PATH` | Library search path |

Example:

```bash
export CUDA_HOME=/usr/local/cuda-12.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Cross-Compilation

Ferrite currently supports only native Linux x86_64 compilation. Cross-compilation is not supported due to CUDA dependencies.

## Troubleshooting

### CUDA not found

```
error: could not find nvcc
```

Solution: Set CUDA_HOME or CUDA_PATH:

```bash
export CUDA_HOME=/usr/local/cuda
./build_spcpp.sh
```

### Linker errors

```
error: linking with `cc` failed
```

Possible causes:

1. Missing CUDA libraries - Install CUDA toolkit
2. Wrong library path - Set LD_LIBRARY_PATH
3. Missing libcudart - Install cuda-runtime package

### Runtime library not found

```
error: cannot open shared object file: libspcpp_launcher.so
```

Solution: Add library path:

```bash
export LD_LIBRARY_PATH=/path/to/ferrite/kernels/external/spcpp/build/lib:$LD_LIBRARY_PATH
```

### Incompatible CUDA version

```
error: CUDA version mismatch
```

Ensure nvcc version matches the installed driver capability. Check compatibility:

```bash
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit CUDA version
```

## Development Setup

### IDE Configuration

For VS Code, create `.vscode/settings.json`:

```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

### Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy
```

## Build Artifacts

After a successful build:

```
ferrite/
├── target/
│   └── release/
│       ├── ferrite              # Main binary
│       ├── stream_demo          # Demo binary
│       ├── cpu_demo             # CPU demo
│       └── libferrite.rlib      # Library
├── kernels/
│   └── external/
│       └── spcpp/
│           └── build/
│               └── lib/
│                   └── spcpp_launcher.so  # GPU backend
```

## Installing Locally

```bash
# Install to cargo bin directory
cargo install --path .

# Or copy manually
cp target/release/ferrite ~/.local/bin/
```
