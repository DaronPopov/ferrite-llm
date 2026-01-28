# Getting Started

This guide covers system requirements, installation, and verification.

## System Requirements

### Hardware

- x86_64 processor
- NVIDIA GPU with Compute Capability 6.0 or higher

Supported GPU generations:

| Generation | Compute Capability | Example GPUs |
|------------|-------------------|--------------|
| Pascal | 6.0, 6.1 | GTX 1080, Tesla P100 |
| Volta | 7.0 | Tesla V100 |
| Turing | 7.5 | RTX 2080, Tesla T4 |
| Ampere | 8.0, 8.6 | A100, RTX 3090 |
| Ada Lovelace | 8.9 | RTX 4090, L40 |
| Hopper | 9.0 | H100 |

### Software

- Linux (Ubuntu 20.04+, RHEL 8+, or equivalent)
- NVIDIA Driver 450.0 or later
- CUDA Toolkit 11.0 or later
- Rust 1.70 or later (for building from source)

## Installation

### Quick Install

The installer auto-detects your GPU and downloads the appropriate binary:

```bash
curl -sSf https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | sh
```

The installer will:

1. Verify CUDA toolkit is present
2. Detect GPU compute capability
3. Download pre-built binaries
4. Configure environment variables

### Manual Installation

Download the release for your GPU architecture:

```bash
# Example for RTX 30xx (sm_86)
wget https://github.com/DaronPopov/ferrite/releases/download/v0.1.0/ferrite-linux-x86_64-sm86.tar.gz
tar -xzf ferrite-linux-x86_64-sm86.tar.gz
cd ferrite-linux-x86_64-sm86

# Add to PATH
export PATH="$PWD/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
```

### Building from Source

```bash
git clone https://github.com/DaronPopov/ferrite.git
cd ferrite

# Build the spcpp backend
./build_spcpp.sh

# Build Ferrite
cargo build --release
```

## Verification

### Check Installation

```bash
ferrite --version
```

### Verify GPU Detection

Create a test file `test.rs`:

```rust
use ferrite::compute::Stream;
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() {
    let device = CudaDevice::new(0).expect("No CUDA device found");
    println!("GPU detected: {:?}", device);

    let allocator = Arc::new(TlsfAllocator::new(device, 64 * 1024 * 1024));
    println!("Allocated 64 MB pool");

    let mut stream = Stream::new(allocator);
    stream.init().expect("Failed to initialize");
    println!("Stream initialized");

    stream.alloc("test", 1024);
    stream.fill("test", 1.0, 1024);
    stream.sync();
    println!("GPU operations successful");
}
```

Run with:

```bash
cargo run --release
```

Expected output:

```
GPU detected: CudaDevice(0)
Allocated 64 MB pool
Stream initialized
GPU operations successful
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA toolkit path | `/usr/local/cuda` |
| `FERRITE_HOME` | Ferrite installation path | `~/.ferrite` |
| `FERRITE_LOG` | Log level (error, warn, info, debug) | `error` |

## Troubleshooting

### CUDA not found

```
error: CUDA toolkit not found
```

Solution: Install CUDA toolkit or set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda-12.0
```

### GPU not detected

```
error: No CUDA device found
```

Verify the NVIDIA driver is loaded:

```bash
nvidia-smi
```

### Library not found

```
error: cannot open shared object file: libspcpp_launcher.so
```

Set the library path:

```bash
export LD_LIBRARY_PATH=/path/to/ferrite/lib:$LD_LIBRARY_PATH
```

## Next Steps

- [Quick Start](quickstart.md) - Write your first Ferrite program
- [Architecture](architecture.md) - Understand how Ferrite works
- [Stream API](api/stream.md) - Core API reference
