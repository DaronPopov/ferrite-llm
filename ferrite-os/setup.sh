#!/bin/bash
# Ferrite-OS Quick Setup Script
#
# Usage:
#   ./setup.sh                           # Auto-detect CUDA and SM
#   ./setup.sh --cuda-version 12.6       # Specify CUDA version
#   ./setup.sh --sm 89                   # Specify compute capability (e.g., 89 for RTX 4090)
#   ./setup.sh --cuda-version 12.6 --sm 89
#
# Environment variables (alternative to flags):
#   CUDA_VERSION=12.6 SM=89 ./setup.sh

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

sudo_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        log_error "Need root privileges (or sudo) to install CUPTI"
        exit 1
    fi
}

has_cupti() {
    if ldconfig -p 2>/dev/null | grep -q 'libcupti\.so'; then
        return 0
    fi
    if [ -f "$CUDA_PATH/targets/x86_64-linux/lib/libcupti.so" ] || \
       [ -f "$CUDA_PATH/targets/aarch64-linux/lib/libcupti.so" ] || \
       [ -f "$CUDA_PATH/extras/CUPTI/lib64/libcupti.so" ]; then
        return 0
    fi
    return 1
}

resolve_cupti_lib_dir() {
    local arch
    arch="$(uname -m)"
    if [ "$arch" = "x86_64" ] && [ -d "$CUDA_PATH/targets/x86_64-linux/lib" ]; then
        echo "$CUDA_PATH/targets/x86_64-linux/lib"
        return 0
    fi
    if [ "$arch" = "aarch64" ] && [ -d "$CUDA_PATH/targets/aarch64-linux/lib" ]; then
        echo "$CUDA_PATH/targets/aarch64-linux/lib"
        return 0
    fi
    if [ -d "$CUDA_PATH/extras/CUPTI/lib64" ]; then
        echo "$CUDA_PATH/extras/CUPTI/lib64"
        return 0
    fi
    return 1
}

install_cupti_linux() {
    local mm_dash="$1"
    if command -v apt-get >/dev/null 2>&1; then
        sudo_cmd apt-get update -y
        sudo_cmd apt-get install -y "cuda-cupti-${mm_dash}" || \
        sudo_cmd apt-get install -y "cuda-cupti-dev-${mm_dash}" || \
        sudo_cmd apt-get install -y "cuda-command-line-tools-${mm_dash}" || \
        sudo_cmd apt-get install -y nvidia-cuda-toolkit || return 1
        return 0
    elif command -v dnf >/dev/null 2>&1; then
        sudo_cmd dnf install -y "cuda-cupti-${mm_dash}" "cuda-cupti-devel-${mm_dash}" || \
        sudo_cmd dnf install -y cuda-toolkit || return 1
        return 0
    elif command -v yum >/dev/null 2>&1; then
        sudo_cmd yum install -y "cuda-cupti-${mm_dash}" "cuda-cupti-devel-${mm_dash}" || \
        sudo_cmd yum install -y cuda-toolkit || return 1
        return 0
    elif command -v pacman >/dev/null 2>&1; then
        sudo_cmd pacman -Sy --noconfirm cuda cuda-tools || return 1
        return 0
    elif command -v zypper >/dev/null 2>&1; then
        sudo_cmd zypper --non-interactive install "cuda-cupti-${mm_dash}" || \
        sudo_cmd zypper --non-interactive install cuda-toolkit || return 1
        return 0
    fi
    return 1
}

ensure_cupti() {
    if has_cupti; then
        log_info "CUPTI detected"
        return 0
    fi
    local mm_dash
    mm_dash="$(echo "$DETECTED_VERSION" | cut -d. -f1,2 | tr '.' '-')"
    if [ -z "$mm_dash" ] || [ "$mm_dash" = "unknown" ]; then
        mm_dash="12-0"
    fi
    log_warn "CUPTI not found; attempting auto-install for CUDA ${mm_dash}"
    if install_cupti_linux "$mm_dash" && has_cupti; then
        log_info "CUPTI installed successfully"
        return 0
    fi
    log_error "Failed to install CUPTI automatically"
    log_error "Install package: cuda-cupti-${mm_dash} (or cuda-cupti-dev-${mm_dash})"
    exit 1
}

# Parse arguments
CUDA_VERSION=""
SM=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --sm)
            SM="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Ferrite-OS Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION   Specify CUDA version (e.g., 12.6, 12.0, 11.8)"
            echo "  --sm SM                  Specify GPU compute capability (e.g., 89, 86, 80)"
            echo "  --verbose                Enable verbose output"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Auto-detect everything"
            echo "  $0 --cuda-version 12.6                # Use CUDA 12.6, auto-detect SM"
            echo "  $0 --sm 89                            # Auto-detect CUDA, use SM 89"
            echo "  $0 --cuda-version 12.6 --sm 89       # Specify both"
            echo ""
            echo "Environment variables:"
            echo "  CUDA_VERSION=12.6 SM=89 $0"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for environment variable overrides
CUDA_VERSION="${CUDA_VERSION:-${CUDA_VERSION_ENV:-}}"
SM="${SM:-${SM_ENV:-}}"

log_info "Starting Ferrite-OS setup"

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for required tools
log_info "Checking dependencies..."

if ! command -v cargo &> /dev/null; then
    log_error "cargo not found. Install Rust from https://rustup.rs/"
    exit 1
fi

if ! command -v make &> /dev/null; then
    log_error "make not found. Install build-essential (Ubuntu/Debian) or base-devel (Arch)"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    log_error "g++ not found. Install build-essential (Ubuntu/Debian) or base-devel (Arch)"
    exit 1
fi

# Detect or verify CUDA installation
log_info "Detecting CUDA installation..."

CUDA_PATH=""
if [ -n "$CUDA_VERSION" ]; then
    # User specified version - try to find it
    CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
    if [ ! -d "$CUDA_PATH" ]; then
        CUDA_PATH="/usr/local/cuda"
        if [ -d "$CUDA_PATH" ]; then
            DETECTED_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
            if [ "$DETECTED_VERSION" != "$CUDA_VERSION" ]; then
                log_warn "Requested CUDA $CUDA_VERSION but found $DETECTED_VERSION at $CUDA_PATH"
                log_warn "Continuing with detected version..."
            fi
        else
            log_error "CUDA $CUDA_VERSION not found at expected locations"
            exit 1
        fi
    fi
else
    # Auto-detect
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
    else
        log_error "CUDA not found. Install from https://developer.nvidia.com/cuda-downloads"
        log_error "Or specify path with: CUDA_PATH=/path/to/cuda $0"
        exit 1
    fi
fi

if [ ! -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
    log_error "CUDA installation at $CUDA_PATH appears incomplete (missing cuda_runtime.h)"
    exit 1
fi

DETECTED_VERSION=$($CUDA_PATH/bin/nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p' || echo "unknown")
log_info "Found CUDA $DETECTED_VERSION at $CUDA_PATH"

export CUDA_PATH
export CUDA_HOME="$CUDA_PATH"
ensure_cupti
CUPTI_LIB_DIR="$(resolve_cupti_lib_dir || true)"
if [ -n "$CUPTI_LIB_DIR" ]; then
    export CUPTI_LIB_DIR
    export LD_LIBRARY_PATH="$CUPTI_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# Detect or verify GPU compute capability
if [ -n "$SM" ]; then
    if [ "$SM" -lt 75 ]; then
        log_error "GPU SM $SM is unsupported by current kernel profile (requires sm_75+)"
        log_error "For Jetson, use Orin-class targets (sm_87) or maintain a legacy kernel profile"
        exit 1
    fi
    # User specified SM
    log_info "Using specified compute capability: sm_$SM"
    export PTX_GPU_SM="sm_$SM"
    export GPU_SM="$SM"
else
    # Auto-detect
    log_info "Detecting GPU compute capability..."
    DETECTED_SM=""

    if command -v nvidia-smi &> /dev/null; then
        DETECTED_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' | tr -d ' ')
    fi

    # Jetson and headless fallback path (uses model + nvcc probe logic).
    if [ -z "$DETECTED_SM" ] && [ -x "./tooling/scripts/ptx_env.sh" ]; then
        DETECTED_SM=$(./tooling/scripts/ptx_env.sh --format env --quiet 2>/dev/null | sed -n 's/^GPU_SM=//p' | head -n1)
    fi

    if [ -n "$DETECTED_SM" ]; then
        if [ "$DETECTED_SM" -lt 75 ]; then
            log_error "Detected GPU SM sm_$DETECTED_SM is unsupported by current kernel profile (requires sm_75+)"
            log_error "For Jetson, use Orin-class targets (sm_87) or maintain a legacy kernel profile"
            exit 1
        fi
        log_info "Detected GPU compute capability: sm_$DETECTED_SM"
        export PTX_GPU_SM="sm_$DETECTED_SM"
        export GPU_SM="$DETECTED_SM"
    else
        log_warn "Could not auto-detect GPU compute capability"
        log_warn "Build will use conservative defaults (set --sm for best results)"
    fi
fi

# Set verbose flag if requested
if [ "$VERBOSE" = true ]; then
    export PTX_SYS_VERBOSE=1
fi

# Build libptx_os.so
log_info "Building libptx_os.so..."
if [ "$VERBOSE" = true ]; then
    make
else
    make > /dev/null 2>&1
fi

if [ ! -f "lib/libptx_os.so" ]; then
    log_error "Failed to build libptx_os.so"
    exit 1
fi
log_info "libptx_os.so built successfully"

# Build Rust workspace
log_info "Building Rust workspace..."
if [ "$VERBOSE" = true ]; then
    cargo build --release
else
    cargo build --release 2>&1 | grep -E "(Compiling|Finished|error|warning:)" || true
fi

if [ $? -eq 0 ]; then
    log_info "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify rpath configuration
log_info "Verifying binary configuration..."
if command -v ldd &> /dev/null; then
    TEST_BIN="target/release/examples/telemetry_demo"
    if [ -f "$TEST_BIN" ]; then
        if ldd "$TEST_BIN" 2>&1 | grep -q "libptx_os.so.*lib/libptx_os.so"; then
            log_info "Runtime library linking configured correctly"
        else
            log_warn "Runtime library linking may need LD_LIBRARY_PATH"
            log_warn "Add to your shell: export LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH"
        fi
    fi
fi

# Print success summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Ferrite-OS Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  CUDA:     $CUDA_PATH ($DETECTED_VERSION)"
[ -n "$PTX_GPU_SM" ] && echo "  GPU SM:   $PTX_GPU_SM"
echo "  Build:    target/release/"
echo ""
echo "Quick Start:"
echo "  # Run an example"
echo "  cargo run --release --example telemetry_demo -p ptx-runtime"
echo ""
echo "  # Run benchmarks"
echo "  cargo run --release --bin memory_pressure -p systemic-benchmarks"
echo ""
echo "  # Start daemon"
echo "  cargo run --release -p ferrite-daemon -- serve --config crates/internal/ptx-daemon/dev-config.toml"
echo ""
echo "Documentation:"
echo "  README.md                - Project overview"
echo "  API_REFERENCE.md         - Complete API documentation"
echo "  DOCUMENTATION_INDEX.md   - Documentation guide"
echo ""
