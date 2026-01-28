#!/bin/bash
# Ferrite Installer
# Usage: curl -sSf https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | sh

set -e

FERRITE_VERSION="${FERRITE_VERSION:-latest}"
FERRITE_HOME="${FERRITE_HOME:-$HOME/.ferrite}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[ferrite]${NC} $1"; }
success() { echo -e "${GREEN}[ferrite]${NC} $1"; }
warn() { echo -e "${YELLOW}[ferrite]${NC} $1"; }
error() { echo -e "${RED}[ferrite]${NC} $1"; exit 1; }

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Ferrite Installer                          ║${NC}"
echo -e "${GREEN}║        High-performance GPU ML runtime for Rust               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================
# SYSTEM CHECKS
# ============================================================

info "Checking system requirements..."

# Check OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "Ferrite currently only supports Linux. Got: $OSTYPE"
fi
success "OS: Linux ✓"

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    error "Ferrite requires x86_64 architecture. Got: $ARCH"
fi
success "Architecture: x86_64 ✓"

# Check for NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    error "NVIDIA driver not found. Please install NVIDIA drivers first."
fi
success "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) ✓"

# Detect GPU and compute capability
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

if [[ -z "$COMPUTE_CAP" ]]; then
    error "Could not detect GPU compute capability"
fi

success "GPU: $GPU_NAME (sm_$COMPUTE_CAP) ✓"

# Check compute capability is supported (>= 7.0 for Tensor Cores, >= 6.0 minimum)
MAJOR_VER=${COMPUTE_CAP:0:1}
if [[ "$MAJOR_VER" -lt 6 ]]; then
    error "GPU compute capability $COMPUTE_CAP is too old. Minimum: 6.0 (Pascal)"
fi

# Check for CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    warn "CUDA toolkit (nvcc) not found in PATH"

    # Try common locations
    CUDA_PATHS="/usr/local/cuda /opt/cuda /usr/lib/cuda"
    for path in $CUDA_PATHS; do
        if [[ -f "$path/bin/nvcc" ]]; then
            export CUDA_HOME="$path"
            export PATH="$CUDA_HOME/bin:$PATH"
            success "Found CUDA at $path"
            break
        fi
    done

    if ! command -v nvcc &> /dev/null; then
        error "CUDA toolkit not found. Please install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
success "CUDA toolkit: $CUDA_VERSION ✓"

# Check CUDA version is sufficient (>= 11.0)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
if [[ "$CUDA_MAJOR" -lt 11 ]]; then
    warn "CUDA $CUDA_VERSION is old. Recommended: CUDA 11.0+"
fi

# ============================================================
# INSTALLATION
# ============================================================

info "Installing Ferrite to $FERRITE_HOME..."

# Create directories
mkdir -p "$FERRITE_HOME/bin"
mkdir -p "$FERRITE_HOME/lib"
mkdir -p "$FERRITE_HOME/kernels"

# Store detected config
cat > "$FERRITE_HOME/config.env" << EOF
# Ferrite Configuration (auto-generated)
FERRITE_HOME=$FERRITE_HOME
FERRITE_VERSION=$FERRITE_VERSION
GPU_NAME="$GPU_NAME"
COMPUTE_CAP=$COMPUTE_CAP
CUDA_VERSION=$CUDA_VERSION
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
INSTALLED_AT=$(date -Iseconds)
EOF

success "Configuration saved to $FERRITE_HOME/config.env"

# Download or build based on availability
RELEASE_URL="https://github.com/DaronPopov/ferrite/releases/download/$FERRITE_VERSION"
BINARY_NAME="ferrite-linux-x86_64-sm${COMPUTE_CAP}.tar.gz"

info "Checking for pre-built binary for sm_$COMPUTE_CAP..."

# Try to download pre-built binary
if curl --output /dev/null --silent --head --fail "$RELEASE_URL/$BINARY_NAME" 2>/dev/null; then
    info "Downloading pre-built binary..."
    curl -sSL "$RELEASE_URL/$BINARY_NAME" | tar -xz -C "$FERRITE_HOME"
    success "Downloaded pre-built binary ✓"
else
    warn "No pre-built binary for sm_$COMPUTE_CAP, building from source..."

    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Clone and build
    TEMP_DIR=$(mktemp -d)
    info "Cloning Ferrite..."
    git clone --depth 1 https://github.com/DaronPopov/ferrite.git "$TEMP_DIR/ferrite"

    info "Building Ferrite (this may take a few minutes)..."
    cd "$TEMP_DIR/ferrite"

    # Build spcpp
    ./build_spcpp.sh

    # Build Rust
    cargo build --release

    # Copy artifacts
    cp target/release/ferrite "$FERRITE_HOME/bin/"
    cp -r kernels/external/spcpp/build/lib/* "$FERRITE_HOME/lib/" 2>/dev/null || true
    cp -r kernels "$FERRITE_HOME/"

    # Cleanup
    rm -rf "$TEMP_DIR"

    success "Built from source ✓"
fi

# ============================================================
# PATH SETUP
# ============================================================

info "Setting up PATH..."

# Detect shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    bash)
        PROFILE="$HOME/.bashrc"
        ;;
    zsh)
        PROFILE="$HOME/.zshrc"
        ;;
    fish)
        PROFILE="$HOME/.config/fish/config.fish"
        ;;
    *)
        PROFILE="$HOME/.profile"
        ;;
esac

# Add to PATH if not already there
EXPORT_LINE="export PATH=\"$FERRITE_HOME/bin:\$PATH\""
EXPORT_LD="export LD_LIBRARY_PATH=\"$FERRITE_HOME/lib:\$LD_LIBRARY_PATH\""

if ! grep -q "FERRITE_HOME" "$PROFILE" 2>/dev/null; then
    echo "" >> "$PROFILE"
    echo "# Ferrite" >> "$PROFILE"
    echo "export FERRITE_HOME=\"$FERRITE_HOME\"" >> "$PROFILE"
    echo "$EXPORT_LINE" >> "$PROFILE"
    echo "$EXPORT_LD" >> "$PROFILE"
    success "Added Ferrite to $PROFILE"
else
    info "Ferrite already in $PROFILE"
fi

# ============================================================
# VERIFICATION
# ============================================================

export PATH="$FERRITE_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$FERRITE_HOME/lib:$LD_LIBRARY_PATH"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 Installation Complete!                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Installed to: $FERRITE_HOME"
echo "  GPU: $GPU_NAME (sm_$COMPUTE_CAP)"
echo "  CUDA: $CUDA_VERSION"
echo ""
echo "  To get started, restart your shell or run:"
echo ""
echo -e "    ${YELLOW}source $PROFILE${NC}"
echo ""
echo "  Then try:"
echo ""
echo -e "    ${YELLOW}ferrite --version${NC}"
echo ""
echo "  Documentation: https://github.com/DaronPopov/ferrite"
echo ""
