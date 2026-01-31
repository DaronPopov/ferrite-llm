#!/usr/bin/env bash
# Ferrite Installer - Seamless deployment with automatic Rust setup
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}Ferrite Installer${NC} - Pure Rust LLM Inference Engine        ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS" in
        Linux*)     OS_TYPE="Linux";;
        Darwin*)    OS_TYPE="macOS";;
        CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows";;
        *)          OS_TYPE="Unknown";;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH_TYPE="x86_64";;
        aarch64|arm64)  ARCH_TYPE="aarch64";;
        armv7l)         ARCH_TYPE="armv7";;
        *)              ARCH_TYPE="Unknown";;
    esac

    print_info "Detected platform: ${OS_TYPE} ${ARCH_TYPE}"
}

# Check if Rust is installed
check_rust() {
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        print_success "Rust is already installed (version ${RUST_VERSION})"
        return 0
    else
        return 1
    fi
}

# Install Rust using rustup
install_rust() {
    print_warning "Rust is not installed on your system"
    echo ""
    read -p "Would you like to install Rust now? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Rust is required to build Ferrite. Exiting."
        exit 1
    fi

    print_info "Installing Rust via rustup..."

    if [ "$OS_TYPE" = "Windows" ]; then
        print_info "Please download and run: https://win.rustup.rs/"
        print_error "Automatic installation not supported on Windows. Please install manually and re-run this script."
        exit 1
    else
        # Install rustup for Unix-like systems
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

        # Source cargo environment
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi

        print_success "Rust installed successfully!"
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        print_info "Installed version: ${RUST_VERSION}"
    fi
}

# Check Rust version compatibility
check_rust_version() {
    RUST_VERSION=$(rustc --version | awk '{print $2}')
    REQUIRED_VERSION="1.70.0"

    # Simple version comparison (works for x.y.z format)
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_success "Rust version ${RUST_VERSION} is compatible (>= ${REQUIRED_VERSION})"
    else
        print_warning "Rust version ${RUST_VERSION} is older than recommended ${REQUIRED_VERSION}"
        read -p "Update Rust now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rustup update
        fi
    fi
}

# Check for CUDA (optional, for GPU acceleration)
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        print_success "NVIDIA GPU detected (Driver: ${CUDA_VERSION})"
        print_info "GPU acceleration will be available"
        HAS_CUDA=true
    else
        print_info "No NVIDIA GPU detected - will use CPU mode"
        HAS_CUDA=false
    fi
}

# Check available disk space
check_disk_space() {
    # Need ~10GB for models + build artifacts
    REQUIRED_SPACE_GB=10

    if [ "$OS_TYPE" = "macOS" ]; then
        AVAILABLE_SPACE=$(df -g . | awk 'NR==2 {print $4}')
    else
        AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi

    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE_GB" ]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (${REQUIRED_SPACE_GB}GB recommended)"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    fi
}

# Build the project
build_project() {
    print_info "Building Ferrite (this may take 5-10 minutes)..."

    if [ "$HAS_CUDA" = true ]; then
        print_info "Building with CUDA support..."
        cargo build --release -p ferrite-examples 2>&1 | grep -v "^   " || true
    else
        print_info "Building for CPU only..."
        cargo build --release -p ferrite-examples --no-default-features 2>&1 | grep -v "^   " || true
    fi

    if [ $? -eq 0 ]; then
        print_success "Build completed successfully!"
    else
        print_error "Build failed. Please check the error messages above."
        exit 1
    fi
}

# Install binaries
install_binaries() {
    INSTALL_DIR="$HOME/.local/bin"

    # Create install directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"

    # Copy the main binary
    if [ -f "target/release/ferrite-chat" ]; then
        cp target/release/ferrite-chat "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/ferrite-chat"
        print_success "Installed ferrite-chat to ${INSTALL_DIR}"
    else
        print_error "ferrite-chat binary not found"
        return 1
    fi

    # Copy other example binaries (optional)
    for binary in gemma_inference qwen_inference mistral_inference tinyllama_inference phi_inference; do
        if [ -f "target/release/$binary" ]; then
            cp "target/release/$binary" "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/$binary"
        fi
    done

    print_success "Installed $(ls target/release/*_inference 2>/dev/null | wc -l) example binaries"
}

# Update PATH if needed
update_path() {
    INSTALL_DIR="$HOME/.local/bin"

    # Check if already in PATH
    if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
        print_success "Installation directory already in PATH"
        return 0
    fi

    print_warning "Installation directory not in PATH"

    # Detect shell and add to appropriate rc file
    SHELL_RC=""
    if [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.profile"
    fi

    read -p "Add ${INSTALL_DIR} to PATH in ${SHELL_RC}? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "" >> "$SHELL_RC"
        echo "# Added by Ferrite installer" >> "$SHELL_RC"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
        print_success "Updated ${SHELL_RC}"
        print_info "Run 'source ${SHELL_RC}' or restart your terminal"
    else
        print_warning "You'll need to manually add ${INSTALL_DIR} to your PATH"
    fi
}

# Configure HuggingFace token (optional)
setup_hf_token() {
    echo ""
    print_info "Some models require a HuggingFace account and token"
    read -p "Do you have a HuggingFace token to configure? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your HuggingFace token: " HF_TOKEN

        if [ -n "$HF_TOKEN" ]; then
            mkdir -p "$HOME/.ferrite"
            echo "$HF_TOKEN" > "$HOME/.ferrite/hf_token"
            chmod 600 "$HOME/.ferrite/hf_token"
            print_success "HuggingFace token saved"
        fi
    else
        print_info "You can configure this later by running: ferrite-chat --login"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${GREEN}Installation Complete!${NC}                                      ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Restart your terminal or run: ${BLUE}source ~/.bashrc${NC}"
    echo -e "  2. Launch Ferrite: ${BLUE}ferrite-chat${NC}"
    echo -e "  3. Or run a specific model: ${BLUE}qwen_inference${NC}"
    echo ""
    echo -e "Documentation:"
    echo -e "  • README: $(pwd)/README.md"
    echo -e "  • Examples: $(pwd)/examples/"
    echo -e "  • GitHub: https://github.com/DaronPopov/ferrite"
    echo ""
    echo -e "Need help?"
    echo -e "  • Run: ${BLUE}ferrite-chat --help${NC}"
    echo -e "  • Report issues: https://github.com/DaronPopov/ferrite/issues"
    echo ""
}

# Main installation flow
main() {
    print_header

    # System checks
    detect_platform
    check_disk_space
    echo ""

    # Rust setup
    if ! check_rust; then
        install_rust
    fi
    check_rust_version
    echo ""

    # Optional: Check for CUDA
    check_cuda
    echo ""

    # Build and install
    build_project
    echo ""

    install_binaries
    echo ""

    update_path

    # Optional: Setup HuggingFace
    setup_hf_token

    # Done!
    print_next_steps
}

# Run main installation
main

# Exit successfully
exit 0
