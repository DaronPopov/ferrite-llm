#!/bin/bash
# Ferrite - One-liner installer
# curl -sSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | bash

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           FERRITE INSTALLER                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "[1/4] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/4] Rust already installed: $(rustc --version)"
fi

# Clone repo
INSTALL_DIR="${FERRITE_INSTALL_DIR:-$HOME/.ferrite}"
echo "[2/4] Cloning ferrite to $INSTALL_DIR..."

if [ -d "$INSTALL_DIR" ]; then
    echo "  Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --ff-only
else
    git clone https://github.com/DaronPopov/ferrite.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Build
echo "[3/4] Building (this may take a few minutes)..."
cargo build --release -p ferrite-examples

# Create symlinks
echo "[4/4] Installing binaries..."
mkdir -p "$HOME/.local/bin"

for bin in ferrite-chat mistral_quantized_inference mistral_inference tinyllama_inference qwen_inference gemma_inference phi_inference; do
    if [ -f "target/release/$bin" ]; then
        ln -sf "$INSTALL_DIR/target/release/$bin" "$HOME/.local/bin/$bin"
    fi
done

# Add to PATH if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Installation complete!"
echo ""
echo "  Run:  ferrite-chat"
echo "  Or:   mistral_quantized_inference"
echo "════════════════════════════════════════════════════════════════"
