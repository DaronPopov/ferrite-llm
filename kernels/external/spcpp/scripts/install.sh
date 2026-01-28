#!/bin/bash
# Install spcpp to /usr/local/bin (Linux/macOS)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPCPP_DIR="$(dirname "$SCRIPT_DIR")"

# Build the runner if needed
if [ ! -f "$SPCPP_DIR/bin/spcpp" ]; then
    echo "Building spcpp runner..."
    bash "$SCRIPT_DIR/build_runner.sh"
fi

# Create symlinks
echo "Installing to /usr/local..."
sudo ln -sf "$SPCPP_DIR/bin/spcpp" /usr/local/bin/spcpp
sudo mkdir -p /usr/local/include/spcpp
sudo ln -sf "$SPCPP_DIR/include/spcpp_portable.hpp" /usr/local/include/spcpp/spcpp.hpp

echo ""
echo "Installed! You can now use 'spcpp' from anywhere."
echo ""
echo "Example:"
echo "  spcpp my_program.cpp"
