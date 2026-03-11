#!/bin/bash
# Test runner for Ferrite-OS
# Runs all tests including GPU integration tests

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          Ferrite-OS Test Suite                            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: No NVIDIA GPU detected or nvidia-smi not available"
    echo "   GPU tests will be skipped"
    GPU_AVAILABLE=false
else
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
    echo ""
fi

# Build first
echo "🔨 Building libraries..."
make -s
echo "✓ C/CUDA libraries built"
echo ""

# Rust unit tests (no GPU required)
echo "🧪 Running Rust unit tests..."
cargo test --workspace --lib
echo "✓ Unit tests passed"
echo ""

# Integration tests (require GPU)
if [ "$GPU_AVAILABLE" = true ]; then
    echo "🚀 Running GPU integration tests..."
    echo ""

    echo "┌─ PTX-Runtime Integration Tests ─────────────────────────┐"
    cargo test --package ptx-runtime --test integration_tests -- --ignored --test-threads=1
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    echo "┌─ PTX-Kernels Guard Tests ────────────────────────────────┐"
    cargo test --package ptx-kernels --test guard_validation -- --ignored --test-threads=1
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    echo "┌─ PTX-Compute API Tests ──────────────────────────────────┐"
    cargo test --package ptx-compute --test api_tests -- --ignored --test-threads=1
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    echo "✅ All GPU integration tests passed!"
else
    echo "⏭️  Skipping GPU integration tests (no GPU available)"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          ✅ ALL TESTS PASSED                              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
