#!/bin/bash
# Build the spcpp backend for Semantic Core RS
# This compiles the launcher that bridges Rust to cuBLAS + JIT CUDA kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPCPP_DIR="${SCRIPT_DIR}/kernels/external/spcpp"
BUILD_DIR="${SPCPP_DIR}/build"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

echo "Building spcpp backend..."
echo "  SPCPP_DIR: ${SPCPP_DIR}"
echo "  CUDA_PATH: ${CUDA_PATH}"

# Create build directories
mkdir -p "${BUILD_DIR}/lib"
mkdir -p "${BUILD_DIR}/ptx"

# Compile the launcher
g++ -shared -fPIC -O3 -std=c++17 \
    -DSPC_SPCPP_DIR="\"${SPCPP_DIR}\"" \
    -I"${SPCPP_DIR}/include" \
    -I"${CUDA_PATH}/include" \
    "${SPCPP_DIR}/spcpp_launcher.cpp" \
    -o "${BUILD_DIR}/lib/spcpp_launcher.so" \
    -L"${CUDA_PATH}/lib64" -lcudart -lcublas -lcuda -ldl

echo "Built: ${BUILD_DIR}/lib/spcpp_launcher.so"
echo ""
echo "spcpp backend ready. CUDA kernels will be JIT compiled on first use."
