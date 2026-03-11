#!/bin/bash
# Environment setup for Ferrite-OS development
# Source this file before building or testing: source env.sh

export FERRITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$FERRITE_ROOT/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "🚀 Ferrite-OS environment loaded"
echo "   FERRITE_ROOT: $FERRITE_ROOT"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   Library path: $LD_LIBRARY_PATH"
