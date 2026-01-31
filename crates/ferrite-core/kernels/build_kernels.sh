#!/usr/bin/env bash
# Ferrite Custom CUDA Kernel Build System
#
# Automatically compiles all .cu files in kernels/ directory
# Users can drop new kernels here and they'll be built automatically!

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Ferrite CUDA Kernel Build System                           ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for required arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output_dir> [architecture]"
    echo "Example: $0 ../target/kernels sm_86"
    exit 1
fi

OUTPUT_DIR="$1"
ARCH="${2:-auto}"
KERNEL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Detect CUDA installation
if [ -z "$CUDA_HOME" ]; then
    if [ -z "$CUDA_PATH" ]; then
        CUDA_HOME="/usr/local/cuda"
    else
        CUDA_HOME="$CUDA_PATH"
    fi
fi

NVCC="$CUDA_HOME/bin/nvcc"

# Check if nvcc exists
if [ ! -f "$NVCC" ]; then
    echo -e "${RED}✗ NVCC not found at $NVCC${NC}"
    echo -e "${YELLOW}  Set CUDA_HOME environment variable or install CUDA toolkit${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found NVCC: $NVCC"

# Detect GPU architecture if set to auto
if [ "$ARCH" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        ARCH="sm_${COMPUTE_CAP}"
        echo -e "${GREEN}✓${NC} Auto-detected GPU architecture: $ARCH"
    else
        ARCH="sm_86"  # Default to RTX 30xx
        echo -e "${YELLOW}⚠${NC} Could not detect GPU, using default: $ARCH"
    fi
else
    echo -e "${GREEN}✓${NC} Using specified architecture: $ARCH"
fi

# Compilation flags
NVCC_FLAGS=(
    "-O3"                    # Maximum optimization
    "--use_fast_math"        # Fast math operations
    "-std=c++17"             # C++17 standard
    "--gpu-architecture=$ARCH"
    "-ptx"                   # Compile to PTX (portable)
)

# Optional: Add debug info in debug builds
if [ "${DEBUG:-0}" = "1" ]; then
    NVCC_FLAGS+=("-g" "-G")
    echo -e "${YELLOW}⚠${NC} Debug mode enabled"
fi

# Find all .cu files
KERNEL_COUNT=0
COMPILED_COUNT=0
FAILED_COUNT=0

echo ""
echo "Scanning for CUDA kernels in: $KERNEL_DIR"
echo ""

for cu_file in "$KERNEL_DIR"/*.cu; do
    if [ ! -f "$cu_file" ]; then
        continue
    fi

    KERNEL_COUNT=$((KERNEL_COUNT + 1))

    kernel_name=$(basename "$cu_file" .cu)
    ptx_file="$OUTPUT_DIR/${kernel_name}.ptx"

    echo -e "${YELLOW}[${KERNEL_COUNT}]${NC} Compiling: ${kernel_name}.cu"

    # Compile
    if "$NVCC" "${NVCC_FLAGS[@]}" "$cu_file" -o "$ptx_file" 2>&1 | grep -v "warning:"; then
        COMPILED_COUNT=$((COMPILED_COUNT + 1))

        # Get file size
        SIZE=$(du -h "$ptx_file" | cut -f1)
        echo -e "    ${GREEN}✓${NC} Success → ${ptx_file} (${SIZE})"

        # Generate Rust constant for PTX path
        echo "pub const ${kernel_name^^}_PTX: &str = include_str!(\"$ptx_file\");" \
            >> "$OUTPUT_DIR/kernel_ptx.rs"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo -e "    ${RED}✗${NC} Failed to compile ${kernel_name}.cu"
    fi

    echo ""
done

# Generate summary
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Build Summary                                               ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Total kernels found: $KERNEL_COUNT"
echo -e "  ${GREEN}Compiled successfully: $COMPILED_COUNT${NC}"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo -e "  ${RED}Failed: $FAILED_COUNT${NC}"
fi
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Architecture: $ARCH"
echo ""

if [ "$COMPILED_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Kernel compilation complete!${NC}"
    exit 0
else
    echo -e "${RED}✗ No kernels compiled successfully${NC}"
    exit 1
fi
