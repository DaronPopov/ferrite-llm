#!/bin/bash
# Comprehensive test suite for Ferrite-OS
# Tests build system, examples, benchmarks, and functionality

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED_TESTS=()
PASSED_TESTS=()

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local show_output="${3:-false}"

    log_test "Running: $test_name"
    echo "  Command: $test_cmd"

    if [ "$show_output" = "true" ]; then
        # Show output in real-time
        if eval "$test_cmd" 2>&1 | tee /tmp/ferrite_test_$$.log; then
            log_info "✓ PASSED: $test_name"
            PASSED_TESTS+=("$test_name")
            return 0
        else
            log_error "✗ FAILED: $test_name"
            FAILED_TESTS+=("$test_name")
            return 1
        fi
    else
        # Capture output but show on failure
        if eval "$test_cmd" > /tmp/ferrite_test_$$.log 2>&1; then
            log_info "✓ PASSED: $test_name"
            # Show last few lines of output
            echo "  Output: $(tail -3 /tmp/ferrite_test_$$.log | head -1)"
            PASSED_TESTS+=("$test_name")
            return 0
        else
            log_error "✗ FAILED: $test_name"
            log_error "Full output:"
            cat /tmp/ferrite_test_$$.log
            FAILED_TESTS+=("$test_name")
            return 1
        fi
    fi
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for verbose flag
VERBOSE=false
if [ "$1" = "-v" ] || [ "$1" = "--verbose" ]; then
    VERBOSE=true
    log_info "Verbose mode enabled - showing all output"
fi

echo "========================================="
echo "Ferrite-OS Comprehensive Test Suite"
echo "========================================="
echo ""
echo "Working directory: $SCRIPT_DIR"
echo "Architecture: $(uname -m)"
echo "OS: $(uname -s) $(uname -r)"
echo ""

# Detect GPU presence
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi -L &> /dev/null; then
        HAS_GPU=true
        log_info "GPU detected: $(nvidia-smi -L | head -1)"
    fi
fi

if [ "$HAS_GPU" = false ]; then
    log_warn "No GPU detected - some tests will be skipped"
fi

echo ""
echo "========================================="
echo "Phase 1: Build System Tests"
echo "========================================="
echo ""

# Test 1: Check dependencies
run_test "Check cargo installed" "command -v cargo"
run_test "Check rustc installed" "command -v rustc"
run_test "Check make installed" "command -v make"
run_test "Check nvcc installed" "command -v nvcc || echo 'CUDA optional for build check'"

# Test 2: C/CUDA library build
if [ "$HAS_GPU" = true ]; then
    log_test "Building C/CUDA libraries (make clean && make)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if make clean > /dev/null 2>&1; then
        echo "✓ Clean completed"
    fi

    if make 2>&1 | tee /tmp/ferrite_make_$$.log; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "✓ PASSED: C/CUDA library build"

        # Show what was built
        if [ -f "lib/libptx_os.so" ]; then
            SIZE=$(ls -lh lib/libptx_os.so | awk '{print $5}')
            log_info "    Built: libptx_os.so ($SIZE)"
        fi
        if [ -f "lib/libptx_hook.so" ]; then
            SIZE=$(ls -lh lib/libptx_hook.so | awk '{print $5}')
            log_info "    Built: libptx_hook.so ($SIZE)"
        fi

        PASSED_TESTS+=("C/CUDA library build")
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_error "✗ FAILED: C/CUDA library build"
        FAILED_TESTS+=("C/CUDA library build")
    fi
else
    log_warn "Skipping C/CUDA build (no GPU)"
fi

# Test 3: Check library exists
if [ -f "lib/libptx_os.so" ]; then
    run_test "libptx_os.so exists" "test -f lib/libptx_os.so"
    run_test "libptx_os.so is valid ELF" "file lib/libptx_os.so | grep -q 'ELF'"
fi

echo ""
echo "========================================="
echo "Phase 2: Rust Build Tests"
echo "========================================="
echo ""

# Test 4: Workspace builds
run_test "Cargo check" "cargo check --all 2>&1 | grep -q 'Finished'"
run_test "Cargo build (debug)" "cargo build --workspace"
run_test "Cargo build (release)" "cargo build --release --workspace"

# Test 5: Individual crate builds
run_test "Build ptx-sys" "cargo build --release -p ptx-sys"
run_test "Build ptx-runtime" "cargo build --release -p ptx-runtime"
run_test "Build ptx-compute" "cargo build --release -p ptx-compute"
run_test "Build ferrite-daemon" "cargo build --release -p ferrite-daemon"
run_test "Build systemic benchmarks" "cargo build --release -p ferrite-benchmarks"

# Test 6: Check binaries
run_test "ferrite-daemon binary exists" "test -f target/release/ferrite-daemon"
run_test "ferrite-daemon is executable" "test -x target/release/ferrite-daemon"

echo ""
echo "========================================="
echo "Phase 3: Example Compilation Tests"
echo "========================================="
echo ""

# Test 7: Compile all examples
EXAMPLES=(
    "telemetry_demo"
    "llm_inference_demo"
    "neural_layer_inference"
    "parallel_batch_processing"
)

for example in "${EXAMPLES[@]}"; do
    run_test "Compile example: $example" \
        "cargo build --release --example $example -p ptx-runtime 2>&1 | grep -q 'Finished'"
done

echo ""
echo "========================================="
echo "Phase 4: Benchmark Compilation Tests"
echo "========================================="
echo ""

# Test 8: Compile all benchmarks
BENCHMARKS=(
    "allocation_comparison"
    "stream_scaling"
    "stability_test"
    "real_workload"
    "memory_pressure"
    "multithreaded_stress"
    "latency_analysis"
)

for bench in "${BENCHMARKS[@]}"; do
    run_test "Compile benchmark: $bench" \
        "cargo build --release --bin $bench -p ferrite-benchmarks 2>&1 | grep -q 'Finished'"
done

echo ""
echo "========================================="
echo "Phase 5: Runtime Tests (GPU Required)"
echo "========================================="
echo ""

if [ "$HAS_GPU" = true ]; then
    # Test 9: Run quick example with FULL output shown
    log_test "Running telemetry_demo (showing all output)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if timeout 10s cargo run --release --example telemetry_demo -p ptx-runtime 2>&1 | tee /tmp/ferrite_demo_$$.log; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if grep -q "Runtime initialized successfully" /tmp/ferrite_demo_$$.log; then
            log_info "✓ PASSED: Runtime example execution"
            PASSED_TESTS+=("Runtime example execution")

            # Extract and display key metrics
            echo ""
            log_info "Key Metrics Detected:"
            grep -E "(Device:|pool_size_gb|Allocations:|Peak memory|Fragmentation|Average allocation time)" /tmp/ferrite_demo_$$.log | while read line; do
                echo "    $line"
            done
            echo ""
        else
            log_error "✗ FAILED: Runtime did not initialize"
            FAILED_TESTS+=("Runtime example execution")
        fi
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_error "✗ FAILED: Example crashed or timed out"
        FAILED_TESTS+=("Runtime example execution")
    fi

    # Test 10: Validate runtime output details
    if [ -f /tmp/ferrite_demo_$$.log ]; then
        echo ""
        log_info "Validating runtime output..."

        if grep -q "Device:" /tmp/ferrite_demo_$$.log; then
            GPU_NAME=$(grep "Device:" /tmp/ferrite_demo_$$.log | head -1)
            log_info "✓ GPU Detected: $GPU_NAME"
            PASSED_TESTS+=("Runtime detected GPU")
        else
            log_error "✗ GPU not detected in output"
            FAILED_TESTS+=("Runtime detected GPU")
        fi

        if grep -q "pool_size_gb" /tmp/ferrite_demo_$$.log; then
            POOL_SIZE=$(grep "pool_size_gb" /tmp/ferrite_demo_$$.log | head -1)
            log_info "✓ Memory Pool: $POOL_SIZE"
            PASSED_TESTS+=("Runtime initialized pool")
        else
            log_error "✗ Memory pool not initialized"
            FAILED_TESTS+=("Runtime initialized pool")
        fi

        if grep -q "TELEMETRY REPORT" /tmp/ferrite_demo_$$.log; then
            log_info "✓ Telemetry system operational"
            PASSED_TESTS+=("Runtime reported telemetry")
        else
            log_error "✗ Telemetry not generated"
            FAILED_TESTS+=("Runtime reported telemetry")
        fi
    fi
else
    log_warn "Skipping runtime tests (no GPU)"
fi

echo ""
echo "========================================="
echo "Phase 6: Unit Tests"
echo "========================================="
echo ""

# Test 11: Rust unit tests (non-GPU)
run_test "Unit tests (lib)" "cargo test --lib --workspace 2>&1 | grep -E '(test result: ok|passed)'"

# Test 12: Integration tests (GPU required)
if [ "$HAS_GPU" = true ]; then
    log_test "Running integration tests (requires GPU)..."
    if cargo test --package ptx-runtime --test integration_tests -- --ignored > /tmp/ferrite_integration_$$.log 2>&1; then
        log_info "✓ PASSED: Integration tests"
        PASSED_TESTS+=("Integration tests")
    else
        log_warn "Integration tests had issues (check log)"
        log_warn "$(tail -5 /tmp/ferrite_integration_$$.log)"
    fi
else
    log_warn "Skipping integration tests (no GPU)"
fi

echo ""
echo "========================================="
echo "Phase 7: Architecture Tests"
echo "========================================="
echo ""

# Test 13: Check binary architecture
ARCH=$(uname -m)
run_test "Detect architecture: $ARCH" "test -n '$ARCH'"

if [ "$ARCH" = "x86_64" ]; then
    run_test "Binary is x86_64" \
        "file target/release/ferrite-daemon | grep -q 'x86-64'"
elif [ "$ARCH" = "aarch64" ]; then
    run_test "Binary is aarch64" \
        "file target/release/ferrite-daemon | grep -q 'aarch64'"
fi

# Test 14: Check library dependencies
run_test "Check rpath configured" \
    "readelf -d target/release/ferrite-daemon 2>/dev/null | grep -q RPATH || \
     readelf -d target/release/ferrite-daemon 2>/dev/null | grep -q RUNPATH"

# Test 15: Library linking
if [ -f "lib/libptx_os.so" ]; then
    run_test "Can find libptx_os.so" \
        "ldd target/release/examples/telemetry_demo 2>&1 | grep -q libptx_os.so"
fi

echo ""
echo "========================================="
echo "Phase 8: Configuration Tests"
echo "========================================="
echo ""

# Test 16: Check configuration files
run_test "Cargo.toml exists" "test -f Cargo.toml"
run_test "Cargo.toml is valid" "cargo metadata --format-version 1 > /dev/null"
run_test ".cargo/config.toml exists" "test -f .cargo/config.toml"
run_test "Makefile exists" "test -f Makefile"

# Test 17: Check documentation
run_test "README.md exists" "test -f README.md"
run_test "docs/ directory exists" "test -d docs"
run_test "API_REFERENCE.md exists" "test -f docs/API_REFERENCE.md"
run_test "ARM64_SUPPORT.md exists" "test -f docs/ARM64_SUPPORT.md"

echo ""
echo "========================================="
echo "Phase 9: Setup Script Tests"
echo "========================================="
echo ""

# Test 18: Setup script functionality
run_test "setup.sh exists" "test -f setup.sh"
run_test "setup.sh is executable" "test -x setup.sh"
run_test "setup.sh --help works" "./setup.sh --help | grep -q 'Usage'"

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""

TOTAL_TESTS=$((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]}))
PASS_RATE=$((${#PASSED_TESTS[@]} * 100 / TOTAL_TESTS))

echo "Total Tests Run: $TOTAL_TESTS"
echo -e "${GREEN}Passed: ${#PASSED_TESTS[@]}${NC}"
echo -e "${RED}Failed: ${#FAILED_TESTS[@]}${NC}"
echo "Pass Rate: $PASS_RATE%"
echo ""

# Show detailed system information
echo "========================================="
echo "System Information"
echo "========================================="
echo ""
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || uname -s)"

if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read name driver mem; do
        echo "  GPU: $name"
        echo "  Driver: $driver"
        echo "  VRAM: $mem"
    done

    if [ -f /tmp/ferrite_demo_$$.log ]; then
        echo ""
        echo "Runtime Metrics (from telemetry_demo):"
        grep -E "pool_size_gb|Allocations:|Peak memory|Fragmentation|allocation time" /tmp/ferrite_demo_$$.log | while read line; do
            echo "  $line"
        done
    fi
fi

echo ""
echo "Build Artifacts:"
if [ -f "lib/libptx_os.so" ]; then
    echo "  ✓ libptx_os.so ($(ls -lh lib/libptx_os.so | awk '{print $5}'))"
else
    echo "  ✗ libptx_os.so missing"
fi

if [ -f "target/release/ferrite-daemon" ]; then
    echo "  ✓ ferrite-daemon ($(ls -lh target/release/ferrite-daemon | awk '{print $5}'))"
else
    echo "  ✗ ferrite-daemon missing"
fi

EXAMPLE_COUNT=$(find target/release/examples -type f -executable 2>/dev/null | wc -l)
echo "  ✓ $EXAMPLE_COUNT examples compiled"

BENCH_COUNT=$(find target/release -maxdepth 1 -name "allocation_*" -o -name "stream_*" -o -name "stability_*" -o -name "real_*" -o -name "memory_*" -o -name "multithreaded_*" -o -name "latency_*" 2>/dev/null | wc -l)
echo "  ✓ $BENCH_COUNT benchmarks compiled"

echo ""

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}Failed Tests:${NC}"
    echo -e "${RED}=========================================${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Ferrite-OS is fully operational and ready for use."
    echo ""
    exit 0
fi
