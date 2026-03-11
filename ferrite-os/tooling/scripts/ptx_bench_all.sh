#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_SH="$ROOT_DIR/tooling/scripts/ptx_env.sh"
BUILD_SH="$ROOT_DIR/tooling/scripts/ptx_build.sh"

if [ ! -x "$ENV_SH" ]; then
  echo "[ptx_bench] ERROR: $ENV_SH not found or not executable" >&2
  exit 1
fi

if [ ! -x "$BUILD_SH" ]; then
  echo "[ptx_bench] ERROR: $BUILD_SH not found or not executable" >&2
  exit 1
fi

OUT_FILE=""
DO_BUILD=1
BASELINE=0
WORKSPACE_DIR="$ROOT_DIR"

usage() {
  cat <<USAGE
Usage: $0 [--no-build] [--baseline] [--out <file>]

  --no-build   Skip build step
  --baseline   Enable baseline env flags for compatible benchmarks
  --out FILE   Write results to FILE (default: benchmarks/ptx_bench_YYYYMMDD_HHMMSS.txt)
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --baseline)
      BASELINE=1
      shift
      ;;
    --out)
      OUT_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# Load centralized env
while IFS='=' read -r key val; do
  if [ -n "$key" ]; then
    export "$key=$val"
  fi
done < <("$ENV_SH" --format env --quiet)

export LD_LIBRARY_PATH="$ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$OUT_FILE" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_FILE="$ROOT_DIR/benchmarks/ptx_bench_${ts}.txt"
fi

mkdir -p "$(dirname "$OUT_FILE")"

run_cmd() {
  local title="$1"
  shift
  echo "=== $title ===" | tee -a "$OUT_FILE"
  echo "cmd: $*" | tee -a "$OUT_FILE"
  "$@" 2>&1 | tee -a "$OUT_FILE"
  echo "" | tee -a "$OUT_FILE"
}

run_example_if_exists() {
  local title="$1"
  local example_path="$2"
  local cmd="$3"

  if [ ! -f "$example_path" ]; then
    echo "=== $title ===" | tee -a "$OUT_FILE"
    echo "SKIP: missing example file: $example_path" | tee -a "$OUT_FILE"
    echo "" | tee -a "$OUT_FILE"
    return 0
  fi

  run_cmd "$title" bash -lc "cd '$WORKSPACE_DIR' && $cmd"
}

{
  echo "PTX-OS Benchmark Run"
  echo "Timestamp: $(date)"
  echo "Host: $(hostname)"
  echo "Workspace: $WORKSPACE_DIR"
  if command -v git >/dev/null 2>&1; then
    echo "Commit: $(git -C "$WORKSPACE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)"
  fi
  echo ""
  echo "Environment:"
  "$ENV_SH" --format env --quiet
  echo ""
} | tee "$OUT_FILE"

if [ "$DO_BUILD" -eq 1 ]; then
  run_cmd "Build (core + rust)" "$BUILD_SH" all
fi

if [ "$BASELINE" -eq 1 ]; then
  export PTX_BENCH_BASELINE=1
  export PTX_BENCH_ASYNC=1
fi

run_example_if_exists \
  "Runtime Jitter Benchmark" \
  "$ROOT_DIR/ptx-runtime/examples/jitter_benchmark.rs" \
  "cargo run --release -p ptx-runtime --example jitter_benchmark"

run_example_if_exists \
  "Fused Kernel Benchmark" \
  "$ROOT_DIR/ptx-runtime/examples/fused_kernel_benchmark.rs" \
  "cargo run --release -p ptx-runtime --example fused_kernel_benchmark"

run_example_if_exists \
  "Candle Performance Benchmark" \
  "$ROOT_DIR/ptx-runtime/examples/candle_performance_benchmark.rs" \
  "cargo run --release -p ptx-runtime --example candle_performance_benchmark"

run_example_if_exists \
  "Dynamic-Shape Benchmark" \
  "$ROOT_DIR/crates/internal/ptx-tensor/examples/bench_dynamic_shapes.rs" \
  "cargo run --release -p ptx-tensor --example bench_dynamic_shapes"

run_example_if_exists \
  "Elementwise Ops Benchmark" \
  "$ROOT_DIR/crates/internal/ptx-tensor/examples/bench_ops.rs" \
  "cargo run --release -p ptx-tensor --example bench_ops"

run_example_if_exists \
  "Matmul Benchmark" \
  "$ROOT_DIR/crates/internal/ptx-tensor/examples/bench_matmul.rs" \
  "cargo run --release -p ptx-tensor --example bench_matmul"

echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
