#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LANG_DIR="$ROOT/ferrite-gpu-lang"
OS_LIB="$ROOT/ferrite-os/lib"

export LD_LIBRARY_PATH="$OS_LIB:${LD_LIBRARY_PATH:-}"

cd "$LANG_DIR"

echo "[foundation] running rust gpu language examples"
cargo run --release --example script_runtime
cargo run --release --example script_handoff
cargo run --release --features torch --example script_cv_depth
cargo run --release --features torch --example script_cv_detect

echo "[foundation] done"
