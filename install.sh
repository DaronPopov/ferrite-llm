#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${FERRITE_REPO_URL:-https://github.com/DaronPopov/ferrite-llm.git}"
INSTALL_ROOT="${FERRITE_INSTALL_ROOT:-$HOME/.local/share/ferrite-llm}"
SRC_DIR="${FERRITE_SRC_DIR:-$INSTALL_ROOT/src/ferrite-llm}"
PREFIX="${FERRITE_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"

need_cmd() {
    command -v "$1" >/dev/null 2>&1
}

log() {
    printf '[ferrite-install] %s\n' "$1"
}

fail() {
    printf '[ferrite-install] error: %s\n' "$1" >&2
    exit 1
}

if ! need_cmd git; then
    fail "git is required"
fi

if ! need_cmd cargo; then
    fail "cargo is required; install Rust first via https://rustup.rs"
fi

if ! need_cmd rustup; then
    fail "rustup is required; install Rust first via https://rustup.rs"
fi

mkdir -p "$INSTALL_ROOT/src" "$BIN_DIR"

if [ -d "$SRC_DIR/.git" ]; then
    log "Updating existing repo at $SRC_DIR"
    git -C "$SRC_DIR" fetch origin
    git -C "$SRC_DIR" checkout main
    git -C "$SRC_DIR" pull --ff-only origin main
else
    log "Cloning $REPO_URL into $SRC_DIR"
    git clone "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"

log "Refreshing Cargo dependencies"
cargo update

log "Installing ferrite runtime into $PREFIX"
cargo install --path crates/ferrite-cli --root "$PREFIX" --force

log "Provisioning WASM toolchain and local prerequisites"
"$BIN_DIR/ferrite-rt" setup --skip-deps-update

log "Building sample guest component"
cargo build -p mistral-inference --target wasm32-wasip1 --release
wasm-tools component embed wit \
  target/wasm32-wasip1/release/mistral_inference.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
wasm-tools component new \
  target/wasm32-wasip1/release/mistral_inference.embed.wasm \
  --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
  -o target/wasm32-wasip1/release/mistral_inference.component.wasm

cat <<EOF

Ferrite install complete.

Repo:
  $SRC_DIR

Binary:
  $BIN_DIR/ferrite-rt

Sample component:
  $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

Suggested PATH update:
  export PATH="$BIN_DIR:\$PATH"

Quick run:
  HF_TOKEN=your_token_here \\
  FERRITE_REQUIRE_CUDA=1 \\
  FERRITE_BACKEND=mistralrs \\
  $BIN_DIR/ferrite-rt run \\
    $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

EOF
