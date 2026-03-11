#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${FERRITE_REPO_URL:-https://github.com/DaronPopov/ferrite-llm.git}"
ARCHIVE_URL="${FERRITE_ARCHIVE_URL:-https://github.com/DaronPopov/ferrite-llm/archive/refs/heads/main.tar.gz}"
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

download_to() {
    local url="$1"
    local out="$2"
    if need_cmd curl; then
        curl -fsSL "$url" -o "$out"
    elif need_cmd wget; then
        wget -qO "$out" "$url"
    else
        fail "curl or wget is required"
    fi
}

bootstrap_rust() {
    if need_cmd cargo && need_cmd rustup; then
        return 0
    fi

    log "Rust toolchain not found; installing rustup into user space"
    local rustup_script
    rustup_script="$(mktemp)"
    download_to "https://sh.rustup.rs" "$rustup_script"
    sh "$rustup_script" -y --profile minimal
    rm -f "$rustup_script"

    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"

    need_cmd cargo || fail "cargo not found after rustup install"
    need_cmd rustup || fail "rustup not found after rustup install"
}

refresh_repo() {
    mkdir -p "$INSTALL_ROOT/src"

    if need_cmd git; then
        if [ -d "$SRC_DIR/.git" ]; then
            log "Updating existing repo at $SRC_DIR"
            git -C "$SRC_DIR" fetch origin
            git -C "$SRC_DIR" checkout main
            git -C "$SRC_DIR" pull --ff-only origin main
            return 0
        fi

        log "Cloning $REPO_URL into $SRC_DIR"
        git clone "$REPO_URL" "$SRC_DIR"
        return 0
    fi

    log "git not found; downloading source archive instead"
    rm -rf "$SRC_DIR"
    local archive
    local unpack_dir
    archive="$(mktemp)"
    unpack_dir="$(mktemp -d)"
    download_to "$ARCHIVE_URL" "$archive"
    tar -xzf "$archive" -C "$unpack_dir"
    rm -f "$archive"
    mkdir -p "$(dirname "$SRC_DIR")"
    mv "$unpack_dir"/ferrite-llm-main "$SRC_DIR"
    rmdir "$unpack_dir" 2>/dev/null || true
}

bootstrap_rust

mkdir -p "$BIN_DIR"
refresh_repo

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
  export PATH="$BIN_DIR:\$PATH"
  HF_TOKEN=your_token_here \\
  FERRITE_REQUIRE_CUDA=1 \\
  FERRITE_BACKEND=mistralrs \\
  ferrite-rt run \\
    $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

EOF
