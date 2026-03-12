#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${FERRITE_REPO_URL:-https://github.com/DaronPopov/ferrite-llm.git}"
ARCHIVE_URL="${FERRITE_ARCHIVE_URL:-https://github.com/DaronPopov/ferrite-llm/archive/refs/heads/main.tar.gz}"
INSTALL_ROOT="${FERRITE_INSTALL_ROOT:-$HOME/.local/share/ferrite-llm}"
SRC_DIR="${FERRITE_SRC_DIR:-$INSTALL_ROOT/src/ferrite-llm}"
PREFIX="${FERRITE_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
ARCH="$(uname -m)"
JETSON_MODEL=""
ENABLE_TLSF_ALLOC="${FERRITE_ENABLE_TLSF_ALLOC:-0}"

detect_jetson() {
    if [ "$ARCH" != "aarch64" ]; then
        return 1
    fi

    if [ -r /proc/device-tree/model ]; then
        JETSON_MODEL="$(tr -d '\0' < /proc/device-tree/model)"
    elif [ -r /etc/nv_tegra_release ]; then
        JETSON_MODEL="NVIDIA Jetson"
    fi

    case "$JETSON_MODEL" in
        *Jetson*|*Orin*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

choose_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME:-}" ]; then
        printf '%s\n' "$CUDA_HOME"
        return 0
    fi

    if [ -n "${CUDA_PATH:-}" ] && [ -d "${CUDA_PATH:-}" ]; then
        printf '%s\n' "$CUDA_PATH"
        return 0
    fi

    if detect_jetson && [ -d /usr/local/cuda-arm64 ]; then
        printf '%s\n' "/usr/local/cuda-arm64"
        return 0
    fi

    if [ -d /usr/local/cuda ]; then
        printf '%s\n' "/usr/local/cuda"
        return 0
    fi

    if need_cmd nvcc; then
        dirname "$(dirname "$(command -v nvcc)")"
        return 0
    fi

    printf '%s\n' "/usr/local/cuda"
}

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

CUDA_HOME="$(choose_cuda_home)"
export CUDA_HOME

if detect_jetson; then
    log "Detected Jetson platform: ${JETSON_MODEL:-aarch64}"
    log "Using CUDA toolkit at $CUDA_HOME"
else
    log "Detected architecture: $ARCH"
    log "Using CUDA toolkit at $CUDA_HOME"
fi

log "Refreshing Cargo dependencies"
cargo update

log "Installing ferrite runtime into $PREFIX"
if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then
    log "Building ferrite runtime with TLSF allocator support"
    cargo install --path crates/ferrite-cli --root "$PREFIX" --force --features tlsf-alloc
    mkdir -p "$PREFIX/lib"
    cp ferrite-os/lib/libptx_os.so "$PREFIX/lib/libptx_os.so"
else
    cargo install --path crates/ferrite-cli --root "$PREFIX" --force
fi

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

log "Verifying Ferrite custom CUDA kernel path"
cargo run -p ferrite-core --features cuda --example custom-kernel-smoke

log "Verifying Ferrite ug-cuda path"
cargo run -p ferrite-core --features cuda --example ug-cuda-smoke

cat <<EOF

Ferrite install complete.

Repo:
  $SRC_DIR

Binary:
  $BIN_DIR/ferrite-rt

Sample component:
  $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

Detected platform:
  $ARCH${JETSON_MODEL:+ ($JETSON_MODEL)}

CUDA toolkit:
  $CUDA_HOME

TLSF allocator build:
  $(if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then printf '%s' "enabled"; else printf '%s' "disabled"; fi)

Suggested PATH update:
  export PATH="$BIN_DIR:\$PATH"

Suggested library path update:
  $(if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then printf 'export LD_LIBRARY_PATH="%s/lib:\\$LD_LIBRARY_PATH"' "$PREFIX"; else printf 'not required'; fi)

Quick run:
  export PATH="$BIN_DIR:\$PATH"
  $(if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then printf 'export LD_LIBRARY_PATH="%s/lib:\\$LD_LIBRARY_PATH"' "$PREFIX"; fi)
  HF_TOKEN=your_token_here \\
  FERRITE_REQUIRE_CUDA=1 \\
  FERRITE_BACKEND=mistralrs \\
  ferrite-rt run \\
    $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

TLSF runtime toggle:
  $(if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then printf 'FERRITE_TLSF_ALLOC=1 ferrite-rt info'; else printf 're-run install with FERRITE_ENABLE_TLSF_ALLOC=1 to build TLSF support'; fi)
  FERRITE_TLSF_ALLOC=1 FERRITE_BACKEND=candle ferrite-rt run \\
    $SRC_DIR/target/wasm32-wasip1/release/mistral_inference.component.wasm

EOF
