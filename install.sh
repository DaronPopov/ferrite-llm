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
LEGACY_BUILD_EVERYTHING="${FERRITE_BUILD_EVERYTHING:-}"
PROFILE="${FERRITE_INSTALL_PROFILE:-}"
GRAPHICS_JOBS="${FERRITE_GRAPHICS_JOBS:-}"
SKIP_GRAPHICS_TESTS="${FERRITE_GRAPHICS_SKIP_TESTS:-0}"
SELECTED_MODULES=()
INSTALL_RESULTS=()
CUDA_HOME=""

usage() {
    cat <<'EOF'
Ferrite installer

Usage:
  install.sh [--profile PROFILE] [--help] [--list-profiles]

Profiles:
  runtime   Build the smallest supported Ferrite runtime slice
  platform  Build runtime + ferrite-os workspace
  full      Build platform + ferrite-gpu-lang
  mega      Build the full Ferrite engine stack

Options:
  --profile PROFILE  Select install profile. Default: mega
  --list-profiles    Print available profiles and exit
  --help, -h         Show this message

Environment:
  FERRITE_INSTALL_PROFILE     Default install profile
  FERRITE_ENABLE_TLSF_ALLOC   Build ferrite-rt with TLSF allocator support
  FERRITE_GRAPHICS_JOBS       Override parallelism for ferrite-graphics build
  FERRITE_GRAPHICS_SKIP_TESTS Skip ferrite-graphics ctest when set to 1

Compatibility:
  FERRITE_BUILD_EVERYTHING=0 maps to --profile runtime
  FERRITE_BUILD_EVERYTHING=1 maps to --profile mega
EOF
}

list_profiles() {
    cat <<'EOF'
runtime
platform
full
mega
EOF
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

record_result() {
    local module="$1"
    local status="$2"
    local detail="${3:-}"
    if [ -n "$detail" ]; then
        INSTALL_RESULTS+=("$module:$status:$detail")
    else
        INSTALL_RESULTS+=("$module:$status")
    fi
}

build_native_runtime() {
    log "Building native PTX-OS runtime library"
    make -C ferrite-os lib/libptx_os.so
    record_result "native-runtime" "built" "ferrite-os/lib/libptx_os.so"
}

build_runtime_workspace() {
    log "Building ferrite-llm workspace"
    cargo build --workspace
    record_result "runtime-workspace" "built"
}

build_platform_workspace() {
    log "Building ferrite-os workspace"
    cargo build --manifest-path ferrite-os/Cargo.toml --workspace
    record_result "platform-workspace" "built"
}

install_runtime_binary() {
    log "Installing ferrite runtime into $PREFIX"
    if [ "$ENABLE_TLSF_ALLOC" = "1" ]; then
        log "Building ferrite runtime with TLSF allocator support"
        cargo install --path crates/ferrite-cli --root "$PREFIX" --force --features runtime,tlsf-alloc
        mkdir -p "$PREFIX/lib"
        cp ferrite-os/lib/libptx_os.so "$PREFIX/lib/libptx_os.so"
    else
        cargo install --path crates/ferrite-cli --root "$PREFIX" --force --features runtime,cuda
    fi
    record_result "ferrite-rt" "installed" "$BIN_DIR/ferrite-rt"
}

setup_wasm_prereqs() {
    log "Provisioning WASM toolchain and local prerequisites"
    "$BIN_DIR/ferrite-rt" setup --skip-deps-update
    record_result "wasm-setup" "built"
}

build_sample_guest() {
    log "Building sample guest component"
    cargo build -p mistral-inference --target wasm32-wasip1 --release
    wasm-tools component embed wit \
      target/wasm32-wasip1/release/mistral_inference.wasm \
      -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
    wasm-tools component new \
      target/wasm32-wasip1/release/mistral_inference.embed.wasm \
      --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
      -o target/wasm32-wasip1/release/mistral_inference.component.wasm
    record_result "wasm-guest" "built" "target/wasm32-wasip1/release/mistral_inference.component.wasm"
}

run_runtime_smokes() {
    log "Verifying Ferrite custom CUDA kernel path"
    cargo run -p ferrite-core --features cuda --example custom-kernel-smoke
    log "Verifying Ferrite ug-cuda path"
    cargo run -p ferrite-core --features cuda --example ug-cuda-smoke
    record_result "runtime-smokes" "built"
}

build_gpu_lang() {
    log "Building ferrite-gpu-lang"
    cargo build --manifest-path ferrite-gpu-lang/Cargo.toml
    record_result "gpu-lang" "built"
}

graphics_jobs() {
    if [ -n "$GRAPHICS_JOBS" ]; then
        printf '%s\n' "$GRAPHICS_JOBS"
        return 0
    fi
    if need_cmd nproc; then
        nproc
        return 0
    fi
    printf '%s\n' "4"
}

build_graphics() {
    need_cmd cmake || fail "mega profile requires cmake"
    need_cmd ctest || fail "mega profile requires ctest"
    need_cmd ffmpeg || fail "mega profile requires ffmpeg"

    local graphics_src="external/ferrite-graphics"
    local graphics_build="$graphics_src/build"
    local jobs
    jobs="$(graphics_jobs)"

    [ -d "$graphics_src" ] || fail "mega profile requires $graphics_src"

    log "Configuring ferrite-graphics"
    cmake -S "$graphics_src" -B "$graphics_build" -DCMAKE_INSTALL_PREFIX="$PREFIX"

    log "Building ferrite-graphics"
    cmake --build "$graphics_build" -j"$jobs"

    if [ "$SKIP_GRAPHICS_TESTS" != "1" ]; then
        log "Running ferrite-graphics tests"
        ctest --test-dir "$graphics_build" --output-on-failure
    else
        log "Skipping ferrite-graphics tests (FERRITE_GRAPHICS_SKIP_TESTS=1)"
    fi

    log "Installing ferrite-graphics into $PREFIX"
    cmake --install "$graphics_build"
    record_result "graphics" "built" "$PREFIX/bin/ferrite_graphics_lab"
}

run_doctor() {
    log "Running Ferrite doctor for profile $PROFILE"
    "$BIN_DIR/ferrite-rt" doctor --profile "$PROFILE" --repo-root "$SRC_DIR"
    record_result "doctor" "built"
}

profile_includes_platform() {
    case "$PROFILE" in
        platform|full|mega)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

profile_includes_full() {
    case "$PROFILE" in
        full|mega)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

profile_includes_mega() {
    [ "$PROFILE" = "mega" ]
}

configure_profile() {
    if [ -z "$PROFILE" ] && [ -n "$LEGACY_BUILD_EVERYTHING" ]; then
        case "$LEGACY_BUILD_EVERYTHING" in
            0)
                PROFILE="runtime"
                ;;
            1)
                PROFILE="mega"
                ;;
            *)
                fail "unsupported FERRITE_BUILD_EVERYTHING value: $LEGACY_BUILD_EVERYTHING"
                ;;
        esac
    fi

    if [ -z "$PROFILE" ]; then
        PROFILE="mega"
    fi

    case "$PROFILE" in
        runtime)
            SELECTED_MODULES=("native-runtime" "runtime-workspace" "ferrite-rt" "wasm-setup" "wasm-guest" "runtime-smokes" "doctor")
            ;;
        platform)
            SELECTED_MODULES=("native-runtime" "runtime-workspace" "platform-workspace" "ferrite-rt" "wasm-setup" "wasm-guest" "runtime-smokes" "doctor")
            ;;
        full)
            SELECTED_MODULES=("native-runtime" "runtime-workspace" "platform-workspace" "ferrite-rt" "wasm-setup" "wasm-guest" "runtime-smokes" "gpu-lang" "doctor")
            ;;
        mega)
            SELECTED_MODULES=("native-runtime" "runtime-workspace" "platform-workspace" "ferrite-rt" "wasm-setup" "wasm-guest" "runtime-smokes" "gpu-lang" "graphics" "doctor")
            ;;
        *)
            fail "unknown profile: $PROFILE"
            ;;
    esac
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --profile)
                [ "$#" -ge 2 ] || fail "--profile requires a value"
                PROFILE="$2"
                shift 2
                ;;
            --list-profiles)
                list_profiles
                exit 0
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                fail "unknown argument: $1"
                ;;
        esac
    done
}

print_profile_header() {
    log "Selected install profile: $PROFILE"
    log "Modules: ${SELECTED_MODULES[*]}"
}

print_summary() {
    local result
    cat <<EOF

Ferrite install complete.

Profile:
  $PROFILE

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

Module results:
EOF

    for result in "${INSTALL_RESULTS[@]}"; do
        IFS=':' read -r module status detail <<<"$result"
        if [ -n "${detail:-}" ]; then
            printf '  %-18s %-9s %s\n' "$module" "$status" "$detail"
        else
            printf '  %-18s %-9s\n' "$module" "$status"
        fi
    done

    cat <<EOF

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
EOF
}

parse_args "$@"
configure_profile
bootstrap_rust

mkdir -p "$BIN_DIR"
refresh_repo

cd "$SRC_DIR"

CUDA_HOME="$(choose_cuda_home)"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"

if detect_jetson; then
    log "Detected Jetson platform: ${JETSON_MODEL:-aarch64}"
    log "Using CUDA toolkit at $CUDA_HOME"
else
    log "Detected architecture: $ARCH"
    log "Using CUDA toolkit at $CUDA_HOME"
fi

print_profile_header

log "Refreshing Cargo dependencies"
cargo update

if profile_includes_platform; then
    log "Refreshing ferrite-os Cargo dependencies"
    cargo update --manifest-path ferrite-os/Cargo.toml
fi

build_native_runtime
build_runtime_workspace

if profile_includes_platform; then
    build_platform_workspace
fi

install_runtime_binary
setup_wasm_prereqs
build_sample_guest
run_runtime_smokes

if profile_includes_full; then
    build_gpu_lang
fi

if profile_includes_mega; then
    build_graphics
fi

run_doctor

print_summary
