#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER_DIR="$ROOT_DIR/installer"

need_cmd() {
    command -v "$1" >/dev/null 2>&1
}

download_to() {
    local url="$1"
    local out="$2"
    if need_cmd curl; then
        curl -fsSL "$url" -o "$out"
    elif need_cmd wget; then
        wget -qO "$out" "$url"
    else
        printf 'bootstrap error: curl or wget is required\n' >&2
        exit 1
    fi
}

bootstrap_rust() {
    if need_cmd cargo && need_cmd rustup; then
        return 0
    fi

    printf '[installer-bootstrap] installing rustup in user space\n'
    local script
    script="$(mktemp)"
    download_to "https://sh.rustup.rs" "$script"
    sh "$script" -y --profile minimal
    rm -f "$script"

    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
}

main() {
    bootstrap_rust
    cargo run --manifest-path "$INSTALLER_DIR/Cargo.toml" -- bootstrap-all "$@"
}

main "$@"
