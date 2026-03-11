#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_OUT="${ROOT_DIR}/tooling/scripts/ptx_env.sh"

if [ ! -x "$ENV_OUT" ]; then
  echo "[ptx_build] ERROR: tooling/scripts/ptx_env.sh not found or not executable" >&2
  exit 1
fi

# Load centralized env
while IFS='=' read -r key val; do
  if [ -n "$key" ]; then
    export "$key=$val"
  fi
done < <("$ENV_OUT" --format env --quiet)

TARGET="all"
STRICT_FREE=0
RUST_DEBUG=0

while [ $# -gt 0 ]; do
  case "$1" in
    all|lib|hook|rust|test|clean|info)
      TARGET="$1"
      shift
      ;;
    --strict-free)
      STRICT_FREE=1
      shift
      ;;
    --debug)
      RUST_DEBUG=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

MAKE_VARS=()
if [ "$STRICT_FREE" -eq 1 ]; then
  MAKE_VARS+=(PTX_STRICT_FREE=1)
fi

run_make() {
  (cd "$ROOT_DIR" && make "${MAKE_VARS[@]}" "$@")
}

case "$TARGET" in
  all)
    run_make all
    if [ "$RUST_DEBUG" -eq 1 ]; then
      run_make rust-debug
    else
      run_make rust
    fi
    ;;
  lib)
    run_make lib/libptx_os.so
    ;;
  hook)
    run_make hook
    ;;
  rust)
    if [ "$RUST_DEBUG" -eq 1 ]; then
      run_make rust-debug
    else
      run_make rust
    fi
    ;;
  test)
    run_make test
    ;;
  clean)
    run_make clean
    ;;
  info)
    run_make info
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    exit 2
    ;;
esac
