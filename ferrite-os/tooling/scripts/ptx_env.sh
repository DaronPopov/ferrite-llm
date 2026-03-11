#!/usr/bin/env bash
set -euo pipefail

FORMAT="env"
QUIET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --format)
      FORMAT="${2:-env}"
      shift 2
      ;;
    --format=*)
      FORMAT="${1#*=}"
      shift
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

log() {
  if [ "$QUIET" -eq 0 ]; then
    echo "$@" >&2
  fi
}

root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "$(cd "$script_dir/.." && pwd)"
}

normalize_sm() {
  local raw="$1"
  local digits
  digits="$(echo "$raw" | tr -cd '0-9')"
  if [ -n "$digits" ]; then
    echo "$digits"
  fi
}

resolve_cuda_path() {
  local cand
  for cand in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "${CUDA_ROOT:-}"; do
    if [ -n "$cand" ] && [ -f "$cand/include/cuda_runtime.h" ]; then
      echo "$cand"
      return 0
    fi
  done

  if [ -f "/usr/local/cuda/include/cuda_runtime.h" ]; then
    echo "/usr/local/cuda"
    return 0
  fi

  local latest
  latest=""
  for cand in /usr/local/cuda-*; do
    if [ -f "$cand/include/cuda_runtime.h" ]; then
      latest="$cand"
    fi
  done
  if [ -n "$latest" ]; then
    echo "$latest"
    return 0
  fi

  if [ -f "/opt/cuda/include/cuda_runtime.h" ]; then
    echo "/opt/cuda"
    return 0
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    local guess
    guess="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
    if [ -f "$guess/include/cuda_runtime.h" ]; then
      echo "$guess"
      return 0
    fi
  fi

  return 1
}

cuda_lib_dir() {
  local base="$1"
  if [ -d "$base/lib64" ]; then
    echo "$base/lib64"
  else
    echo "$base/lib"
  fi
}

cupti_lib_dir() {
  local base="$1"
  local arch
  arch="$(uname -m)"
  if [ "$arch" = "x86_64" ] && [ -d "$base/targets/x86_64-linux/lib" ]; then
    if [ -f "$base/targets/x86_64-linux/lib/libcupti.so" ] || [ -f "$base/targets/x86_64-linux/lib/libcupti.so.12" ]; then
      echo "$base/targets/x86_64-linux/lib"
      return 0
    fi
  fi
  if [ "$arch" = "aarch64" ] && [ -d "$base/targets/aarch64-linux/lib" ]; then
    if [ -f "$base/targets/aarch64-linux/lib/libcupti.so" ] || [ -f "$base/targets/aarch64-linux/lib/libcupti.so.12" ]; then
      echo "$base/targets/aarch64-linux/lib"
      return 0
    fi
  fi
  if [ -d "$base/extras/CUPTI/lib64" ]; then
    echo "$base/extras/CUPTI/lib64"
    return 0
  fi
  return 1
}

nvcc_path() {
  local base="$1"
  if [ -x "$base/bin/nvcc" ]; then
    echo "$base/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
  fi
}

cuda_version() {
  local nvcc="$1"
  if [ -x "$nvcc" ]; then
    "$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1
  fi
}

detect_sm_nvidia_smi() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$cap" ]; then
    normalize_sm "$cap"
    return 0
  fi

  cap="$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$cap" ]; then
    normalize_sm "$cap"
    return 0
  fi

  return 1
}

detect_sm_jetson_model() {
  local model=""
  if [ -r /proc/device-tree/model ]; then
    model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null | tr '[:upper:]' '[:lower:]')"
  elif [ -r /sys/firmware/devicetree/base/model ]; then
    model="$(tr -d '\0' < /sys/firmware/devicetree/base/model 2>/dev/null | tr '[:upper:]' '[:lower:]')"
  fi

  if [ -z "$model" ]; then
    return 1
  fi

  case "$model" in
    *orin*) echo "87"; return 0 ;;
    *xavier*) echo "72"; return 0 ;;
    *tx2*) echo "62"; return 0 ;;
    *nano*|*tx1*) echo "53"; return 0 ;;
  esac

  return 1
}

detect_sm_nvcc() {
  local nvcc="$1"
  local root="$2"
  local detect_src="$root/scripts/detect_sm.cu"

  if [ ! -x "$nvcc" ] || [ ! -f "$detect_src" ]; then
    return 1
  fi

  local tmp
  tmp="$(mktemp -t ptx_detect_sm.XXXXXX)"
  if "$nvcc" -O2 -std=c++11 "$detect_src" -o "$tmp" >/dev/null 2>&1; then
    local sm
    sm="$($tmp 2>/dev/null | tr -cd '0-9')"
    rm -f "$tmp"
    if [ -n "$sm" ]; then
      echo "$sm"
      return 0
    fi
  else
    rm -f "$tmp"
  fi

  return 1
}

CUDA_PATH_RESOLVED="$(resolve_cuda_path || true)"
if [ -z "$CUDA_PATH_RESOLVED" ]; then
  log "[ptx_env] ERROR: CUDA toolkit not found. Set CUDA_PATH/CUDA_HOME."
  exit 1
fi

CUDA_LIB="$(cuda_lib_dir "$CUDA_PATH_RESOLVED")"
CUPTI_LIB="$(cupti_lib_dir "$CUDA_PATH_RESOLVED" || true)"
NVCC_BIN="$(nvcc_path "$CUDA_PATH_RESOLVED" || true)"
CUDA_VERSION="$(cuda_version "$NVCC_BIN")"

GPU_SM=""
for cand in "${PTX_GPU_SM:-}" "${GPU_SM:-}" "${CUDA_SM:-}"; do
  if [ -n "$cand" ]; then
    GPU_SM="$(normalize_sm "$cand")"
    if [ -n "$GPU_SM" ]; then
      break
    fi
  fi
done

if [ -z "$GPU_SM" ]; then
  GPU_SM="$(detect_sm_nvidia_smi || true)"
fi

if [ -z "$GPU_SM" ]; then
  GPU_SM="$(detect_sm_jetson_model || true)"
fi

if [ -z "$GPU_SM" ]; then
  GPU_SM="$(detect_sm_nvcc "$NVCC_BIN" "$(root_dir)" || true)"
fi

PTX_GPU_SM=""
if [ -n "$GPU_SM" ]; then
  PTX_GPU_SM="sm_${GPU_SM}"
fi

case "$FORMAT" in
  env)
    echo "CUDA_PATH=$CUDA_PATH_RESOLVED"
    echo "CUDA_LIB=$CUDA_LIB"
    [ -n "$CUPTI_LIB" ] && echo "CUPTI_LIB=$CUPTI_LIB"
    [ -n "$NVCC_BIN" ] && echo "NVCC=$NVCC_BIN"
    [ -n "$CUDA_VERSION" ] && echo "CUDA_VERSION=$CUDA_VERSION"
    [ -n "$GPU_SM" ] && echo "GPU_SM=$GPU_SM"
    [ -n "$PTX_GPU_SM" ] && echo "PTX_GPU_SM=$PTX_GPU_SM"
    exit 0
    ;;
  make)
    echo "CUDA_PATH:=$CUDA_PATH_RESOLVED"
    echo "CUDA_LIB:=$CUDA_LIB"
    [ -n "$CUPTI_LIB" ] && echo "CUPTI_LIB:=$CUPTI_LIB"
    [ -n "$NVCC_BIN" ] && echo "NVCC:=$NVCC_BIN"
    [ -n "$CUDA_VERSION" ] && echo "CUDA_VERSION:=$CUDA_VERSION"
    [ -n "$GPU_SM" ] && echo "GPU_SM:=$GPU_SM"
    [ -n "$PTX_GPU_SM" ] && echo "PTX_GPU_SM:=$PTX_GPU_SM"
    exit 0
    ;;
  json)
    printf '{"CUDA_PATH":"%s","CUDA_LIB":"%s"' "$CUDA_PATH_RESOLVED" "$CUDA_LIB"
    if [ -n "$CUPTI_LIB" ]; then printf ',"CUPTI_LIB":"%s"' "$CUPTI_LIB"; fi
    if [ -n "$NVCC_BIN" ]; then printf ',"NVCC":"%s"' "$NVCC_BIN"; fi
    if [ -n "$CUDA_VERSION" ]; then printf ',"CUDA_VERSION":"%s"' "$CUDA_VERSION"; fi
    if [ -n "$GPU_SM" ]; then printf ',"GPU_SM":"%s"' "$GPU_SM"; fi
    if [ -n "$PTX_GPU_SM" ]; then printf ',"PTX_GPU_SM":"%s"' "$PTX_GPU_SM"; fi
    printf '}'
    exit 0
    ;;
  *)
    echo "Unknown format: $FORMAT" >&2
    exit 2
    ;;
esac
