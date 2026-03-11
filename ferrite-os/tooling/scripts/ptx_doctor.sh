#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_SH="${ROOT_DIR}/tooling/scripts/ptx_env.sh"
DIAG_LIB="${ROOT_DIR}/../scripts/install/lib/diag.sh"
DIAG_FORMAT="${FERRITE_DIAG_FORMAT:-plain}"

if [[ -f "$DIAG_LIB" ]]; then
  # shellcheck disable=SC1090
  source "$DIAG_LIB"
fi

emit_diag() {
  local status="$1"
  local code="$2"
  local summary="$3"
  local remediation="$4"
  local stream="${5:-stderr}"
  if declare -F diag_emit >/dev/null 2>&1; then
    FERRITE_DIAG_FORMAT="$DIAG_FORMAT" diag_emit "doctor.ptx" "$status" "$code" "$summary" "$remediation" "$stream"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --diag-format)
        DIAG_FORMAT="${2:-plain}"
        shift 2
        ;;
      --diag-format=*)
        DIAG_FORMAT="${1#*=}"
        shift
        ;;
      *)
        echo "[ptx_doctor] ERROR: unknown argument: $1" >&2
        emit_diag "FAIL" "DOC-ARGS-0001" "unknown argument: $1" "use --diag-format <plain|json>"
        exit 1
        ;;
    esac
  done
}

parse_args "$@"

if [ ! -x "$ENV_SH" ]; then
  echo "[ptx_doctor] ERROR: tooling/scripts/ptx_env.sh not found or not executable" >&2
  emit_diag "FAIL" "DOC-PREF-0001" "tooling/scripts/ptx_env.sh not found or not executable" "restore or chmod +x ferrite-os/tooling/scripts/ptx_env.sh"
  exit 1
fi

if ! "$ENV_SH" --format env --quiet >/dev/null 2>&1; then
  echo "[ptx_doctor] ERROR: CUDA toolkit not found. Set CUDA_PATH/CUDA_HOME." >&2
  emit_diag "FAIL" "DOC-PREF-0002" "CUDA toolkit not found" "set CUDA_PATH/CUDA_HOME and ensure nvcc is available"
  exit 1
fi

while IFS='=' read -r key val; do
  if [ -n "$key" ]; then
    export "$key=$val"
  fi
done < <("$ENV_SH" --format env --quiet)

echo "PTX-OS Doctor"
echo "CUDA_PATH=${CUDA_PATH}"
echo "CUDA_LIB=${CUDA_LIB}"
echo "NVCC=${NVCC}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "GPU_SM=${GPU_SM}"
echo "PTX_GPU_SM=${PTX_GPU_SM}"
echo

fail=0

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ptx_doctor] WARN: nvidia-smi not found. GPU driver may be missing."
  emit_diag "WARN" "DOC-GPU-0001" "nvidia-smi not found" "install NVIDIA driver tooling or verify GPU driver state"
else
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 | \
    sed 's/^/[ptx_doctor] GPU: /'
  emit_diag "PASS" "DOC-GPU-0002" "nvidia-smi query succeeded" "none" stdout
fi

if [ ! -x "$NVCC" ]; then
  echo "[ptx_doctor] ERROR: nvcc not found at ${NVCC}" >&2
  emit_diag "FAIL" "DOC-CUDA-0001" "nvcc not found at ${NVCC}" "fix CUDA toolkit install or update CUDA_PATH/CUDA_HOME"
  fail=1
else
  emit_diag "PASS" "DOC-CUDA-0002" "nvcc present at ${NVCC}" "none" stdout
fi

if [ ! -f "${ROOT_DIR}/lib/libptx_os.so" ]; then
  echo "[ptx_doctor] WARN: lib/libptx_os.so not built (run ./tooling/scripts/ptx_build.sh all)"
  emit_diag "WARN" "DOC-BUILD-0001" "libptx_os.so not built" "run ./tooling/scripts/ptx_build.sh all from ferrite-os"
else
  emit_diag "PASS" "DOC-BUILD-0002" "libptx_os.so present" "none" stdout
fi
if [ ! -f "${ROOT_DIR}/lib/libptx_hook.so" ]; then
  echo "[ptx_doctor] WARN: lib/libptx_hook.so not built (run ./tooling/scripts/ptx_build.sh all)"
  emit_diag "WARN" "DOC-BUILD-0003" "libptx_hook.so not built" "run ./tooling/scripts/ptx_build.sh all from ferrite-os"
else
  emit_diag "PASS" "DOC-BUILD-0004" "libptx_hook.so present" "none" stdout
fi

if [ "$fail" -ne 0 ]; then
  emit_diag "FAIL" "DOC-OVERALL-0001" "ptx_doctor completed with blocking failures" "resolve FAIL diagnostics and rerun"
  exit 1
fi

echo "[ptx_doctor] OK"
emit_diag "PASS" "DOC-OVERALL-0002" "ptx_doctor completed successfully" "none" stdout
