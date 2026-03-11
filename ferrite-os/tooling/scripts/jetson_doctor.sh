#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_SH="${ROOT_DIR}/tooling/scripts/ptx_env.sh"
LOG_FILE=""
STRICT=0

usage() {
  cat <<'USAGE'
Usage: tooling/scripts/jetson_doctor.sh [--log-file PATH] [--strict]

Checks Jetson bring-up prerequisites and validates runtime log signatures.

Options:
  --log-file PATH   Validate expected runtime signatures in PATH.
  --strict          Treat warnings as failures.
  -h, --help        Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    --log-file=*)
      LOG_FILE="${1#*=}"
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[jetson_doctor] ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PASS=0
WARN=0
FAIL=0

pass() {
  echo "[jetson_doctor] PASS: $*"
  PASS=$((PASS + 1))
}

warn() {
  echo "[jetson_doctor] WARN: $*" >&2
  WARN=$((WARN + 1))
}

fail() {
  echo "[jetson_doctor] FAIL: $*" >&2
  FAIL=$((FAIL + 1))
}

detect_jetson_model() {
  local model=""
  if [[ -r /proc/device-tree/model ]]; then
    model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
  elif [[ -r /sys/firmware/devicetree/base/model ]]; then
    model="$(tr -d '\0' < /sys/firmware/devicetree/base/model 2>/dev/null || true)"
  fi
  echo "$model"
}

expected_sm_for_model() {
  local model_lc="$1"
  case "$model_lc" in
    *orin*) echo "87" ;;
    *xavier*) echo "72" ;;
    *tx2*) echo "62" ;;
    *nano*|*tx1*) echo "53" ;;
    *) echo "" ;;
  esac
}

if [[ ! -x "$ENV_SH" ]]; then
  fail "missing executable: $ENV_SH"
  echo "[jetson_doctor] Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
  exit 1
fi

if ! "$ENV_SH" --format env --quiet >/dev/null 2>&1; then
  fail "ptx_env failed to resolve CUDA toolkit; set CUDA_PATH/CUDA_HOME"
  echo "[jetson_doctor] Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
  exit 1
fi

while IFS='=' read -r key val; do
  if [[ -n "$key" ]]; then
    export "$key=$val"
  fi
done < <("$ENV_SH" --format env --quiet)

ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]]; then
  pass "host architecture is aarch64"
else
  fail "host architecture is $ARCH (expected aarch64 for Jetson)"
fi

JETSON_MODEL="$(detect_jetson_model)"
JETSON_MODEL_LC="$(echo "$JETSON_MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ -n "$JETSON_MODEL" ]]; then
  pass "device tree model detected: ${JETSON_MODEL}"
else
  warn "device tree model unavailable; cannot confirm Jetson SKU"
fi

EXPECTED_SM="$(expected_sm_for_model "$JETSON_MODEL_LC")"
if [[ -n "$EXPECTED_SM" ]]; then
  pass "expected SM from model: sm_${EXPECTED_SM}"
  if [[ "$EXPECTED_SM" -lt 75 ]]; then
    fail "model maps to sm_${EXPECTED_SM}, but current kernel profile requires sm_75+"
  fi
else
  warn "could not map model to expected SM (Orin/Xavier/TX2/Nano map)"
fi

if [[ -n "${GPU_SM:-}" ]]; then
  pass "detected SM from ptx_env: sm_${GPU_SM}"
  if [[ "${GPU_SM}" -lt 75 ]]; then
    fail "detected SM sm_${GPU_SM} is unsupported by current kernel profile (sm_75+ required)"
  fi
  if [[ -n "$EXPECTED_SM" && "$GPU_SM" != "$EXPECTED_SM" ]]; then
    fail "SM mismatch: model suggests sm_${EXPECTED_SM}, ptx_env reports sm_${GPU_SM}"
  fi
else
  fail "ptx_env did not report GPU_SM; pass --sm explicitly to installer/build"
fi

if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/targets/aarch64-linux/lib" ]]; then
  pass "CUDA aarch64 target libs present: ${CUDA_PATH}/targets/aarch64-linux/lib"
else
  warn "CUDA aarch64 target lib dir missing under CUDA_PATH=${CUDA_PATH:-<unset>}"
fi

if [[ -f "${ROOT_DIR}/lib/libptx_os.so" ]]; then
  pass "runtime library present: lib/libptx_os.so"
else
  warn "runtime library missing: lib/libptx_os.so (run make in ferrite-os)"
fi

if [[ -f "${ROOT_DIR}/lib/libptx_hook.so" ]]; then
  pass "hook library present: lib/libptx_hook.so"
else
  warn "hook library missing: lib/libptx_hook.so (optional but recommended)"
fi

if [[ -n "$LOG_FILE" ]]; then
  if [[ ! -f "$LOG_FILE" ]]; then
    fail "log file not found: $LOG_FILE"
  else
    pass "log file found: $LOG_FILE"

    if grep -Eq "\\[Ferrite-OS\\] (Orin unified-memory mode active|Embedded managed-pool mode active)" "$LOG_FILE"; then
      pass "runtime mode signature present"
    else
      fail "missing runtime mode signature (Orin unified-memory or Embedded managed-pool)"
    fi

    if grep -Eq "\\[Ferrite-OS\\] Managed pool allocated:" "$LOG_FILE"; then
      pass "managed pool allocation signature present"
    else
      warn "managed pool allocation signature missing"
    fi

    if grep -Eq "Booting persistent kernel|ptx_os_boot_persistent_kernel|\\[PTX-OS-ORIN-UM\\]" "$LOG_FILE"; then
      if grep -Eq "\\[PTX-OS-ORIN-UM\\] Launching persistent kernel" "$LOG_FILE"; then
        pass "Orin persistent kernel launch signature present"
      else
        fail "persistent kernel requested but Orin launch signature missing"
      fi
      if grep -Eq "\\[GPU-OS-ORIN-UM\\] Kernel initialized" "$LOG_FILE"; then
        pass "Orin kernel initialization signature present"
      else
        fail "persistent kernel requested but kernel init signature missing"
      fi
    else
      warn "no persistent-kernel markers found in log (acceptable if boot kernel disabled)"
    fi
  fi
fi

echo
echo "[jetson_doctor] Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
if [[ "$STRICT" -eq 1 && "$WARN" -gt 0 ]]; then
  echo "[jetson_doctor] STRICT mode enabled: warnings treated as failures" >&2
  FAIL=$((FAIL + WARN))
fi

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi

exit 0
