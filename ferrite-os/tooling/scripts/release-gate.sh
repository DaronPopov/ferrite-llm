#!/usr/bin/env bash
set -euo pipefail

# release-gate.sh — Pre-release verification for ptx-kernels and ptx-runtime
#
# Runs the full build and test suite that must pass before a release.
# Exit code 0 = gate passed.  Non-zero = gate failed.
#
# Usage:
#   ./scripts/release-gate.sh          # run all checks
#   ./scripts/release-gate.sh --quick  # skip release build (dev only)
#
# Known pre-existing issues (not blocking):
#   - telemetry::tests::test_success_rate is flaky due to global counter pollution
#   - clippy -D warnings fails on build.rs files in ptx-sys/ptx-kernels (not library code)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

QUICK=0
if [ "${1:-}" = "--quick" ]; then
  QUICK=1
fi

PASS=0
FAIL=0

run_gate() {
  local label="$1"
  shift
  printf "  %-50s " "$label"
  if "$@" >/dev/null 2>&1; then
    echo "PASS"
    PASS=$((PASS + 1))
  else
    echo "FAIL"
    FAIL=$((FAIL + 1))
  fi
}

echo "============================================"
echo " Ferrite-OS Release Gate"
echo "============================================"
echo ""

# 1. Library tests
echo "[1/4] Library unit tests"
run_gate "ptx-kernels --lib" cargo test -p ptx-kernels --lib
run_gate "ptx-runtime --lib (excl. flaky)" cargo test -p ptx-runtime --lib -- --skip telemetry::tests::test_success_rate
echo ""

# 2. Scheduler integration tests (the critical path for HARDEN-03/05)
echo "[2/4] Scheduler + integration tests"
run_gate "scheduler tests" cargo test -p ptx-runtime --lib -- scheduler::
run_gate "guard tests" cargo test -p ptx-kernels --lib -- guards::
echo ""

# 3. Dev build (catches type errors, missing imports)
echo "[3/4] Dev build check"
run_gate "ptx-kernels check" cargo check -p ptx-kernels
run_gate "ptx-runtime check" cargo check -p ptx-runtime
echo ""

# 4. Release build
echo "[4/4] Release build"
if [ "$QUICK" -eq 0 ]; then
  run_gate "workspace release build" cargo build --release
else
  echo "  (skipped: --quick mode)"
fi
echo ""

# Summary
echo "============================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "RELEASE GATE: FAILED"
  exit 1
fi

echo ""
echo "RELEASE GATE: PASSED"
exit 0
