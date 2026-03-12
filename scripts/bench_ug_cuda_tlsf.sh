#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export LD_LIBRARY_PATH="$ROOT_DIR/ferrite-os/lib:${LD_LIBRARY_PATH:-}"

run_once() {
    local mode="$1"
    shift

    local tlsf_flag="0"
    if [ "$mode" = "tlsf" ]; then
        tlsf_flag="1"
    fi

    FERRITE_TLSF_ALLOC="$tlsf_flag" \
        cargo run -p ferrite-core --features tlsf-alloc --example ug-cuda-bench -- "$@"
}

extract_metric() {
    local label="$1"
    awk -F': ' -v key="$label" '$1 ~ key { print $2 }'
}

echo "[ferrite] ug-cuda TLSF comparison"
echo "[ferrite] repo: $ROOT_DIR"
echo

baseline_output="$(run_once baseline)"
tlsf_output="$(run_once tlsf)"

baseline_latency="$(printf '%s\n' "$baseline_output" | extract_metric "Avg latency")"
baseline_throughput="$(printf '%s\n' "$baseline_output" | extract_metric "Estimated throughput")"
tlsf_latency="$(printf '%s\n' "$tlsf_output" | extract_metric "Avg latency")"
tlsf_throughput="$(printf '%s\n' "$tlsf_output" | extract_metric "Estimated throughput")"

echo "Baseline:"
printf '%s\n' "$baseline_output"
echo
echo "TLSF:"
printf '%s\n' "$tlsf_output"
echo
echo "Summary:"
echo "  baseline latency:    $baseline_latency"
echo "  tlsf latency:        $tlsf_latency"
echo "  baseline throughput: $baseline_throughput"
echo "  tlsf throughput:     $tlsf_throughput"
