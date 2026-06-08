#!/usr/bin/env bash
# Usage: sudo ./scripts/profile.sh <kernel_id> [M] [N] [K]
# (GeForce requires sudo or NVreg_RestrictProfilingToAdminUsers=0)
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL_ID="${1:-5}"
M="${2:-4096}"
N="${3:-$M}"
K="${4:-$M}"
BIN=build/gemm

# sudo strips PATH, so resolve ncu absolutely.
NCU="$(command -v ncu || true)"
if [[ -z "$NCU" ]]; then
  for c in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/cuda/bin/ncu; do
    [[ -x "$c" ]] && NCU="$c" && break
  done
fi
if [[ -z "$NCU" ]]; then
  echo "ncu not found" >&2
  exit 1
fi

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static"

"$NCU" --target-processes all \
    -k "regex:sgemm_" \
    --launch-count 1 \
    --metrics "$METRICS" \
    --section WarpStateStats \
    "$BIN" "$KERNEL_ID" "$M" "$N" "$K" 3
