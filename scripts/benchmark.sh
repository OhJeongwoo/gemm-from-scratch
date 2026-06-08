#!/usr/bin/env bash
# Usage: ./scripts/benchmark.sh
#   env: KERNELS="0 3 5"  ITERS=50
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/gemm
OUT=benchmarks/results.csv
ITERS="${ITERS:-20}"
KERNELS="${KERNELS:-0 1 2 3 4 5}"
SHAPES=("1024 1024 1024" "2048 2048 2048" "4096 4096 4096")

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  exit 1
fi

mkdir -p benchmarks
echo "kernel_id,name,M,N,K,ms,gflops,pct,status" > "$OUT"

for shape in "${SHAPES[@]}"; do
  for k in $KERNELS; do
    echo ">> kernel $k @ $shape"
    "$BIN" "$k" $shape "$ITERS" | grep '^CSV,' | sed 's/^CSV,//' >> "$OUT"
  done
done

echo
echo "=== results ($OUT) ==="
column -t -s, "$OUT"
