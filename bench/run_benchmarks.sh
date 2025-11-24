#!/usr/bin/env bash
set -euo pipefail

# Simple benchmark runner for this repo.
# Usage: run in project root (WSL). Edit sizes below or pass env vars:
#   export BENCH_2D_W=128
#   export BENCH_2D_H=128
#   export BENCH_2D_IT=10
#   export BENCH_3D_W=24
#   export BENCH_3D_H=24
#   export BENCH_3D_D=6
#   export BENCH_3D_IT=10
# Then: bash bench/run_benchmarks.sh

OUT_CSV=bench/results.csv
mkdir -p bench
echo "scenario,backend,elapsed_s" > "$OUT_CSV"

# defaults (small, quick)
: ${BENCH_2D_W:=128}
: ${BENCH_2D_H:=128}
: ${BENCH_2D_IT:=10}
: ${BENCH_3D_W:=24}
: ${BENCH_3D_H:=24}
: ${BENCH_3D_D:=6}
: ${BENCH_3D_IT:=10}

BACKENDS=(cpu openmp avx2 cuda)

run_case() {
  scen="$1"
  backend="$2"
  cmd="$3"

  echo "Running: $scen --backend $backend"
  start=$(date +%s.%N)
  # run the program; redirect stdout/stderr to avoid clutter; the script measures wall time
  eval "$cmd" >/dev/null 2>&1 || true
  end=$(date +%s.%N)
  elapsed=$(echo "$end - $start" | bc -l)
  echo "$scen,$backend,$elapsed" >> "$OUT_CSV"
}

rebuild_and_run() {
  scen="$1"; backend="$2"; cmd="$3";
  # Rebuild main with appropriate flags for each backend
  echo "Rebuilding for backend: $backend"
  case "$backend" in
    cpu)
      make clean >/dev/null 2>&1 || true
      make -j4 CXXFLAGS='-std=c++17 -Wall -Wextra -O2 -Iinclude -DCUDA_STUB' >/dev/null 2>&1 || true
      ;;
    openmp)
      make clean >/dev/null 2>&1 || true
      make -j4 CXXFLAGS='-std=c++17 -Wall -Wextra -O2 -Iinclude -fopenmp -DCUDA_STUB' >/dev/null 2>&1 || true
      ;;
    avx2)
      make clean >/dev/null 2>&1 || true
      make -j4 CXXFLAGS='-std=c++17 -Wall -Wextra -O3 -Iinclude -mavx2 -march=native -DCUDA_STUB' >/dev/null 2>&1 || true
      ;;
    cuda)
      # Try to build CUDA target (Makefile has `cuda` target which uses nvcc if available)
      make clean >/dev/null 2>&1 || true
      make -j4 cuda >/dev/null 2>&1 || true
      ;;
    *)
      make clean >/dev/null 2>&1 || true
      make -j4 >/dev/null 2>&1 || true
      ;;
  esac

  run_case "$scen" "$backend" "$cmd"
}

# 2D cases
for backend in "${BACKENDS[@]}"; do
  cmd="./main 2d --width ${BENCH_2D_W} --height ${BENCH_2D_H} --iterations ${BENCH_2D_IT} --density 0.12 --backend ${backend} --output_every 0"
  rebuild_and_run "2d_${BENCH_2D_W}x${BENCH_2D_H}_it${BENCH_2D_IT}" "$backend" "$cmd"
done

# 3D cases
for backend in "${BACKENDS[@]}"; do
  cmd="./main 3d --width ${BENCH_3D_W} --height ${BENCH_3D_H} --depth ${BENCH_3D_D} --iterations ${BENCH_3D_IT} --density 0.12 --backend ${backend} --output_every 0"
  rebuild_and_run "3d_${BENCH_3D_W}x${BENCH_3D_H}x${BENCH_3D_D}_it${BENCH_3D_IT}" "$backend" "$cmd"
done

echo "Benchmark finished. Results in $OUT_CSV"

echo "To compute speedups run: python3 bench/compute_speedups.py $OUT_CSV"
