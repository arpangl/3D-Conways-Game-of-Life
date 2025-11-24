#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1"
cd "$ROOT"
export CXXFLAGS='-O3 -march=native -mavx2 -mtune=native'
echo "Using CXXFLAGS=$CXXFLAGS"
make clean
make -j4
OUT="$ROOT/bench/results_large.csv"
echo "backend,run,seconds" > "$OUT"
for backend in cpu avx2 openmp; do
  for i in 1 2 3; do
    seed=12345
    case "$i" in
      2) seed=23456 ;;
      3) seed=34567 ;;
    esac
    TMP="/tmp/${backend}_run_${i}.txt"
    if [ "$backend" = "openmp" ]; then
      export OMP_NUM_THREADS=$(nproc || echo 4)
    fi
    echo
    echo "=== $backend run $i seed $seed (OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}) ==="
    "$ROOT/main" 3d --width 256 --height 256 --depth 128 --iterations 200 --backend "$backend" --seed "$seed" --density 0.2 --output_every 1000000 2>&1 | tee "$TMP"
    v=$(grep -oE "[0-9]+\.[0-9]+" "$TMP" | tail -n1 || true)
    echo "Recorded: $backend,$i,$v"
    echo "$backend,$i,$v" >> "$OUT"
  done
done

echo
echo "All runs done. Results:"
cat "$OUT"
