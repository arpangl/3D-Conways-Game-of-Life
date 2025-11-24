#!/bin/bash
set -euo pipefail
ROOT="/mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1"
cd "$ROOT"
# Ensure AVX2/CUDA builds
make avx2 || true
make cuda || true
# Test parameters (smaller)
W=128
H=128
D=64
ITER=100
OUT=bench/micro_results.csv
echo "test,backend,threads,seconds" > "$OUT"
NPROCS=$(nproc || echo 1)
# Backends to test
BACKENDS=(cpu avx2 openmp cuda)
# For OpenMP vary threads 1..min(8,NPROCS)
MAX_T=8
if [ "$NPROCS" -lt "$MAX_T" ]; then MAX_T=$NPROCS; fi
for backend in "${BACKENDS[@]}"; do
  if [ "$backend" = "openmp" ]; then
    for t in $(seq 1 $MAX_T); do
      echo "Running $backend with OMP_NUM_THREADS=$t"
      export OMP_NUM_THREADS=$t
      /usr/bin/time -f "%e" -o /tmp/micro_time.txt ./main 3d --width $W --height $H --depth $D --iterations $ITER --backend $backend --seed 12345 --density 0.2 --output_every 1000000 2>&1 | tee /tmp/micro_${backend}_t${t}.txt || true
      sec=$(cat /tmp/micro_time.txt)
      echo "micro,$backend,$t,$sec" >> "$OUT"
    done
  else
    echo "Running $backend (single variant)"
    # unset OMP env to default
    unset OMP_NUM_THREADS || true
    /usr/bin/time -f "%e" -o /tmp/micro_time.txt ./main 3d --width $W --height $H --depth $D --iterations $ITER --backend $backend --seed 12345 --density 0.2 --output_every 1000000 2>&1 | tee /tmp/micro_${backend}.txt || true
    sec=$(cat /tmp/micro_time.txt)
    echo "micro,$backend,1,$sec" >> "$OUT"
  fi
done

echo "Micro results:"
cat "$OUT"
