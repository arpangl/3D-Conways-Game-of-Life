#!/bin/bash
set -euo pipefail
ROOT="/mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1"
cd "$ROOT"
echo "Starting no-I/O benchmark run in: $ROOT"
# Clean and build
make clean || true
echo "Building AVX2 target (may be slow)"
make avx2 || true
echo "Building CUDA target (if nvcc available)"
make cuda || true
# Set threads
export OMP_NUM_THREADS=$(nproc || echo 1)
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
OUT=bench/results_noio.csv
echo "backend,run,seconds" > "$OUT"
BACKENDS=(cpu avx2 openmp cuda)
for backend in "${BACKENDS[@]}"; do
  for r in 1 2 3; do
    echo "Running $backend run $r"
    ./main 3d --width 256 --height 256 --depth 128 --iterations 200 --backend ${backend} --seed 12345 --density 0.2 --output_every 1000000 2>&1 | tee "/tmp/run_${backend}_${r}.txt" || true
    t=$(grep -oE "[0-9]+\.[0-9]+" "/tmp/run_${backend}_${r}.txt" | tail -n1 || true)
    if [ -z "$t" ]; then t=0; fi
    echo "$backend,$r,$t" >> "$OUT"
  done
done

echo "Results file contents:"
cat "$OUT"

# Compute stats with python
python3 - <<'PY'
import csv,statistics
f='bench/results_noio.csv'
rows=[]
with open(f) as fh:
    rdr=csv.DictReader(f)
    for r in rdr:
        rows.append(r)
backends={}
for r in rows:
    b=r['backend']; s=float(r['seconds']); backends.setdefault(b,[]).append(s)
cpu_mean=None
if 'cpu' in backends and len(backends['cpu'])>0:
    cpu_mean=statistics.mean(backends['cpu'])
for b,vals in backends.items():
    mean=statistics.mean(vals)
    std=statistics.stdev(vals) if len(vals)>1 else 0.0
    speedup=(cpu_mean/mean) if cpu_mean else None
    print(f"{b}: mean={mean:.3f}s std={std:.3f}s speedup_vs_cpu={speedup:.3f}")
PY

echo "Done"
