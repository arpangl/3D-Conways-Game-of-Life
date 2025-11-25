#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1"
cd "$ROOT"

echo "Rebuilding avx2/cuda targets..."
make avx2 || true
make cuda || true

# Deterministic correctness runs
echo "Starting deterministic runs (128x128x64, iter=100)"
rm -rf frames* || true
for backend in cpu avx2 openmp cuda; do
  echo "--- backend=$backend"
  ./main 3d --width 128 --height 128 --depth 64 --iterations 100 --backend ${backend} --seed 42 --density 0.2 --output_every 1000 || true
  # move outputs to per-backend dirs
  if [ -d frames ]; then mv frames "frames_${backend}"; fi
  if [ -d frames_3d ]; then mv frames_3d "frames_3d_${backend}"; fi
  if [ -f "frames_${backend}/frame_0099.ppm" ]; then
    md5sum "frames_${backend}/frame_0099.ppm" || true
  else
    echo "frames_${backend}/frame_0099.ppm missing"
  fi
done

# Warmed no-IO benchmark
echo "\nStarting warmed no-IO benchmark (256x256x128, iter=200)"
OUT=bench/results_noio_warm.csv
echo "backend,run,seconds" > "$OUT"
BACKENDS=(cpu avx2 openmp cuda)
for backend in "${BACKENDS[@]}"; do
  echo "Warming backend: $backend"
  # warm-up: 1 iteration (not timed)
  SKIP_OUTPUT=1 ./main 3d --width 256 --height 256 --depth 128 --iterations 1 --backend ${backend} --seed 12345 --density 0.2 --output_every 1000000 >/tmp/warm_${backend}.txt 2>&1 || true
  for r in 1 2 3; do
    echo "Running $backend run $r"
    SKIP_OUTPUT=1 ./main 3d --width 256 --height 256 --depth 128 --iterations 200 --backend ${backend} --seed 12345 --density 0.2 --output_every 1000000 2>&1 | tee "/tmp/run_${backend}_${r}.txt" || true
    t=$(grep -oE "[0-9]+\.[0-9]+" "/tmp/run_${backend}_${r}.txt" | tail -n1 || true)
    if [ -z "$t" ]; then t=0; fi
    echo "$backend,$r,$t" >> "$OUT"
  done
done

echo "Results file contents:"
cat "$OUT"

# Compute basic stats
python3 - <<'PY'
import csv,statistics
f='bench/results_noio_warm.csv'
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
