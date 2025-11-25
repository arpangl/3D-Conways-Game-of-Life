#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# remove old frames
rm -f /tmp/run_*.txt
rm -f frames/frame_*.ppm
BACKENDS=(cpu avx2 openmp cuda)
for backend in "${BACKENDS[@]}"; do
  echo "=== RUN backend=$backend ==="
  ./main 3d --width 16 --height 16 --depth 4 --iterations 3 --backend ${backend} --seed 12345 --density 0.2 --output_every 1000 2>&1 | tee "/tmp/run_${backend}.txt" || true
  last=$(grep -oE "Wrote frames/frame_[0-9]{4}\.ppm \(iter=[0-9]+\)" "/tmp/run_${backend}.txt" | tail -n1 | sed -E "s/Wrote (frames\/frame_[0-9]{4}\.ppm) \(iter=[0-9]+\)/\1/")
  if [ -n "${last}" ] && [ -f "${last}" ]; then
    echo "Backend ${backend} final frame: ${last}"
    md5sum "${last}"
  else
    echo "Backend ${backend} did not write final frame or not found"
  fi
  echo
done
