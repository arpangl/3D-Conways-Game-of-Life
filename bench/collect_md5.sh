#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/d/0-IOC-1/PP/final/3D-Conways-Game-of-Life-0.0.1"
cd "$ROOT"
echo "Collecting md5 of final frames"
for d in frames_cpu frames_avx2 frames_openmp frames_cuda; do
  echo "== $d"
  if [ -f "$d/frame_0099.ppm" ]; then
    md5sum "$d/frame_0099.ppm"
  else
    echo "$d/frame_0099.ppm missing"
  fi
done
