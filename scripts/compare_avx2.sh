#!/usr/bin/env bash
set -e
echo "== AVX2 correctness test: generate CPU frames"
rm -rf frames_cpu_avx2 frames_avx2 frames
./main 2d --width 256 --height 256 --iterations 20 --backend cpu --species 1 --seed 12345 --output_every 5
mv frames frames_cpu_avx2

echo "== AVX2: generate frames"
./main 2d --width 256 --height 256 --iterations 20 --backend avx2 --species 1 --seed 12345 --output_every 5
mv frames frames_avx2

echo "== MD5 sums and diff"
md5sum frames_cpu_avx2/*.ppm | sort > /tmp/cpu_hashes
md5sum frames_avx2/*.ppm | sort > /tmp/avx2_hashes

echo "CPU hashes:"
cat /tmp/cpu_hashes || true

echo "AVX2 hashes:"
cat /tmp/avx2_hashes || true

echo "Diff (if empty, identical):"
diff /tmp/cpu_hashes /tmp/avx2_hashes || true

echo "== done"
