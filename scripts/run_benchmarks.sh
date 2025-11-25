#!/usr/bin/env bash
set -e
OUT=/tmp/bench_results.csv
echo "size,iterations,backend,time_s" > $OUT
SIZES=(256 512 1024)
ITERS=(200 100 50)
BACKENDS=(cpu avx2 cuda)
SEED=12345
# Ensure CUDA objects are built
make cuda >/dev/null 2>&1 || true
for idx in ${!SIZES[@]}; do
  SIZE=${SIZES[$idx]}
  ITERS=${ITERS[$idx]}
  for bk in "${BACKENDS[@]}"; do
    echo "Running: ${bk} ${SIZE}x${SIZE} iters=${ITERS}"
    # avoid frame IO: set output_every large
    LOG=/tmp/bench_${bk}_${SIZE}.log
    ./main 2d --width ${SIZE} --height ${SIZE} --iterations ${ITERS} --backend ${bk} --species 1 --seed ${SEED} --output_every 100000 > ${LOG} 2>&1 || true
    # extract timing
    if grep -q "CUDA host-run time" ${LOG}; then
      t=$(grep "CUDA host-run time" ${LOG} | tail -n1 | sed -E 's/.*: *([0-9.]+) s/\1/')
    elif grep -q "AVX2 run time" ${LOG}; then
      t=$(grep "AVX2 run time" ${LOG} | tail -n1 | sed -E 's/.*: *([0-9.]+) s/\1/')
    elif grep -q "CPU run time" ${LOG}; then
      t=$(grep "CPU run time" ${LOG} | tail -n1 | sed -E 's/.*: *([0-9.]+) s/\1/')
    else
      # fallback: try to find any 'real' from /usr/bin/time output
      t=$(grep -Eo "real [0-9.]+" ${LOG} | awk '{print $2}' | tail -n1 || echo "0")
    fi
    echo "${SIZE}x${SIZE},${ITERS},${bk},${t}" >> $OUT
    echo " -> ${bk} time: ${t}s"
  done
done

echo "Benchmark completed. Results:" 
cat $OUT
echo "Results saved to $OUT"
