#!/usr/bin/env bash
set -e
frame=$1
if [ -z "$frame" ]; then
  echo "Usage: $0 frame_0000.ppm"
  exit 1
fi
f1=frames_cpu_avx2/$frame
f2=frames_avx2/$frame
if [ ! -f "$f1" ] || [ ! -f "$f2" ]; then
  echo "Files missing: $f1 or $f2"
  exit 1
fi
# skip ppm header (assumes header ends with '255\n')
# find offset of data
offset1=$(awk 'NR==1{if($0!~/P6/) exit 1} NR<=3{if(NR==3) print length($0)+1}' "$f1" || true)
# simpler: use binary parse: header ends at first occurrence of '\n' after '255'
skip=$( (awk 'BEGIN{c=0} {print NR":"$0} NR==3{exit}' "$f1") >/dev/null 2>&1; echo 0)
# fallback: use a fixed header length search
data_offset=$(python3 - <<'PY'
import sys
p='frames_cpu_avx2/'+sys.argv[1]
with open(p,'rb') as f:
    s=f.read()
idx=s.find(b'255\n')
if idx==-1:
    print(0)
else:
    print(idx+4)
PY
"$frame")

echo "data_offset=$data_offset"
# use cmp -l on binary data parts
cmp -l <(dd if="$f1" bs=1 skip=$data_offset status=none) <(dd if="$f2" bs=1 skip=$data_offset status=none) | head -n 50
echo "(showing up to 50 differing bytes)"
