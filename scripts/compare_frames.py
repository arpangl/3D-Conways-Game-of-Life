#!/usr/bin/env python3
import filecmp,sys,os
fdir1="frames_cpu_test"
fdir2="frames_avx2_test"
files=sorted([f for f in os.listdir(fdir1) if f.endswith('.ppm')])
if not files:
    print('No frames found in',fdir1); sys.exit(1)
allok=True
for f in files:
    p1=os.path.join(fdir1,f); p2=os.path.join(fdir2,f)
    if not os.path.exists(p2):
        print('Missing',p2); allok=False; continue
    if not filecmp.cmp(p1,p2,shallow=False):
        print('DIFFER:',f); allok=False
    else:
        print('SAME:',f)
if allok:
    print('ALL FRAMES IDENTICAL')
else:
    sys.exit(2)
