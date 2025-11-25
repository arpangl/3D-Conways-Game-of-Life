#!/usr/bin/env python3
import sys
if len(sys.argv)!=2:
    print('usage: diff_frames_py.py frame_XXXX.ppm')
    sys.exit(1)
frame=sys.argv[1]
f1='frames_cpu_avx2/'+frame
f2='frames_avx2/'+frame
with open(f1,'rb') as a, open(f2,'rb') as b:
    A=a.read(); B=b.read()
# find header end by locating b'255\n'
h1=A.find(b'255\n')
h2=B.find(b'255\n')
if h1==-1 or h2==-1:
    print('Cannot find header')
    sys.exit(1)
off1=h1+4; off2=h2+4
D=A[off1:]; E=B[off2:]
minlen=min(len(D),len(E))
if D==E:
    print('IDENTICAL')
    sys.exit(0)
count=0
for i in range(minlen):
    if D[i]!=E[i]:
        print('diff byte offset', i, 'cpu=', D[i], 'avx2=', E[i])
        count+=1
        if count>=50:
            break
if len(D)!=len(E):
    print('different lengths:',len(D), len(E))
