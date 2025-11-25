#!/usr/bin/env python3
import csv,statistics,sys
f='bench/results_noio.csv'
try:
    with open(f,'r',newline='') as fh:
        first = fh.readline()
        if not first:
            print('Empty results file')
            sys.exit(1)
        headers = [h.strip().lstrip('\ufeff').lower() for h in first.strip().split(',')]
        rows = []
        rdr = csv.DictReader(fh, fieldnames=headers)
        for r in rdr:
            rows.append({k: (v.strip() if v is not None else '') for k,v in r.items()})
except FileNotFoundError:
    print(f'File not found: {f}')
    sys.exit(1)
backends = {}
for r in rows:
    b = r.get('backend')
    s = r.get('seconds','')
    if b is None or b=='':
        continue
    try:
        sec = float(s) if s!='' else 0.0
    except:
        sec = 0.0
    backends.setdefault(b,[]).append(sec)
cpu_mean = None
if 'cpu' in backends and len(backends['cpu'])>0:
    cpu_mean = statistics.mean(backends['cpu'])
print('\nComputed stats:')
for b,vals in backends.items():
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals)>1 else 0.0
    speedup = (cpu_mean/mean) if cpu_mean else None
    print(f"{b}: mean={mean:.3f}s  std={std:.3f}s  speedup_vs_cpu={speedup:.3f}")
