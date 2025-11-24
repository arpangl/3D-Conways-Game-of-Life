#!/usr/bin/env python3
import sys
import csv
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: compute_speedups.py results.csv")
    sys.exit(1)

path = sys.argv[1]
rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Group by scenario
by_scenario = defaultdict(dict)
for r in rows:
    scen = r['scenario']
    backend = r['backend']
    elapsed = float(r['elapsed_s'])
    by_scenario[scen][backend] = elapsed

print("scenario,cpu,backend,time_s,speedup")
for scen, data in by_scenario.items():
    cpu = data.get('cpu')
    if cpu is None:
        print(f"{scen},MISSING_CPU,,")
        continue
    for backend, t in data.items():
        speedup = cpu / t if t>0 else float('inf')
        print(f"{scen},{cpu:.6f},{backend},{t:.6f},{speedup:.4f}")
