"""Section 6.2 provenance: dimensions and per-solve time of the
facility-opening subproblem on the Table 2 testbed (n=16, m=30, W=500,
seed=100).

The manuscript states "each subproblem LP has 510 columns and 46 rows and
solves in about 1.7 ms" for the study machine (Python 3.10 / SciPy 1.15.3).
This probe reports the LP dimensions read off the instance and times 500
subproblem solves at a fixed incumbent (mean and 95th percentile) on the
machine that runs this script. Per-solve times are machine- and
load-dependent: the archived full-evaluation runs of Table 2 imply
1.4--1.7 ms per solve on the study machine (e.g. 5.3 s over 3,000 solves at
W=500, vol=0, with the master at about 4% of the total), while this probe
typically measures about 2 ms under a newer Python/SciPy stack. The
dimensions themselves are exact and stack-independent.
"""
import os
import sys
import time

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

from real_lp import RealLPInstance  # noqa: E402

inst = RealLPInstance(n=16, m=30, W=500, sparse_vol=0.0, seed=100)
y = np.full(inst.n, 0.5)

n_cols = len(inst.bounds)                      # n*m transport + m shortage
n_rows = inst.A_eq.shape[0] + inst.A_ub.shape[0]

for w in range(20):                            # warm-up
    inst.solve_sub(w, y)

N = 500
times = []
t0 = time.perf_counter()
for w in range(N):
    t = time.perf_counter()
    inst.solve_sub(w % inst.W, y)
    times.append((time.perf_counter() - t) * 1e3)
wall = time.perf_counter() - t0

lines = [
    f"instance: facility opening, n={inst.n}, m={inst.m}, W={inst.W}, seed=100 (Table 2 testbed)",
    f"subproblem LP: {n_cols} columns ({inst.n*inst.m} transport + {inst.m} shortage), "
    f"{n_rows} rows ({inst.m} demand equalities + {inst.n} capacity inequalities)",
    f"timing on this machine: {N} solves at y=0.5*1, mean {np.mean(times):.2f} ms, "
    f"p95 {np.percentile(times, 95):.2f} ms, total {wall:.2f} s",
    "study-machine reference: the archived full-evaluation runs of Table 2 imply "
    "1.4-1.7 ms per solve (5.3 s / 3,000 solves at W=500 vol=0; 1.5 s / 1,000 at "
    "W=500 vol=1; 6.3 s / 4,000 at W=2000 vol=1, master at about 4% of the total)",
    f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}",
]
print("\n".join(lines))
with open(os.path.join(RESULTS, "e2_subsolve_timing.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
