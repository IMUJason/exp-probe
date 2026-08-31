"""T3: (a) eval-count separation at scale on the sparse-trap family;
(b) recourse LP timing at the problem sizes used in E2."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from scipy.optimize import linprog
from run_t1_sparse import make_trap_sparse
from spike_core import eval_counts

out = []

# --- T3a: separation at scale -------------------------------------------
for W, K in [(2000, 20), (10000, 50)]:
    inst, groups = make_trap_sparse(W=W, per_group=5)
    t0 = time.time()
    res = eval_counts(inst, eps=1e-3, K=K, reps=3, max_iter=800)
    dt = time.time() - t0
    o = res['oracle'][0]; r = res['random'][0]
    estdet = "STALL" if res['est-det'][2] >= 800 else f"{res['est-det'][0]:.0f}"
    out.append(f"[T3a] W={W} K={K} (owners=15): full={res['full'][0]:.0f} "
               f"oracle={o:.0f} ({res['oracle'][2]:.0f} it) random={r:.0f} "
               f"({res['random'][2]:.0f} it) est-det={estdet} "
               f"ratio random/oracle={r/max(o,1):.1f}x  [{dt:.0f}s wall]")

# --- T3b: real recourse-LP timing (ARCH-like subproblems) ----------------
rng = np.random.default_rng(0)
n_fac, n_cus = 32, 48
nv = n_fac * n_cus + n_cus
c_lp = rng.uniform(1, 5, nv)
rows, rhs = [], []
for j in range(n_cus):                       # sum_i x_ij + u_j = d_j
    row = np.zeros(nv); row[j*n_fac:(j+1)*n_fac] = 1.0; row[n_fac*n_cus + j] = 1.0
    rows.append(row); rhs.append(rng.uniform(10, 30))
for i in range(n_fac):                       # sum_j x_ij <= K_i (y fixed => RHS)
    row = np.zeros(nv)
    for j in range(n_cus): row[j*n_fac + i] = 1.0
    rows.append(row); rhs.append(rng.uniform(5, 25))
A_eq = np.array(rows[:n_cus]); b_eq = np.array(rhs[:n_cus])
A_ub = np.array(rows[n_cus:]); b_ub = np.array(rhs[n_cus:])
times = []
for trial in range(200):
    b_eq_t = b_eq * rng.uniform(0.8, 1.2, n_cus)
    t0 = time.perf_counter()
    linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq_t, bounds=(0, None), method="highs")
    times.append(time.perf_counter() - t0)
ms = np.mean(times) * 1000
out.append(f"[T3b] recourse LP (32x48, {nv} vars, {n_cus+n_fac} rows): {ms:.2f} ms mean "
           f"(p95 {np.percentile(times,95)*1000:.2f} ms) over 200 solves")
out.append(f"[T3b] projected per-iteration full eval: W=10000 -> {10000*ms/1000:.1f}s ; "
           f"partial K=50 -> {50*ms/1000:.3f}s  (headroom {10000/50:.0f}x)")

open(os.path.join(RESULTS, "t3_results.txt"), "w").write("\n".join(out))
print("\n".join(out))
