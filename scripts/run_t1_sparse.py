"""Spike-T1b: sparse-relevance variant — rest scenarios are single-piece
(deep == shallow, always exactly known), only the 15 trap owners matter.
Oracle should pay O(#owners), random pays a coupon collector over W."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from spike_core import make_trap_family, eval_counts, PWLInstance, RNG

def make_trap_sparse(W=200, per_group=5, depth=600.0, tau=0.85):
    inst, groups = make_trap_family(n=2, W=W, per_group=per_group, depth=depth, tau=tau)
    rest = [w for w in range(W) if not any(w in g for g in groups)]
    rest = np.array(rest)
    inst.dp_b[rest] = inst.sh_b[rest]
    inst.dp_a[rest] = inst.sh_a[rest]      # single-piece: exactly known from the start
    return inst, groups

inst, groups = make_trap_sparse()
print(f"W={inst.W}, owners={sum(len(g) for g in groups)}, K=4, eps=1e-3")
res = eval_counts(inst, eps=1e-3, K=4, reps=5, max_iter=3000)
print(f"{'rule':<12} {'evals(mean±sd)':<18} {'iters(mean±sd)':<18}")
for rule, (em, es, im, isd) in res.items():
    print(f"{rule:<12} {em:>8.0f} ± {es:<6.0f} {im:>8.1f} ± {isd:<5.1f}")
o = res['oracle'][0]; r = res['random'][0]
print(f"\nrandom/oracle = {r/max(o,1):.1f}x   (oracle pays ~{o:.0f} evals vs W={inst.W})")
json.dump(res, open("t1_sparse_results.json", "w"), indent=1)
