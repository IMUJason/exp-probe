"""Instance-cluster bootstrap: is oracle's advantage over random significant?"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from spike_core import make_trap_family, eval_counts

def make_trap_sparse(W=2000, per_group=5, depth=600.0, tau=0.85):
    inst, groups = make_trap_family(n=2, W=W, per_group=per_group, depth=depth, tau=tau)
    rest = [w for w in range(W) if not any(w in g for g in groups)]
    rest = np.array(rest)
    inst.dp_b[rest] = inst.sh_b[rest]
    inst.dp_a[rest] = inst.sh_a[rest]
    return inst

# repeated runs on sparse-trap
rng = np.random.default_rng(42)
diffs = []
for rep in range(30):
    inst = make_trap_sparse(W=2000)
    res = eval_counts(inst, eps=1e-3, K=20, reps=1, max_iter=1200)
    # diff = random - oracle (positive means oracle wins)
    d = res['random'][0] - res['oracle'][0]
    diffs.append(d)
diffs = np.array(diffs)

# cluster bootstrap on the 30 repetitions
B = 10000
boots = []
for b in range(B):
    sample = rng.choice(diffs, size=len(diffs), replace=True)
    boots.append(sample.mean())
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"random - oracle (sparse-trap W=2000): mean={diffs.mean():.1f}, 95% CI=[{lo:.1f}, {hi:.1f}]")
print(f"significant (CI excludes 0): {lo > 0 or hi < 0}")
json.dump({"mean_diff": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
           "n_reps": len(diffs)}, open(os.path.join(RESULTS, "bootstrap_significance.json"), "w"))
