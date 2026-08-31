"""Large-scale sparse-trap: W=100000."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
from spike_core import make_trap_family, eval_counts

def make_trap_sparse_100k(W=100000, per_group=5, depth=None, tau=0.85):
    inst, groups = make_trap_family(n=3, W=W, per_group=per_group, depth=depth, tau=tau)
    rest = [w for w in range(W) if not any(w in g for g in groups)]
    import numpy as np
    rest = np.array(rest)
    inst.dp_b[rest] = inst.sh_b[rest]
    inst.dp_a[rest] = inst.sh_a[rest]
    return inst, groups

inst, _ = make_trap_sparse_100k()
res = eval_counts(inst, eps=1e-3, K=50, reps=1, max_iter=3000)
out = {k: v for k, v in res.items()}
print(json.dumps(out, indent=1))
open(os.path.join(RESULTS, "large_scale_w100k.json"), "w").write(json.dumps(out, indent=1))
