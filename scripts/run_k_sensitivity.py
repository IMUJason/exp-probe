"""K-sensitivity probe: W=500 facility-opening testbed, K in {10, 25, 50, 100}.

Complements the T_chk sweep: full evaluation is invariant to K, while the
partial rules' cost scales with K through the selection budget.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

out = []
for vol in [0.0, 1.0]:
    for K in [10, 25, 50, 100]:
        for rule in ["full", "oracle", "random"]:
            evs, its, dts = [], [], []
            for r in range(2):
                inst = RealLPInstance(n=16, m=30, W=500, sparse_vol=vol, seed=100 + r)
                sim = RealPartialBenders(inst, eps=1e-3, T_chk=100)
                e, it, dt = sim.run(rule, K=K, max_iter=1000, rng=np.random.default_rng(r))
                evs.append(e); its.append(it); dts.append(dt)
            out.append(f"vol={vol} K={K:<4} {rule:<8} evals={np.mean(evs):>7.0f} "
                       f"iters={np.mean(its):>5.1f} wall={np.mean(dts):>6.1f}s")
            print(out[-1], flush=True)

with open(os.path.join(RESULTS, "k_sensitivity.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
