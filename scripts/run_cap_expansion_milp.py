"""Capacity expansion with a genuinely binary (MILP) first stage.

A smaller-scale probe of the regime-map prediction that combinatorial
masters, whose iteration counts are larger, are the favorable quadrant for
selection. At the W=500 scale of the LP study the MILP master becomes the
bottleneck (cut accumulation slows every solve quadratically), so this
probe uses W=200, n=12 binary projects, and a certification tolerance of
1e-2; full-evaluation iteration counts on the binary master are then large
enough to compare against the partial rules' certification lag.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealPartialBenders
from run_cap_expansion import CapacityExpansionInstance

out = []
R = {}
for vol in [0.0, 1.0]:
    for rule in ["full", "oracle", "random", "est-det"]:
        evs, its, dts, certs = [], [], [], []
        for r in range(2):
            inst = CapacityExpansionInstance(n=12, m=20, W=200, sparse_vol=vol, seed=100 + r)
            sim = RealPartialBenders(inst, eps=1e-2, T_chk=50, binary=True)
            e, it, dt = sim.run(rule, K=25, max_iter=200, rng=np.random.default_rng(r))
            evs.append(e); its.append(it); dts.append(dt); certs.append(it < 200)
        R[f"W200_v{vol}_{rule}"] = (float(np.mean(evs)), float(np.mean(its)),
                                     float(np.mean(dts)), all(certs))
        out.append(f"W=200 vol={vol} {rule:<8} evals={np.mean(evs):>6.0f} "
                   f"iters={np.mean(its):>5.1f} wall={np.mean(dts):>6.1f}s certified={all(certs)}")
        print(out[-1], flush=True)

with open(os.path.join(RESULTS, "cap_expansion_milp.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
json.dump(R, open(os.path.join(RESULTS, "cap_expansion_milp.json"), "w"), indent=1)
