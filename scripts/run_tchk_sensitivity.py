"""Sensitivity of the E2 conclusions to the certification period T_chk.

W=500 facility-opening testbed, vol in {0, 1}, T_chk in {20, 50, 100, 200},
rules {full, oracle, random, est-det}, 2 seeds. est-det is the implementable
stale-cut-extrapolation proxy (no extra solves).
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

out = []
R = {}
for vol in [0.0, 1.0]:
    for T_chk in [20, 50, 100, 200]:
        for rule in ["full", "oracle", "random", "est-det"]:
            evs, its, dts, certs = [], [], [], []
            for r in range(2):
                inst = RealLPInstance(n=16, m=30, W=500, sparse_vol=vol, seed=100 + r)
                sim = RealPartialBenders(inst, eps=1e-3, T_chk=T_chk)
                e, it, dt = sim.run(rule, K=50, max_iter=1000, rng=np.random.default_rng(r))
                evs.append(e); its.append(it); dts.append(dt); certs.append(it < 1000)
            R[f"vol{vol}_T{T_chk}_{rule}"] = (float(np.mean(evs)), float(np.mean(its)),
                                              float(np.mean(dts)), all(certs))
            out.append(f"vol={vol} Tchk={T_chk:<4} {rule:<8} evals={np.mean(evs):>7.0f} "
                       f"iters={np.mean(its):>5.1f} wall={np.mean(dts):>6.1f}s certified={all(certs)}")
            print(out[-1], flush=True)

with open(os.path.join(RESULTS, "tchk_sensitivity.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
json.dump(R, open(os.path.join(RESULTS, "tchk_sensitivity.json"), "w"), indent=1)
