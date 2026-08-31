"""E2: real-LP regime map with pricing-device oracle semantics."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

out = []
def grid(W, vol, K, T_chk, rules, reps=2, max_iter=1000):
    rows = {}
    for rule in rules:
        evs, its, dts, certified = [], [], [], []
        for r in range(reps):
            inst = RealLPInstance(n=16, m=30, W=W, sparse_vol=vol, seed=100 + r)
            sim = RealPartialBenders(inst, eps=1e-3, T_chk=T_chk)
            e, it, dt = sim.run(rule, K=K, max_iter=max_iter, rng=np.random.default_rng(r))
            evs.append(e); its.append(it); dts.append(dt)
            certified.append(it < max_iter)
        rows[rule] = (float(np.mean(evs)), float(np.std(evs)), float(np.mean(its)), float(np.mean(dts)), all(certified))
        out.append(f"W={W} vol={vol} K={K} Tchk={T_chk} {rule:<9} evals={rows[rule][0]:>7.0f}+-{rows[rule][1]:<5.0f} iters={rows[rule][2]:>5.1f} wall={rows[rule][3]:>6.1f}s certified={rows[rule][4]}")
        print(out[-1], flush=True)
    return rows

R = {}
R['W500_v0']  = grid(500, 0.0, 50, 100, ["full", "oracle", "random"])
R['W500_v1']  = grid(500, 1.0, 50, 100, ["full", "oracle", "random"])
R['W2000_v1'] = grid(2000, 1.0, 50, 100, ["full", "oracle", "random"])
for k, rows in R.items():
    f = rows['full'][0]
    for rule in rows:
        if rule != 'full':
            out.append(f"  {k} {rule}: {rows[rule][0]/f:.2f}x full (certified={rows[rule][4]})")
open(os.path.join(RESULTS, "e2e_results.txt"), "w").write("\n".join(out))
json.dump(R, open(os.path.join(RESULTS, "e2e_results.json"), "w"), indent=1)
print("saved")
