"""E1 sparse-trap scaling and the E2 concentration sweep.

E1  PWL separation (sparse-trap family) with EXP-Probe included, W in {2000, 10000}.
E2  Real-LP testbed: concentration sweep — sparse_vol in {0 (flat), 0.5, 1.0 (sparse-hot)},
    W=500, K=25, rules {full, random, oracle, exp-probe}; reports evaluations to
    certification and wall time.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np

out = []

# ---------------- E1: PWL separation with EXP-Probe ----------------
from run_t1_sparse import make_trap_sparse
from spike_core import eval_counts

for W, K in [(2000, 20), (10000, 50)]:
    inst, groups = make_trap_sparse(W=W)
    res = eval_counts(inst, eps=1e-3, K=K, reps=3, max_iter=1200)
    o = res['oracle'][0]; r = res['random'][0]; xp = res['exp-probe'][0]
    estdet = "STALL" if res['est-det'][2] >= 1200 else f"{res['est-det'][0]:.0f}"
    out.append(f"[E1] W={W} K={K}: full={res['full'][0]:.0f} oracle={o:.0f} "
               f"exp-probe={xp:.0f} random={r:.0f} est-det={estdet} | "
               f"random/oracle={r/max(o,1):.1f}x exp-probe/oracle={xp/max(o,1):.1f}x")
    print(out[-1], flush=True)

# ---------------- E2: real-LP concentration sweep ----------------
from real_lp import RealLPInstance, RealPartialBenders

def e2_run(sparse_vol, W=500, K=25, reps=3, seed0=100):
    rows = {}
    for rule in ["full", "random", "oracle", "exp-probe"]:
        evs, its, dts = [], [], []
        for r in range(reps):
            inst = RealLPInstance(n=16, m=30, W=W, sparse_vol=sparse_vol, seed=seed0 + r)
            sim = RealPartialBenders(inst, eps=1e-3, T_chk=10)
            e, it, dt = sim.run(rule, K=K, max_iter=400, rng=np.random.default_rng(r))
            evs.append(e); its.append(it); dts.append(dt)
        rows[rule] = (float(np.mean(evs)), float(np.std(evs)),
                      float(np.mean(its)), float(np.mean(dts)))
        print(f"  vol={sparse_vol} {rule:<10} evals={rows[rule][0]:>7.0f}±{rows[rule][1]:<5.0f} "
              f"iters={rows[rule][2]:>5.1f} wall={rows[rule][3]:>6.1f}s", flush=True)
    return rows

E2 = {}
for vol in [0.0, 0.5, 1.0]:
    out.append(f"[E2] sparse_vol={vol} (W=500, K=25):")
    E2[vol] = e2_run(vol)
    full_e = E2[vol]['full'][0]
    for rule in ["random", "oracle", "exp-probe"]:
        out.append(f"    {rule}: {E2[vol][rule][0]:.0f} evals "
                   f"({E2[vol][rule][0]/full_e:.2f}x full), wall {E2[vol][rule][3]:.1f}s "
                   f"vs full {E2[vol]['full'][3]:.1f}s")
    print(out[-1], flush=True)

open(os.path.join(RESULTS, "final_study_results.txt"), "w").write("\n".join(out))
json.dump({"E2": {str(k): v for k, v in E2.items()}}, open(os.path.join(RESULTS, "final_study_results.json"), "w"), indent=1)
print("saved final_study_results.txt/json")
