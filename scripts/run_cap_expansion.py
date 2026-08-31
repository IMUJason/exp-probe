"""Second problem class: capacity expansion (2-stage LP).

A scaled-capacity variant of the facility-opening generator: expansion
capacities are 1.5x larger and lumpier, on the same continuous master and
the same recourse structure. This varies the recourse geometry and cost
scale; structural generality beyond facility opening is not established
here. A genuinely binary master is probed separately in
run_cap_expansion_milp.py.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

class CapacityExpansionInstance(RealLPInstance):
    """Scaled-capacity variant on the same continuous master and recourse
    structure: expansion capacities are 1.5x the facility-opening ones."""

    def __init__(self, n=16, m=30, W=500, sparse_vol=0.5, seed=0):
        super().__init__(n=n, m=m, W=W, sparse_vol=sparse_vol, seed=seed)
        self.K = self.K * 1.5

def run():
    out = []
    for vol in [0.0, 1.0]:
        for W in [500, 2000]:
            for rule in ["full", "oracle", "random"]:
                evs, its, dts, certs = [], [], [], []
                for r in range(2):
                    inst = CapacityExpansionInstance(n=16, m=30, W=W, sparse_vol=vol, seed=100 + r)
                    sim = RealPartialBenders(inst, eps=1e-3, T_chk=100)
                    e, it, dt = sim.run(rule, K=50, max_iter=1000, rng=np.random.default_rng(r))
                    evs.append(e); its.append(it); dts.append(dt); certs.append(it < 1000)
                out.append((W, vol, rule, float(np.mean(evs)), float(np.mean(its)), float(np.mean(dts)), all(certs)))
                print(f"W={W} vol={vol} {rule}: {evs[-1]:.0f} evals, {its[-1]} iters, {dts[-1]:.1f}s, certified={certs[-1]}", flush=True)
    print("\n=== Capacity expansion summary ===")
    lines = []
    for W, vol, rule, ev, it, dt, cert in out:
        l = f"W={W} vol={vol} {rule:8s} evals={ev:7.0f} iters={it:5.0f} wall={dt:5.1f}s certified={cert}"
        print(l); lines.append(l)
    with open(os.path.join(RESULTS, "cap_expansion.log"), "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    run()
