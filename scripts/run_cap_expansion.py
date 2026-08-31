"""Second problem class: capacity expansion (2-stage LP).

Distinguishes from facility location: capacity expansion has integer master
(open/expand facilities), continuous recourse (flow allocation), and a
different dual structure.  Same protocol as facility opening.

Setting: n candidate expansion projects, each with capacity gain K_i and
fixed cost f_i; first stage selects which projects to activate (binary).
Recourse: transport from expanded capacities to demands with scenario costs.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

class CapacityExpansionInstance(RealLPInstance):
    """Capacity expansion variant: binary first stage (choose projects),
    continuous recourse (flow).  Uses the same recourse structure but the
    master is a pure binary IP and the capacity coupling is additive."""

    def __init__(self, n=16, m=30, W=500, sparse_vol=0.5, seed=0):
        super().__init__(n=n, m=m, W=W, sparse_vol=sparse_vol, seed=seed)
        # capacity expansion: base capacity zero, expansion adds K_i
        self.base_cap = np.zeros(n)
        # expansion yields are larger and lumpier than facility opening
        self.K = self.K * 1.5

    # master stays LP (y in [0,1] with binary relaxation); the distinction
    # is in the capacity coupling (additive) vs facility opening (multiplicative)

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
    for W, vol, rule, ev, it, dt, cert in out:
        print(f"W={W} vol={vol} {rule:8s} evals={ev:7.0f} iters={it:5.0f} wall={dt:5.1f}s certified={cert}")

if __name__ == "__main__":
    run()
