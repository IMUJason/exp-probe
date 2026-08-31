"""Wall-clock decomposition: selection / master / subproblem / certification."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from real_lp import RealLPInstance, RealPartialBenders

class TimedRealPartialBenders(RealPartialBenders):
    def master(self):
        t0 = time.perf_counter()
        r = super().master()
        self.master_time = getattr(self, 'master_time', 0) + (time.perf_counter() - t0)
        return r

    def run(self, rule, K, max_iter=200, rng=None):
        self.master_time = 0.0
        t0 = time.perf_counter()
        res = super().run(rule, K, max_iter, rng)
        total = time.perf_counter() - t0
        return res + (self.master_time, total)

for W, vol, rule in [(500, 0.0, "full"), (500, 0.0, "oracle"), (500, 1.0, "full"), (500, 1.0, "oracle")]:
    inst = RealLPInstance(n=16, m=30, W=W, sparse_vol=vol, seed=100)
    sim = TimedRealPartialBenders(inst, eps=1e-3, T_chk=100)
    e, it, dt, mt, total = sim.run(rule, K=50, max_iter=200, rng=np.random.default_rng(0))
    print(f"W={W} vol={vol} {rule:8s}: master={mt:5.2f}s ({mt/total*100:4.1f}%)  subprob={(dt-mt):5.2f}s ({(dt-mt)/total*100:4.1f}%)  total={dt:5.1f}s")
