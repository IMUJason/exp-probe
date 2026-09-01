"""Quantitative proxies for the two regime-map axes.

Mass concentration: at the first master iterate y_1, the revealed masses
g_w = p_w (Q_w(y_1) - theta_{w,1}) are computed for every scenario; we report
the top-K mass share (fraction of total positive mass held by the K largest)
and the normalized entropy of the positive-mass distribution (1 = uniform,
0 = a single scenario). Piece complexity: for the real-LP instances we report
the number of distinct active dual bases seen across the certified run, a
proxy for how many cuts breadth must supply.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np
from run_t1_sparse import make_trap_sparse
from spike_core import PartialBenders
from real_lp import RealLPInstance, RealPartialBenders


def concentration(g, K):
    g = np.maximum(np.asarray(g, float), 0.0)
    tot = g.sum()
    if tot <= 0:
        return 0.0, 1.0
    share = np.sort(g)[::-1][:K].sum() / tot
    p = g[g > 0] / tot
    ent = -(p * np.log(p)).sum() / np.log(len(g)) if len(g) > 1 else 0.0
    return float(share), float(ent)


out = []

# ---- E1 sparse-trap: mass at y_1 = 1 (all coordinates active) ----
for W, K in [(2000, 20), (10000, 50)]:
    inst, groups = make_trap_sparse(W=W)
    sim = PartialBenders(inst, eps=1e-3)
    y, theta, lb = sim.master()
    g = inst.p * (inst.values(y[None, :])[0] - theta)
    share, ent = concentration(g, K)
    out.append(f"E1 W={W} K={K}: topK_share={share:.3f} norm_entropy={ent:.3f} "
               f"(owners={sum(len(x) for x in groups)}/{W})")
    print(out[-1], flush=True)

# ---- E2 real-LP: mass at y_1 + active-dual-basis count over a full run ----
for W, vol, tag in [(500, 0.0, "diffuse"), (500, 1.0, "concentrated")]:
    inst = RealLPInstance(n=16, m=30, W=W, sparse_vol=vol, seed=100)
    sim = RealPartialBenders(inst, eps=1e-3, T_chk=100)
    y, theta, lb = sim.master()
    vals, grads = inst.eval_all(y)
    g = inst.p * (vals - theta)
    share, ent = concentration(g, 50)
    # distinct active dual bases (rounded subgradients) at y_1
    bases = {tuple(np.round(grads[w], 6)) for w in range(W)}
    out.append(f"E2 W={W} vol={vol} ({tag}): topK_share={share:.3f} "
               f"norm_entropy={ent:.3f} distinct_dual_bases={len(bases)}/{W}")
    print(out[-1], flush=True)

with open(os.path.join(RESULTS, "concentration.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
