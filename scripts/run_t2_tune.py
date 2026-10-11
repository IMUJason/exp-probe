"""T2b: can hyperparameter tuning repair the slopes? (eta grid stationary,
alpha grid rotating; T=2000 fixed, reps=8)

The eta grid probes whether rates more aggressive than the analyzed
horizon-constant rate sqrt(K ln W / (T W)) (~0.014 at T=2000, W=50, K=5)
help; the alpha grid is the fixed-share sweep summarized in the manuscript.
Output is archived to results/t2_tune.txt.
"""
import os
import numpy as np
from run_t2 import make_gains, exp3m, baselines

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

lines = []
W, K, T, reps = 50, 5, 2000, 8
lines.append(f"analyzed horizon-constant rate eta* = sqrt(K lnW/(TW)) = "
             f"{np.sqrt(K*np.log(W)/(T*W)):.5f} (T={T}, W={W}, K={K})")
lines.append("stationary: regret vs best-fixed for eta grid")
for eta in [0.005, 0.01, 0.02, 0.05, 0.1]:
    regs = []
    for r in range(reps):
        rng = np.random.default_rng(1000*r + 17)
        G = make_gains(T, W, rng, mode="stationary")
        bf, _ = baselines(G, K)
        regs.append(bf - exp3m(G, K, eta, 0.0))
    lines.append(f"  eta={eta:<6} regret={np.mean(regs):8.2f} +- {np.std(regs):6.2f}")

lines.append("rotating: regret vs best-fixed for alpha grid (eta=0.02)")
for alpha in [0.0, 0.01, 0.02, 0.05]:
    regs = []
    for r in range(reps):
        rng = np.random.default_rng(1000*r + 23)
        G = make_gains(T, W, rng, mode="rotating")
        bf, oracle = baselines(G, K)
        regs.append(bf - exp3m(G, K, 0.02, alpha))
    lines.append(f"  alpha={alpha:<5} regret={np.mean(regs):8.2f} +- {np.std(regs):6.2f}")

print("\n".join(lines))
with open(os.path.join(RESULTS, "t2_tune.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
