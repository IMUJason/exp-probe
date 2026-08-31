"""T2b: can hyperparameter tuning repair the slopes? (eta grid stationary,
alpha grid rotating; T=2000 fixed, reps=8)"""
import numpy as np
from run_t2 import make_gains, exp3m, baselines

W, K, T, reps = 50, 5, 2000, 8
print("stationary: regret vs best-fixed for eta grid")
for eta in [0.005, 0.01, 0.02, 0.05, 0.1]:
    regs = []
    for r in range(reps):
        rng = np.random.default_rng(1000*r + 17)
        G = make_gains(T, W, rng, mode="stationary")
        bf, _ = baselines(G, K)
        regs.append(bf - exp3m(G, K, eta, 0.0))
    print(f"  eta={eta:<6} regret={np.mean(regs):8.2f} ± {np.std(regs):6.2f}")

print("rotating: regret vs best-fixed for alpha grid (eta=0.02)")
for alpha in [0.0, 0.01, 0.02, 0.05]:
    regs = []
    for r in range(reps):
        rng = np.random.default_rng(1000*r + 23)
        G = make_gains(T, W, rng, mode="rotating")
        bf, oracle = baselines(G, K)
        regs.append(bf - exp3m(G, K, 0.02, alpha))
    print(f"  alpha={alpha:<5} regret={np.mean(regs):8.2f} ± {np.std(regs):6.2f}")
