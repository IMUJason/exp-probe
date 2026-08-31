"""T2: conditional (fixed-trajectory) regret numerics.

Semi-bandit EXP3.M over W scenarios, K selected per round, gains in [0,1].
Stationary and block-rotating (non-stationary) gain processes.
Verifies that regret against the best fixed K-set scales ~sqrt(T ln W) and
that a fixed-share variant tracks rotating heavy sets where plain EXP3.M
degrades.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from sampling import marginals_and_round

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

def make_gains(T, W, rng, mode="stationary", n_heavy=6, n_blocks=5):
    base = np.zeros(W)
    heavy = rng.choice(W, n_heavy, replace=False)
    base[heavy] = rng.uniform(0.6, 1.0, n_heavy)
    G = np.zeros((T, W))
    if mode == "stationary":
        G = np.clip(base[None, :] + 0.1 * rng.standard_normal((T, W)), 0, 1)
    else:  # rotating heavy set across blocks
        B = T // n_blocks
        for b in range(n_blocks):
            base_b = np.zeros(W)
            hb = rng.permutation(W)[:n_heavy]
            base_b[hb] = rng.uniform(0.6, 1.0, n_heavy)
            G[b*B:(b+1)*B] = np.clip(base_b[None, :] + 0.1*rng.standard_normal((B, W)), 0, 1)
    return G

def exp3m(G, K, eta=None, alpha=0.0):
    """EXP3.M via dependent rounding: exact inclusion marginals make the
    gain estimates unbiased; eta=None uses the theorem's constant rate
    sqrt(K ln W / (T W)). Gains and benchmark are totals over the K-set."""
    T, W = G.shape
    if eta is None:
        eta = np.sqrt(K * np.log(W) / (T * W))
    logw = np.zeros(W)
    gain_policy = 0.0
    rng = np.random.default_rng(1)
    for t in range(1, T + 1):
        wts = np.exp(logw - logw.max())
        p = wts / wts.sum()
        incl, S = marginals_and_round(p, K, rng)   # exact marginals
        ghat = np.zeros(W)
        ghat[S] = G[t - 1, S] / np.maximum(incl[S], 1e-12)
        logw = logw + eta * ghat
        logw -= logw.max()
        if alpha > 0:
            wts2 = np.exp(logw - logw.max())
            wts2 = (1 - alpha) * wts2 + alpha / W
            logw = np.log(np.maximum(wts2, 1e-10))
        gain_policy += G[t - 1, S].sum()
    return gain_policy

def baselines(G, K):
    T, W = G.shape
    cum = G.sum(axis=0)
    best_fixed_idx = np.argsort(cum)[::-1][:K]
    bf = G[:, best_fixed_idx].sum(axis=1).sum()
    oracle = np.sort(G, axis=1)[:, ::-1][:, :K].sum(axis=1).sum()
    return bf, oracle

def regret_scan(mode, W=50, K=5, Ts=(500, 1000, 2000, 4000), reps=10):
    out = {}
    for T in Ts:
        reg_plain, reg_fs = [], []
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + T)
            G = make_gains(T, W, rng, mode=mode)
            bf, oracle = baselines(G, K)
            gp = exp3m(G, K, alpha=0.0)
            gf = exp3m(G, K, alpha=0.05)
            reg_plain.append(bf - gp)
            reg_fs.append(bf - gf)
        out[T] = (float(np.mean(reg_plain)), float(np.std(reg_plain)),
                  float(np.mean(reg_fs)), float(np.std(reg_fs)))
    return out

def loglog_slope(d, col=0):
    T = np.array(sorted(d), float)
    y = np.array([d[t][col] for t in sorted(d)]) + 1e-9
    return float(np.polyfit(np.log(T), np.log(y), 1)[0])

results = {}
for mode in ["stationary", "rotating"]:
    res = regret_scan(mode)
    results[mode] = res
    print(f"\n=== {mode} (W=50, K=5) — regret vs best-fixed-K ===")
    print(f"{'T':<7} {'EXP3.M(mean±sd)':<20} {'fixed-share(mean±sd)':<22}")
    for T, (pm, ps, fm, fs) in res.items():
        print(f"{T:<7} {pm:>8.2f} ± {ps:<7.2f} {fm:>8.2f} ± {fs:<7.2f}")
    print(f"EXP3.M regret log-log slope (vs T): {loglog_slope(res):.3f}  [sqrt(T) => 0.5]")

# dynamic-oracle gap on rotating: how far policy is from per-round oracle
rng = np.random.default_rng(7)
G = make_gains(2000, 50, rng, mode="rotating")
bf, oracle = baselines(G, 5)
gp = exp3m(G, 5, alpha=0.05)
print(f"\nrotating T=2000: policy={gp:.1f} best-fixed={bf:.1f} dynamic-oracle={oracle:.1f}")
json.dump(results, open(os.path.join(RESULTS, "t2_results.json"), "w"), indent=1)
