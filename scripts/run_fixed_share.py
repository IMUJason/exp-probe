"""Fixed-share ablation: does fixed-share repair rotating non-stationarity?"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
import numpy as np

def make_gains(T, W, rng, mode="stationary", n_heavy=6, n_blocks=5):
    base = np.zeros(W)
    heavy = rng.choice(W, n_heavy, replace=False)
    base[heavy] = rng.uniform(0.6, 1.0, n_heavy)
    G = np.zeros((T, W))
    if mode == "stationary":
        G = np.clip(base[None, :] + 0.1 * rng.standard_normal((T, W)), 0, 1)
    else:
        B = T // n_blocks
        for b in range(n_blocks):
            base_b = np.zeros(W)
            hb = rng.permutation(W)[:n_heavy]
            base_b[hb] = rng.uniform(0.6, 1.0, n_heavy)
            G[b*B:(b+1)*B] = np.clip(base_b[None, :] + 0.1*rng.standard_normal((B, W)), 0, 1)
    return G

def exp3m(G, K, eta, alpha=0.0):
    """Same core as run_t2.exp3m: dependent rounding with exact marginals
    and a fixed-share parameter alpha; gains are K-set totals."""
    from sampling import marginals_and_round
    T, W = G.shape
    logw = np.zeros(W)
    gain_policy = 0.0
    rng = np.random.default_rng(1)
    for t in range(T):
        wts = np.exp(logw - logw.max())
        p = wts / wts.sum()
        P, S = marginals_and_round(p, K, rng)
        ghat = np.zeros(W)
        ghat[S] = G[t, S] / np.maximum(P[S], 1e-12)
        logw = logw + eta * ghat
        logw -= logw.max()
        if alpha > 0:
            wts2 = np.exp(logw - logw.max())
            wts2 = (1 - alpha) * wts2 + alpha / W
            logw = np.log(np.maximum(wts2, 1e-10))
        gain_policy += G[t, S].sum()
    return gain_policy

def baselines(G, K):
    cum = G.sum(axis=0)
    bf = G[:, np.argsort(cum)[::-1][:K]].sum(axis=1).sum()
    oracle = np.sort(G, axis=1)[:, ::-1][:, :K].sum(axis=1).sum()
    return bf, oracle

def scan(mode, alpha_vals, eta_vals, W=50, K=5, T=2000, reps=8):
    out = {}
    for alpha in alpha_vals:
        for eta in eta_vals:
            regs = []
            for r in range(reps):
                rng = np.random.default_rng(1000 * r + T)
                G = make_gains(T, W, rng, mode=mode)
                bf, _ = baselines(G, K)
                gp = exp3m(G, K, eta, alpha)
                regs.append(bf - gp)
            out[f"alpha={alpha}_eta={eta}"] = (float(np.mean(regs)), float(np.std(regs)))
    return out

for mode in ["stationary", "rotating"]:
    res = scan(mode, [0.0, 0.01, 0.05], [0.01, 0.05, 0.1])
    print(f"\n=== {mode} ===")
    for k, (m, s) in res.items():
        print(f"  {k}: regret={m:8.2f} ± {s:6.2f}")
    json.dump(res, open(os.path.join(RESULTS, f"fixed_share_{mode}.json"), "w"), indent=1)
