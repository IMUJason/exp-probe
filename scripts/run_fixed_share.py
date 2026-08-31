"""Fixed-share ablation: does fixed-share repair rotating non-stationarity?"""
import sys
sys.path.insert(0, "../src")
import numpy as np, json

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

def exp3m(G, K, eta, alpha=0.0, logspace=True):
    T, W = G.shape
    if logspace:
        logw = np.zeros(W)
    else:
        w = np.ones(W)
    gain_policy = 0.0
    rng = np.random.default_rng(1)
    for t in range(T):
        if logspace:
            wts = np.exp(logw - logw.max())
            wts = np.maximum(wts, 1e-10)   # floor
            p = wts / wts.sum()
        else:
            p = w / w.sum()
        S = rng.choice(W, size=K, replace=False, p=p)
        phat = np.minimum(1.0, K * p)
        ghat = np.zeros(W)
        ghat[S] = G[t, S] / np.maximum(phat[S], 1e-12)
        if logspace:
            logw = logw + eta * ghat / K
            if alpha > 0:
                wts2 = np.exp(logw - logw.max())
                wts2 = (1 - alpha) * wts2 + alpha * wts2.sum() / W
                wts2 = np.maximum(wts2, 1e-10)          # floor to avoid zero probs
                logw = np.log(wts2)
        # ensure at least K positive probs
        if logspace:
            wts = np.exp(logw - logw.max())
            if np.count_nonzero(wts) < K:
                wts = np.maximum(wts, 1e-8)
        else:
            w = w * np.exp(eta * ghat / K)
            if alpha > 0:
                w = (1 - alpha) * w + alpha * w.sum() / W
        gain_policy += G[t, S].mean()
    return gain_policy

def baselines(G, K):
    cum = G.sum(axis=0)
    bf = G[:, np.argsort(cum)[::-1][:K]].mean(axis=1).sum()
    oracle = np.sort(G, axis=1)[:, ::-1][:, :K].mean(axis=1).sum()
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
                gp = exp3m(G, K, eta, alpha, logspace=True)
                regs.append(bf - gp)
            out[f"alpha={alpha}_eta={eta}"] = (float(np.mean(regs)), float(np.std(regs)))
    return out

for mode in ["stationary", "rotating"]:
    res = scan(mode, [0.0, 0.01, 0.05], [0.01, 0.05, 0.1])
    print(f"\n=== {mode} ===")
    for k, (m, s) in res.items():
        print(f"  {k}: regret={m:8.2f} ± {s:6.2f}")
    json.dump(res, open(f"../results/fixed_share_{mode}.json", "w"), indent=1)
