"""T2-matched: certified-regime validation of the regret guarantee.

Theorem 3 requires the balance condition eta <= K gamma / W. The decoupled
sandbox runs in the paper use plain exponential weights (gamma = 0), which lies
outside that condition; this script instantiates the theorem's own regime by
using the matching exploration schedule

    gamma = sqrt(W ln W / (T K))   (available whenever T >= 4 W ln W / K),
    eta   = sqrt(K ln W / (T W)),

which satisfies the balance with equality, and running the gamma-mixed
semi-bandit rule (dependent rounding with exact inclusion marginals) on the
stationary and block-rotating gain processes of run_t2.py. It reports the
measured regret against the best fixed K-set, the bound of Theorem 3---both
with the pooled mean capping deficit (the number quoted in the paper) and run
by run with each run's own deficit---and the two proof diagnostics: the
per-round exponent x_t = eta g_t / pi_t (must stay <= 1 under the balance)
and the capping deficit D_t.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

from sampling import marginals_and_round  # noqa: E402

E_CONST = float(np.e - 2.0)


def make_gains(T, W, rng, mode="stationary", n_heavy=6):
    """Same process as run_t2.py: a few heavy scenarios plus noise."""
    base = np.zeros(W)
    heavy = rng.choice(W, n_heavy, replace=False)
    base[heavy] = rng.uniform(0.6, 1.0, n_heavy)
    G = np.zeros((T, W))
    if mode == "stationary":
        G = np.clip(base[None, :] + 0.1 * rng.standard_normal((T, W)), 0, 1)
    else:
        B = T // 5
        for b in range(5):
            base_b = np.zeros(W)
            hb = rng.permutation(W)[:n_heavy]
            base_b[hb] = rng.uniform(0.6, 1.0, n_heavy)
            G[b * B:(b + 1) * B] = np.clip(base_b[None, :] + 0.1 * rng.standard_normal((B, W)), 0, 1)
    return G


def run_mixed(G, K, gamma, eta, rng):
    """gamma-mixed EXP3.M with exact inclusion marginals; returns
    (regret vs best fixed K-set, summed D_t, max exponent)."""
    T, W = G.shape
    logw = np.zeros(W)
    played = 0.0
    d_sum = 0.0
    x_max = 0.0
    for t in range(T):
        wts = np.exp(logw - logw.max())
        rho = wts / wts.sum()
        q = (1 - gamma) * rho + gamma / W
        P, S = marginals_and_round(q, K, rng)
        g = G[t]
        d_sum += float(np.sum(np.maximum(K * q - 1.0, 0.0)))
        x_max = max(x_max, float(np.max(eta * g / np.maximum(P, 1e-300))))
        ghat = np.zeros(W)
        ghat[S] = g[S] / np.maximum(P[S], 1e-12)
        logw = logw + eta * ghat
        logw -= logw.max()
        played += float(g[S].sum())
    cum = G.sum(axis=0)
    best_fixed = float(np.sort(cum)[::-1][:K].sum())
    return best_fixed - played, d_sum, x_max


def main():
    out = {}
    reps = 20
    for (W, K, T) in [(50, 5, 4000), (50, 5, 20000), (200, 10, 8000)]:
        threshold = 4.0 * W * np.log(W) / K
        assert T >= threshold, (W, K, T, threshold)
        gamma = float(np.sqrt(W * np.log(W) / (T * K)))
        eta = float(np.sqrt(K * np.log(W) / (T * W)))
        balance_slack = float(K * gamma / W - eta)   # >= 0 required
        assert gamma <= 0.5 + 1e-12 and balance_slack >= -1e-12
        rng = np.random.default_rng(2026)
        regs, xs, ds = [], [], []
        for r in range(reps):
            mode = "stationary" if r % 2 == 0 else "rotating"
            G = make_gains(T, W, np.random.default_rng(1000 * r + T), mode=mode)
            reg, d_sum, x_max = run_mixed(G, K, gamma, eta, rng)
            regs.append(reg)
            xs.append(x_max)
            ds.append(d_sum / T)
        mean_d = float(np.mean(ds))
        main_term = ((K / (1 - gamma)) * (np.log(W) / eta + E_CONST * eta * T * (2 * W / K + 1))
                     + K * gamma * T)
        bound = main_term + mean_d * T
        bounds_per_run = [main_term + d * T for d in ds]
        excess = [r - b for r, b in zip(regs, bounds_per_run)]
        out[f"W{W}_K{K}_T{T}"] = {
            "W": W, "K": K, "T": T,
            "gamma": gamma, "eta": eta,
            "balance_slack_Kgamma_over_W_minus_eta": balance_slack,
            "feasibility_threshold_4WlnW_over_K": threshold,
            "reps": reps,
            "mean_regret": float(np.mean(regs)),
            "sd_regret": float(np.std(regs)),
            "min_regret": float(np.min(regs)),
            "max_regret": float(np.max(regs)),
            "regrets": [float(r) for r in regs],
            "bound_main_term_excl_deficit": float(main_term),
            "bound": bound,
            "bound_holds_on_mean": bool(np.mean(regs) <= bound),
            "bound_holds_on_every_rep_pooled_deficit": bool(np.max(regs) <= bound),
            "bound_per_run_min": float(np.min(bounds_per_run)),
            "bound_per_run_max": float(np.max(bounds_per_run)),
            "max_excess_regret_minus_own_bound": float(np.max(excess)),
            "bound_holds_on_every_rep_own_deficit": bool(np.max(excess) <= 0.0),
            "bound_over_mean_regret": float(bound / np.mean(regs)),
            "max_exponent_seen": float(np.max(xs)),
            "exponent_le_one_on_all_runs": bool(np.max(xs) <= 1.0 + 1e-9),
            "mean_capping_deficit_per_round": mean_d,
            "main_term_scale_sqrt_TKWlnW": float(np.sqrt(T * K * W * np.log(W))),
        }
        print(f"W={W} K={K} T={T}: gamma={gamma:.4f} eta={eta:.5f} "
              f"regret={np.mean(regs):.1f}+-{np.std(regs):.1f} bound={bound:.0f} "
              f"ratio={bound/np.mean(regs):.1f}x max_x={np.max(xs):.4f} "
              f"D/round={mean_d:.5f} per-run-bound=[{min(bounds_per_run):.0f},{max(bounds_per_run):.0f}] "
              f"max_excess={np.max(excess):.1f}", flush=True)
    with open(os.path.join(RESULTS, "t2_matched.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/t2_matched.json")


if __name__ == "__main__":
    main()
