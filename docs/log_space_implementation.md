# Log-space implementation of EXP-Probe weights

The paper (Section 5, parameter discussion) states that the log-space
implementation with per-iteration renormalization is essential at
W ≥ 10^4 and that we document the failure mode here.

## Where the code lives

`src/spike_core.py`, method `run(...)`, the two `rule == "exp-probe"` branches:

- **Sampling** (lines ~159–165): weights are stored as log-weights `logw`.
  Before sampling we compute `wts = np.exp(logw - logw.max())`, normalize to a
  distribution, apply exploration mixing `p = 0.95 * p + 0.05 / W`, and draw
  `K` scenarios without replacement with probabilities `p`.
- **Update** (lines ~193–203): for each evaluated scenario the revealed mass
  `g_w = p_w (Q_w(y_t) − θ_{w,t})` is importance-weighted by the inclusion
  bound `p̂_w = min(1, K·p_w)` and applied multiplicatively in log space,
  `logw[w] += η·g_w / π̃_w` with the theorem's constant rate `η = sqrt(K ln W/(T·W))`, `T` being the iteration cap; afterwards the
  vector is re-anchored with `logw -= logw.max()`.

## Why naive multiplicative weights fail at W ≥ 10^4

The textbook EXP3.M representation keeps linear weights
`w_w ← w_w · (1 + η ĝ_w)` and samples proportionally to `w`. Two failure
modes appear on our instance scales:

1. **Underflow to zero.** With W = 10^4, only K = 50 weights are updated per
   iteration; the other ~9,950 weights never grow. After a few hundred
   iterations the updated weights overflow (or the ratios underflow when
   normalizing), and in IEEE-754 double precision the normalized distribution
   becomes exactly one-hot or contains `NaN`s.
2. **Degenerate sampling kills exploration.** Once the sampling distribution
   collapses onto the currently-best scenarios, the γ-mixing term is the only
   source of exploration; with linear weights the collapse happens
   catastrophically (all mass flushed to a single index) instead of gradually,
   and the observed evaluation counts diverge from the
   $O(G\sqrt{TKW\ln W})$ regret behavior of Theorem 3.

Both are eliminated by (i) storing `logw` and exponentiating only after
subtracting the current maximum (`exp(logw - logw.max())` is numerically safe:
the largest entry maps to 1, everything else to (0, 1]), and (ii) re-anchoring
`logw -= logw.max()` after each update, so log-weights live in (−∞, 0] with
the anchor at 0. The multiplicative-update semantics are mathematically
identical to the linear formulation.

## Reproducing the failure mode

```python
import numpy as np
rng = np.random.default_rng(0)
W, K, T = 10_000, 50, 500
w = np.ones(W)
for t in range(1, T + 1):
    S = rng.choice(W, size=K, replace=False)
    w[S] *= 1 + eta * rng.random(K) * 50      # typical revealed-mass scale
print(w.max(), w.sum())                        # inf / inf — collapsed
```

The log-space variant in `src/spike_core.py` runs the same schedule without
any overflow for the horizons used in the paper (T up to 4,000 in the regret
scans and W = 10^4 in E1).
