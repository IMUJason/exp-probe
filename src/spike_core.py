"""Partial-evaluation Benders on piecewise-linear two-stage LP instances.

Setting A (anchor-partial): min_y c'y + sum_w p_w Q_w(y), y in [0,1]^n,
Q_w(y) = max_j (a_wj' y + b_wj).  Each iteration the master LP over revealed
cuts gives y_t; the rule picks K scenarios; selected scenarios reveal their
active piece at y_t.  Ground truth (true gap) is tracked by the simulator.

Families:
  trap       - shallow pieces carry no signal; each coordinate group of
               owners hides a deep piece 10 + d*(y_s - tau), d = W.  A single
               owner reveal repels its coordinate; certification requires
               hitting every group, so deterministic oblivious rules
               (lowest-index ties) never certify, the oracle clears one
               coordinate per round, and random pays a coupon collector.
  informative - deep piece continues the shallow trend and dominates the box;
               a nearest-point surrogate is exact for any evaluated scenario,
               so informed exploration costs ~W while random pays coupon-
               collector overhead.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from sampling import marginals_and_round

RNG = np.random.default_rng


class PWLInstance:
    """Pieces are stored as (a, b) with value a'y + b; the convention is
    shared by values(), the cut pools, and the master LP."""

    def __init__(self, n, W, sh_b, sh_a, dp_b, dp_a, c, p=None):
        self.n, self.W = n, W
        self.sh_b, self.sh_a = sh_b, sh_a
        self.dp_b, self.dp_a = dp_b, dp_a
        self.c, self.p = c, (np.full(W, 1.0 / W) if p is None else p)

    def values(self, Y):
        sh = self.sh_b[None, :] + Y @ self.sh_a.T
        dp = self.dp_b[None, :] + Y @ self.dp_a.T
        return np.maximum(sh, dp)

    def F(self, Y):
        return Y @ self.c + self.values(Y) @ self.p

    def active_is_deep(self, Y):
        return (self.dp_b[None, :] + Y @ self.dp_a.T) >= (self.sh_b[None, :] + Y @ self.sh_a.T)


def make_trap_family(n=3, W=40, depth=None, per_group=5, tau=0.85, rng=None):
    """Coordinate-staircase traps: group s binds coordinate s. An owner of
    group s has deep piece 10 + d*(y_s - tau), dominated by the shallow
    piece on y_s <= tau and active on the upper face y_s > tau. With
    d = W (the default) a single owner reveal repels the master from
    y_s = 1 (uplift (1/W)*d*(1-tau) > |c_s|), the uncleared coordinates
    stay at 1 with constant gap, and certification requires hitting every
    group -- one coordinate at a time."""
    rng = RNG(7) if rng is None else rng
    d = float(W) if depth is None else float(depth)
    sh_b = 10.0 + 1e-3 * rng.standard_normal(W)
    sh_a = 1e-3 * rng.standard_normal((W, n))               # flat: no signal
    dp_b = sh_b.copy()                                       # non-owners: single piece
    dp_a = sh_a.copy()
    idx = rng.permutation(W)
    groups = []
    for s in range(n):
        owners = idx[W - per_group * (s + 1): W - per_group * s]
        dp_a[owners, :] = 0.0
        dp_a[owners, s] = d
        dp_b[owners] = 10.0 - d * tau
        groups.append(owners)
    c = -0.05 * np.ones(n)                                   # greedy corner (1,..,1)
    return PWLInstance(n, W, sh_b, sh_a, dp_b, dp_a, c), groups


def make_informative_family(n=2, W=40, depth=60.0, rng=None):
    """Deep piece = continuation of the heterogeneous shallow trend, active over
    the whole box: any evaluated point pins scenario w exactly."""
    rng = RNG(11) if rng is None else rng
    slopes = rng.uniform(1.0, 3.0, size=(W, n))
    sh_b = 20.0 * np.ones(W)
    sh_a = -slopes.copy()                                    # value 20 - s'y
    dp_b = sh_b + depth * rng.uniform(0.5, 1.5, W)
    dp_a = -1.02 * slopes                                    # value (20+du) - 1.02 s'y
    c = 0.3 * np.ones(n)
    return PWLInstance(n, W, sh_b, sh_a, dp_b, dp_a, c)


class PartialBenders:
    def __init__(self, inst: PWLInstance, eps=0.05):
        self.I, self.eps = inst, eps
        self.pools = [[(inst.sh_a[w].copy(), float(inst.sh_b[w]))] for w in range(inst.W)]
        self.hist_points = [[] for _ in range(inst.W)]       # evaluated points per scenario

    def master(self):
        I = self.I
        c_obj = np.concatenate([I.c, I.p])
        rows, rhs = [], []
        for w in range(I.W):
            for (a, b) in self.pools[w]:
                row = np.zeros(I.n + I.W)
                row[:I.n] = a; row[I.n + w] = -1.0
                rows.append(row); rhs.append(-b)
        bounds = [(0.0, 1.0)] * I.n + [(None, None)] * I.W
        res = linprog(c_obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        return res.x[:I.n], res.x[I.n:], res.fun

    def reveal(self, S, y):
        vals = self.I.values(y[None, :])[0]
        deep = self.I.active_is_deep(y[None, :])[0]
        new = 0
        for w in S:
            cut = ((self.I.dp_a[w].copy(), float(self.I.dp_b[w])) if deep[w]
                   else (self.I.sh_a[w].copy(), float(self.I.sh_b[w])))
            if not any(np.allclose(a, cut[0], atol=1e-12) and abs(b - cut[1]) < 1e-12
                       for (a, b) in self.pools[w]):
                self.pools[w].append(cut); new += 1
            self.hist_points[w].append(y.copy())
        return new

    def _surrogate_nn(self, y):
        """Nearest-point surrogate: est_w(y) = Q_w at w's closest evaluated point
        (exact continuation); unevaluated -> +inf error proxy, envelope value est."""
        est = np.empty(self.I.W); has_hist = np.zeros(self.I.W, bool)
        for w in range(self.I.W):
            pts = self.hist_points[w]
            if pts:
                P = np.array(pts)
                j = int(np.argmin(np.linalg.norm(P - y[None, :], axis=1)))
                est[w] = self.I.values(P[j:j + 1])[0, w]
                has_hist[w] = True
            else:
                est[w] = self.envelope_at(w, y)
        return est, has_hist

    def envelope_at(self, w, y):
        return max(a @ y + b for (a, b) in self.pools[w])

    def envelope(self, y):
        return np.array([self.envelope_at(w, y) for w in range(self.I.W)])

    def run(self, rule, K, max_iter=600, rng=None):
        rng = RNG(0) if rng is None else rng
        evals = it = 0
        stall = 0
        logw = np.zeros(self.I.W)                          # EXP-Probe log-weights
        eta = np.sqrt(K * np.log(self.I.W) / (max_iter * self.I.W))  # theorem rate, T = cap
        for it in range(1, max_iter + 1):
            y, theta, lb = self.master()
            true_vals = self.I.values(y[None, :])[0]
            ub = float(y @ self.I.c + true_vals @ self.I.p)
            if ub - lb <= self.eps * max(1.0, abs(ub)):
                return evals, it - 1
            if rule == "full":
                S = np.arange(self.I.W)
            elif rule == "random":
                S = rng.choice(self.I.W, size=min(K, self.I.W), replace=False)
            elif rule == "oracle":
                mass = self.I.p * (true_vals - theta)
                S = np.argsort(mass)[::-1][:K]
            elif rule == "exp-probe":
                # semi-bandit EXP3.M: DepRound sampling with exact marginals
                wts = np.exp(logw - logw.max())
                p = wts / wts.sum()
                q = 0.95 * p + 0.05 / self.I.W      # gamma-mixed sampling distribution
                incl, S = marginals_and_round(q, min(K, self.I.W), rng)
            elif rule in ("est-det", "est-rand", "est-nn", "surr-error"):
                if rule == "est-nn":
                    est, has = self._surrogate_nn(y)
                    mass_est = np.where(has, self.I.p * np.maximum(est - theta, 0.0), np.inf)
                else:
                    last = np.array([self.pools[w][-1][0] @ y + self.pools[w][-1][1]
                                     for w in range(self.I.W)])
                    if rule == "surr-error":
                        mass_est = np.abs(last - theta)
                    else:
                        mass_est = self.I.p * np.maximum(last - theta, 0.0)
                if rule == "est-rand":
                    perm = rng.permutation(self.I.W)
                    key = np.empty(self.I.W); key[perm] = np.arange(self.I.W)
                    S = np.lexsort((key, -mass_est))[:K]     # random tie-break
                else:
                    S = np.argsort(-mass_est, kind="stable")[:K]  # lowest-index ties
            else:
                raise ValueError(rule)
            # protocol accounting: the rule is charged for all K selections
            # each round (re-evaluations at a stationary point cost budget
            # and add no new cut); est-det's repetition is what stalls it
            S_eff = S
            new = self.reveal(S_eff, y)
            if rule == "exp-probe":
                # unbiased semi-bandit update: divide by the exact marginal
                # computed at selection time (incl is deterministic in q)
                for w in S_eff:
                    mass_w = float(self.I.p[w] * (true_vals[w] - theta[w]))
                    gain = max(mass_w, 0.0)
                    logw[w] += eta * gain / max(incl[w], 1e-12)
                logw -= logw.max()
            evals += len(S_eff)
            stall = 0 if new else stall + 1
            # early exit is sound only for rules that repeat a fixed
            # selection (deterministic ties): they provably add no new cut.
            # Randomized rules keep a nonzero exploration mass and stay live.
            if rule in ("est-det", "surr-error") and stall >= 30:
                return evals, max_iter
            if stall >= 300:
                return evals, max_iter
        return evals, max_iter


def eval_counts(inst, eps=0.05, K=4, reps=5, max_iter=600):
    out = {}
    for rule in ["full", "random", "oracle", "exp-probe", "est-det", "est-nn", "surr-error"]:
        if rule in ("est-nn", "surr-error"):
            reps_rule = max(1, reps // 2)
        else:
            reps_rule = reps
        evs, its = [], []
        for r in range(reps_rule):
            sim = PartialBenders(inst, eps=eps)
            e, it = sim.run(rule, K, max_iter=max_iter, rng=RNG(100 + r))
            evs.append(e); its.append(it)
        out[rule] = (float(np.mean(evs)), float(np.std(evs)),
                     float(np.mean(its)), float(np.std(its)))
    return out
