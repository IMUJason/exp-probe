"""Real-LP testbed: partial-evaluation Benders with genuine recourse LPs.

Two-stage LP with continuous first stage (capacitated facility opening):
    min  f'y + sum_w p_w Q_w(y)
    Q_w(y) = min  sum c^w x + pi u
              s.t. sum_i x_ij + u_j = d^w_j        (demand rows)
                   sum_j x_ij      <= K_i y_i      (capacity rows, RHS couples y)
Master: LP over y in [0,1]^n with multicut epigraph theta_w.
Benders cut from subproblem duals: dQ/dy_i = K_i * mu_i (mu = capacity duals).

Concentration knob: scenario volatility vol_w is heterogeneous — a sparse set
of high-volatility scenarios carries most of the recourse mass, so informed
selection should win; the homogeneous case reproduces the diffuse-mass regime
in which per-scenario differences are small.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from sampling import marginals_and_round
from scipy.sparse import coo_matrix

RNG = np.random.default_rng


class RealLPInstance:
    def __init__(self, n=16, m=30, W=500, sparse_vol=0.5, seed=0):
        rng = RNG(seed)
        self.n, self.m, self.W = n, m, W
        fx = rng.uniform(0, 100, n); fy = rng.uniform(0, 100, n)
        cx = rng.uniform(0, 100, m); cy = rng.uniform(0, 100, m)
        D = np.linalg.norm(fx[:, None] - cx[None, :], axis=0) + np.abs(fy[:, None] - cy[None, :])
        base_cost = 1.0 + D / 20.0                            # (n, m) base transport
        self.base_demand = rng.uniform(10, 30, m)
        self.K = rng.uniform(0.8, 1.4, n)
        self.K = self.K / self.K.sum() * 1.10 * self.base_demand.sum()
        self.f = rng.uniform(20, 60, n)
        self.pi = 50.0
        # heterogeneous volatility: fraction (1-sparse_vol) scenarios are calm
        n_hot = max(1, int(0.05 * W)) if sparse_vol > 0 else 0
        vol = np.full(W, 0.02)
        hot = rng.choice(W, n_hot, replace=False)
        vol[hot] = 0.02 + sparse_vol * 2.0                    # hot scenarios swing
        self.demand = np.stack([
            self.base_demand * (1.0 + vol[w] * rng.standard_normal(m)) for w in range(W)
        ]).clip(1.0, None)                                    # (W, m)
        self.cost = base_cost[None, :, :] * (1.0 + 0.1 * rng.standard_normal((W, n, m))).clip(0.5)
        self.p = np.full(W, 1.0 / W)
        self._build_static()

    def _build_static(self):
        n, m = self.n, self.m
        nv = n * m + m
        self.c_lp = np.concatenate([self.cost[0].ravel(), np.full(m, self.pi)])
        rows, cols, vals, i = [], [], [], 0
        for j in range(m):                                    # demand: sum_i x_ij + u_j = d_j
            for ii in range(n):
                rows.append(j); cols.append(ii * m + j); vals.append(1.0)
            rows.append(j); cols.append(n * m + j); vals.append(1.0)
        A_eq = coo_matrix((vals, (rows, cols)), shape=(m, nv)).tocsr()
        rows, cols, vals = [], [], []
        for ii in range(n):                                   # capacity: sum_j x_ij <= K_i y_i
            for j in range(m):
                rows.append(ii); cols.append(ii * m + j); vals.append(1.0)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr()
        self.A_eq, self.A_ub = A_eq, A_ub
        self.bounds = [(0.0, None)] * nv

    def solve_sub(self, w, y):
        """Solve Q_w(y); return value and subgradient w.r.t. y."""
        c_lp = np.concatenate([self.cost[w].ravel(), np.full(self.m, self.pi)])
        b_ub = self.K * y
        res = linprog(c_lp, A_ub=self.A_ub, b_ub=b_ub, A_eq=self.A_eq,
                      b_eq=self.demand[w], bounds=self.bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        mu = res.ineqlin.marginals                            # (n,), <=0 sign convention
        grad = self.K * mu                                    # dQ/dy
        return res.fun, grad

    def eval_all(self, y):
        vals = np.empty(self.W); grads = np.empty((self.W, self.n))
        for w in range(self.W):
            vals[w], grads[w] = self.solve_sub(w, y)
        return vals, grads


class RealPartialBenders:
    def __init__(self, inst, eps=1e-3, T_chk=5, binary=False):
        self.I, self.eps, self.T_chk = inst, eps, T_chk
        self.binary = binary                    # binary (MILP) first stage
        self.cuts = [[] for _ in range(inst.W)]               # (g, b): theta >= g'y + b
        self.n_master = 0

    def master(self):
        I = self.I
        nv = I.n + I.W
        c_obj = np.concatenate([I.f, I.p])
        rows, cols, vals, rhs = [], [], [], []
        r = 0
        for w in range(I.W):
            for (g, b) in self.cuts[w]:
                for i in range(I.n):
                    rows.append(r); cols.append(i); vals.append(g[i])
                rows.append(r); cols.append(I.n + w); vals.append(-1.0)
                rhs.append(-b); r += 1
        A = coo_matrix((vals, (rows, cols)), shape=(max(r, 1), nv)).tocsr()
        lb = np.zeros(nv)
        ub = np.concatenate([np.ones(I.n), np.full(I.W, np.inf)])   # Q_w >= 0 valid global bound
        if self.binary:
            cons = [LinearConstraint(A, -np.inf, np.array(rhs))] if r > 0 else []
            res = milp(c=c_obj, constraints=cons,
                       integrality=np.concatenate([np.ones(I.n), np.zeros(I.W)]),
                       bounds=Bounds(lb, ub))
        elif r > 0:
            res = linprog(c_obj, A_ub=A, b_ub=np.array(rhs), bounds=list(zip(lb, ub)), method="highs")
        else:
            res = linprog(c_obj, bounds=list(zip(lb, ub)), method="highs")
        self.n_master += 1
        if not res.success:
            raise RuntimeError(res.message)
        return res.x[:I.n], res.x[I.n:], res.fun

    def run(self, rule, K, max_iter=400, rng=None):
        rng = RNG(0) if rng is None else rng
        I = self.I
        evals = 0
        t0 = time.perf_counter()
        logw = np.zeros(I.W)
        eta = np.sqrt(K * np.log(I.W) / (max_iter * I.W))  # theorem rate, T = cap
        for it in range(1, max_iter + 1):
            y, theta, lb = self.master()
            if rule == "full":
                vals, grads = I.eval_all(y); evals += I.W
                for w in range(I.W):
                    g = grads[w]; cut = (g.copy(), float(vals[w] - g @ y))
                    if not any(np.allclose(gg, g, atol=1e-9) and abs(bb - cut[1]) < 1e-9
                               for (gg, bb) in self.cuts[w]):
                        self.cuts[w].append(cut)
                ub = float(y @ I.f + vals @ I.p)
                lb2 = self.master()[2]
                if ub - lb2 <= self.eps * max(1.0, abs(ub)):
                    return evals, it, time.perf_counter() - t0
                continue
            check = (it % self.T_chk == 0) or (it == 1)
            if check:
                vals, grads = I.eval_all(y); evals += I.W
                for w in range(I.W):
                    g = grads[w]; cut = (g.copy(), float(vals[w] - g @ y))
                    if not any(np.allclose(gg, g, atol=1e-9) and abs(bb - cut[1]) < 1e-9
                               for (gg, bb) in self.cuts[w]):
                        self.cuts[w].append(cut)
                ub = float(y @ I.f + vals @ I.p)
                lb2 = self.master()[2]
                if ub - lb2 <= self.eps * max(1.0, abs(ub)):
                    return evals, it, time.perf_counter() - t0
            # selection
            if rule == "full":
                S = np.arange(I.W)
            elif rule == "random":
                S = rng.choice(I.W, size=K, replace=False)
            elif rule == "oracle":
                # pricing device: ranking uses true masses at zero cost (only K evals charged)
                vals_free, _ = I.eval_all(y)
                mass_free = I.p * (vals_free - theta)
                S = np.argsort(mass_free)[::-1][:K]
            elif rule == "est-det":
                # stale-cut extrapolation: rank by last cut's value at y (no extra solves)
                last = np.array([(self.cuts[w][-1][0] @ y + self.cuts[w][-1][1]) if self.cuts[w]
                                 else -np.inf for w in range(I.W)])
                mass_est = I.p * np.maximum(last - theta, 0.0)
                S = np.argsort(-mass_est, kind="stable")[:K]   # deterministic ties
            elif rule == "exp-probe":
                wts = np.exp(logw - logw.max()); p = wts / wts.sum()
                q = 0.95 * p + 0.05 / I.W          # gamma-mixed sampling distribution
                incl, S = marginals_and_round(q, K, rng)   # exact marginals
            else:
                raise ValueError(rule)
            # evaluate selected scenarios and add cuts
            for w in S:
                v, g = I.solve_sub(w, y); evals += 1
                cut = (g.copy(), float(v - g @ y))
                if not any(np.allclose(gg, g, atol=1e-9) and abs(bb - cut[1]) < 1e-9
                           for (gg, bb) in self.cuts[w]):
                    self.cuts[w].append(cut)
                if rule == "exp-probe":
                    m_w = float(I.p[w] * (v - theta[w]))
                    logw[w] += eta * max(m_w, 0.0) / max(incl[w], 1e-12)
            if rule == "exp-probe":
                logw -= logw.max()
        return evals, max_iter, time.perf_counter() - t0
