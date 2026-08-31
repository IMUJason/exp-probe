"""Dependent rounding with prescribed marginals (pipage rounding).

Standard EXP3.M sampling layer (Uchiya--Nakamura--Kudo 2010; Audibert--
Bubeck--Lugosi 2014): given a distribution q over W arms and a play count K,
construct the fractional vector P = cap(K*q) summing to K with entries in
[0,1], then round it to a 0/1 vector S of cardinality K whose marginals are
exactly E[S_w] = P_w. The estimator g/P_w is then unbiased.
"""
import numpy as np


def marginals_and_round(q, K, rng):
    """Return (P, S): P is the exact marginal vector, S a sample with
    E[S] = P, |S| = K, where P >= min(K q, 1) coordinatewise."""
    q = np.asarray(q, dtype=float)
    W = q.size
    P = np.minimum(K * q, 1.0)
    # redistribute any deficit created by capping, proportionally to q
    for _ in range(W + 1):
        deficit = K - P.sum()
        if deficit < 1e-12:
            break
        frac = np.where((P > 1e-12) & (P < 1.0 - 1e-12))[0]
        if frac.size == 0:
            zero = np.where(P < 1e-12)[0]
            if zero.size == 0:
                break
            P[zero] += deficit / zero.size  # degenerate tie-break
            break
        wsum = q[frac].sum()
        share = (q[frac] / wsum) if wsum > 0 else np.full(frac.size, 1.0 / frac.size)
        add = np.minimum(1.0 - P[frac], deficit * share)
        P[frac] += add
    # dependent rounding (Gandhi et al. 2006): merge fractional pairs with
    # marginals-preserving probabilities; each step makes one coordinate 0/1
    S = P.copy()
    while True:
        frac = np.where((S > 1e-9) & (S < 1.0 - 1e-9))[0]
        if frac.size <= 1:
            break
        i, j = int(frac[0]), int(frac[1])
        si, sj = S[i], S[j]
        if si + sj >= 1.0:
            # w.p. (1-si)/(2-si-sj): (si+sj-1, 1); else (1, si+sj-1)
            if rng.random() < (1.0 - si) / (2.0 - si - sj):
                S[i], S[j] = si + sj - 1.0, 1.0
            else:
                S[i], S[j] = 1.0, si + sj - 1.0
        else:
            # w.p. si/(si+sj): (si+sj, 0); else (0, si+sj)
            if rng.random() < si / (si + sj):
                S[i], S[j] = si + sj, 0.0
            else:
                S[i], S[j] = 0.0, si + sj
    S = np.rint(S).astype(int)
    if S.sum() != K:  # numerical safety: fix cardinality by swapping
        ones, zeros = np.where(S == 1)[0], np.where(S == 0)[0]
        rng.shuffle(ones); rng.shuffle(zeros)
        while S.sum() > K and ones.size:
            S[ones[-1]] = 0; ones = ones[:-1]
        while S.sum() < K and zeros.size:
            S[zeros[-1]] = 1; zeros = zeros[:-1]
    return P, np.where(S == 1)[0]
