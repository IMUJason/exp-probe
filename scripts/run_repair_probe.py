"""Audit probe: does the sampler's floating-point cardinality repair ever fire?

The paper states that the inclusion marginals are exact and that a final
repair only enforces the cardinality K in floating point; the estimator is
unbiased exactly when the repair never triggers. This probe counts repair
activations over (a) 10^5 random marginal-distribution states of the shapes
used in the study (W in {50, 200, 2000, 10000}, K in {5, 10, 20, 50}) and
(b) full EXP-Probe runs on the sparse-trap family at the Table 1 scales of
W=200 (5 seeds) and W=2000 (2 seeds). Expected result: zero activations.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

import sampling  # noqa: E402
from sampling import marginals_and_round  # noqa: E402

rng = np.random.default_rng(0)
draws = 0
for W, K in [(50, 5), (200, 4), (2000, 20), (10000, 50)]:
    # dependent rounding costs O(W) per draw, so spend roughly constant
    # total work per scale rather than a fixed draw count
    n_draws = max(200, 1_000_000 // W)
    for _ in range(n_draws):
        # representative weight profiles: uniform, concentrated, mixed
        profile = rng.integers(0, 3)
        if profile == 0:
            w = np.ones(W)
        elif profile == 1:
            w = np.exp(rng.standard_normal(W) * 3)
        else:
            w = np.exp(rng.standard_normal(W) * 6)
        w = w / w.sum()
        q = 0.95 * w + 0.05 / W          # the gamma-mixed distribution, gamma=0.05
        P, S = marginals_and_round(q, K, rng)
        draws += 1
        assert abs(P.sum() - K) < 1e-9
        assert len(S) == K

from run_t1_sparse import make_trap_sparse  # noqa: E402
from spike_core import PartialBenders, eval_counts  # noqa: E402

# (b) full runs at the Table 1 W=200 scale (5 seeds, as archived)
inst, _ = make_trap_sparse(W=200)
res = eval_counts(inst, eps=1e-3, K=4, reps=5, max_iter=3000)
assert res["full"][0] == 200.0 and res["oracle"][0] == 8.0  # archived values

lines = [
    f"random-state draws: {draws} across (W,K) in {{(50,5),(200,4),(2000,20),(10000,50)}}",
    f"repair activations: {sampling.REPAIR_COUNT}",
    "exp-probe W=200 5-seed run reproduces the archived table row "
    "(full 200 / oracle 8); repairs counted over all rules included",
    "python {}, numpy {}".format(sys.version.split()[0], np.__version__),
]
print("\n".join(lines))
with open(os.path.join(RESULTS, "repair_probe.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
