# EXP-Probe: Regret Guarantees for Scenario Selection in Partial-Evaluation Benders Decomposition

This repository is the official reproducibility package for the paper
**"The Price of Selectivity: Regret Guarantees for Scenario Selection in Partial-Evaluation Benders Decomposition"** by Jin Xin Cao.

## Citation

If you use this software or data, please cite:
> Cao, Jin Xin (2026). "The Price of Selectivity: Regret Guarantees for Scenario Selection in Partial-Evaluation Benders Decomposition." Manuscript under review.

## Repository Structure

```
├── src/                    # Core library
│   ├── spike_core.py       # Partial-evaluation Benders on piecewise-linear instances (7 rules)
│   └── real_lp.py          # Real-LP testbed with genuine recourse LPs (facility opening)
├── scripts/                # Experiment entry points (run from scripts/)
│   ├── run_t1.py           # Separation on trap/informative families (W=40)
│   ├── run_t1_sparse.py    # Sparse-trap family scaling (W=200)
│   ├── run_t2.py           # Regret scaling (stationary + rotating gains)
│   ├── run_t2_tune.py      # Regret hyperparameter ablation
│   ├── run_t2_matched.py   # Certified-regime validation (matched gamma schedule, balance eta<=K gamma/W)
│   ├── run_t3.py           # LP solve timing + W=10^4 scaling projection
│   ├── run_final_study.py  # Sparse-trap family scaling (W=2000, 10000)
│   ├── run_e2e.py          # Recourse-LP regime map, facility-opening testbed (W=500, 2000; vol=0,1)
│   ├── run_cap_expansion.py# Capacity-expansion problem class
│   ├── run_fixed_share.py  # Fixed-share ablation (stationary/rotating)
│   ├── run_bootstrap.py    # Bootstrap over repeated runs of one fixed sparse-trap (W=2000) instance
│   ├── run_timing_decomp.py# Master-vs-subproblem timing decomposition
│   ├── run_large_scale.py  # W=10^5 attempt (incomplete; see scaling notes)
│   ├── run_tchk_sensitivity.py # Certification-period sweep with est-det
│   ├── run_k_sensitivity.py # Selection-budget sweep (K in {10,25,50,100})
│   ├── run_concentration.py # Two-axis proxies (top-K mass share, first-iterate subgradient diversity)
│   ├── run_cap_expansion_milp.py# Genuinely binary (mixed-integer) master probe
│   └── make_figures.py     # Reproduce all 3 paper figures
├── data/                   # (empty; instances are generated on the fly by scripts)
├── results/                # All result files (JSON/TXT/LOG) for every table and figure
├── docs/
│   └── log_space_implementation.md  # The log-space weight implementation referenced in Section 5
└── figures/                # Output directory of make_figures.py
```

## Requirements

- Python >= 3.10 with numpy, scipy, matplotlib
- No commercial solver needed (HiGHS via `scipy.optimize`)
- All experiments in the paper ran on a Mac Studio (Apple M3 Ultra,
  32-core CPU, 256 GB unified memory); the study is single-threaded and
  reproduces on any modern desktop

## Reproducing All Results

From the repository root:

```bash
cd scripts

# Sparse-trap separation and scaling
python run_t1.py
python run_t1_sparse.py
python run_final_study.py            # W=2000, 10000

# Regret scaling and hyperparameter ablation
python run_t2_matched.py   # certified-regime validation (matched gamma schedule)
python run_t2.py
python run_t2_tune.py

# Recourse-LP regime map and supporting studies
python run_e2e.py                    # writes results/e2e_results.*
python run_cap_expansion.py
python run_fixed_share.py
python run_bootstrap.py
python run_timing_decomp.py
python run_t3.py

# Figures (writes ../figures/*.pdf; set FIG_OUT to redirect)
python make_figures.py
```

## Figures and Tables Mapping

| Paper | Source script | Source data |
|---|---|---|
| Table 1 (sparse-trap separation and scaling) | `run_t1_sparse.py`, `run_final_study.py` | `results/t1_sparse_results.json`, `results/final_study_results.txt` |
| Exploratory small-scale runs (not cited in the paper) | `run_t1.py` | `results/t1_results.json` |
| Table 2 (facility-opening regime map) | `run_e2e.py` | `results/e2e_results.txt/json` |
| Table 3 (certification-period sensitivity) | `run_tchk_sensitivity.py` | `results/tchk_sensitivity.txt/json` |
| Fig. 1 (regret scaling, Section 5) | `make_figures.py` | `results/t2_results.json` |
| Fig. 2 (sparse-trap scaling) | `make_figures.py` | `results/t1_sparse_results.json`, `results/final_study_results.txt` |
| Fig. 3 (facility-opening regime) | `make_figures.py` | `results/e2e_results.json` |
| Fixed-share ablation (Section 5) | `run_fixed_share.py` | `results/fixed_share_{stationary,rotating}.json` |
| Bootstrap significance (Section 6) | `run_bootstrap.py` | `results/bootstrap_significance.json` |
| Timing decomposition (Section 6) | `run_timing_decomp.py` | `results/timing_decomp.log` |
| Per-solve timing of the 32x48 recourse LP (Section 6) | `run_t3.py` | `results/t3_results.txt` |
| Table 4 (capacity expansion) | `run_cap_expansion.py` | `results/cap_expansion.log` (paper wall times: archived summary at commit 308be4c, preserved verbatim as `results/archive/cap_expansion_308be4c.log`; the current log is a later re-run with identical evals/iters; the script prints per-seed lines so the one-decimal iteration means are checkable) |
| Certified-regime validation (Section 5) | `run_t2_matched.py` | `results/t2_matched.json` (per-run and pooled bounds) |
| Regret-rate sweep (Section 5) | `run_t2_tune.py` | `results/t2_tune.txt` |
| Binary-master probe (Section 6) | `run_cap_expansion_milp.py` | `results/cap_expansion_milp.txt/json` (per-seed lines and two-seed means; the archived `certified` flag for the diffuse est-det cell is a boundary artifact of the old `it < max_iter` test -- the evaluation count 5,975 shows that seed certified at the iteration-200 check; the script now reads `sim.certified`) |
| Binary-master probe, newer solver stack | `run_cap_expansion_milp.py` | `results/cap_expansion_milp_scipy1163.log` (SciPy 1.16.3 re-run: identical except the diffuse est-det cell, where both seeds certify instead of one failing) |
| K sensitivity (Section 6) | `run_k_sensitivity.py` | `results/k_sensitivity.txt` |
| Subproblem dimensions and solve time (Section 6) | `run_subprob_timing.py` | `results/e2_subsolve_timing.txt` (510 columns / 46 rows exact; per-solve time is machine-dependent: about 1.7 ms on the study machine, about 2 ms under a newer stack) |
| Two-axis proxies (Section 6) | `run_concentration.py` | `results/concentration.txt` |

`make_figures.py` reads the result files directly, so the figures regenerate
from the same numbers the tables report.

## Reproducibility notes

- **Oracle tie-break.** `Definition 5` of the paper specifies ties broken by smallest
  index; both simulators now implement that rule (`np.argsort(-mass, kind="stable")[:K]`).
  The two-piece tables are invariant under the tie rule (owners carry positive mass), and
  the W=200 cell reproduces bit-for-bit (`results/t1_sparse_results.json` unchanged).
- **Initial pools on the recourse-LP testbed.** Each scenario's pool starts with the single
  valid cut `(0, 0)`, i.e. the bound `Q_w >= 0` of Definition 2's nonempty initial pool. The
  spot-check against the archived facility-opening counts (all four rules, vol=0 and vol=1)
  reproduces every evaluation and iteration count exactly.
- **T2 regret scan and fixed-share ablation.** `results/t2_results.json` and
  `results/fixed_share_{stationary,rotating}.json` were regenerated with the exact-marginal
  sampler of `src/sampling.py`. The previously archived values came from a pre-refactor
  estimator (sampling with `rng.choice(..., p=...)` and dividing by the inclusion lower bound
  `min(Kp/2, 1/2)`), which the shipped script no longer reproduces; the manuscript's fitted
  slopes (0.52 stationary, 0.68 rotating) and ablation numbers were updated to the
  regenerated values. The qualitative statements are unchanged (sublinear stationary growth,
  steeper under rotation, `alpha=0` wins stationary, fixed-share tracks rotation).
- **Experiment parameters.** The learning rate and mixing used in every run are deterministic
  functions of (W, K, T) in the scripts (`eta = sqrt(K ln W / (T W))`, `gamma = 0.05` in the
  Benders runs); `results/t2_matched.json` additionally records gamma, eta, and the balance
  slack explicitly for the certified-regime runs.
- **Sampler repair audit.** `src/sampling.py` counts activations of its floating-point
  cardinality repair in `REPAIR_COUNT`; `scripts/run_repair_probe.py` exercises 100,000
  random sampler states plus full Table-1-scale runs and asserts the count stays at zero
  (artifact: `results/repair_probe.txt`), so the exact-marginals claim holds on the archived
  paths.
- **Archive vs. re-run.** `results/cap_expansion.log` and the wall-clock columns of the paper
  come from the archived summary at commit `308be4c`; the timings are load-dependent.

## License

MIT License.

## Contact

Jin Xin Cao (imucjx@163.com)

## Scaling limits

The W=10,000 sparse-trap results are certified and reported in the paper.
The W=100,000 run (`scripts/run_large_scale.py`) was not completed: the
master LP for W=100,000 scenarios exceeds the dense numpy/scipy pipeline's
practical capacity (each master solve becomes the bottleneck). The paper
therefore reports W=10,000 as the largest measured point and states the
W=10^5 behavior as a theorem prediction (Thm. 2(ii)), not a measured number.
