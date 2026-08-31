# EXP-Probe: Regret-Optimal Scenario Selection in Partial-Evaluation Benders Decomposition

This repository is the official reproducibility package for the paper
**"The Price of Selectivity: Regret-Optimal Scenario Selection in Partial-Evaluation Benders Decomposition"** by Jin Xin Cao.

## Citation

If you use this software or data, please cite:
> Cao, Jin Xin (2026). "The Price of Selectivity: Regret-Optimal Scenario Selection in Partial-Evaluation Benders Decomposition." Manuscript under review.

## Repository Structure

```
├── src/                    # Core library
│   ├── spike_core.py       # Partial-evaluation Benders on piecewise-linear instances (7 rules)
│   └── real_lp.py          # Real-LP testbed with genuine recourse LPs (facility opening)
├── scripts/                # Experiment entry points (run from scripts/)
│   ├── run_t1.py           # E1: separation on trap/informative families (W=40)
│   ├── run_t1_sparse.py    # E1: sparse-trap scaling (W=200)
│   ├── run_t2.py           # T2: regret scaling (stationary + rotating gains)
│   ├── run_t2_tune.py      # T2: hyperparameter ablation
│   ├── run_t3.py           # T3: LP solve timing + W=10^4 scaling projection
│   ├── run_final_study.py  # E1: sparse-trap scaling (W=2000, 10000)
│   ├── run_e2e.py          # E2: real-LP regime map (W=500, 2000; vol=0,1)
│   ├── run_cap_expansion.py# E2b: capacity-expansion problem class
│   ├── run_fixed_share.py  # Fixed-share ablation (stationary/rotating)
│   ├── run_bootstrap.py    # Cluster bootstrap significance for E2
│   ├── run_timing_decomp.py# Master-vs-subproblem timing decomposition
│   ├── run_large_scale.py  # W=10^5 attempt (incomplete; see scaling notes)
│   ├── run_tchk_sensitivity.py # E2: certification-period sweep with est-det
│   ├── run_k_sensitivity.py # E2: selection-budget sweep (K in {10,25,50,100})
│   ├── run_cap_expansion_milp.py# E2b: genuinely binary (MILP) master probe
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

# E1: separation families and sparse-trap scaling
python run_t1.py
python run_t1_sparse.py
python run_final_study.py            # W=2000, 10000

# T2: regret scaling and hyperparameter ablation
python run_t2.py
python run_t2_tune.py

# E2: real-LP regime map and supporting studies
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
| Table 1 (E1 separation) | `run_t1.py`, `run_t1_sparse.py`, `run_final_study.py` | `results/t1_results.json`, `results/t1_sparse_results.json`, `results/final_study_results.txt/json` |
| Table 2 (E2 regime map) | `run_e2e.py` | `results/e2e_results.txt/json` |
| Table 3 (capacity expansion) | `run_cap_expansion.py` | `results/cap_expansion.log` |
| Fig. 1 (regret scaling, Section 5) | `make_figures.py` | `results/t2_results.json` |
| Fig. 2 (E1 scaling) | `make_figures.py` | `results/t1_sparse_results.json`, `results/final_study_results.txt` |
| Fig. 3 (E2 regime) | `make_figures.py` | `results/e2e_results.json` |
| Fixed-share ablation (Section 5) | `run_fixed_share.py` | `results/fixed_share_{stationary,rotating}.json` |
| Bootstrap significance (Section 6) | `run_bootstrap.py` | `results/bootstrap_significance.json` |
| Timing decomposition (Section 6) | `run_timing_decomp.py` | `results/timing_decomp.log` |
| T_chk sensitivity (Section 6) | `run_tchk_sensitivity.py` | `results/tchk_sensitivity.txt/json` |
| Binary-master probe (Section 6) | `run_cap_expansion_milp.py` | `results/cap_expansion_milp.txt/json` |
| K sensitivity (Section 6) | `run_k_sensitivity.py` | `results/k_sensitivity.txt` |

`make_figures.py` reads the result files directly, so the figures regenerate
from the same numbers the tables report.

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
