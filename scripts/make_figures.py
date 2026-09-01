"""Publication figures for the paper, built from the result files in results/.

Run from anywhere: figures go to <repo>/figures by default, or to $FIG_OUT.
Data sources (see README for the table/figure mapping):
  fig_e1_scaling  <- t1_sparse_results.json (W=200), final_study_results.txt (W>=2000)
  fig_t2_regret   <- t2_results.json
  fig_e2_regime   <- e2e_results.json
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.environ.get("FIG_OUT", os.path.normpath(os.path.join(HERE, "..", "figures")))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C = {"full": "#444444", "oracle": "#0072B2", "random": "#E69F00",
     "exp": "#009E73", "fs": "#CC79A7", "sqrt": "#999999"}
LBL = {"full": "Full", "oracle": "Oracle", "random": "Random",
       "exp": "Exp-Probe", "fs": "Fixed-share"}

# ---------------- E1: separation scaling ----------------
with open(os.path.join(RESULTS, "t1_sparse_results.json")) as f:
    t1s = json.load(f)
E = {"full": [t1s["full"][0]], "oracle": [t1s["oracle"][0]],
     "random": [t1s["random"][0]], "exp": [t1s["exp-probe"][0]]}
W_list = [200]
pat = re.compile(r"\[E1\] W=(\d+) K=\d+: full=(\d+) oracle=(\d+) exp-probe=(\d+) random=(\d+)")
for line in open(os.path.join(RESULTS, "final_study_results.txt")):
    m = pat.search(line)
    if m:
        W_list.append(int(m.group(1)))
        E["full"].append(float(m.group(2)))
        E["oracle"].append(float(m.group(3)))
        E["exp"].append(float(m.group(4)))
        E["random"].append(float(m.group(5)))
W = np.array(W_list, dtype=float)
E = {k: np.round(np.array(v)) for k, v in E.items()}

fig, ax = plt.subplots(figsize=(4.6, 3.4))
for k in ["full", "random", "exp", "oracle"]:
    ax.plot(W, E[k], "o-", color=C[k], label=LBL[k], lw=1.6, ms=5,
            zorder=3 if k == "oracle" else 2)
ax.plot(W, W, ":", color=C["sqrt"], lw=1, zorder=1, label=r"$\Theta(W)$ (full)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"number of scenarios $W$")
ax.set_ylabel("scenario evaluations to certification")
ax.set_xticks(W); ax.set_xticklabels(["200", "2,000", "10,000"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(frameon=False, loc="upper left")
ax.set_title("Sparse-trap family ($K$ = 4/20/50; $r{=}3$, $g{=}5$)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_e1_scaling.pdf"))
plt.close(fig)

# ---------------- T2: regret scaling ----------------
with open(os.path.join(RESULTS, "t2_results.json")) as f:
    t2 = json.load(f)
Ts = np.array(sorted(int(k) for k in t2["stationary"]))
stationary = {T: round(t2["stationary"][str(T)][0], 2) for T in Ts}
rotating = {T: round(t2["rotating"][str(T)][0], 2) for T in Ts}

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(Ts, [stationary[t] for t in Ts], "o-", color=C["oracle"], label="Stationary gains", lw=1.6, ms=5)
ax.plot(Ts, [rotating[t] for t in Ts], "s-", color=C["exp"], label="Rotating gains", lw=1.6, ms=5)
ref = Ts ** 0.5
ax.plot(Ts, ref / ref[0] * stationary[500], ":", color=C["sqrt"], lw=1,
        label=r"$\sqrt{T}$ reference")
lin = Ts ** (2.0 / 3.0)
ax.plot(Ts, lin / lin[0] * rotating[500], "--", color=C["sqrt"], lw=1,
        label=r"$T^{2/3}$ reference")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"horizon $T$"); ax.set_ylabel("regret vs. best fixed $K$-set")
ax.set_xticks(Ts); ax.set_xticklabels(["500", "1,000", "2,000", "4,000"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(frameon=False, loc="lower right")
ax.set_title(r"Semi-bandit EXP3.M ($W{=}50$, $K{=}5$)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_t2_regret.pdf"))
plt.close(fig)

# ---------------- E2: regime map (two panels) ----------------
with open(os.path.join(RESULTS, "e2e_results.json")) as f:
    e2 = json.load(f)
settings = [("W500_v0", "$W{=}500$\nvol$=0$\n(diffuse)"),
            ("W500_v1", "$W{=}500$\nvol$=1$\n(concentrated)"),
            ("W2000_v1", "$W{=}2000$\nvol$=1$")]
rules = ["full", "oracle", "random", "exp"]
rule_keys = {"full": "full", "oracle": "oracle", "random": "random", "exp": "exp-probe"}
evals = np.round(np.array([[e2[s][rule_keys[r]][0] for s, _ in settings] for r in rules]))
wall = np.round(np.array([[e2[s][rule_keys[r]][3] for s, _ in settings] for r in rules]), 1)
labels = [lab for _, lab in settings]

x = np.arange(3); wdt = 0.2
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
handles = []
for j, (r, col) in enumerate(zip(rules, [C["full"], C["oracle"], C["random"], C["exp"]])):
    off = (j - 1.5) * wdt
    h = axes[0].bar(x + off, evals[j], wdt, color=col, label=LBL[r])
    handles.append(h[0])
    axes[1].bar(x + off, wall[j], wdt, color=col)
for ax, ylab in zip(axes, ["scenario evaluations", "wall time (s)"]):
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylab); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_yscale("log")
axes[0].set_title("Cost to certification", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_title("Wall time (oracle incl.\nfree ranking solves)", fontsize=8.5)
fig.legend(handles=handles, labels=[LBL[r] for r in rules],
           loc="lower center", ncol=4, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "fig_e2_regime.pdf"))
plt.close(fig)
print("figures written to", OUT)
