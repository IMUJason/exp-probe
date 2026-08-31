"""Publication-quality figures for the regret-probe paper (matplotlib, vector PDF)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Figures are written to <repo>/figures by default (independent of the
# caller's working directory); set FIG_OUT to redirect, e.g. to the
# manuscript's figures directory.
HERE = os.path.dirname(os.path.abspath(__file__))
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

# ---------------- Fig 1: E1 separation scaling ----------------
W = np.array([200, 2000, 10000])
E = {"full": [200, 2000, 10000], "oracle": [4, 20, 50],
     "random": [30, 161, 1615], "exp": [44, 335, 1147]}
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for k in ["full", "random", "exp", "oracle"]:
    ax.plot(W, E[k], "o-", color=C[k], label=LBL[k], lw=1.6, ms=5,
            zorder=3 if k == "oracle" else 2)
ax.plot(W, W, ":", color=C["sqrt"], lw=1, zorder=1, label=r"$\Theta(W)$ (full)")
from matplotlib.ticker import NullFormatter
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

# ---------------- Fig 2: T2 regret scaling ----------------
Ts = np.array([500, 1000, 2000, 4000])
stationary = {500: 249.71, 1000: 423.55, 2000: 652.18, 4000: 959.08}
rotating = {500: 107.69, 1000: 200.67, 2000: 355.17, 4000: 625.22}
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(Ts, [stationary[t] for t in Ts], "o-", color=C["oracle"], label="Stationary gains", lw=1.6, ms=5)
ax.plot(Ts, [rotating[t] for t in Ts], "s-", color=C["exp"], label="Rotating gains", lw=1.6, ms=5)
ref = Ts ** 0.5
ax.plot(Ts, ref / ref[0] * stationary[500], ":", color=C["sqrt"], lw=1,
        label=r"$\sqrt{T}$ reference")
lin = Ts ** 0.844
ax.plot(Ts, lin / lin[0] * rotating[500], "--", color=C["sqrt"], lw=1,
        label=r"$T^{0.84}$ reference")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"horizon $T$"); ax.set_ylabel("regret vs. best fixed $K$-set")
ax.set_xticks(Ts); ax.set_xticklabels(["500", "1,000", "2,000", "4,000"])
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(frameon=False, loc="lower right")
ax.set_title(r"Semi-bandit EXP3.M ($W{=}50$, $K{=}5$)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_t2_regret.pdf"))
plt.close(fig)

# ---------------- Fig 3: E2 regime map (two panels) ----------------
settings = ["$W{=}500$\nvol$=0$\n(diffuse)", "$W{=}500$\nvol$=1$\n(concentrated)", "$W{=}2000$\nvol$=1$"]
rules = ["full", "oracle", "random"]
evals = np.array([[3000, 1000, 4000], [5950, 5950, 8950], [5950, 5950, 8950]], dtype=float)
wall = np.array([[5.2, 1.6, 6.0], [95.1, 91.4, 334.8], [12.1, 10.5, 16.6]])
x = np.arange(3); wdt = 0.26
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
handles = []
for j, (r, col) in enumerate(zip(rules, [C["full"], C["oracle"], C["random"]])):
    h = axes[0].bar(x + (j - 1) * wdt, evals[j], wdt, color=col, label=LBL[r])
    handles.append(h[0])
    axes[1].bar(x + (j - 1) * wdt, wall[j], wdt, color=col)
for ax, ylab in zip(axes, ["scenario evaluations", "wall time (s)"]):
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=8)
    ax.set_ylabel(ylab); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_yscale("log")
axes[0].set_title("Cost to certification", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_title("Wall time (oracle incl.\nfree ranking solves)", fontsize=8.5)
fig.legend(handles=handles, labels=[LBL[r] for r in rules],
           loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "fig_e2_regime.pdf"))
plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
