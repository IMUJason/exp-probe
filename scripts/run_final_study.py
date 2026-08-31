"""E1 sparse-trap scaling with EXP-Probe, W in {2000, 10000}."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
out = []

# ---------------- E1: PWL separation with EXP-Probe ----------------
from run_t1_sparse import make_trap_sparse
from spike_core import eval_counts

for W, K in [(2000, 20), (10000, 50)]:
    inst, groups = make_trap_sparse(W=W)
    res = eval_counts(inst, eps=1e-3, K=K, reps=3, max_iter=1200)
    o = res['oracle'][0]; r = res['random'][0]; xp = res['exp-probe'][0]
    estdet = "STALL" if res['est-det'][2] >= 1200 else f"{res['est-det'][0]:.0f}"
    out.append(f"[E1] W={W} K={K}: full={res['full'][0]:.0f} oracle={o:.0f} "
               f"exp-probe={xp:.0f} random={r:.0f} est-det={estdet} | "
               f"random/oracle={r/max(o,1):.1f}x exp-probe/oracle={xp/max(o,1):.1f}x")
    print(out[-1], flush=True)

open(os.path.join(RESULTS, "final_study_results.txt"), "w").write("\n".join(out))
print("saved final_study_results.txt")
