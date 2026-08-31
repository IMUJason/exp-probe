"""Spike-T1: separation between selection rules on two instance families."""
import sys, json
sys.path.insert(0, ".")
from spike_core import make_trap_family, make_informative_family, eval_counts

results = {}
instA, groups = make_trap_family()
instB = make_informative_family()
for name, inst in [("trap (no signal)", instA), ("informative (signal exists)", instB)]:
    res = eval_counts(inst, eps=1e-3, K=4, reps=5)
    results[name] = res
    print(f"\n=== {name}  (W={inst.W}, K=4) ===")
    print(f"{'rule':<12} {'evals(mean±sd)':<18} {'iters(mean±sd)':<18}")
    for rule, (em, es, im, isd) in res.items():
        print(f"{rule:<12} {em:>8.0f} ± {es:<6.0f} {im:>8.1f} ± {isd:<5.1f}")

for name, res in results.items():
    o = res['oracle'][0]; r = res['random'][0]
    print(f"\n[{name}] random/oracle={r/max(o,1):.1f}x  est-det/oracle={res['est-det'][0]/max(o,1):.1f}x  est-nn/oracle={res['est-nn'][0]/max(o,1):.1f}x")
json.dump(results, open("t1_results.json", "w"), indent=1)
