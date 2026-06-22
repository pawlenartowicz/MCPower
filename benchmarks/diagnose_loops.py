"""Diagnose the statsmodels ConvergenceWarnings from the LME DIY-loop baselines.

Question it answers: are those warnings random per-sim hiccups (a fit occasionally
fails to converge — benign, already excluded from the power denominator), or do
they mark a systematically unfittable ("impossible") case (statsmodels MixedLM
cannot fit the design, so the loop's power/timing for that point is meaningless)?

It runs the REAL loop fit code (loops_py._lme_best_chunk and lme_naive) so the
failures are exactly the ones the benchmark hits — not a re-implementation — but
in-process (loops_py.DIAG is a parent-process global; the production `*_best`
path forks workers, which would not update it). Per (case, n, tier) it reports
the share of fits that are usable / non-converged / raised, and which warning
categories fired.

Usage (from mcpower/benchmarks/, venv active):
    python diagnose_loops.py                 # loop tier's own sim budget per point
    python diagnose_loops.py --scale 0.1     # 10% of that budget (faster, noisier)
    python diagnose_loops.py --sims 100      # fixed sims/point (overrides --scale)

Reads which cases to probe from benchmark_cases.json (family == "lme").
"""
import argparse

from cases import load_cases
import loops_py

SEED = 2137


def _probe(fn, case, n, n_sims, seed):
    """Run one loop tier in-process with diagnostics on; return the DIAG dict."""
    loops_py.DIAG = {}
    try:
        fn(case, n, n_sims, seed)
    finally:
        diag = loops_py.DIAG
        loops_py.DIAG = None
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of each tier's native sim budget (default 1.0)")
    ap.add_argument("--sims", type=int, default=None,
                    help="fixed sims/point, overriding --scale")
    args = ap.parse_args()

    cases = [c for c in load_cases("benchmark_cases.json") if c.family == "lme"]
    tiers = (("best", loops_py._lme_best_chunk), ("naive", loops_py.lme_naive))

    worst = {}  # case.id -> (min usable%, at n, tier)
    for case in cases:
        print(f"\n=== {case.id}   {case.formula}   "
              f"ICC={case.cluster['ICC']} n_clusters={case.cluster['n_clusters']} ===")
        print(f"{'n':>5} {'tier':>5} {'sims':>5} | {'usable':>7} {'noconv':>7} {'exc':>5} | warnings fired")
        for n in case.n_grid:
            for tier, fn in tiers:
                budget = args.sims if args.sims is not None else max(1, round(args.scale * case.n_sims[tier]))
                d = _probe(fn, case, n, budget, SEED + n)
                tot = d.get("total", 0) or 1
                usable = d.get("usable", 0)
                noconv = d.get("not_converged", 0)
                exc = sum(v for k, v in d.items() if k.startswith("exception:"))
                warns = {k[len("warn:"):]: v for k, v in d.items() if k.startswith("warn:")}
                wstr = ", ".join(f"{k}×{v}" for k, v in sorted(warns.items())) or "—"
                up = 100.0 * usable / tot
                print(f"{n:>5} {tier:>5} {tot:>5} | {up:>6.1f}% {100.0*noconv/tot:>6.1f}% "
                      f"{100.0*exc/tot:>4.0f}% | {wstr}")
                key = (up, n, tier)
                if case.id not in worst or up < worst[case.id][0]:
                    worst[case.id] = key

    print("\n=== verdict (lowest usable rate per case) ===")
    for cid, (up, n, tier) in worst.items():
        if up >= 90.0:
            tag = "benign noise (fits almost always usable)"
        elif up <= 10.0:
            tag = "IMPOSSIBLE / degenerate — fitter fails almost everywhere"
        else:
            tag = "PARTIAL — unreliable at this point, check if it's only small n"
        print(f"  {cid:18s} min usable {up:5.1f}% at n={n} ({tier}) -> {tag}")


if __name__ == "__main__":
    main()
