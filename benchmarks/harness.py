"""Python benchmark harness: build MCPower models from Cases, time the
recorded methods (mcpower, tool selector, DIY loops) over each case's n grid,
and write {meta, records} results JSON.
"""
from __future__ import annotations
# Pin BLAS to 1 thread before any import that could load numpy.
# cpu_count() pool workers × multi-threaded BLAS oversubscribe the machine,
# inflating the loop baseline and flattering MCPower; the engine reads
# RAYON_NUM_THREADS instead. setdefault keeps user overrides possible.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, json, pathlib, platform, statistics, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
import mcpower
from cases import Case, load_cases
from loops_py import LOOPS

def build_model(case: Case):
    # The benchmark "glmm" family is a logistic model with clusters — mcpower
    # has no separate GLMM family; it dispatches GLMM off logit + set_cluster.
    mc_family = "logit" if case.family == "glmm" else case.family
    m = mcpower.MCPower(case.formula, family=mc_family)
    for var, spec in case.variable_types.items():
        m.set_variable_type(f"{var}={spec}")
    m.set_effects(case.effects)
    if case.correlations is not None:
        m.set_correlations(case.correlations)
    if case.baseline_p is not None:
        m.set_baseline_probability(case.baseline_p)
    if case.cluster is not None:
        cl = case.cluster
        # Random slopes ride inside the cluster dict (uncorrelated REs in the
        # benchmark suite — correlation recovery is validation's job). Python's
        # set_cluster takes a single scalar slope_variance applied to every
        # listed slope; harness.R mirrors via its per-slope list. Keep in sync.
        if cl.get("random_slopes"):
            m.set_cluster(cl["var"], ICC=cl["ICC"], n_clusters=cl["n_clusters"],
                          random_slopes=cl["random_slopes"],
                          slope_variance=cl.get("slope_variance", 0.0),
                          slope_intercept_corr=cl.get("slope_intercept_corr", 0.0))
        else:
            m.set_cluster(cl["var"], ICC=cl["ICC"], n_clusters=cl["n_clusters"])
    if case.max_failed_frac is not None:
        m.set_max_failed_simulations(case.max_failed_frac)
    # Honest-compare override (benchmark-only): the loops and dedicated tools
    # draw factor levels randomly per sim, while mcpower's optimistic baseline
    # uses exact-count allocation. Flip to sampled so all tiers answer the
    # same random-allocation question. No-op for factor-free cases; the
    # engine default is unchanged. Keep in sync with harness.R.
    m.set_scenario_configs({"optimistic": {"sampled_factor_proportions": True}})
    return m


CASES_PATH = pathlib.Path(__file__).parent / "benchmark_cases.json"
RECORDED_METHODS = ["mcpower_find_power", "mcpower_find_sample_size",
                    "tool", "loop_naive", "loop_best"]

# Scaling for every tier's sim count (mcpower / loops / tools). 1.0 = the real
# (publishable) run; set e.g. 0.1 for a 1/10-sims quick preview. Recorded rows
# carry the scaled n_sims, so per-sim normalization in combine.py stays honest
# either way; meta records the scale.
N_SIMS_SCALE = 0.1


def scaled_sims(count):
    return max(1, round(N_SIMS_SCALE * count))
_REPS = {"mcpower_find_power": 3, "mcpower_find_sample_size": 3,
         "loop_best": 3, "loop_naive": 1}


def seed_for(n: int) -> int:
    return 2137 + int(n)


def _cpu_info():
    """(model name, physical core count) from /proc/cpuinfo; graceful fallback."""
    model, pairs, phys = "unknown", set(), None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and model == "unknown":
                model = line.split(":", 1)[1].strip()
            elif line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                pairs.add((phys, line.split(":", 1)[1].strip()))
    except OSError:
        pass
    return model, (len(pairs) or os.cpu_count())


def build_meta(threads_mode):
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "unknown"

    cpu_model, cores_physical = _cpu_info()
    return {
        "lang": "py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu_model": cpu_model,
        "cores_physical": cores_physical,
        "cores_logical": os.cpu_count(),
        "threads_mode": threads_mode,
        "n_sims_scale": N_SIMS_SCALE,
        "lang_version": platform.python_version(),
        "packages": {p: ver(p) for p in ("mcpower", "numpy", "scipy", "statsmodels")},
    }


def fmt_vec(v) -> str:
    arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    return "[" + ", ".join(f"{x:.3f}" for x in arr) + "]"


def fmt_first_achieved(fa) -> str:
    items = sorted(fa.items(), key=lambda kv: str(kv[0]))
    return "[" + ", ".join("--" if v is None else str(v) for _, v in items) + "]"


def time_call(fn, *, warmup, reps):
    if warmup:
        fn()
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def run_mcpower_find_power(case, n, n_sims, seed):
    m = build_model(case)
    res = m.find_power(sample_size=n, n_sims=n_sims, seed=seed,
                       progress_callback=False, verbose=False)
    return [float(x) for x in res["power_uncorrected"][0]]


def run_mcpower_find_sample_size(case, n_sims, seed):
    m = build_model(case)
    g = case.n_grid
    res = m.find_sample_size(target_power=case.target_power, from_size=g[0],
                             to_size=g[-1], by=g[1] - g[0], mode="linear",
                             n_sims=n_sims, seed=seed,
                             progress_callback=False, verbose=False)
    n_pts = int(res.get("n_sample_sizes", 0))
    fa = {str(k): (None if v is None else int(v))
          for k, v in res.get("first_achieved", {}).items()}
    # Full per-grid-point power curve (diagnostic: must agree with the
    # find_power tier at the same n within MC noise — seeds differ).
    curve = [[float(p) for p in pt] for pt in res.get("power_uncorrected", [])]
    return fa, n_pts, curve


def run_loop(case, kind, n, n_sims, seed):
    out = LOOPS[case.family][kind](case, n, n_sims, seed)
    return [float(x) for x in np.atleast_1d(np.asarray(out[0], dtype=float))]


def record(case, method, n, n_sims, elapsed, power):
    return {"case_id": case.id, "family": case.family, "lang": "py", "method": method,
            "n": n, "n_sims": n_sims, "time_s": elapsed,
            "per_sim_s": elapsed / n_sims, "power": power}


def main(argv):
    parser = argparse.ArgumentParser(description="MCPower benchmark harness")
    parser.add_argument("--case", default="all",
                        help="Case id to run, or 'all' (default)")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated methods (default: all recorded methods)")
    parser.add_argument("--out", default="results/py.json",
                        help="Output JSON path (default: results/py.json)")
    parser.add_argument("--threads", choices=["auto", "1"], default="auto",
                        help="Thread mode: auto (rayon multi-core) or 1 (re-exec under RAYON_NUM_THREADS=1)")
    args = parser.parse_args(argv)

    # Single-thread mode: re-exec under RAYON_NUM_THREADS=1 — the env var must
    # be set before the engine spins up its thread pool. Mirrors harness.R.
    if args.threads == "1" and os.environ.get("RAYON_NUM_THREADS") != "1":
        env = {**os.environ, "RAYON_NUM_THREADS": "1"}
        sys.exit(subprocess.call([sys.executable, __file__, *argv], env=env))
    if args.threads == "1":
        from mcpower import _engine
        _engine.set_n_threads(1)

    all_cases = load_cases(CASES_PATH)
    if args.case == "all":
        cases = all_cases
    else:
        matches = [c for c in all_cases if c.id == args.case]
        if not matches:
            ids = [c.id for c in all_cases]
            sys.exit(f"Case {args.case!r} not found. Available: {ids}")
        cases = matches

    methods = RECORDED_METHODS if args.methods is None else args.methods.split(",")
    threads = args.threads
    out_path = args.out

    records = []

    for case in cases:
        for method in methods:

            if method == "mcpower_find_power":
                n_sims = scaled_sims(case.n_sims["mcpower"])
                print(f"\n=== {case.id} mcpower find_power ({n_sims} sims/n, threads={threads}) ===")
                print(f"{'n':>6} | {'time(s)':>10} | {'per-sim(s)':>12} | power")
                print("-" * 60)

                for n in case.n_grid:
                    t, pwr = time_call(
                        lambda n=n: run_mcpower_find_power(case, n, n_sims, seed_for(n)),
                        warmup=True, reps=_REPS[method],
                    )
                    records.append(record(case, method, n, n_sims, t, pwr))
                    print(f"{n:>6} | {t:>10.4f} | {t/n_sims:>12.6f} | {fmt_vec(pwr)}")

            elif method in ("loop_best", "loop_naive"):
                # No Python GLMM loop baseline: statsmodels has no frequentist
                # Laplace GLMM matching glmer's Wald-z rule, so a fair Python
                # comparator does not exist. The R harness runs the glmer loop;
                # combine.py renders the missing py rows as dashes. (The Python
                # `tool` tier is skipped for the same no-equivalent reason.)
                if case.family == "glmm":
                    print(f"\n=== {case.id} {method} (glmm): skipped — no Python GLMM loop baseline ===")
                    continue
                kind = "best" if method == "loop_best" else "naive"
                tier = kind
                n_sims = scaled_sims(case.n_sims[tier])
                reps = _REPS[method]
                print(f"\n=== {case.id} {method} ({n_sims} sims/n) ===")
                print(f"{'n':>6} | {'time(s)':>10} | {'per-sim(s)':>12} | power")
                print("-" * 60)
                for n in case.n_grid:
                    t, pwr = time_call(
                        lambda n=n: run_loop(case, kind, n, n_sims, seed_for(n)),
                        warmup=True, reps=reps,
                    )
                    records.append(record(case, method, n, n_sims, t, pwr))
                    print(f"{n:>6} | {t:>10.4f} | {t/n_sims:>12.6f} | {fmt_vec(pwr)}")

            elif method == "tool":
                label = case.tool or "none"
                print(f"\n=== {case.id} tool ({label}): skipped — no Python simulation-based power tool ===")

            elif method == "mcpower_find_sample_size":
                # One call evaluates the FULL grid from n_sims shared draws
                # (budget = n_sims total, NOT x grid — see orchestrator
                # find_sample_size.rs). Same sims/point as the find_power tier,
                # so combine.py compares it grid-vs-grid against that tier.
                n_sims = scaled_sims(case.n_sims["mcpower"])
                print(f"\n=== {case.id} mcpower find_sample_size ({n_sims} sims, full grid, threads={threads}) ===")
                print(f"{'target':>8} | {'time(s)':>10} | {'grid pts':>10} | first_achieved")
                print("-" * 60)

                elapsed, (fa, n_pts, curve) = time_call(
                    lambda: run_mcpower_find_sample_size(case, n_sims, 2137),
                    warmup=True, reps=_REPS[method],
                )
                # One row per case: n = recommended n (max over targets; 0 if any
                # target never achieves); power = the full per-point curve
                # (diagnostic against the find_power tier's powers at the same n).
                fa_vals = list(fa.values())
                n_rec = max(fa_vals) if fa_vals and all(v is not None for v in fa_vals) else 0
                records.append(record(case, method, n_rec, n_sims, elapsed, curve))
                print(f"{case.target_power:>8.2f} | {elapsed:>10.4f} | {n_pts:>10} | {fmt_first_achieved(fa)}")

            else:
                print(f"WARNING: unknown method {method!r}, skipping", file=sys.stderr)

    abs_out = os.path.abspath(out_path)
    parent = os.path.dirname(abs_out)
    os.makedirs(parent or ".", exist_ok=True)
    with open(abs_out, "w") as f:
        json.dump({"meta": build_meta(threads), "records": records}, f)
    print(f"\nWrote {len(records)} records to {abs_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
