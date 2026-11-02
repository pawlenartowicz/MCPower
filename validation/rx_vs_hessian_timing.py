"""Wall-clock cost of wald_se="hessian" against the default "rx" on logit GLMMs.

Times find_power on one core for random-intercept designs (the cells of
rx_vs_hessian_type1.py) and random-slope designs (the cells of
glmm_slope_glmer.R), both standard errors on the same seed. The arms alternate
inside each pass, so drift hits both. The first pass is a warm-up and is dropped,
and each arm reports its minimum, since interference only ever slows a run.

Needs a stabilized machine with the clock locked. Run from the repo root:
    pin --p 1 -- python validation/rx_vs_hessian_timing.py

Writes validation/data/rx_vs_hessian_timing.csv, one row per design and prints it.
"""
from __future__ import annotations

import csv
import pathlib
import time

import mcpower

N_SIMS = 2000
PASSES = 5                  # the first is dropped
SEED = 2137
ARMS = ("rx", "hessian")
OUT_CSV = pathlib.Path(__file__).parent / "data" / "rx_vs_hessian_timing.csv"

# (kind, clusters, cluster size, slope variance)
DESIGNS = [
    ("intercept", 10, 20, 0.0), ("intercept", 20, 20, 0.0), ("intercept", 50, 20, 0.0),
    ("slope", 20, 20, 0.3), ("slope", 60, 8, 0.3), ("slope", 60, 20, 0.1),
]


def build(kind: str, g: int, slope_var: float):
    if kind == "intercept":
        m = mcpower.MCPower("y = x + z + (1|g)", family="logit")
        m.set_effects("x=0, z=0")
        m.set_cluster("g", ICC=0.2, n_clusters=g, cluster_level_vars=["z"])
        return m, "x, z"
    m = mcpower.MCPower("y = x * w + (1 + x|g)", family="logit")
    m.set_variable_type("w=binary")
    m.set_effects("x=0, w=0.3, x:w=0")
    m.set_cluster("g", ICC=0.3, n_clusters=g, cluster_level_vars=["w"],
                  random_slopes=["x"], slope_variance=slope_var)
    return m, "x, x:w"


def timed(kind: str, g: int, size: int, slope_var: float, arm: str) -> float:
    m, targets = build(kind, g, slope_var)
    m.set_baseline_probability(0.5)
    m.set_max_failed_simulations(1.0)
    t0 = time.perf_counter()
    m.find_power(sample_size=g * size, target_test=targets, n_sims=N_SIMS, seed=SEED,
                 wald_se=arm, verbose=False, progress_callback=False)
    return time.perf_counter() - t0


def main():
    fields = ["kind", "g", "cluster_size", "slope_var", "n", "n_sims", "rx_min_s", "hessian_min_s",
              "ratio", "rx_all_s", "hessian_all_s"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for kind, g, size, v in DESIGNS:
            runs = {a: [] for a in ARMS}
            for p in range(PASSES):
                for arm in (ARMS if p % 2 == 0 else ARMS[::-1]):
                    t = timed(kind, g, size, v, arm)
                    if p > 0:
                        runs[arm].append(t)
            rx, hess = min(runs["rx"]), min(runs["hessian"])
            row = {"kind": kind, "g": g, "cluster_size": size, "slope_var": v, "n": g * size,
                   "n_sims": N_SIMS, "rx_min_s": round(rx, 3), "hessian_min_s": round(hess, 3),
                   "ratio": round(hess / rx, 3),
                   "rx_all_s": " ".join(f"{t:.3f}" for t in runs["rx"]),
                   "hessian_all_s": " ".join(f"{t:.3f}" for t in runs["hessian"])}
            writer.writerow(row)
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
