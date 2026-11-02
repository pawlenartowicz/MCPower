"""Type-I error of the tests a logit GLMM applies to terms with a random slope.

The random-slope grid of df_calibration.R covers linear mixed models only. This
script runs the same cells on the logit path: g clusters in {5, 10, 20, 30, 60}
x cluster size in {8, 20} x slope variance in {.05, .10, .30}, ICC .3, formula
y ~ x * w + (1 + x | g), w binary and cluster-level, effects x = 0, x:w = 0,
w = 0.3, baseline probability .5. The targets are x (a within-cluster effect
with a random slope) and x:w (a cross-level interaction).

Arms, each on the same seed:
  auto, rx          the default: between-within t on one grouping, rx standard error
  asymptotic, rx    Wald z with the rx standard error
  asymptotic, hessian  Wald z with the Hessian standard error, which is glmer's test
Slope variances are on the log-odds scale, where the level-1 residual variance
is pi^2/3, so the same numbers as the linear grid give smaller slopes relative
to the residual.

A failed or separated fit counts as not significant and stays in the
denominator, as in any power run. The failure limit is lifted so that the
smallest cells report a rate and their convergence rate instead of stopping.

Run from the repo root with the repo's venv:
    pin 8 -- python validation/glmm_slope_type1.py

Writes validation/data/glmm_slope_type1.csv, one row per cell x arm x target,
adding each cell's rows as it finishes. GLMM_SLOPE_K sets the simulations per
cell (default 10000).
"""
from __future__ import annotations

import csv
import os
import pathlib
import time

import mcpower

K = int(os.environ.get("GLMM_SLOPE_K", "10000"))
G_VALUES = [5, 10, 20, 30, 60]
CLUSTER_SIZES = [8, 20]
SLOPE_VARIANCES = [0.05, 0.1, 0.3]
ARMS = [("auto", "rx"), ("asymptotic", "rx"), ("asymptotic", "hessian")]
TARGETS = ["x", "x:w"]
ICC = 0.3
BASELINE_P = 0.5
BASE_SEED = 60000000

OUT_CSV = pathlib.Path(__file__).parent / "data" / "glmm_slope_type1.csv"


def run_cell(g: int, m: int, slope_var: float, df_method: str, wald_se: str, seed: int, k: int = K):
    model = mcpower.MCPower("y = x * w + (1 + x|g)", family="logit")
    model.set_variable_type("w=binary")
    model.set_baseline_probability(BASELINE_P)
    model.set_effects("x=0, w=0.3, x:w=0")
    model.set_cluster("g", ICC=ICC, n_clusters=g, cluster_level_vars=["w"],
                      random_slopes=["x"], slope_variance=slope_var)
    model.set_max_failed_simulations(1.0)
    t0 = time.time()
    r = model.find_power(sample_size=g * m, target_test="x, x:w", n_sims=k, seed=seed,
                         df_method=df_method, wald_se=wald_se, verbose=False)
    return r, time.time() - t0


def main():
    fields = ["label", "g", "cluster_size", "slope_var", "n", "df_method", "wald_se", "target", "rejection_rate",
              "ci_lo", "ci_hi", "convergence_rate", "n_sims", "seed", "elapsed_sec"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        cell_index = 0
        for g in G_VALUES:
            for m in CLUSTER_SIZES:
                for v in SLOPE_VARIANCES:
                    seed = BASE_SEED + 100000 * cell_index
                    cell_index += 1
                    label = f"glmm_slope_g{g}_m{m}_v{v:g}"
                    for df_method, wald_se in ARMS:
                        r, elapsed = run_cell(g, m, v, df_method, wald_se, seed)
                        for i, target in enumerate(TARGETS):
                            writer.writerow({
                                "label": label, "g": g, "cluster_size": m, "slope_var": v,
                                "n": g * m, "df_method": df_method, "wald_se": wald_se, "target": target,
                                "rejection_rate": r["power_uncorrected"][0][i],
                                "ci_lo": r["ci_uncorrected"][0][i][0],
                                "ci_hi": r["ci_uncorrected"][0][i][1],
                                "convergence_rate": r["convergence_rate"][0],
                                "n_sims": K, "seed": seed, "elapsed_sec": elapsed,
                            })
                        print(f"{label} {df_method} {wald_se}: " + " ".join(
                            f"{t}={r['power_uncorrected'][0][i]:.2f}" for i, t in enumerate(TARGETS))
                            + f" conv={r['convergence_rate'][0]:.4f} {elapsed:.0f}s", flush=True)
                    f.flush()


if __name__ == "__main__":
    main()
