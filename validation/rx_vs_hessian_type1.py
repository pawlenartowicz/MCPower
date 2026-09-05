"""Type-I error under wald_se="rx" vs "hessian" on the clustered binary GLMM
path, at the true null (both effects = 0), across cluster counts
g in {5, 10, 20, 30, 50}. Measures whether the standard-error mode moves the
rejection rate; the companion study of the reference distribution (Wald z vs
between-within t) is df_calibration.R.

Run (from the repo's venv, at the repo root):
    python validation/rx_vs_hessian_type1.py

Writes validation/data/rx_vs_hessian_type1.csv (one row per cell x arm x
target; elapsed_sec is per find_power call, so it repeats across the x and z
rows of a cell) and prints the same table as markdown to stdout.
"""
from __future__ import annotations

import csv
import pathlib
import time

import mcpower

SEED = 2137
ALPHA = 0.05
CLUSTER_SIZE = 20
G_VALUES = [5, 10, 20, 30, 50]
ARMS = ["rx", "hessian"]
N_SIMS = 4000
ICC = 0.2
BASELINE_P = 0.5

OUT_CSV = pathlib.Path(__file__).parent / "data" / "rx_vs_hessian_type1.csv"


def run_cell(g: int, arm: str):
    n = g * CLUSTER_SIZE
    m = mcpower.MCPower("y = x + z + (1|g)", family="logit")
    m.set_baseline_probability(BASELINE_P)
    m.set_effects("x=0, z=0")
    m.set_cluster("g", ICC=ICC, n_clusters=g, cluster_level_vars=["z"])
    t0 = time.time()
    r = m.find_power(
        sample_size=n,
        target_test="x, z",
        n_sims=N_SIMS,
        seed=SEED,
        wald_se=arm,
        verbose=False,
    )
    elapsed = time.time() - t0
    return r, elapsed, n


def main():
    rows = []
    target_labels = ["x", "z"]  # within-cluster, between-cluster (cluster_level_vars=["z"])
    for g in G_VALUES:
        for arm in ARMS:
            r, elapsed, n = run_cell(g, arm)
            power = r["power_uncorrected"][0]
            ci = r["ci_uncorrected"][0]
            conv = r["convergence_rate"][0]
            for i, label in enumerate(target_labels):
                rows.append({
                    "g": g,
                    "n": n,
                    "cluster_size": CLUSTER_SIZE,
                    "icc": ICC,
                    "baseline_prob": BASELINE_P,
                    "alpha": ALPHA,
                    "n_sims": N_SIMS,
                    "seed": SEED,
                    "arm": arm,
                    "target": label,
                    "rejection_rate": power[i],
                    "ci_lo": ci[i][0],
                    "ci_hi": ci[i][1],
                    "convergence_rate": conv,
                    "elapsed_sec": elapsed,
                })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print markdown table.
    header = ("g", "n", "arm", "target", "rejection_rate", "ci_lo", "ci_hi",
              "convergence_rate", "elapsed_sec")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in header) + " |")


if __name__ == "__main__":
    main()
