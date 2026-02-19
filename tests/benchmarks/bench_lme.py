"""Benchmark all 3 LME solver paths: q=1, q>1 (general), nested.

Measures single-call latency and full-pipeline throughput.
Stores results to JSON for before/after comparison.

Usage:
    python MCPower/tests/benchmarks/bench_lme.py [--output results.json]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def generate_q1_data(K=20, n=1000, p=3, seed=42):
    """Generate data for q=1 random intercept benchmark."""
    rng = np.random.RandomState(seed)
    obs_per_cluster = n // K
    cluster_ids = np.repeat(np.arange(K), obs_per_cluster).astype(np.int32)
    n_actual = len(cluster_ids)
    X = rng.randn(n_actual, p)
    beta_true = rng.randn(p + 1) * 0.5  # with intercept
    X_int = np.column_stack([np.ones(n_actual), X])
    tau = 0.5
    cluster_effects = rng.randn(K) * tau
    y = X_int @ beta_true + cluster_effects[cluster_ids] + rng.randn(n_actual)
    return X, y, cluster_ids, K


def generate_general_data(K=30, n=1500, p=2, q=2, seed=42):
    """Generate data for q>1 random slopes benchmark."""
    rng = np.random.RandomState(seed)
    obs_per_cluster = n // K
    cluster_ids = np.repeat(np.arange(K), obs_per_cluster).astype(np.int32)
    n_actual = len(cluster_ids)
    X = rng.randn(n_actual, p)
    beta_true = np.array([1.0] + [0.5] * p)  # intercept + fixed
    X_int = np.column_stack([np.ones(n_actual), X])
    # Z = [1, x1] for random intercept + slope on first predictor
    Z = np.column_stack([np.ones(n_actual), X[:, 0]])
    # Random effects covariance
    G = np.array([[0.25, 0.05], [0.05, 0.1]])
    L_G = np.linalg.cholesky(G)
    b = rng.randn(K, q) @ L_G.T
    y = X_int @ beta_true + np.sum(Z * b[cluster_ids], axis=1) + rng.randn(n_actual)
    return X, y, Z, cluster_ids, K, q


def generate_nested_data(K_parent=10, K_child=30, n=1500, p=2, seed=42):
    """Generate data for nested random intercepts benchmark."""
    rng = np.random.RandomState(seed)
    children_per_parent = K_child // K_parent
    obs_per_child = n // K_child

    parent_ids = []
    child_ids = []
    child_to_parent = np.zeros(K_child, dtype=np.int32)

    for pj in range(K_parent):
        for cj_local in range(children_per_parent):
            cj = pj * children_per_parent + cj_local
            child_to_parent[cj] = pj
            parent_ids.extend([pj] * obs_per_child)
            child_ids.extend([cj] * obs_per_child)

    parent_ids = np.array(parent_ids, dtype=np.int32)
    child_ids = np.array(child_ids, dtype=np.int32)
    n_actual = len(parent_ids)

    X = rng.randn(n_actual, p)
    beta_true = np.array([1.0] + [0.5] * p)
    X_int = np.column_stack([np.ones(n_actual), X])

    tau_parent = 0.4
    tau_child = 0.3
    b_parent = rng.randn(K_parent) * tau_parent
    b_child = rng.randn(K_child) * tau_child
    y = X_int @ beta_true + b_parent[parent_ids] + b_child[child_ids] + rng.randn(n_actual)

    return X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent


def bench_single_call_q1(n_reps=100):
    """Benchmark q=1 single analysis call."""
    from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_full

    X, y, cluster_ids, K = generate_q1_data()
    p = X.shape[1]
    n_targets = p
    target_indices = np.arange(n_targets, dtype=np.int32)
    chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

    # Warmup
    for _ in range(5):
        lme_analysis_full(X, y, cluster_ids, K, target_indices, chi2_crit, z_crit, correction_z_crits, 0)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        lme_analysis_full(X, y, cluster_ids, K, target_indices, chi2_crit, z_crit, correction_z_crits, 0)
        times.append(time.perf_counter() - t0)

    return {
        "name": "q1_single_call",
        "params": {"K": 20, "n": 1000, "p": 3},
        "n_reps": n_reps,
        "mean_ms": np.mean(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def bench_single_call_general(n_reps=100):
    """Benchmark q>1 single analysis call."""
    from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_general

    X, y, Z, cluster_ids, K, q = generate_general_data()
    p = X.shape[1]
    n_targets = p
    target_indices = np.arange(n_targets, dtype=np.int32)
    chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

    # Warmup
    for _ in range(3):
        lme_analysis_general(X, y, cluster_ids, K, q, Z, target_indices, chi2_crit, z_crit, correction_z_crits, 0)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        lme_analysis_general(X, y, cluster_ids, K, q, Z, target_indices, chi2_crit, z_crit, correction_z_crits, 0)
        times.append(time.perf_counter() - t0)

    return {
        "name": "general_single_call",
        "params": {"K": 30, "n": 1500, "p": 2, "q": 2},
        "n_reps": n_reps,
        "mean_ms": np.mean(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def bench_single_call_nested(n_reps=100):
    """Benchmark nested single analysis call."""
    from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_nested

    X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent = generate_nested_data()
    p = X.shape[1]
    n_targets = p
    target_indices = np.arange(n_targets, dtype=np.int32)
    chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

    # Warmup
    for _ in range(3):
        lme_analysis_nested(
            X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent,
            target_indices, chi2_crit, z_crit, correction_z_crits, 0,
        )

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        lme_analysis_nested(
            X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent,
            target_indices, chi2_crit, z_crit, correction_z_crits, 0,
        )
        times.append(time.perf_counter() - t0)

    return {
        "name": "nested_single_call",
        "params": {"K_parent": 10, "K_child": 30, "n": 1500, "p": 2},
        "n_reps": n_reps,
        "mean_ms": np.mean(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def bench_pipeline_q1(n_sims=50):
    """Benchmark q=1 full pipeline (MCPower)."""
    from mcpower import MCPower

    t0 = time.perf_counter()
    model = MCPower("y ~ x1 + x2 + (1|g)")
    model.set_cluster("g", ICC=0.2, n_clusters=20)
    model.set_effects("x1=0.5, x2=0.3")
    model.set_max_failed_simulations(0.15)
    model.set_simulations(n_sims)
    model.find_power(sample_size=1200, print_results=False, progress_callback=False)
    elapsed = time.perf_counter() - t0

    return {
        "name": "q1_pipeline",
        "params": {"n_sims": n_sims, "sample_size": 1200, "K": 20},
        "total_seconds": elapsed,
        "per_sim_ms": elapsed / n_sims * 1000,
    }


def bench_pipeline_general(n_sims=50):
    """Benchmark q>1 full pipeline (MCPower)."""
    from mcpower import MCPower

    t0 = time.perf_counter()
    model = MCPower("y ~ x1 + (1 + x1|g)")
    model.set_cluster("g", ICC=0.2, n_clusters=30, random_slopes=["x1"], slope_variance=0.1, slope_intercept_corr=0.3)
    model.set_effects("x1=0.5")
    model.set_max_failed_simulations(0.15)
    model.set_simulations(n_sims)
    model.find_power(sample_size=1500, print_results=False, progress_callback=False)
    elapsed = time.perf_counter() - t0

    return {
        "name": "general_pipeline",
        "params": {"n_sims": n_sims, "sample_size": 1500, "K": 30, "q": 2},
        "total_seconds": elapsed,
        "per_sim_ms": elapsed / n_sims * 1000,
    }


def bench_pipeline_nested(n_sims=50):
    """Benchmark nested full pipeline (MCPower)."""
    from mcpower import MCPower

    t0 = time.perf_counter()
    model = MCPower("y ~ x1 + (1|school/classroom)")
    model.set_cluster("school", ICC=0.15, n_clusters=10)
    model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
    model.set_effects("x1=0.5")
    model.set_max_failed_simulations(0.15)
    model.set_simulations(n_sims)
    model.find_power(sample_size=1500, print_results=False, progress_callback=False)
    elapsed = time.perf_counter() - t0

    return {
        "name": "nested_pipeline",
        "params": {"n_sims": n_sims, "sample_size": 1500, "K_parent": 10, "K_child": 30},
        "total_seconds": elapsed,
        "per_sim_ms": elapsed / n_sims * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LME solvers")
    parser.add_argument("--output", default=None, help="JSON output file path")
    parser.add_argument("--reps", type=int, default=100, help="Single-call reps")
    parser.add_argument("--sims", type=int, default=50, help="Pipeline simulation count")
    args = parser.parse_args()

    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "benchmarks": []}

    print("=" * 60)
    print("LME Solver Benchmarks")
    print("=" * 60)

    # Single-call benchmarks
    print("\n--- Single-call benchmarks ---")

    r = bench_single_call_q1(n_reps=args.reps)
    results["benchmarks"].append(r)
    print(f"q=1:     {r['mean_ms']:.3f} ms/call (median {r['median_ms']:.3f}, std {r['std_ms']:.3f})")

    r = bench_single_call_general(n_reps=args.reps)
    results["benchmarks"].append(r)
    print(f"q>1:     {r['mean_ms']:.3f} ms/call (median {r['median_ms']:.3f}, std {r['std_ms']:.3f})")

    r = bench_single_call_nested(n_reps=args.reps)
    results["benchmarks"].append(r)
    print(f"nested:  {r['mean_ms']:.3f} ms/call (median {r['median_ms']:.3f}, std {r['std_ms']:.3f})")

    # Pipeline benchmarks
    print("\n--- Pipeline benchmarks ---")

    r = bench_pipeline_q1(n_sims=args.sims)
    results["benchmarks"].append(r)
    print(f"q=1 pipeline:     {r['total_seconds']:.2f}s total ({r['per_sim_ms']:.1f} ms/sim)")

    r = bench_pipeline_general(n_sims=args.sims)
    results["benchmarks"].append(r)
    print(f"q>1 pipeline:     {r['total_seconds']:.2f}s total ({r['per_sim_ms']:.1f} ms/sim)")

    r = bench_pipeline_nested(n_sims=args.sims)
    results["benchmarks"].append(r)
    print(f"nested pipeline:  {r['total_seconds']:.2f}s total ({r['per_sim_ms']:.1f} ms/sim)")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(Path(__file__).parent / "bench_lme_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
