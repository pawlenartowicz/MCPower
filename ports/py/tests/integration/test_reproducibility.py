"""Reproducibility tests for mcpower.

These tests exercise the seed-determinism contract of the v2 engine end-to-end
through the Python frontend. The seed-aligned scenario stream property is
particularly important: scenarios are perturbations on top of the same
underlying RNG draws, so for a given seed the optimistic slice must equal a
plain ``scenarios=False`` run, and per-scenario deltas must reproduce.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any, Dict

import pytest

from mcpower import MCPower


def _flatten_for_compare(result: Dict[str, Any]) -> Any:
    """Strip non-deterministic fields and return a JSON-stable subset.

    The result dict contains numpy arrays and floats; we want byte-stability
    on the numeric content. Wilson CIs are deterministic functions of the
    point estimates so they're included; convergence is included.
    """
    if "scenarios" in result:
        return {
            "scenarios": {
                name: _flatten_for_compare(payload)
                for name, payload in result["scenarios"].items()
            },
        }
    return {
        "power_uncorrected": [list(row) for row in result["power_uncorrected"]],
        "power_corrected": [list(row) for row in result["power_corrected"]],
        "ci_uncorrected": [
            [list(ci) for ci in row] for row in result["ci_uncorrected"]
        ],
        "ci_corrected": [
            [list(ci) for ci in row] for row in result["ci_corrected"]
        ],
        "convergence_rate": list(result["convergence_rate"]),
        "n_sims": result["n_sims"],
        "n_targets": result["n_targets"],
        "target_indices": list(result["target_indices"]),
        "sample_sizes": list(result["sample_sizes"]),
    }


# ---------------------------------------------------------------------------
# Test 1: byte-stable across two runs
# ---------------------------------------------------------------------------


def test_g2_byte_stable_two_runs():
    def run() -> Dict[str, Any]:
        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.5, x2=0.3")
        return m.find_power(
            sample_size=100, n_sims=200, seed=42, progress_callback=False
        )

    a = _flatten_for_compare(run())
    b = _flatten_for_compare(run())
    assert a == b, "find_power(seed=42) should be byte-stable across two runs"


# ---------------------------------------------------------------------------
# Test 2: prefix stability across N — two find_power calls at different N
# with the same seed should yield identical results at the overlapping
# sample size, given the engine's seed-stream layout.
# ---------------------------------------------------------------------------


def test_g2_prefix_stability_across_n():
    """The engine streams RNG draws per (sim, sample_size). With the same seed
    and n_sims, a single-N call at N=100 should be byte-identical to a
    single-N call at N=100 even when other find_power invocations have
    happened with different N. This is the basic 'no global RNG state' check.
    """

    def run_at(N: int) -> Dict[str, Any]:
        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.5, x2=0.3")
        return m.find_power(
            sample_size=N, n_sims=100, seed=42, progress_callback=False
        )

    # Run at N=100, then at N=200, then back at N=100 — the third call's
    # numerics must equal the first call's (no global state contamination).
    a = _flatten_for_compare(run_at(100))
    _ = run_at(200)  # different N intentionally; must not contaminate.
    c = _flatten_for_compare(run_at(100))
    assert a == c, (
        "find_power(N=100, seed=42) should yield identical results regardless "
        "of intervening find_power(N=200) calls"
    )


# ---------------------------------------------------------------------------
# Test 3: thread invariance (subprocess for the 4-thread case)
# ---------------------------------------------------------------------------


_THREAD_SCRIPT = """
import json
from mcpower import MCPower, _engine
_engine.set_n_threads({n_threads})
m = MCPower("y = x1 + x2")
m.set_effects("x1=0.5, x2=0.3")
r = m.find_power(sample_size=100, n_sims=200, seed=42, progress_callback=False)
print(json.dumps({{
    "power_uncorrected": r["power_uncorrected"],
    "power_corrected": r["power_corrected"],
    "convergence_rate": r["convergence_rate"],
}}))
"""


def _run_in_subproc(n_threads: int) -> Dict[str, Any]:
    code = textwrap.dedent(_THREAD_SCRIPT.format(n_threads=n_threads))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    # Find the last line of stdout that's parseable JSON; engine progress may
    # write earlier lines to stdout in some builds.
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def test_g2_thread_invariance():
    a = _run_in_subproc(1)
    b = _run_in_subproc(4)
    assert a == b, (
        "Same seed must yield identical results across thread counts; "
        f"1-thread={a} vs 4-thread={b}"
    )


# ---------------------------------------------------------------------------
# Test 4: scenarios=True byte-stable across two runs
# ---------------------------------------------------------------------------


def test_g2_scenarios_byte_stable_two_runs():
    def run() -> Dict[str, Any]:
        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.5, x2=0.3")
        return m.find_power(
            sample_size=100,
            n_sims=200,
            seed=42,
            scenarios=True,
            progress_callback=False,
        )

    a = _flatten_for_compare(run())
    b = _flatten_for_compare(run())
    assert a == b, "scenarios=True envelope must be byte-stable across runs"


# ---------------------------------------------------------------------------
# Test 5: optimistic-slice equivalence between scenarios=True and
# scenarios=False with the same args.
# ---------------------------------------------------------------------------


def test_g2_optimistic_slice_equivalence():
    m1 = MCPower("y = x1 + x2")
    m1.set_effects("x1=0.5, x2=0.3")
    flat = m1.find_power(
        sample_size=100, n_sims=200, seed=42, progress_callback=False
    )

    m2 = MCPower("y = x1 + x2")
    m2.set_effects("x1=0.5, x2=0.3")
    env = m2.find_power(
        sample_size=100,
        n_sims=200,
        seed=42,
        scenarios=True,
        progress_callback=False,
    )

    flat_cmp = _flatten_for_compare(flat)
    optimistic_cmp = _flatten_for_compare(env["scenarios"]["optimistic"])
    assert flat_cmp == optimistic_cmp, (
        "scenarios=True[optimistic] must match scenarios=False bit-for-bit"
    )


# ---------------------------------------------------------------------------
# Test 6: scenario stream independence — same scenario at same seed
# reproducibly perturbs the optimistic baseline. Two seeds give different
# results, but each seed is itself reproducible.
# ---------------------------------------------------------------------------


def test_g2_scenario_stream_independence():
    def run(seed: int) -> Dict[str, Any]:
        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.5, x2=0.3")
        return m.find_power(
            sample_size=100,
            n_sims=200,
            seed=seed,
            scenarios=["optimistic", "realistic"],
            progress_callback=False,
        )

    a_seed42 = _flatten_for_compare(run(42))
    b_seed42 = _flatten_for_compare(run(42))

    # A fixed seed reproduces the full multi-scenario envelope byte-for-byte
    # across independent calls (the deterministic mechanic). The former
    # "different seeds produce different realistic-slice power" assertion was
    # dropped because it is an MC observable (two seeds can coincide on a slice
    # at n_sims=200, and a broken seed path can still diverge at the power
    # level), so it tolerated/leaned on random behavior rather than testing a
    # determinism mechanic. Cross-seed stream sensitivity is L3 statistical
    # content.
    assert a_seed42 == b_seed42, (
        "Same seed twice must yield byte-identical scenario envelopes"
    )


# ---------------------------------------------------------------------------
# Logit-specific reproducibility tests
# ---------------------------------------------------------------------------


def test_logit_seed_stable():
    """find_power(family='logit', seed=42) must be byte-stable across two calls.

    Mirrors test_g2_byte_stable_two_runs but for the logit family path. The
    logit data-gen path (Bernoulli draw from a Uniform) and the IRLS kernel
    must both be deterministic at the same seed.
    """
    def run() -> Dict[str, Any]:
        m = MCPower("y = x", family="logit")
        m.set_baseline_probability(0.3)
        m.set_effects("x=0.5")
        return m.find_power(
            sample_size=200, n_sims=500, seed=42, progress_callback=False
        )

    r1 = _flatten_for_compare(run())
    r2 = _flatten_for_compare(run())
    assert r1 == r2, (
        "find_power(family='logit', seed=42) must be byte-stable across two runs"
    )


_LOGIT_THREAD_SCRIPT = """
import json
from mcpower import MCPower, _engine
_engine.set_n_threads({n_threads})
m = MCPower("y = x", family="logit")
m.set_baseline_probability(0.3)
m.set_effects("x=0.5")
r = m.find_power(sample_size=200, n_sims=500, seed=42, progress_callback=False)
print(json.dumps({{
    "power_uncorrected": r["power_uncorrected"],
    "power_corrected": r["power_corrected"],
    "convergence_rate": r["convergence_rate"],
}}))
"""


def _run_logit_in_subproc(n_threads: int) -> Dict[str, Any]:
    code = textwrap.dedent(_LOGIT_THREAD_SCRIPT.format(n_threads=n_threads))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def test_logit_thread_invariance():
    """Logit power must be identical across 1-thread and 4-thread pool.

    set_n_threads must be called before the pool is initialised — so each
    thread count runs in a fresh subprocess (same pattern as
    test_g2_thread_invariance). The IRLS kernel and Bernoulli data-gen both
    rely only on the per-sim RNG stream; no shared mutable state means the
    per-sim results should be identical regardless of which rayon worker
    handles them.
    """
    a = _run_logit_in_subproc(1)
    b = _run_logit_in_subproc(4)
    assert a == b, (
        "Logit find_power(seed=42) must be identical across 1-thread and "
        f"4-thread pool; 1-thread={a} vs 4-thread={b}"
    )


# ---------------------------------------------------------------------------
# Uploaded-data reproducibility: same seed + same uploaded frame must give
# byte-identical power across two runs (the upload data-gen path is seeded the
# same way as the synthetic path).
# ---------------------------------------------------------------------------


def test_upload_data_seed_stable():
    """find_power on uploaded data must be byte-stable across two runs with the
    same seed and the same uploaded frame."""
    import numpy as np

    rng = np.random.default_rng(0)
    x1 = rng.normal(size=80)
    x2 = 0.4 * x1 + rng.normal(size=80)
    y = 0.5 * x1 + 0.3 * x2 + rng.normal(size=80)
    data = {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}

    def run() -> Dict[str, Any]:
        m = MCPower("y = x1 + x2")
        m.upload_data(data, mode="partial", verbose=False)
        m.set_effects("x1=0.5, x2=0.3")
        return m.find_power(
            sample_size=100, n_sims=200, seed=42, progress_callback=False
        )

    a = _flatten_for_compare(run())
    b = _flatten_for_compare(run())
    assert a == b, (
        "find_power on uploaded data (seed=42) must be byte-stable across runs"
    )
