"""Engine-owned snap-and-warn for cluster alignment.

The snap-and-warn is inside the Rust engine (find_power) rather than in
Python. find_power with N not divisible by n_clusters returns the floored N
and emits a grid_warnings entry that the Python port surfaces as a UserWarning.
These tests verify the observable behaviour through the public API.
"""

from __future__ import annotations

import warnings

import pytest

from mcpower import MCPower


# ---------------------------------------------------------------------------
# find_power: engine snaps N and emits a warning
# ---------------------------------------------------------------------------


def test_find_power_snaps_sample_size_for_lme():
    """N=95, n_clusters=10 → engine floors to 90, emits UserWarning."""
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, n_clusters=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.find_power(95, n_sims=20, verbose=False)
    # Engine floors to the nearest multiple of n_clusters (atom=10).
    assert res["sample_sizes"][0] == 90
    # The engine emits a grid_warning that Python surfaces as a UserWarning.
    assert any("90" in str(w.message) or "95" in str(w.message) for w in caught if issubclass(w.category, UserWarning))


def test_find_power_no_warn_when_divisible():
    """Exactly divisible N=100, n_clusters=10 → no snap warning."""
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, n_clusters=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.find_power(100, n_sims=20, verbose=False)
    # N=100 is a multiple of 10 — no snap, no warning.
    assert res["sample_sizes"][0] == 100
    snap_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    # Should have no snap-related warnings (no grid_warnings from engine).
    snap_warns_msg = [w for w in snap_warns if "not evenly divisible" in str(w.message).lower() or "lowered" in str(w.message).lower() or "snapped" in str(w.message).lower()]
    assert len(snap_warns_msg) == 0


def test_find_power_snap_floors_not_rounds():
    """N=99, n_clusters=10 → floors to 90, not rounds to 100."""
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, n_clusters=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.find_power(99, n_sims=20, verbose=False)
    # 99 // 10 * 10 = 90 (floor), not 100 (round up).
    assert res["sample_sizes"][0] == 90


def test_snap_does_not_apply_to_ols():
    """OLS models are not snapped; N=95 passes through unchanged."""
    model = MCPower("y ~ x1")
    model.set_effects("x1=0.5")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.find_power(95, n_sims=20, verbose=False)
    snap_warns = [w for w in caught if issubclass(w.category, UserWarning)
                  and ("not evenly divisible" in str(w.message).lower()
                       or "lowered" in str(w.message).lower())]
    assert len(snap_warns) == 0
    assert res["sample_sizes"][0] == 95


# ---------------------------------------------------------------------------
# find_sample_size: FixedSize + cluster_size (previously NotImplementedError)
# ---------------------------------------------------------------------------


def test_find_sample_size_cluster_size_runs():
    """find_sample_size with cluster_size (FixedSize regime) no longer raises NotImplementedError."""
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, cluster_size=10)
    # atom=10, hard_min=5*10=50; from_size=60 is valid.
    res = model.find_sample_size(from_size=60, to_size=200, n_sims=20, verbose=False)
    assert res is not None
    # Grid N's must all be multiples of 10 (the cluster atom).
    sample_sizes = res["sample_sizes"]
    assert len(sample_sizes) > 0
    assert all(n % 10 == 0 for n in sample_sizes), f"non-multiple in grid: {sample_sizes}"


# ---------------------------------------------------------------------------
# L2 binding tests: fitted / fitted_joint / cluster_atom in result payload
# ---------------------------------------------------------------------------

_ALLOWED_STATUSES = {"fitted", "at_or_below_min", "not_reached", "non_monotone"}


def test_find_sample_size_result_carries_fitted_keys():
    """find_sample_size result dict carries fitted, fitted_joint, cluster_atom.

    L2 contract: the engine always emits these three keys, populated according
    to the grid search outcome. For an unclustered OLS run:
      - fitted has len == n_targets with statuses in the allowed set.
      - fitted_joint has len == len(first_joint_achieved).
      - cluster_atom == 1 (no clustering).
    """
    model = MCPower("y ~ x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")
    res = model.find_sample_size(from_size=40, to_size=300, by=20,
                                 n_sims=400, verbose=False)
    fitted = res.get("fitted")
    fitted_joint = res.get("fitted_joint")
    cluster_atom = res.get("cluster_atom")

    assert fitted is not None, "fitted key missing from result"
    assert fitted_joint is not None, "fitted_joint key missing from result"
    assert cluster_atom is not None, "cluster_atom key missing from result"

    n_targets = len(res["target_indices"])
    assert len(fitted) == n_targets, (
        f"fitted has {len(fitted)} entries but n_targets={n_targets}"
    )
    for pos, f in fitted.items():
        assert f.get("status") in _ALLOWED_STATUSES, (
            f"fitted[{pos}] status {f.get('status')!r} not in {_ALLOWED_STATUSES}"
        )

    assert len(fitted_joint) == len(res.get("first_joint_achieved", {})), (
        "fitted_joint length does not match first_joint_achieved"
    )
    assert cluster_atom == 1, f"expected cluster_atom=1 for OLS, got {cluster_atom}"


def test_clustered_find_sample_size_cluster_atom_and_n_achievable_alignment():
    """Clustered result: cluster_atom == cluster size; fitted n_achievable % atom == 0.

    Guards the invariant that the engine atom-ceils headline N values, so every
    "Required N" the display layer shows is a valid grid point for that design.
    """
    cluster_size = 10
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, cluster_size=cluster_size)
    res = model.find_sample_size(from_size=60, to_size=300, by=20, n_sims=50, verbose=False)

    assert res.get("cluster_atom") == cluster_size, (
        f"cluster_atom={res.get('cluster_atom')}, expected {cluster_size}"
    )
    fitted = res.get("fitted") or {}
    for pos, f in fitted.items():
        if f.get("status") == "fitted":
            n_achievable = f["n_achievable"]
            assert n_achievable % cluster_size == 0, (
                f"fitted[{pos}].n_achievable={n_achievable} not divisible by "
                f"cluster_atom={cluster_size}"
            )
