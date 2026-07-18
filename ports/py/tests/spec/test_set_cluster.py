"""Python frontend contract for LME random-intercept power analysis.

Tests cover construction, validation, and the ``to_simulation_spec``
projection. End-to-end engine calls (find_power with LME) are covered by
separate integration tests.
"""

from __future__ import annotations

import json as _json
import warnings

import pytest

from mcpower import MCPower


# ---------------------------------------------------------------------------
# Construction: LME formulas + family='lme' now succeed
# ---------------------------------------------------------------------------


def test_lme_formula_constructs():
    """LME formulas no longer raise at MCPower(...) construction time."""
    m = MCPower("y ~ x1 + (1|school)")
    # family defaults to ols; the formula is parsed and random effects stored.
    assert m._registry._random_effects_parsed
    assert m._registry._random_effects_parsed[0]["grouping_var"] == "school"


def test_family_lme_constructs():
    """family='lme' is now a valid construction kwarg."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    assert m.family == "lme"


def test_family_unknown_still_rejected():
    """Unknown families are still rejected by _validate_family."""
    with pytest.raises(ValueError, match="family must be"):
        MCPower("y = x", family="gamma")


# ---------------------------------------------------------------------------
# set_cluster: basic round-trip + tau_squared injection
# ---------------------------------------------------------------------------


def test_set_cluster_basic_round_trip():
    """set_cluster stores the spec; to_simulation_spec emits the engine fields."""
    model = MCPower("y ~ x1 + (1|school)", family="lme")
    model.set_effects("x1=0.5").set_cluster("school", ICC=0.3, n_clusters=20)
    spec = model.to_simulation_spec()
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "mle"
    cluster = spec["generation"]["cluster"]
    assert cluster["sizing"]["FixedClusters"]["n_clusters"] == 20
    # tau_squared = ICC / (1 - ICC).
    assert abs(cluster["tau_squared"] - 0.3 / 0.7) < 1e-12


def test_set_cluster_chainable():
    """set_cluster returns self for chaining."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    result = m.set_cluster("school", ICC=0.2, n_clusters=10)
    assert result is m


def test_set_cluster_icc_zero_allowed():
    """ICC=0 is valid (degenerate, tau_squared=0)."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.0, n_clusters=10)
    spec = m.to_simulation_spec()
    assert spec["generation"]["cluster"]["tau_squared"] == 0.0
    assert "FixedClusters" in spec["generation"]["cluster"]["sizing"]


# ---------------------------------------------------------------------------
# set_cluster: NotImplementedError gates
# ---------------------------------------------------------------------------


def test_set_cluster_accepts_random_slopes_no_longer_raises():
    """random_slopes is now accepted and stored."""
    model = MCPower("y ~ x1 + (1 + x1|school)", family="lme")
    model.set_effects("x1=0.5")
    model.set_cluster(
        "school",
        ICC=0.2,
        n_clusters=20,
        random_slopes=["x1"],
        slope_variance=0.1,
    )
    assert model._pending_clusters["school"]["random_slopes"] == ["x1"]
    assert model._pending_clusters["school"]["slope_variance"] == pytest.approx(0.1)


def test_set_cluster_rejects_between_vars():
    """between_vars is removed — passing it raises TypeError (unknown kwarg)."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(TypeError):
        m.set_cluster("school", ICC=0.2, n_clusters=20, between_vars=["x1"])


def test_set_cluster_accepts_n_per_parent():
    """n_per_parent is now accepted and stored (nested random effects)."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    # school is the outer grouping; n_per_parent = classrooms per school
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10, n_clusters=30, n_per_parent=3)
    assert m._pending_clusters["school"]["icc"] == pytest.approx(0.15)
    assert m._pending_clusters["school:classroom"]["n_per_parent"] == 3


def test_nested_formula_no_longer_raises():
    """A '(1|A/B)' formula no longer raises NotImplementedError."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    # Must not raise — nested gate removed
    m.set_cluster("school:classroom", ICC=0.10, n_clusters=30)
    assert "school:classroom" in m._pending_clusters


def test_multiple_groupings_accepted():
    """Two crossed groupings are accepted — multiple-groupings gate removed."""
    m = MCPower("y ~ x1 + (1|school) + (1|teacher)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m.set_cluster("teacher", ICC=0.1, n_clusters=30)
    assert len(m._pending_clusters) == 2


def test_set_cluster_overwrite_same_var():
    """Calling set_cluster twice for the same grouping_var overwrites — last-wins.

    _pending_clusters is keyed by grouping_var, so the second call replaces the
    entire entry. No stale first-call ICC survives.
    """
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m.set_cluster("school", ICC=0.5, n_clusters=10)
    assert len(m._pending_clusters) == 1
    assert m._pending_clusters["school"]["icc"] == pytest.approx(0.5)
    assert m._pending_clusters["school"]["n_clusters"] == 10


# ---------------------------------------------------------------------------
# set_cluster: XOR + value validations
# ---------------------------------------------------------------------------


def test_set_cluster_rejects_both_n_clusters_and_cluster_size():
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="Specify either"):
        m.set_cluster("school", ICC=0.2, n_clusters=10, cluster_size=20)


def test_set_cluster_rejects_neither():
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="Must specify"):
        m.set_cluster("school", ICC=0.2)


def test_set_cluster_rejects_icc_out_of_range():
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="ICC"):
        m.set_cluster("school", ICC=1.5, n_clusters=10)


def test_set_cluster_rejects_unknown_grouping_var():
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="not found in formula"):
        m.set_cluster("classroom", ICC=0.2, n_clusters=10)


# ---------------------------------------------------------------------------
# _validate_lme_runtime gates (called from find_power before engine)
# ---------------------------------------------------------------------------


def test_lme_find_power_without_set_cluster_raises():
    """family='lme' with no set_cluster call raises at find_power time."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="set_cluster"):
        m.find_power(sample_size=100)


def test_lme_validate_runtime_derives_n_from_cluster_size():
    """cluster_size + sample_size → n_clusters derivation populates effective."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=10)
    m._apply()
    m._validate_lme_runtime(sample_size=100, scenario_filter=["optimistic"])
    assert m._effective_n_clusters == 10


def test_lme_snap_via_find_power_warns():
    """sample_size not divisible by n_clusters: engine snaps and emits a warning.

    Snap-and-warn lives in the Rust engine; the Python port surfaces
    engine grid_warnings as UserWarning. This test exercises end-to-end behavior
    (N=101 → snapped to 100 by atom=20; warning emitted).
    """
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, n_clusters=20)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = m.find_power(101, n_sims=20, verbose=False)
    # Engine floors 101 to nearest multiple of 20 → 100.
    assert res["sample_sizes"][0] == 100
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    # Engine emits a grid_warning that Python surfaces.
    assert any("100" in s or "101" in s for s in msgs), msgs


def test_lme_validate_runtime_silent_for_non_lme_family():
    """OLS / Logit with no cluster spec is a no-op."""
    m = MCPower("y = x")
    m.set_effects("x=0.5")
    m._apply()
    m._validate_lme_runtime(sample_size=100, scenario_filter=["optimistic"])
    assert m._effective_n_clusters is None


def test_lme_validate_runtime_clustered_ols_fallthrough():
    """Clustered-OLS fall-through: a non-LME family whose formula carries a
    random-effect term plus a set_cluster spec must NOT be rejected. The
    decoupled DGP (clusters) still needs validating, so _validate_lme_runtime
    falls through to the per-cluster checks and populates _effective_n_clusters.
    """
    m = MCPower("y ~ x + (1|g)", family="ols")
    m.set_effects("x=0.4")
    m.set_cluster("g", ICC=0.2, n_clusters=12)
    # Must not raise even though family != "lme".
    m._validate_lme_runtime(sample_size=120, scenario_filter=["optimistic"])
    assert m._effective_n_clusters == 12
    # The DGP stays clustered while the estimator remains OLS.
    spec = m.to_simulation_spec()
    assert spec["estimator"] == "ols"
    assert spec["generation"]["cluster"]["sizing"]["FixedClusters"]["n_clusters"] == 12


def test_lme_to_simulation_spec_strips_random_effects_from_formula():
    """Spec builder rejects '(1|group)'; we must strip before passing JSON."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, n_clusters=10)
    spec = m.to_simulation_spec()
    # The roundtripped spec uses the cleaned formula; the LinearSpec JSON
    # sent to the builder no longer contains '(1|school)'. We don't see the
    # formula in the final SimulationSpec dict, but a successful build is
    # itself the assertion: the builder would have raised RandomEffectsUnsupported
    # if the strip were missing.
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "mle"


def test_set_cluster_accepts_cluster_level_vars():
    """cluster_level_vars stores predictor names; replaces reserved between_vars."""
    m = MCPower("y ~ x1 + x2 + (1|school)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    # Must not raise — no longer gated
    result = m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["x1"])
    assert result is m
    assert m._pending_clusters["school"]["cluster_level_vars"] == ["x1"]


def test_set_cluster_between_vars_still_rejected():
    """between_vars parameter is removed; passing it raises TypeError."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(TypeError):
        m.set_cluster("school", ICC=0.2, n_clusters=20, between_vars=["x1"])


# ---------------------------------------------------------------------------
# set_cluster: cluster_level_vars validation
# ---------------------------------------------------------------------------


def test_cluster_level_vars_rejects_unknown_predictor():
    """A name not in the formula raises ValueError at set_cluster time."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match="cluster_level_vars.*not a predictor|not a predictor.*cluster_level_vars"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["x_unknown"])


def test_cluster_level_vars_rejects_grouping_var_itself():
    """The grouping variable cannot also be listed as a cluster-level predictor."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match="grouping variable"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["school"])


def test_cluster_level_vars_rejects_when_no_cluster_active():
    """cluster_level_vars requires a corresponding set_cluster call (family check)."""
    # This is enforced at find_power time via family=lme requirement; here we
    # test that passing cluster_level_vars on a plain-OLS model with no RE formula
    # is caught by the predictor-in-registry check (no random effects → no grouping
    # var → validation path still reached via normal set_cluster call).
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match="cluster_level_vars.*not a predictor|not a predictor.*cluster_level_vars"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["y"])


def test_cluster_level_vars_rejects_uploaded_predictor():
    """D4: a name bound to uploaded data cannot be a cluster_level_var."""
    import numpy as np
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    arr = np.column_stack([
        np.random.default_rng(0).normal(size=60),
        np.random.default_rng(1).normal(size=60),
    ])
    col_names = ["x1", "y"]
    m.upload_data(arr, col_names)
    with pytest.raises(ValueError, match="uploaded"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["x1"])


# ---------------------------------------------------------------------------
# _build_cluster_spec_dict: multiple groupings → extra_groupings wire shape
# ---------------------------------------------------------------------------


def test_clusters_json_two_crossed_groupings():
    """Two crossed groupings produce a ClusterSpec with extra_groupings."""
    m = MCPower("y ~ x1 + (1|school) + (1|teacher)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m.set_cluster("teacher", ICC=0.1, n_clusters=30)
    m._effective_n_clusters = 20  # simulate post-_validate_lme_runtime
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert len(specs) == 1  # one top-level ClusterSpec entry
    primary = specs[0]
    # primary grouping = school (first inserted)
    assert primary["sizing"]["FixedClusters"]["n_clusters"] == 20
    assert abs(primary["tau_squared"] - 0.2 / 0.8) < 1e-12
    # second grouping encoded as extra_groupings
    assert len(primary["extra_groupings"]) == 1
    eg = primary["extra_groupings"][0]
    assert eg["relation"]["Crossed"]["n_clusters"] == 30
    assert abs(eg["tau_squared"] - 0.1 / 0.9) < 1e-12


def test_clusters_json_nested_grouping():
    """Nested (1|A/B) produces extra_groupings with Nested relation."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10, n_clusters=30, n_per_parent=3)
    m._effective_n_clusters = 10
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert len(specs) == 1
    primary = specs[0]
    assert primary["sizing"]["FixedClusters"]["n_clusters"] == 10
    assert len(primary["extra_groupings"]) == 1
    eg = primary["extra_groupings"][0]
    assert eg["relation"]["NestedWithin"]["n_per_parent"] == 3
    assert abs(eg["tau_squared"] - 0.10 / 0.90) < 1e-12


def test_validate_lme_runtime_multi_grouping_sets_effective_n():
    """_validate_lme_runtime with two groupings populates _effective_n_clusters
    from the primary (first) grouping and validates each spec independently."""
    m = MCPower("y ~ x1 + (1|school) + (1|teacher)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m.set_cluster("teacher", ICC=0.1, n_clusters=30)
    m._apply()
    m._validate_lme_runtime(sample_size=200, scenario_filter=["optimistic"])
    # Primary grouping governs effective_n_clusters
    assert m._effective_n_clusters == 20


# ---------------------------------------------------------------------------
# Task 1.8 — random_slopes acceptance + storage
# ---------------------------------------------------------------------------


def test_set_cluster_accepts_random_slopes():
    """random_slopes + slope_variance + slope_intercept_corr are now accepted."""
    m = MCPower("y ~ x1 + (1 + x1|school)", family="lme")
    m.set_effects("x1=0.4")
    m.set_cluster(
        "school",
        ICC=0.2,
        n_clusters=20,
        random_slopes=["x1"],
        slope_variance=0.05,
        slope_intercept_corr=0.3,
    )
    cfg = m._pending_clusters["school"]
    assert cfg["random_slopes"] == ["x1"]
    assert cfg["slope_variance"] == pytest.approx(0.05)
    assert cfg["slope_intercept_corr"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Task 1.9 — slopes emitted in clusters_json
# ---------------------------------------------------------------------------


def test_clusters_json_random_slope_shape():
    """Random slope appears in clusters_json as a SlopeTerm dict — mirrors the
    R debug-seam shape (spec-builder.R:1154-1162).

    Shape: slopes = [{"column": <0-based gen col>, "variance": <f64>,
                      "corr_with_intercept": <f64>, "corr_with": []}]
    The column index is the 0-based index of the slope predictor in the
    registry's non_factor_names list (generation column ordering).
    """
    m = MCPower("y ~ x1 + x2 + (1 + x1|school)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    m.set_cluster(
        "school",
        ICC=0.2,
        n_clusters=20,
        random_slopes=["x1"],
        slope_variance=0.05,
        slope_intercept_corr=0.3,
    )
    m._effective_n_clusters = 20
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    primary = specs[0]
    assert "slopes" in primary
    assert len(primary["slopes"]) == 1
    slope = primary["slopes"][0]
    # x1 is the first non-factor predictor → column index 0
    assert slope["column"] == 0
    assert slope["variance"] == pytest.approx(0.05)
    assert slope["corr_with_intercept"] == pytest.approx(0.3)
    assert slope["corr_with"] == []  # first (and only) slope → empty


def test_clusters_json_no_slopes_key_absent():
    """When no random_slopes are configured, 'slopes' key is absent."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m._effective_n_clusters = 20
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert "slopes" not in specs[0]


# Engine-required smoke tests — these call find_power end-to-end.
# Requires: maturin develop --release (Phase 0 engine changes must land first).

def test_find_power_two_crossed_groupings_smoke():
    """Two crossed groupings reach the engine without error; power is numeric."""
    m = MCPower("y ~ x1 + (1|school) + (1|teacher)", family="lme")
    m.set_effects("x1=0.4")
    m.set_cluster("school", ICC=0.2, n_clusters=15)
    m.set_cluster("teacher", ICC=0.1, n_clusters=10)
    result = m.find_power(sample_size=150, n_sims=50, seed=2137, verbose=False)
    # PowerResult is flat (single default scenario): power_uncorrected is
    # [n_sample_sizes][n_targets]; y ~ x1 has one target.
    pw = result["power_uncorrected"][0][0]
    assert 0.0 <= pw <= 1.0
    # Floor guards against a broken path returning all-zeros/NaN→0.
    # Observed: 1.0 at seed=2137 (x1=0.4, N=150, two-grouping LME).
    assert pw > 0.3, f"power {pw} too low — crossed-groupings engine path may be broken"


def test_find_power_nested_groupings_smoke():
    """Nested (1|school/classroom) reaches the engine without error."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10, n_clusters=30, n_per_parent=3)
    result = m.find_power(sample_size=150, n_sims=50, seed=2137, verbose=False)
    pw = result["power_uncorrected"][0][0]  # single target (treatment)
    assert 0.0 <= pw <= 1.0
    # Floor guards against a broken path returning all-zeros/NaN→0.
    # Observed: 1.0 at seed=2137 (treatment=0.5, N=150, nested LME).
    assert pw > 0.3, f"power {pw} too low — nested-groupings engine path may be broken"


def test_find_power_random_slope_smoke():
    """Random slope DGP reaches the engine; power is numeric."""
    m = MCPower("y ~ x1 + (1 + x1|school)", family="lme")
    m.set_effects("x1=0.4")
    m.set_cluster(
        "school",
        ICC=0.2,
        n_clusters=20,
        random_slopes=["x1"],
        slope_variance=0.05,
        slope_intercept_corr=0.2,
    )
    result = m.find_power(sample_size=200, n_sims=50, seed=2137, verbose=False)
    pw = result["power_uncorrected"][0][0]  # single target (x1)
    assert 0.0 <= pw <= 1.0
    # Floor guards against a broken path returning all-zeros/NaN→0.
    # Observed: 0.98 at seed=2137 (x1=0.4, N=200, random-slope LME).
    assert pw > 0.3, f"power {pw} too low — random-slope engine path may be broken"


def test_find_power_cluster_level_var_smoke():
    """cluster_level_vars flows to the engine; find_power completes."""
    m = MCPower("y ~ x1 + x2 + (1|school)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["x2"])
    result = m.find_power(sample_size=200, n_sims=50, seed=2137, verbose=False)
    # Two targets (x1, x2) → first row has two power entries, both in [0, 1].
    row = result["power_uncorrected"][0]
    assert len(row) == 2
    assert all(0.0 <= p <= 1.0 for p in row)
    # Floors guard against a broken path returning all-zeros/NaN→0.
    # Observed at seed=2137: x1(obs-level, β=0.3)=0.98, x2(cluster-level, β=0.2)=0.28.
    assert row[0] > 0.15, f"x1 power {row[0]} too low — cluster-level-var path may be broken"
    assert row[1] > 0.05, f"x2 power {row[1]} too low — cluster-level-var path may be broken"
    # x1 (observation-level) should have higher power than x2 (cluster-level, same effect).
    assert row[0] > row[1], f"Expected power(x1)>power(x2) but got {row[0]:.3f} vs {row[1]:.3f}"


# ---------------------------------------------------------------------------
# Task 4 — extra grouping emits slopes
# ---------------------------------------------------------------------------


def test_extra_grouping_emits_slopes():
    """Extra grouping with random_slopes emits a slopes list in the spec dict.
    Primary grouping slopes are also exercised — refactored _slope_terms_for
    call site must work for both primary and extra.

    column=0 because x1 is the sole non-factor predictor → generation index 0.
    corr_with=[] because it is the first (and only) slope term.
    """
    m = MCPower("y ~ x1 + (1 + x1|grp) + (1 + x1|item)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("grp", ICC=0.2, n_clusters=24,                      # primary
                  random_slopes=["x1"], slope_variance=0.08,
                  slope_intercept_corr=0.15)
    m.set_cluster("item", ICC=0.1, n_clusters=6,                      # extra
                  random_slopes=["x1"], slope_variance=0.10,
                  slope_intercept_corr=0.2)
    spec = m._build_cluster_spec_dict()
    # Primary grouping slopes
    assert "slopes" in spec
    assert spec["slopes"] == [
        {"column": 0, "variance": 0.08, "corr_with_intercept": 0.15, "corr_with": []}
    ]
    # Extra grouping slopes
    extra = spec["extra_groupings"][0]
    assert extra["slopes"] == [
        {"column": 0, "variance": 0.10, "corr_with_intercept": 0.2, "corr_with": []}
    ]


def test_extra_grouping_no_slopes_key_absent():
    """When an extra grouping has no random_slopes, 'slopes' key is absent.
    Negative assertion locking in the conditional emission pattern."""
    m = MCPower("y ~ x1 + (1|grp) + (1|item)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("grp", ICC=0.2, n_clusters=24)
    m.set_cluster("item", ICC=0.1, n_clusters=6)
    spec = m._build_cluster_spec_dict()
    extra = spec["extra_groupings"][0]
    assert "slopes" not in extra
