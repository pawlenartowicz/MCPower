"""Python frontend contract for LME random-intercept power analysis.

Tests cover construction, validation, and the ``to_simulation_spec``
projection. End-to-end engine calls (find_power with LME) are covered by
separate integration tests.
"""

from __future__ import annotations

import json as _json

import pytest

from mcpower import MCPower


# ---------------------------------------------------------------------------
# Construction: LME formulas + family='lme' now succeed
# ---------------------------------------------------------------------------


def test_family_lme_constructs():
    """family='lme' is a valid construction kwarg, as are the non-default GLM
    families probit and poisson."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    assert m.family == "lme"
    assert MCPower("y = x", family="probit").family == "probit"
    assert MCPower("y = x", family="poisson").family == "poisson"


def test_family_unknown_still_rejected():
    """Unknown families are still rejected by _validate_family."""
    with pytest.raises(ValueError, match="family must be"):
        MCPower("y = x", family="gamma")


# ---------------------------------------------------------------------------
# set_cluster: basic round-trip + tau_squared injection
# ---------------------------------------------------------------------------


def test_set_cluster_basic_round_trip():  # APIC-52
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
# set_cluster: removed / formerly gated kwargs
# ---------------------------------------------------------------------------


def test_set_cluster_rejects_between_vars():
    """between_vars is removed — passing it raises TypeError (unknown kwarg)."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5")
    with pytest.raises(TypeError):
        m.set_cluster("school", ICC=0.2, n_clusters=20, between_vars=["x1"])


def test_nested_formula_no_longer_raises():
    """A '(1|A/B)' formula no longer raises NotImplementedError."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    # Must not raise — nested gate removed
    m.set_cluster("school:classroom", ICC=0.10, n_per_parent=3)
    assert "school:classroom" in m._pending_clusters


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


@pytest.mark.parametrize(
    "family, kwargs, message",
    [
        ("lme", {"ICC": float("nan")}, "ICC must be a finite number, got nan"),
        ("lme", {"ICC": 0.2, "random_slopes": ["x1"], "slope_variance": float("nan")},
         "slope_variance must be a finite number, got nan"),
        ("lme", {"ICC": 0.2, "random_slopes": ["x1"], "slope_variance": float("inf")},
         "slope_variance must be a finite number, got inf"),
        ("lme", {"ICC": 0.2, "random_slopes": ["x1"], "slope_intercept_corr": float("nan")},
         "slope_intercept_corr must be a finite number, got nan"),
        ("poisson", {"tau_squared": float("inf")},
         "tau_squared must be a finite non-negative number, got inf"),
    ],
)
def test_set_cluster_names_a_non_finite_number(family, kwargs, message):
    # A NaN or infinity used to reach the engine as invalid clusters JSON.
    m = MCPower("y ~ x1 + (1 + x1|school)", family=family)
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match=message):
        m.set_cluster("school", n_clusters=10, **kwargs)


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


def test_lme_validate_runtime_keeps_cluster_size_sizing():
    """A cluster_size spec stays FixedSize after a find_power-style check: the
    cluster count grows with N, it is not frozen at sample_size // cluster_size
    for later runs."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=10)
    m._apply()
    m._validate_lme_runtime(sample_size=100, scenario_filter=["optimistic"])
    sizing = m.to_simulation_spec()["generation"]["cluster"]["sizing"]
    assert sizing == {"FixedSize": {"cluster_size": 10}}


def test_find_power_rejects_derived_cluster_count_below_min_clusters():
    """cluster_size=10 passes the rows-per-cluster check, so only the derived
    cluster count (40 // 10 = 4, below the min_clusters limit of 5) can raise."""
    m = MCPower("y ~ x + (1|g)", family="lme")
    m.set_effects("x=0.3")
    m.set_cluster("g", ICC=0.2, cluster_size=10)
    with pytest.raises(ValueError, match=r"derived n_clusters=4 is below 5"):
        m.find_power(sample_size=40, n_sims=10, seed=1, verbose=False)


def test_lme_validate_runtime_silent_for_non_lme_family():
    """OLS / Logit with no cluster spec is a no-op."""
    m = MCPower("y = x")
    m.set_effects("x=0.5")
    m._apply()
    m._validate_lme_runtime(sample_size=100, scenario_filter=["optimistic"])


def test_lme_validate_runtime_clustered_ols_fallthrough():
    """Clustered-OLS fall-through: a non-LME family whose formula carries a
    random-effect term plus a set_cluster spec must NOT be rejected. The
    decoupled DGP (clusters) still needs validating, so _validate_lme_runtime
    falls through to the per-cluster checks.
    """
    m = MCPower("y ~ x + (1|g)", family="ols")
    m.set_effects("x=0.4")
    m.set_cluster("g", ICC=0.2, n_clusters=12)
    # Must not raise even though family != "lme".
    m._validate_lme_runtime(sample_size=120, scenario_filter=["optimistic"])
    # The DGP stays clustered while the estimator remains OLS.
    spec = m.to_simulation_spec()
    assert spec["estimator"] == "ols"
    assert spec["generation"]["cluster"]["sizing"]["FixedClusters"]["n_clusters"] == 12


# ---------------------------------------------------------------------------
# set_cluster: cluster_level_vars validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["x_unknown", "y"])
def test_cluster_level_vars_rejects_unknown_predictor(name):
    """A name that is not a predictor (unknown, or the dependent variable)
    raises ValueError at set_cluster time."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match=f"cluster_level_vars: '{name}' is not a predictor"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=[name])


def test_cluster_level_vars_rejects_grouping_var_itself():
    """The grouping variable cannot also be listed as a cluster-level predictor."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match="grouping variable"):
        m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["school"])


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
    m.set_cluster("school:classroom", ICC=0.10, n_per_parent=3)
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert len(specs) == 1
    primary = specs[0]
    assert primary["sizing"]["FixedClusters"]["n_clusters"] == 10
    assert len(primary["extra_groupings"]) == 1
    eg = primary["extra_groupings"][0]
    assert eg["relation"]["NestedWithin"]["n_per_parent"] == 3
    assert abs(eg["tau_squared"] - 0.10 / 0.90) < 1e-12


def test_set_cluster_accepts_n_per_parent_alone():
    """A nested child needs only n_per_parent — no n_clusters/cluster_size of
    its own, since its size is fully determined by n_per_parent against the
    parent's cluster count (matches the documented example)."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10, n_per_parent=4)  # no size arg
    assert m._pending_clusters["school:classroom"]["n_per_parent"] == 4
    assert m._pending_clusters["school:classroom"]["n_clusters"] is None
    assert m._pending_clusters["school:classroom"]["cluster_size"] is None

    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    primary = specs[0]
    eg = primary["extra_groupings"][0]
    assert eg["relation"]["NestedWithin"]["n_per_parent"] == 4


def test_nested_child_without_n_per_parent_uses_config_default():
    """A nested child with no n_per_parent is NestedWithin at the config
    default, not Crossed."""
    from mcpower.config import get_simulation_defaults

    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10)
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    eg = _json.loads(clusters_json)[0]["extra_groupings"][0]
    default = get_simulation_defaults()["nested_n_per_parent"]
    assert eg["relation"] == {"NestedWithin": {"n_per_parent": default}}


def test_nested_child_rejects_n_clusters():
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    with pytest.raises(ValueError, match="classroom is nested in school; give n_per_parent"):
        m.set_cluster("school:classroom", ICC=0.10, n_clusters=30)


def test_nested_child_rejects_cluster_size():
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    with pytest.raises(ValueError, match="give n_per_parent .*, not cluster_size"):
        m.set_cluster("school:classroom", ICC=0.10, cluster_size=30)


def test_set_cluster_n_per_parent_alone_passes_runtime_validation():
    """_validate_lme_runtime accepts a nested child with only n_per_parent set."""
    m = MCPower("y ~ treatment + (1|school/classroom)", family="lme")
    m.set_effects("treatment=0.5")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster("school:classroom", ICC=0.10, n_per_parent=6)
    m._apply()
    # Must not raise "Must specify either n_clusters or cluster_size".
    m._validate_lme_runtime(sample_size=60, scenario_filter=["optimistic"])


def test_validate_lme_runtime_multi_grouping_passes():
    """_validate_lme_runtime with two groupings validates each spec
    independently."""
    m = MCPower("y ~ x1 + (1|school) + (1|teacher)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m.set_cluster("teacher", ICC=0.1, n_clusters=30)
    m._apply()
    m._validate_lme_runtime(sample_size=200, scenario_filter=["optimistic"])


# ---------------------------------------------------------------------------
# Slopes emitted in clusters_json
# ---------------------------------------------------------------------------


def test_clusters_json_random_slope_shape():
    """Random slope appears in clusters_json as a SlopeTerm dict — mirrors the
    R debug-seam shape.

    Shape: slopes = [{"column": <0-based gen col>, "variance": <f64>,
                      "corr_with_intercept": <f64>, "corr_with": []}]
    The column index is the 0-based index of the slope predictor in the
    registry's non_factor_names list (generation column ordering): the slope
    sits on x2, the second non-factor predictor, so column 1.
    """
    m = MCPower("y ~ x1 + x2 + (1 + x2|school)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    m.set_cluster(
        "school",
        ICC=0.2,
        n_clusters=20,
        random_slopes=["x2"],
        slope_variance=0.05,
        slope_intercept_corr=0.3,
    )
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert specs[0]["slopes"] == [
        {"column": 1, "variance": 0.05, "corr_with_intercept": 0.3, "corr_with": []}
    ]


def test_clusters_json_no_slopes_key_absent():
    """When no random_slopes are configured, 'slopes' key is absent."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert "slopes" not in specs[0]


def test_formula_slope_without_set_cluster_slope_is_refused():
    """A slope written only in the formula is refused before anything runs,
    naming the variable and the set_cluster fix, rather than being fitted as a
    random intercept."""
    m = MCPower("y = x1 + x2 + (1 + x2|g)", family="lme")
    m.set_effects("x1=0.3, x2=0.3")
    m.set_cluster("g", ICC=0.2, n_clusters=20)
    with pytest.raises(ValueError, match=r"random slope on 'x2'.*add 'x2' to random_slopes"):
        m.find_power(200, n_sims=20, progress_callback=False, verbose=False)


def test_set_cluster_slope_without_formula_slope_is_refused():
    """The reverse: set_cluster gives x2 a random slope that the formula's
    '(1|g)' does not declare, so the stated model and the fitted one differ."""
    m = MCPower("y = x1 + x2 + (1|g)", family="lme")
    m.set_effects("x1=0.3, x2=0.3")
    m.set_cluster("g", ICC=0.2, n_clusters=20, random_slopes=["x2"], slope_variance=0.1)
    with pytest.raises(ValueError, match=r"write '\(1 \+ x2\|g\)' in the formula"):
        m.find_power(200, n_sims=20, progress_callback=False, verbose=False)


def test_nested_child_slope_from_set_cluster_alone_runs():
    """The formula cannot write a slope on a nested child ('(1 + x|school/classroom)'
    does not parse), so set_cluster is the only route: a nested-child slope with
    a plain '(1|school/classroom)' formula runs and reaches the engine."""
    m = MCPower("y ~ x + (1|school/classroom)", family="lme")
    m.set_effects("x=0.4")
    m.set_cluster("school", ICC=0.15, n_clusters=10)
    m.set_cluster(
        "school:classroom", ICC=0.10, n_per_parent=3, random_slopes=["x"], slope_variance=0.1
    )
    _, _, _, _, clusters_json = m._encode_outcome_and_clusters()
    assert len(_json.loads(clusters_json)[0]["extra_groupings"][0]["slopes"]) == 1
    res = m.find_power(300, n_sims=20, seed=2137, progress_callback=False, verbose=False)
    assert 0.0 <= res["power_uncorrected"][0][0] <= 1.0


def test_double_bar_random_effect_is_refused():
    """lme4's uncorrelated '||' syntax gets its own message, not a syntax error."""
    with pytest.raises(ValueError, match=r"uncorrelated random effects \('\|\|'\)"):
        MCPower("y = x1 + (1 + x1 || g)", family="lme")


# Engine-required smoke tests — these call find_power end-to-end.
# Requires: maturin develop --release (Phase 0 engine changes must land first).

def test_find_power_two_crossed_groupings_smoke():  # APIC-65
    """Two crossed groupings reach the engine without error; power is numeric.
    The single-scenario result is unwrapped to the flat inner dict, so the
    top-level ``power_uncorrected`` index would fail on the envelope."""
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
    m.set_cluster("school:classroom", ICC=0.10, n_per_parent=3)
    result = m.find_power(sample_size=150, n_sims=50, seed=2137, verbose=False)
    pw = result["power_uncorrected"][0][0]  # single target (treatment)
    assert 0.0 <= pw <= 1.0
    # Floor guards against a broken path returning all-zeros/NaN→0.
    # Observed: 1.0 at seed=2137 (treatment=0.5, N=150, nested LME).
    assert pw > 0.3, f"power {pw} too low — nested-groupings engine path may be broken"


def test_nested_cluster_size_design_keeps_cluster_size_across_runs():
    """Schools of 30 with 3 classes each: N=600 is 20 whole schools, so it is
    not snapped, and a later find_sample_size on the same model still steps by
    one school (atom 30), not by a school count frozen by find_power."""
    import warnings

    m = MCPower("y ~ treatment + (1|school/class)", family="lme")
    m.set_variable_type("treatment=binary")
    m.set_effects("treatment=0.4")
    m.set_cluster("school", ICC=0.15, cluster_level_vars=["treatment"], cluster_size=30)
    m.set_cluster("school:class", ICC=0.10, n_per_parent=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.find_power(sample_size=600, n_sims=50, seed=2137, verbose=False)
    assert not [w for w in caught if "cluster atom" in str(w.message)]
    result = m.find_sample_size(
        from_size=600, to_size=660, n_sims=50, seed=2137, verbose=False
    )
    assert result["cluster_atom"] == 30


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


@pytest.mark.parametrize(
    "family, formula, setup, message",
    [
        ("logit", "y = x + (1 + x|g)", lambda m: m.set_baseline_probability(0.3),
         r"^the random-effect term on 'g' has no set_cluster call"),
        ("logit", "y = x + (1|g)", lambda m: m.set_baseline_probability(0.3),
         r"^the random-effect term on 'g' has no set_cluster call"),
        ("poisson", "y = x + (1|g)", lambda m: m.set_baseline_rate(1.0),
         r"^the random-effect term on 'g' has no set_cluster call"),
        ("lme", "y = x + (1|g)", lambda m: None,
         r"^family='lme' requires set_cluster\(\.\.\.\) for each random effect"),
    ],
)
def test_random_effect_without_set_cluster_is_refused(family, formula, setup, message):
    """A random-effect term with no set_cluster call stops the run in every
    family; a non-lme family used to run unclustered, dropping the random
    intercept and slope without notice."""
    m = MCPower(formula, family=family)
    m.set_effects("x=0.5")
    setup(m)
    with pytest.raises(ValueError, match=message):
        m.find_power(200, n_sims=20, verbose=False)


def test_random_effect_without_set_cluster_names_the_nested_grouping():
    """The refusal names the grouping as the formula declares it, not a
    rewritten '(1|g)' term."""
    m = MCPower("y = x + (1|school/class)", family="lme")
    m.set_effects("x=0.5")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    with pytest.raises(
        ValueError,
        match=r"^the random-effect term on 'school:class' has no set_cluster call",
    ):
        m.find_power(1200, n_sims=20, verbose=False)


def test_non_psd_slope_correlations_are_a_user_error():
    """One intercept correlation over two uncorrelated slopes is not a valid
    correlation matrix; the contract rule reaches Python as ValueError with
    its own message, not as an internal RuntimeError."""
    m = MCPower("y = x1 + x2 + (1 + x1 + x2|g)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    m.set_cluster(
        "g", ICC=0.2, n_clusters=20, random_slopes=["x1", "x2"],
        slope_variance=0.1, slope_intercept_corr=0.9,
    )
    with pytest.raises(ValueError) as exc:
        m.find_power(400, n_sims=20, verbose=False)
    assert str(exc.value).startswith(
        "the random-slope correlations are not a valid correlation matrix"
    )
    assert "internal" not in str(exc.value)


def test_clustered_ols_runs_with_scenarios():
    """OLS fitted to clustered data runs the scenario analysis with its
    random-effect settings: the port hands the engine realistic and doomer
    scenarios with a nonzero ICC jitter (data generation draws the random
    effects for it as for a mixed model), the engine accepts them, and power
    falls from optimistic to realistic to doomer."""
    m = MCPower("y ~ x + (1|g)")
    m.set_effects("x=0.3")
    m.set_cluster("g", ICC=0.2, n_clusters=10)
    payload = m._to_linear_spec_dict(["realistic", "doomer"])
    assert all(sc["icc_noise_sd"] > 0 for sc in payload["scenarios"])
    res = m.find_power(
        100, n_sims=200, seed=2137, scenarios=True,
        progress_callback=False, verbose=False,
    )
    power = {
        name: res["scenarios"][name]["power_uncorrected"][0][0]
        for name in ("optimistic", "realistic", "doomer")
    }
    assert power["optimistic"] > power["realistic"] > power["doomer"], power


def test_get_effects_from_data_with_one_group_is_a_user_error():
    """The upload path pins n_clusters to the data's distinct groups, so one
    group reaches the engine's cluster-count rule as a ValueError with its
    own message."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    m = MCPower("y ~ x + (1|group)", family="lme")
    m.set_cluster("group", ICC=0.2, n_clusters=4)
    m.upload_data(
        pd.DataFrame({"y": 0.4 * x + rng.normal(size=60), "x": x, "group": 0}),
        mode="partial", verbose=False,
    )
    with pytest.raises(ValueError) as exc:
        m.get_effects_from_data("y", verbose=False)
    assert str(exc.value).startswith("n_clusters must be an integer >= 2, got 1")


def test_random_slopes_need_a_positive_variance():
    m = MCPower("y ~ x + (1 + x|g)", family="lme")
    for kwargs in ({}, {"slope_variance": 0.0}):
        with pytest.raises(ValueError, match="random_slopes need a slope_variance above 0"):
            m.set_cluster("g", ICC=0.2, n_clusters=20, random_slopes=["x"], **kwargs)


def test_repeated_random_slope_is_refused():
    m = MCPower("y ~ x + (1 + x + x|g)", family="lme")
    m.set_effects("x=0.3")
    m.set_cluster("g", ICC=0.2, n_clusters=20, random_slopes=["x", "x"], slope_variance=0.1)
    with pytest.raises(ValueError, match="list each slope variable once"):
        m.find_power(400, n_sims=20, seed=1)


def test_crossed_grouping_slopes_need_a_valid_covariance():
    def model(icc, corr):
        m = MCPower("y ~ x1 + x2 + (1|subject) + (1 + x1 + x2|item)", family="lme")
        m.set_effects("x1=0.3, x2=0.3")
        m.set_cluster("subject", ICC=0.2, n_clusters=20)
        m.set_cluster("item", ICC=icc, n_clusters=12, random_slopes=["x1", "x2"],
                      slope_variance=0.1, slope_intercept_corr=corr)
        return m

    with pytest.raises(ValueError, match="not a valid correlation matrix"):
        model(0.2, 0.9).find_power(480, n_sims=20, seed=1)
    with pytest.raises(ValueError, match="random-intercept variance above 0"):
        model(0.0, 0.3).find_power(480, n_sims=20, seed=1)


def test_ols_test_formula_may_drop_a_slope_variable():
    m = MCPower("y ~ x1 + x2 + (1 + x2|g)", family="ols")
    m.set_effects("x1=0.3, x2=0.3")
    m.set_cluster("g", ICC=0.2, n_clusters=20, random_slopes=["x2"], slope_variance=0.1)
    res = m.find_power(240, n_sims=20, seed=1, test_formula="y ~ x1", target_test="x1")
    assert len(res["power_uncorrected"][0]) == 1


def test_estimator_glm_on_a_continuous_outcome_is_a_user_error():
    m = MCPower("y ~ x1", family="ols", estimator="glm")
    m.set_effects("x1=0.3")
    with pytest.raises(ValueError, match="needs a binary or count outcome"):
        m.find_power(100, n_sims=20, seed=1)
