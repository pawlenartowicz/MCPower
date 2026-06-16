"""APIC gap tests — Python chained API surface.

Property / invariant / shape / error-path / determinism tests for APIC
catalogue rows that the pre-existing suite did not cover. Python is the
canonical reference for the chained surface, so these become the mirror target
the R port is later held to.

None of these assert a numeric power value, oracle match, or frozen snapshot
(D1/D2). They assert validator error-kinds, parser/spec shapes, transform
identities (ratios/logit), and structural envelopes that a broken kernel would
fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcpower import MCPower, _engine
from mcpower.spec.parsers import (
    _assignment_to_legacy,
    _normalise_correlation_input,
    _parse_equation,
    _parse_independent_variables,
)
from mcpower.output.results import unwrap_scenario_result
from mcpower.spec.validators import (
    _validate_alpha,
    _validate_correction_method,
    _validate_correlation_matrix,
    _validate_estimator,
    _validate_family,
    _validate_sample_size,
    _validate_sample_size_for_model,
    _validate_sample_size_range,
    _validate_test_formula,
    _validate_upload_data,
    _warn_logit_effect_scale,
)


# ---------------------------------------------------------------------------
# Validators (APIC-01..06, 14..17)
# ---------------------------------------------------------------------------


def test_validate_alpha_soft_warn_above_quarter():  # APIC-01
    # The 0.25 ceiling is a soft warning now, not a hard reject; the (0,1)
    # range is enforced by the engine, so the validator never rejects on range.
    r_quarter = _validate_alpha(0.25)
    assert r_quarter.is_valid and r_quarter.warnings == []   # at/below threshold: clean
    r_high = _validate_alpha(0.30)
    assert r_high.is_valid and r_high.warnings                # above 0.25: valid + warns
    assert _validate_alpha(0.0).is_valid                      # range reject moved to the engine


def test_validate_family_non_string():  # APIC-02 type reject
    assert not _validate_family(5).is_valid
    assert not _validate_family(None).is_valid
    assert _validate_family("OLS").is_valid        # case-insensitive accept


def test_validate_estimator_accepts_none_and_canonical_rejects_others():  # APIC-03
    assert _validate_estimator(None).is_valid
    for ok in ("ols", "GLM", "Mle"):
        assert _validate_estimator(ok).is_valid
    for bad in ("poisson", "", "linear"):
        assert not _validate_estimator(bad).is_valid
    assert not _validate_estimator(5).is_valid     # non-string non-None


def test_validate_sample_size_bounds_and_type():  # APIC-04
    assert not _validate_sample_size(19).is_valid
    assert _validate_sample_size(20).is_valid
    assert not _validate_sample_size(100_001).is_valid
    assert not _validate_sample_size(50.0).is_valid   # non-integer
    assert not _validate_sample_size("50").is_valid


def test_validate_sample_size_for_model_greens_rule():  # APIC-05
    # min required = 15 + n_variables.
    assert not _validate_sample_size_for_model(19, 5).is_valid   # < 20
    assert _validate_sample_size_for_model(20, 5).is_valid       # == 15 + 5
    assert not _validate_sample_size_for_model(20, 6).is_valid   # < 15 + 6


def test_validate_correction_method_whitelist():  # APIC-06
    assert _validate_correction_method(None).is_valid
    for ok in ("Bonferroni", "holm", "BH", "fdr", "Benjamini-Hochberg", "tukey"):
        assert _validate_correction_method(ok).is_valid
    assert not _validate_correction_method("sidak").is_valid


def test_validate_correlation_matrix_rejects_malformed():  # APIC-14
    # The validator now guards only the wire-unrepresentable structural
    # properties (square, symmetric, unit diagonal). Range (|r| <= 1) and PSD
    # are enforced downstream by the engine, so they are NOT checked here.
    assert _validate_correlation_matrix(np.eye(3)).is_valid
    assert not _validate_correlation_matrix(np.ones((2, 3))).is_valid      # non-square
    assert not _validate_correlation_matrix(
        np.array([[1.0, 0.5], [0.2, 1.0]])
    ).is_valid                                                              # non-symmetric
    assert not _validate_correlation_matrix(
        np.array([[2.0, 0.0], [0.0, 1.0]])
    ).is_valid                                                              # non-unit diagonal
    assert not _validate_correlation_matrix(None).is_valid
    # In-range, symmetric, unit-diagonal but indefinite (non-PSD): passes the
    # structural validator — rejection is delegated to the engine (see APIC-83).
    assert _validate_correlation_matrix(
        np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    ).is_valid


def test_validate_sample_size_range_invariants():  # APIC-15
    assert _validate_sample_size_range(30, 50, 5).is_valid
    assert not _validate_sample_size_range(50, 30, 5).is_valid   # from >= to
    assert not _validate_sample_size_range(30, 50, 40).is_valid  # by > range
    assert not _validate_sample_size_range(-1, 50, 5).is_valid   # non-positive
    assert not _validate_sample_size_range(30, 50, 5.0).is_valid  # non-integer


def test_validate_upload_data_rejects_bad_arrays():  # APIC-16
    assert not _validate_upload_data(np.ones(30)).is_valid           # 1-D
    assert not _validate_upload_data(np.ones((19, 2))).is_valid      # < min_rows (20)
    assert _validate_upload_data(np.ones((20, 2))).is_valid          # == min_rows (20) → ok
    assert not _validate_upload_data(np.ones((1_000_001, 1))).is_valid  # > max_rows (1M, native cap)
    nan_arr = np.ones((30, 2))
    nan_arr[0, 0] = np.nan
    assert not _validate_upload_data(nan_arr).is_valid               # NaN
    inf_arr = np.ones((30, 2))
    inf_arr[1, 1] = np.inf
    assert not _validate_upload_data(inf_arr).is_valid               # Inf
    assert _validate_upload_data(np.ones((30, 2))).is_valid          # clean


def test_validate_test_formula_error_paths():  # APIC-17
    assert not _validate_test_formula("", ["x1"]).is_valid           # empty
    assert not _validate_test_formula(123, ["x1"]).is_valid          # non-string
    assert not _validate_test_formula("x1 + z", ["x1"]).is_valid     # unknown ident
    assert _validate_test_formula("x1", ["x1", "x2"]).is_valid


# ---------------------------------------------------------------------------
# Parsers (APIC-18..26)
# ---------------------------------------------------------------------------


def test_parse_equation_triple_and_re_tagging():  # APIC-18
    dep, fixed, res = _parse_equation("y ~ x1 + (1|g)")
    assert dep == "y"
    assert "x1" in fixed and "(1|g)" not in fixed   # RE stripped from fixed RHS
    assert res == [{"type": "random_intercept", "grouping_var": "g"}]


def test_parse_equation_equals_synonym():  # APIC-20
    dep_eq, fixed_eq, _ = _parse_equation("y = x1 + x2")
    dep_tilde, fixed_tilde, _ = _parse_equation("y ~ x1 + x2")
    assert dep_eq == dep_tilde == "y"
    assert fixed_eq == fixed_tilde


def test_parse_independent_variables_shape():  # APIC-19
    variables, effects = _parse_independent_variables("x1 + x2 + x1:x2")
    assert list(variables.keys()) == ["variable_1", "variable_2"]
    assert all(k.startswith("effect_") for k in effects)
    interactions = [e for e in effects.values() if e.get("type") == "interaction"]
    assert interactions and "var_names" in interactions[0]
    assert "column_indices" in interactions[0]
    mains = [e for e in effects.values() if e.get("type") != "interaction"]
    assert all("column_index" in e for e in mains)


def _var_type_items(spec, predictors=("x", "y")):
    """Parse a variable-type assignment string through the engine; return
    ({name: info_dict}, errors). The design build/value-shape parsing is
    single-sourced in Rust (engine-spec-builder assignments)."""
    out = _engine.parse_assignments(
        spec, "variable_type",
        {"predictors": list(predictors), "interaction_terms": []},
    )
    info = {it["key"]["name"]: it["value"]["variable_type"] for it in out["items"]}
    return info, list(out["errors"])


def test_split_assignments_top_level_only():  # APIC-21
    # split_assignments is now the shared engine splitter.
    assert _engine.split_assignments("a=1, b=(2,3), c=4") == ["a=1", "b=(2,3)", "c=4"]
    with pytest.raises(ValueError):
        _engine.split_assignments("a=1)")   # unbalanced close paren


def test_normalise_correlation_input_rewrites_bare_form():  # APIC-22
    assert _normalise_correlation_input("(x1,x2)=0.3") == "corr(x1,x2)=0.3"
    # Already-prefixed entries pass through unchanged (modulo join spacing).
    assert _normalise_correlation_input("corr(x1,x2)=0.3") == "corr(x1,x2)=0.3"


def test_assignment_to_legacy_key_shapes():  # APIC-23
    name_item = {"key": {"name": "x1"}, "value": {"effect": 0.5}}
    key, val = _assignment_to_legacy(name_item)
    assert key == "x1" and val == 0.5
    pair_item = {"key": {"pair": ["x2", "x1"]}, "value": {"correlation": 0.4}}
    pkey, pval = _assignment_to_legacy(pair_item)
    assert pkey == ("x1", "x2") and pval == 0.4   # sorted tuple key


def test_parse_variable_type_value_bare_defaults():  # APIC-24
    info, err = _var_type_items("x=binary, y=factor")
    assert err == []
    assert info["x"] == {"type": "binary", "proportion": 0.5}
    assert info["y"]["type"] == "factor" and info["y"]["n_levels"] == 3
    assert info["y"]["proportions"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_parse_variable_type_value_factor_normalisation():  # APIC-25
    info, err = _var_type_items("x=(factor,1,1,2)", predictors=("x",))
    assert err == []
    assert info["x"]["proportions"] == pytest.approx([0.25, 0.25, 0.5])
    assert sum(info["x"]["proportions"]) == pytest.approx(1.0)
    _, neg_err = _var_type_items("x=(factor,0.5,-0.3)", predictors=("x",))
    assert neg_err   # non-positive proportion rejected (soft error)


def test_parse_variable_type_value_error_paths():  # APIC-26
    assert _var_type_items("x=weird", predictors=("x",))[1]            # unknown type
    assert _var_type_items("x=(binary,0.2,0.3)", predictors=("x",))[1]  # arity
    assert _var_type_items(
        "x=(factor," + ",".join(["1"] * 21) + ")", predictors=("x",)
    )[1]                                                               # > max levels
    # Single-element factor → n_levels < 2.
    assert _var_type_items("x=(factor,1)", predictors=("x",))[1]


# ---------------------------------------------------------------------------
# MCPower setters / construction (APIC-29..35, 44, 73..77)
# ---------------------------------------------------------------------------


def test_set_seed_validation():  # APIC-29
    m = MCPower("y = x1")
    m.set_seed(None)        # accepted (random seeding)
    m.set_seed(0)           # accepted
    with pytest.raises(ValueError):
        m.set_seed(-1)
    with pytest.raises(TypeError):
        m.set_seed(1.5)


def test_set_effects_dict_and_string_equivalent():  # APIC-30
    a = MCPower("y = x1 + x2")
    a.set_effects("x1=0.5, x2=0.3")
    b = MCPower("y = x1 + x2")
    b.set_effects({"x1": 0.5, "x2": 0.3})
    assert (
        a.to_simulation_spec()["outcome"]["coefficients"]
        == b.to_simulation_spec()["outcome"]["coefficients"]
    )
    with pytest.raises(Exception):
        MCPower("y = x1").set_effects("")   # empty string raises


def test_set_correlation_input_forms():  # APIC-31
    spec_string = (
        MCPower("y = x1 + x2")
        .set_effects("x1=0.5, x2=0.3")
    )
    spec_string.set_correlations("corr(x1,x2)=0.3")
    str_corr = spec_string._to_linear_spec_dict(["optimistic"])["correlations"]

    spec_arr = MCPower("y = x1 + x2")
    spec_arr.set_effects("x1=0.5, x2=0.3")
    spec_arr.set_correlations(np.array([[1.0, 0.3], [0.3, 1.0]]))
    arr_corr = spec_arr._to_linear_spec_dict(["optimistic"])["correlations"]

    # Both forms produce the same single non-zero pair.
    assert len(str_corr) == 1 and len(arr_corr) == 1
    assert str_corr[0]["value"] == pytest.approx(arr_corr[0]["value"])


def test_apply_collects_all_unknown_effect_errors():  # APIC-33
    m = MCPower("y = x1")
    m.set_effects("nope1=0.2, nope2=0.3")
    with pytest.raises(ValueError) as exc:
        m._apply()
    msg = str(exc.value)
    assert "nope1" in msg and "nope2" in msg   # not fail-fast: both reported


def test_apply_string_correlation_needs_two_vars():  # APIC-34
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    m.set_correlations("corr(x1,x2)=0.3")
    with pytest.raises(ValueError):
        m._apply()


def test_apply_ndarray_correlation_shape_mismatch():  # APIC-35
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m.set_correlations(np.eye(3))   # 3x3 but only 2 non-factor predictors
    with pytest.raises(ValueError):
        m._apply()


def test_apply_ndarray_correlation_asymmetric_raises():  # APIC-81
    # Symmetry is wire-unrepresentable (only the upper triangle crosses the
    # boundary), so the port must reject an asymmetric full matrix loudly.
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m.set_correlations(np.array([[1.0, 0.5], [0.2, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        m._apply()


def test_apply_ndarray_correlation_nonunit_diagonal_raises():  # APIC-82
    # Unit diagonal is likewise wire-unrepresentable.
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m.set_correlations(np.array([[2.0, 0.3], [0.3, 1.0]]))
    with pytest.raises(ValueError, match="[Dd]iagonal"):
        m._apply()


def test_find_power_rejects_non_psd_correlation_every_scenario():  # APIC-83
    # In-range, symmetric, unit-diagonal but indefinite: passes the port's
    # structural guards and is rejected by the ENGINE — consistently across
    # scenarios, because the up-front PSD check sits before scenario dispatch.
    # Parity with the pre-cleanup Python message ("positive semi-definite").
    for scenarios in (False, ["doomer"]):
        m = MCPower("y = x1 + x2 + x3")
        m.set_effects("x1=0.5, x2=0.3, x3=0.2")
        m.set_correlations(
            np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        )
        with pytest.raises(ValueError, match="positive semi-definite"):
            m.find_power(
                sample_size=100,
                n_sims=20,
                scenarios=scenarios,
                progress_callback=False,
            )


def test_set_scenario_configs_non_dict_raises():  # APIC-44
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    with pytest.raises(TypeError):
        m.set_scenario_configs(["not", "a", "dict"])


def test_set_residual_distribution_whitelist():  # APIC-73 (all canonical five accepted)
    m = MCPower("y = x1")
    # All five canonical names must be accepted.
    for name in ("normal", "right_skewed", "left_skewed", "high_kurtosis", "uniform"):
        m.set_residual_distribution(name)
    # v1 aliases and unknown names must be rejected.
    for bad in ("heavy_tailed", "skewed", "t", "garbage"):
        with pytest.raises(ValueError):
            m.set_residual_distribution(bad)


def test_set_heteroskedasticity_driver_constant_and_bad_var():  # APIC-74
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m.set_heteroskedasticity_driver(var=None)
    assert m._heteroskedasticity == {"driver_var_index": None}
    m.set_heteroskedasticity_driver(var="x1")
    assert m._heteroskedasticity == {"driver_var_index": 0}
    with pytest.raises(ValueError):
        m.set_heteroskedasticity_driver(var="not_a_var")


def test_set_heteroskedasticity_driver_honored():  # APIC-74b
    """driver_var_index flows through to the encoded spec (catches the v1 bug
    that ignored the driver and always used the LP)."""
    m1 = MCPower("y = x1 + x2")
    m1.set_effects("x1=0.5, x2=0.3")
    m1.set_heteroskedasticity_driver(var="x1")
    payload1 = m1._to_linear_spec_dict(["optimistic"])
    assert payload1["heteroskedasticity"]["driver_var_index"] == 0

    m2 = MCPower("y = x1 + x2")
    m2.set_effects("x1=0.5, x2=0.3")
    m2.set_heteroskedasticity_driver(var="x2")
    payload2 = m2._to_linear_spec_dict(["optimistic"])
    assert payload2["heteroskedasticity"]["driver_var_index"] == 1

    assert (
        payload1["heteroskedasticity"]["driver_var_index"]
        != payload2["heteroskedasticity"]["driver_var_index"]
    )


def test_set_heteroskedasticity_driver_no_ratio_key():  # APIC-74c
    """λ is scenario-only — the stored dict must never have a 'ratio' key."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    assert "ratio" not in m._heteroskedasticity
    m.set_heteroskedasticity_driver(var="x1")
    assert "ratio" not in m._heteroskedasticity


def test_to_simulation_spec_requires_effects():  # APIC-75
    m = MCPower("y = x1")
    with pytest.raises(RuntimeError):
        m.to_simulation_spec()


def test_upload_data_bad_mode():  # APIC-76 (renamed: preserve_correlation → mode)
    m = MCPower("y = x1 + x2")
    data = np.random.RandomState(0).normal(size=(30, 2))
    with pytest.raises(ValueError):
        m.upload_data(data, columns=["x1", "x2"], mode="maybe")
    # A valid, implemented choice must not raise on the mode gate.
    m.upload_data(data, columns=["x1", "x2"], mode="partial")


def test_set_max_failed_simulations_range():  # APIC-77
    m = MCPower("y = x1")
    m.set_max_failed_simulations(0.0)
    m.set_max_failed_simulations(1.0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            m.set_max_failed_simulations(bad)


def test_scenario_dict_rejects_unknown_distribution():  # APIC-80
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    m._scenario_configs["realistic"]["new_distributions"] = ["weird_dist"]
    with pytest.raises(ValueError):
        m._scenario_dict("realistic")

    m2 = MCPower("y = x1")
    m2.set_effects("x1=0.5")
    # Completely unknown name → residual_dists rejection.
    m2._scenario_configs["realistic"]["residual_dists"] = ["nonexistent_dist"]
    with pytest.raises(ValueError):
        m2._scenario_dict("realistic")


# ---------------------------------------------------------------------------
# Cluster / spec shape (APIC-42/43, 51, 52, 61..64)
# ---------------------------------------------------------------------------


def test_resolve_scenarios_empty_list_raises():  # APIC-43
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError):
        m._resolve_scenarios_arg([])


def test_build_cluster_spec_tau_transform():  # APIC-52
    m = MCPower("y ~ x1 + (1|g)", family="lme")
    m.set_effects("x1=0.5").set_cluster("g", ICC=0.2, n_clusters=10)
    spec = m._build_cluster_spec_dict()
    tau = spec["tau_squared"]
    assert tau == pytest.approx(0.2 / 0.8)   # ICC/(1-ICC) transform identity
    # FixedClusters shape check.
    assert spec["sizing"]["FixedClusters"]["n_clusters"] == 10
    m0 = MCPower("y ~ x1 + (1|g)", family="lme")
    m0.set_effects("x1=0.5").set_cluster("g", ICC=0.0, n_clusters=10)
    assert m0._build_cluster_spec_dict()["tau_squared"] == 0.0


def test_build_cluster_spec_fixed_size_shape():  # APIC-51b
    """cluster_size → FixedSize sizing dict."""
    m = MCPower("y ~ x1 + (1|g)", family="lme")
    m.set_effects("x1=0.5").set_cluster("g", ICC=0.2, cluster_size=15)
    spec = m._build_cluster_spec_dict()
    assert spec["sizing"]["FixedSize"]["cluster_size"] == 15


def test_find_sample_size_grid_multiples_of_atom():  # APIC-51 replacement
    """Engine-owned atom snapping: returned grid N's are all multiples of n_clusters."""
    m = MCPower("y ~ x1 + (1|g)", family="lme")
    m.set_effects("x1=0.5").set_cluster("g", ICC=0.2, n_clusters=20)
    res = m.find_sample_size(from_size=60, to_size=200, n_sims=20, verbose=False)
    sample_sizes = res["sample_sizes"]
    assert len(sample_sizes) > 0
    assert all(n % 20 == 0 for n in sample_sizes), f"non-multiple in grid: {sample_sizes}"


def test_to_linear_spec_correlations_nonzero_only():  # APIC-61
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    assert m._to_linear_spec_dict(["optimistic"])["correlations"] == []
    m.set_correlations("corr(x1,x2)=0.4")
    pairs = m._to_linear_spec_dict(["optimistic"])["correlations"]
    assert len(pairs) == 1 and pairs[0]["value"] == pytest.approx(0.4)


def test_to_linear_spec_correction_alias_mapping():  # APIC-63
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    assert m._to_linear_spec_dict(["optimistic"], correction="bh")["correction"] == "benjamini_hochberg"
    assert m._to_linear_spec_dict(["optimistic"], correction="fdr")["correction"] == "benjamini_hochberg"
    assert m._to_linear_spec_dict(["optimistic"], correction=None)["correction"] == "none"


def test_to_linear_spec_strips_random_effects():  # APIC-62
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, n_clusters=10)
    formula = m._to_linear_spec_dict(["optimistic"])["formula"]
    assert "(1|school)" not in formula
    assert "x1" in formula


def test_encode_clusters_json_present_for_clustered_ols():  # APIC-64
    m = MCPower("y ~ x + (1|g)", family="lme", estimator="ols")
    m.set_effects("x=0.4")
    m.set_cluster("g", ICC=0.2, n_clusters=12)
    _, estimator_wire, _, clusters_json = m._encode_outcome_and_clusters()
    assert estimator_wire == "ols"                 # estimator stays OLS
    assert clusters_json not in ("", "[]")         # cluster block still emitted


# ---------------------------------------------------------------------------
# Registry (APIC-57..59) & warnings (APIC-60)
# ---------------------------------------------------------------------------


def test_expand_factors_reference_coding_shape():  # APIC-57
    m = MCPower("y = a + f")
    m.set_variable_type("f=(factor,0.5,0.3,0.2)")
    m.set_effects("a=0.3")
    m._apply()
    dummies = [n for n in m._registry.effect_names if n.startswith("f[")]
    assert len(dummies) == 2                        # n_levels - 1
    assert all(n.startswith("f[") and n.endswith("]") for n in dummies)


def test_registry_set_correlation_symmetric():  # APIC-58
    m = MCPower("y = x1 + x2")
    m._registry.set_correlation("x1", "x2", 0.4)
    cm = m._registry.get_correlation_matrix()
    assert cm[0, 1] == cm[1, 0] == 0.4


def test_registry_set_variable_type_factor_info():  # APIC-59
    m = MCPower("y = a + f")
    m._registry.set_variable_type(
        "f", "factor", n_levels=3, proportions=[0.5, 0.3, 0.2]
    )
    info = m._registry._factors["f"]
    assert info["n_levels"] == 3
    assert info["proportions"] == [0.5, 0.3, 0.2]
    assert "level_labels" in info and "reference_level" in info


def test_warn_logit_effect_scale_two_tiers():  # APIC-60
    m = MCPower("y = x", family="logit")
    big = _warn_logit_effect_scale({"x": 6.0}, m._registry)
    assert big and any("|β|>5" in w for w in big)
    mid = _warn_logit_effect_scale({"x": 4.0}, m._registry)
    assert mid and any("|β|>3" in w for w in mid)
    none = _warn_logit_effect_scale({"x": 0.5}, m._registry)
    assert none == []


# ---------------------------------------------------------------------------
# unwrap_scenario_result (APIC-65/66) — direct results.py units
# ---------------------------------------------------------------------------


def test_unwrap_single_scenario_returns_inner():  # APIC-65
    raw = {"scenarios": {"optimistic": {"n_sims": 5}}}
    out = unwrap_scenario_result(raw, ["optimistic"])
    assert "scenarios" not in out      # inner dict, not the envelope
    assert out["n_sims"] == 5


def test_unwrap_multi_scenario_keeps_envelope():  # APIC-66
    raw = {"scenarios": {"optimistic": {"n_sims": 5}, "realistic": {"n_sims": 6}}}
    out = unwrap_scenario_result(raw, ["optimistic", "realistic"])
    assert "scenarios" in out          # full envelope unchanged
    assert set(out["scenarios"].keys()) == {"optimistic", "realistic"}


# ---------------------------------------------------------------------------
# find_* entry validation (APIC-78)
# ---------------------------------------------------------------------------


def test_find_power_validates_sample_size_before_engine():  # APIC-78
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    # sample_size=5 fails _validate_sample_size (< 20) before any engine call.
    with pytest.raises(ValueError):
        m.find_power(sample_size=5, n_sims=10, progress_callback=False)


# Sanity: logit baseline → intercept transform identity (APIC-41 anchor used
# elsewhere) is a transform shape, included here as the canonical mirror anchor.
def test_logit_baseline_intercept_is_logit_transform():  # APIC-41
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m._apply()
    assert m.intercept == pytest.approx(math.log(0.3 / 0.7))
