"""Tests for set_tests DSL, defaults, and grouped print format.

Tests:
  1. Default (no set_tests) → wire targets=["overall"], report_overall=True,
     contrast_pairs=[] (linear family default).
  2. set_tests("all") explicit → same as default for linear.
  3. set_tests("x1") → just Marginal(x1).
  4. set_tests("all, -x2") → all-but-x2.
  5. set_tests("all-posthoc") on a 3-level treatment factor →
     posthoc_requests=[{"factor": "treatment"}], no contrast_pairs, targets=[].
  6. set_tests("all, all-posthoc") → omnibus + every β + posthoc_requests.
  7. set_tests("treatment[low_dose] vs treatment[placebo]") → contrast pair.
  8. set_tests("group[1] vs group[2]") (numeric integer labels) → contrast pair.
  9. set_tests("overall, baseline_pain, treatment[placebo] vs treatment[high_dose]")
     → mixed selection round-trips.
  10. Unknown name raises ValueError.
  11. all-contrasts emits posthoc_requests (one per factor).
  12. all-posthoc is an alias for all-contrasts.
  13. all-contrasts on a model with no factors raises ValueError.
  14. all-contrasts + "all" → posthoc_requests and omnibus + every β.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mcpower import MCPower


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ols_with_treatment(named_levels=True):
    """OLS model with continuous baseline_pain + 3-level treatment factor.

    treatment has levels: placebo, low_dose, high_dose (when named_levels=True)
    or integer indices 1, 2, 3 (when named_levels=False).
    """
    m = MCPower("y = baseline_pain + treatment")
    m.set_variable_type("treatment=(factor,0.33,0.33,0.34)")
    m.set_effects("baseline_pain=0.3")
    # After _apply() the factor has integer labels 1/2/3.
    # To get named labels we patch the registry directly (the public API only
    # supports named labels from upload_data; internal API is fine for tests).
    m._apply()
    if named_levels:
        labels = ["placebo", "low_dose", "high_dose"]
        m._registry._factors["treatment"]["level_labels"] = labels
        m._registry._factors["treatment"]["reference_level"] = "placebo"
        # Reset applied so _resolve_tests sees the registry in its final state.
        m._applied = True
        # Also need to populate effect sizes for the dummies after re-expansion.
        # The effects after apply are treatment[1], treatment[2] (integer dummy names).
        # We need to set the effect sizes for the named dummy effects.
        # The simplest approach: do NOT rely on _apply doing this again;
        # instead set effects on the already-expanded registry.
        # Since we patched level_labels AFTER apply, the effect names in the
        # registry are still treatment[1]/treatment[2]. Let's alias them.
        # Actually, set named level effects directly:
        for eff_name in list(m._registry._effects.keys()):
            if eff_name.startswith("treatment["):
                m._registry._effects[eff_name].effect_size = 0.2
    return m


def _make_ols_integer_factor():
    """OLS with baseline_pain + group factor (3 integer-labeled levels)."""
    m = MCPower("y = baseline_pain + group")
    m.set_variable_type("group=(factor,0.33,0.33,0.34)")
    m.set_effects("baseline_pain=0.3")
    m._apply()
    for eff_name in list(m._registry._effects.keys()):
        if eff_name.startswith("group["):
            m._registry._effects[eff_name].effect_size = 0.2
    return m


def _make_ols_two_factors():
    """OLS with two factors: drug (3 levels) + gender (2 levels), no continuous."""
    m = MCPower("y = drug + gender")
    m.set_variable_type("drug=(factor,0.34,0.33,0.33), gender=(factor,0.5,0.5)")
    m._apply()
    for eff_name in list(m._registry._effects.keys()):
        if eff_name.startswith("drug[") or eff_name.startswith("gender["):
            m._registry._effects[eff_name].effect_size = 0.2
    return m


def _make_continuous_only_model():
    """OLS with two continuous predictors, no factor."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.2")
    m._apply()
    return m


def _get_wire_payload(m: MCPower, target_test=None) -> Dict[str, Any]:
    """Build the wire payload dict, driving the DSL through ``target_test=``.

    ``target_test`` threads through ``_to_linear_spec_dict`` exactly as the
    ``find_power`` / ``find_sample_size`` kwarg does; ``None`` reproduces the
    family default.
    """
    # Re-apply to ensure fresh state.
    if not m._applied:
        m._apply()
    return m._to_linear_spec_dict(["optimistic"], target_test=target_test)


# ---------------------------------------------------------------------------
# 1. Default (no set_tests) → linear family default
# ---------------------------------------------------------------------------


def test_default_no_set_tests_is_family_default():
    """No set_tests call → targets=['overall'] (magic), report_overall=True,
    contrast_pairs=[]."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m)
    assert payload["targets"] == ["overall"]
    assert payload["report_overall"] is True
    assert payload["contrast_pairs"] == []


# ---------------------------------------------------------------------------
# 2. set_tests("all") → same as default for linear
# ---------------------------------------------------------------------------


def test_set_tests_all_equals_default():
    """set_tests('all') is equivalent to the family default for linear."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m, target_test="all")
    # "all" expands to "overall" + every beta → targets=["x1","x2"],
    # report_overall=True, contrast_pairs=[].
    assert payload["report_overall"] is True
    assert set(payload["targets"]) == {"x1", "x2"}
    assert payload["contrast_pairs"] == []


# ---------------------------------------------------------------------------
# 3. set_tests("x1") → just that effect
# ---------------------------------------------------------------------------


def test_set_tests_single_effect():
    """set_tests('x1') → targets=['x1'], report_overall=False,
    contrast_pairs=[]."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m, target_test="x1")
    assert payload["targets"] == ["x1"]
    assert payload["report_overall"] is False
    assert payload["contrast_pairs"] == []


# ---------------------------------------------------------------------------
# 4. set_tests("all, -x2") → all-but-x2
# ---------------------------------------------------------------------------


def test_set_tests_all_minus_x2():
    """set_tests('all, -x2') expands 'all' then removes x2."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m, target_test="all, -x2")
    assert payload["report_overall"] is True
    assert "x1" in payload["targets"]
    assert "x2" not in payload["targets"]
    assert payload["contrast_pairs"] == []


# ---------------------------------------------------------------------------
# 5. set_tests("all-posthoc") on a 3-level factor → posthoc_requests (new path)
# ---------------------------------------------------------------------------


def test_set_tests_all_posthoc_named_factor():
    """all-posthoc on treatment → posthoc_requests=[{"factor": "treatment"}],
    no contrast_pairs expansion, no marginal β targets, targets=[]."""
    m = _make_ols_with_treatment(named_levels=True)
    payload = _get_wire_payload(m, target_test="all-posthoc")
    assert payload["report_overall"] is False
    assert payload["targets"] == []
    assert payload["contrast_pairs"] == []
    assert payload["posthoc_requests"] == [{"factor": "treatment"}]


# ---------------------------------------------------------------------------
# 6. set_tests("all, all-posthoc") → omnibus + every β + posthoc_requests
# ---------------------------------------------------------------------------


def test_set_tests_all_and_all_posthoc():
    """all + all-posthoc → report_overall=True, all betas in targets,
    posthoc_requests=[{"factor": "treatment"}], no contrast_pairs expansion."""
    m = _make_ols_with_treatment(named_levels=True)
    payload = _get_wire_payload(m, target_test="all, all-posthoc")
    assert payload["report_overall"] is True
    # "all" expands to every β, so "baseline_pain" must appear.
    assert "baseline_pain" in payload["targets"]
    assert payload["contrast_pairs"] == []
    assert payload["posthoc_requests"] == [{"factor": "treatment"}]


# ---------------------------------------------------------------------------
# 7. Explicit contrast pair with named levels → reference-collapse expected
# ---------------------------------------------------------------------------


def test_set_tests_explicit_contrast_pair_named():
    """set_tests('treatment[low_dose] vs treatment[placebo]') stores the pair."""
    m = _make_ols_with_treatment(named_levels=True)
    payload = _get_wire_payload(
        m, target_test="treatment[low_dose] vs treatment[placebo]"
    )
    assert payload["report_overall"] is False
    assert payload["targets"] == []
    pairs = payload["contrast_pairs"]
    assert len(pairs) == 1
    pos, neg = pairs[0]
    assert pos == "treatment[low_dose]"
    assert neg == "treatment[placebo]"


# ---------------------------------------------------------------------------
# 8. set_tests("group[1] vs group[2]") with integer-indexed levels
# ---------------------------------------------------------------------------


def test_set_tests_integer_level_contrast():
    """set_tests('group[1] vs group[2]') works for integer-labeled factors."""
    m = _make_ols_integer_factor()
    payload = _get_wire_payload(m, target_test="group[1] vs group[2]")
    assert payload["report_overall"] is False
    pairs = payload["contrast_pairs"]
    assert len(pairs) == 1
    pos, neg = pairs[0]
    assert pos == "group[1]"
    assert neg == "group[2]"


# ---------------------------------------------------------------------------
# 9. Mixed selection round-trips
# ---------------------------------------------------------------------------


def test_set_tests_mixed_selection():
    """'overall, baseline_pain, treatment[placebo] vs treatment[high_dose]'
    → report_overall=True, targets=['baseline_pain'], 1 contrast pair."""
    m = _make_ols_with_treatment(named_levels=True)
    payload = _get_wire_payload(
        m, target_test="overall, baseline_pain, treatment[placebo] vs treatment[high_dose]"
    )
    assert payload["report_overall"] is True
    assert payload["targets"] == ["baseline_pain"]
    pairs = payload["contrast_pairs"]
    assert len(pairs) == 1
    pos, neg = pairs[0]
    assert pos == "treatment[placebo]"
    assert neg == "treatment[high_dose]"


# ---------------------------------------------------------------------------
# 10. Unknown name raises ValueError
# ---------------------------------------------------------------------------


def test_set_tests_unknown_name_raises():
    """Unknown effect name in target_test raises ValueError."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m._apply()
    with pytest.raises(ValueError, match="Unknown test name"):
        m._resolve_tests("x3_nonexistent")


def test_set_tests_unknown_exclusion_raises():
    """Excluding a name that's not in the expanded set raises ValueError."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m._apply()
    with pytest.raises(ValueError, match="Exclusion"):
        m._resolve_tests("x1, -x2")  # x2 not in the explicit set (only x1)


# ---------------------------------------------------------------------------
# Alias tests: "y" and dep_var_name map to "overall"
# ---------------------------------------------------------------------------


def test_set_tests_y_alias_maps_to_overall():
    """Token 'y' is an alias for 'overall'."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m, target_test="y")
    assert payload["report_overall"] is True
    assert payload["targets"] == []


def test_set_tests_dep_var_alias_maps_to_overall():
    """Token equal to the dep var name is an alias for 'overall'."""
    m = MCPower("score = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    payload = _get_wire_payload(m, target_test="score")
    assert payload["report_overall"] is True
    assert payload["targets"] == []


# ---------------------------------------------------------------------------
# Migration shim: the removed set_tests setter raises an instructive error
# ---------------------------------------------------------------------------


def test_removed_set_tests_raises_instructive_error():
    """Calling the removed set_tests setter points at the target_test kwarg."""
    m = MCPower("y = x1")
    with pytest.raises(AttributeError, match="target_test"):
        m.set_tests("x1")


# ---------------------------------------------------------------------------
# 11–14. all-contrasts keyword (new) and alias / error cases
# ---------------------------------------------------------------------------


def test_all_contrasts_emits_posthoc_requests_one_per_factor():
    """all-contrasts on a single factor → posthoc_requests=[{"factor": name}],
    no contrast_pairs; targets=[] (engine natively handles n_targets==0)."""
    m = _make_ols_with_treatment(named_levels=True)
    payload = _get_wire_payload(m, target_test="overall, all-contrasts")
    assert payload["posthoc_requests"] == [{"factor": "treatment"}]
    assert payload["report_overall"] is True
    assert payload["contrast_pairs"] == []
    assert payload["targets"] == []


def test_all_contrasts_two_factors_emits_two_requests():
    """all-contrasts on a two-factor model → one request per factor."""
    m = _make_ols_two_factors()
    payload = _get_wire_payload(m, target_test="all-contrasts")
    factors = {r["factor"] for r in payload["posthoc_requests"]}
    assert factors == {"drug", "gender"}
    assert payload["contrast_pairs"] == []
    assert payload["targets"] == []


def test_all_posthoc_is_alias_for_all_contrasts():
    """all-posthoc and all-contrasts produce identical posthoc_requests."""
    m = _make_ols_with_treatment(named_levels=True)
    a = _get_wire_payload(m, target_test="all-posthoc")
    b = _get_wire_payload(m, target_test="all-contrasts")
    assert a["posthoc_requests"] == b["posthoc_requests"]


def test_all_contrasts_no_factor_errors():
    """all-contrasts on a model with no factors raises ValueError mentioning
    'at least one factor'."""
    m = _make_continuous_only_model()
    with pytest.raises(ValueError, match="at least one factor"):
        _get_wire_payload(m, target_test="all-contrasts")


# ---------------------------------------------------------------------------
# Overall-test availability gate (Choice 1): the omnibus is exposed only for
# OLS / unclustered GLM fits. Mixed-effects fits (LME, clustered GLMM) suppress
# it; an OLS fit on clustered data keeps the F-test.
# ---------------------------------------------------------------------------


def _make_lme():
    """LME: linear outcome with a random intercept, fit by MLE."""
    m = MCPower("y ~ x + (1|group)", family="lme")
    m.set_effects("x=0.5")
    m.set_cluster("group", ICC=0.3, n_clusters=20)
    return m


def _make_glmm():
    """GLMM: binary outcome with a random intercept (clustered logistic)."""
    m = MCPower("y ~ x + (1|group)", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m.set_cluster("group", ICC=0.3, n_clusters=20)
    return m


def _make_clustered_ols():
    """Clustered data fit with OLS (the 'cost of ignoring clustering' study)."""
    m = MCPower("y ~ x + (1|group)", family="lme", estimator="ols")
    m.set_effects("x=0.5")
    m.set_cluster("group", ICC=0.3, n_clusters=20)
    return m


def test_lme_default_suppresses_overall():
    """LME family default → marginals kept, omnibus suppressed."""
    payload = _get_wire_payload(_make_lme())
    assert payload["report_overall"] is False
    assert payload["targets"] == ["overall"]  # sentinel = every β, no omnibus


def test_glmm_default_suppresses_overall():
    """Clustered-logistic GLMM default → omnibus suppressed."""
    payload = _get_wire_payload(_make_glmm())
    assert payload["report_overall"] is False


def test_lme_explicit_overall_raises():
    """Explicit target_test='overall' on a mixed model raises ValueError."""
    m = _make_lme()
    m._apply()
    with pytest.raises(ValueError, match="not available for mixed-effects"):
        m._resolve_tests("overall")


def test_lme_y_alias_overall_raises():
    """The dep-var/'y' alias for the omnibus is rejected on a mixed model too."""
    m = _make_lme()
    m._apply()
    with pytest.raises(ValueError, match="not available for mixed-effects"):
        m._resolve_tests("y")


def test_glmm_all_drops_overall_keeps_betas():
    """'all' on a mixed model expands to fixed-effect βs with no omnibus."""
    payload = _get_wire_payload(_make_glmm(), target_test="all")
    assert payload["report_overall"] is False
    assert "x" in payload["targets"]


def test_clustered_ols_keeps_overall():
    """An OLS fit on clustered data keeps the F-test: the naive omnibus matches
    the naive fit (family='lme', estimator='ols')."""
    payload = _get_wire_payload(_make_clustered_ols())
    assert payload["report_overall"] is True


def test_clustered_ols_explicit_overall_allowed():
    """Explicit 'overall' is allowed for a clustered-OLS fit."""
    payload = _get_wire_payload(_make_clustered_ols(), target_test="overall")
    assert payload["report_overall"] is True
