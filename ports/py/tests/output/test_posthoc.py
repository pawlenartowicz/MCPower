"""Tests: engine surfaces posthoc block in the Python result dict.

These tests exercise the all-contrasts keyword and Tukey HSD correction
end-to-end through the full Python→Rust round-trip.
"""

from __future__ import annotations

import pytest

from mcpower import MCPower


@pytest.fixture
def simple_anova_model():
    """One-way ANOVA: pain_reduction ~ dose_group, a 3-level factor.

    Factor levels are integer-indexed after _apply(): dose_group[1] is the
    reference (omitted), dose_group[2] and dose_group[3] are the dummies.
    """
    m = MCPower("pain_reduction = dose_group")
    m.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")
    m.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")
    return m


def _named_three_level_factor_model():
    """OLS model with continuous baseline_pain + 3-level treatment factor.

    treatment has named levels: placebo (reference), low_dose, high_dose.
    This helper fully renames the integer-indexed dummy effects/predictors to
    named-level form so the Rust spec builder accepts them in find_power.
    """
    from mcpower.spec.variables import Effect

    m = MCPower("y = baseline_pain + treatment")
    m.set_variable_type("treatment=(factor,0.33,0.33,0.34)")
    m.set_effects("baseline_pain=0.3")
    m._apply()

    labels = ["placebo", "low_dose", "high_dose"]
    non_ref_labels = ["low_dose", "high_dose"]
    m._registry._factors["treatment"]["level_labels"] = labels
    m._registry._factors["treatment"]["reference_level"] = "placebo"

    # Rename integer-indexed dummy effects → named-level effects.
    # _apply() created treatment[2] (col 1) and treatment[3] (col 2).
    old_dummy_names = ["treatment[2]", "treatment[3]"]
    for old_name, level_label in zip(old_dummy_names, non_ref_labels):
        new_name = f"treatment[{level_label}]"
        col_idx = m._registry._effects[old_name].column_index
        new_eff = Effect(
            name=new_name,
            effect_type="main",
            effect_size=0.3,
            var_names=[new_name],
            column_index=col_idx,
            column_indices=[],
            factor_source="treatment",
            factor_level=level_label,
        )
        m._registry._effects[new_name] = new_eff
        del m._registry._effects[old_name]
        # Update _predictors
        if old_name in m._registry._predictors:
            pred = m._registry._predictors.pop(old_name)
            pred.name = new_name
            pred.factor_level = level_label
            m._registry._predictors[new_name] = pred
        # Update _factor_dummies
        if old_name in m._registry._factor_dummies:
            entry = m._registry._factor_dummies.pop(old_name)
            entry["level"] = level_label
            m._registry._factor_dummies[new_name] = entry

    m._applied = True
    return m


def test_engine_surfaces_posthoc_block(simple_anova_model):
    m = simple_anova_model
    r = m.find_power(sample_size=200, target_test="overall, all-contrasts", verbose=False)
    # PowerResult is a dict subclass; posthoc is a list of per-factor blocks.
    posthoc = r["posthoc"]
    assert len(posthoc) == 1            # one factor -> one block
    ph = posthoc[0]
    assert ph["n_levels"] == 3
    assert len(ph["power_uncorrected"]) == 3   # C(3,2) = 3 pairwise contrasts


def test_tukey_accepted_for_posthoc_only(simple_anova_model):
    """correction='tukey' works for post-hoc; corrected power ≤ uncorrected."""
    m = simple_anova_model
    r = m.find_power(
        sample_size=200,
        target_test="overall, all-contrasts",
        correction="tukey",
        verbose=False,
    )
    ph = r["posthoc"][0]
    for u, c in zip(ph["power_uncorrected"], ph["power_corrected"]):
        assert c <= u + 1e-9   # Tukey is at least as conservative as uncorrected


def test_tukey_with_marginal_targets_warns_not_errors(simple_anova_model):
    """correction='tukey' + marginal β targets → UserWarning, not an error."""
    m = simple_anova_model
    with pytest.warns(UserWarning, match="Tukey HSD"):
        r = m.find_power(
            sample_size=200,
            target_test="all, all-contrasts",
            correction="tukey",
            verbose=False,
        )
    assert len(r["posthoc"]) >= 1   # post-hoc still computed


def test_pure_posthoc_only_no_marginal_targets(simple_anova_model):
    """Pure all-contrasts (no 'overall', no named β targets) runs without error.

    Engine must handle n_targets==0 natively.  The main power_uncorrected table
    should contain no β entries; the posthoc block should have one factor block.
    """
    m = simple_anova_model
    r = m.find_power(sample_size=200, target_test="all-contrasts", verbose=False)
    result = dict(r)
    # posthoc block present with one factor
    assert len(result["posthoc"]) == 1
    assert result["posthoc"][0]["n_levels"] == 3
    # no marginal β entries in the main table
    power_main = result["power_uncorrected"]
    assert power_main == [] or power_main == [[]]


# ---------------------------------------------------------------------------
# Rendering tests (Task 5.4)
# ---------------------------------------------------------------------------


def test_short_form_renders_posthoc_factor_grouped(capsys):
    """Short-form repr nests the contrasts into the main per-test table under a
    '<factor>  (pairwise)' span (like factor levels), not a separate section."""
    m = _named_three_level_factor_model()
    m.find_power(sample_size=200, target_test="overall, all-contrasts",
                 correction="tukey")
    out = capsys.readouterr().out
    assert "treatment  (pairwise)" in out
    assert "low_dose vs placebo" in out
    assert "high_dose vs low_dose" in out
    # Tukey correction → the main table carries both columns
    assert "uncorrected" in out
    assert "corrected" in out
    # No standalone post-hoc section any more
    assert "Post-hoc pairwise contrasts" not in out
    assert "factor: treatment" not in out


def test_short_form_posthoc_no_correction_single_power_column(simple_anova_model):
    """Without correction the nested contrasts show a single Power column."""
    m = simple_anova_model
    r = m.find_power(sample_size=200, target_test="all-contrasts", verbose=False)
    text = repr(r)
    assert "dose_group  (pairwise)" in text
    # Integer-indexed: labels should be 2 vs 1, 3 vs 1, 3 vs 2
    assert "2 vs 1" in text
    assert "3 vs 1" in text
    assert "3 vs 2" in text
    # No correction → single Power column (no "uncorrected"/"corrected" split)
    assert "uncorrected" not in text
    assert "Post-hoc pairwise contrasts" not in text


def test_long_form_nests_posthoc():
    """Long-form summary() nests post-hoc contrasts into the Per-test power table."""
    m = _named_three_level_factor_model()
    r = m.find_power(sample_size=200, target_test="overall, all-contrasts",
                     correction="tukey", verbose=False)
    text = str(r.summary())
    assert "treatment  (pairwise)" in text
    assert "low_dose vs placebo" in text
    assert "high_dose vs low_dose" in text
    assert "Post-hoc pairwise contrasts" not in text


def test_posthoc_meta_has_posthoc_factors():
    """_report_meta includes posthoc_factors with correct levels list."""
    m = _named_three_level_factor_model()
    r = m.find_power(sample_size=200, target_test="overall, all-contrasts",
                     correction="tukey", verbose=False)
    meta = r._meta
    assert "posthoc_factors" in meta
    pf = meta["posthoc_factors"]
    assert len(pf) == 1
    assert pf[0]["name"] == "treatment"
    assert pf[0]["levels"] == ["placebo", "low_dose", "high_dose"]


def test_posthoc_meta_integer_indexed_levels():
    """_report_meta synthesises numeric labels when level_labels is None."""
    m = MCPower("pain_reduction = dose_group")
    m.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")
    m.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")
    r = m.find_power(sample_size=200, target_test="all-contrasts", verbose=False)
    meta = r._meta
    pf = meta["posthoc_factors"]
    assert len(pf) == 1
    assert pf[0]["name"] == "dose_group"
    assert pf[0]["levels"] == ["1", "2", "3"]
