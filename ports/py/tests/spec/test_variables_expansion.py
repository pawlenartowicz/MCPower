"""Factor-level expansion edge cases for VariableRegistry.expand_factors().

Covers lines ~139-146 (empty/random-only formula) and lines ~366-395
(factor interaction expansion with/without level_labels).
"""

import pytest

from mcpower.spec.variables import VariableRegistry


# ---------------------------------------------------------------------------
# Lines ~139-146: formula with only random effects / empty RHS
# ---------------------------------------------------------------------------


def test_random_effects_only_no_fixed_raises():
    """Formula '(1|cluster)' with no fixed predictors must raise ValueError
    naming the grouping variable and suggesting an example."""
    with pytest.raises(ValueError, match=r"random effects") as exc_info:
        VariableRegistry("y ~ (1|cluster)")
    msg = str(exc_info.value)
    # Must name the grouping var and give the usage hint
    assert "cluster" in msg
    assert "Power analysis requires at least one fixed effect" in msg


def test_empty_rhs_raises():
    """Formula 'y ~ ' with an entirely empty RHS must raise ValueError."""
    with pytest.raises(ValueError):
        VariableRegistry("y ~ ")


# ---------------------------------------------------------------------------
# Lines ~366-395: factor interaction expansion — with level_labels
# ---------------------------------------------------------------------------


def _make_registry_with_labeled_factor_interaction():
    """Return a registry with 'cyl * x' (cyl is a labeled factor, x is continuous)
    using preserve_factor_level_names semantics (level_labels provided)."""
    reg = VariableRegistry("y ~ cyl:x")
    reg.set_variable_type(
        "cyl",
        "factor",
        n_levels=3,
        labels=["4", "6", "8"],
        reference="4",
    )
    reg.expand_factors()
    return reg


def test_factor_interaction_with_labels_expands_to_correct_dummy_names():
    """cyl:x with labels=['4','6','8'] and reference='4' must expand to
    'cyl[6]:x' and 'cyl[8]:x' — the two non-reference dummy interactions."""
    reg = _make_registry_with_labeled_factor_interaction()
    effect_names = reg.effect_names
    assert "cyl[6]:x" in effect_names, f"Expected 'cyl[6]:x' in {effect_names}"
    assert "cyl[8]:x" in effect_names, f"Expected 'cyl[8]:x' in {effect_names}"


def test_factor_interaction_reference_level_excluded():
    """Reference level 'cyl[4]' must NOT appear as an interaction term."""
    reg = _make_registry_with_labeled_factor_interaction()
    effect_names = reg.effect_names
    assert "cyl[4]:x" not in effect_names, (
        f"Reference-level dummy 'cyl[4]:x' must not appear; got {effect_names}"
    )


def test_factor_interaction_original_interaction_removed():
    """After expand_factors the raw 'cyl:x' interaction must not remain."""
    reg = _make_registry_with_labeled_factor_interaction()
    effect_names = reg.effect_names
    assert "cyl:x" not in effect_names, (
        f"Unexpanded 'cyl:x' must be gone after expand_factors; got {effect_names}"
    )


# ---------------------------------------------------------------------------
# Lines ~380-381: factor interaction expansion — WITHOUT level_labels
# (integer-indexed fallback, preserve_factor_level_names=False semantics)
# ---------------------------------------------------------------------------


def _make_registry_without_labeled_factor_interaction():
    """Factor 'group' without level_labels — integer index path."""
    reg = VariableRegistry("y ~ group:x")
    reg.set_variable_type(
        "group",
        "factor",
        n_levels=3,
        # no labels → integer-indexed 1..k
    )
    reg.expand_factors()
    return reg


def test_factor_interaction_without_labels_expands_to_integer_indexed_names():
    """Without level_labels, interaction dummies use integer indices:
    'group[2]:x' and 'group[3]:x'."""
    reg = _make_registry_without_labeled_factor_interaction()
    effect_names = reg.effect_names
    assert "group[2]:x" in effect_names, f"Expected 'group[2]:x' in {effect_names}"
    assert "group[3]:x" in effect_names, f"Expected 'group[3]:x' in {effect_names}"


def test_factor_interaction_without_labels_reference_excluded():
    """'group[1]:x' (reference level) must not appear."""
    reg = _make_registry_without_labeled_factor_interaction()
    effect_names = reg.effect_names
    assert "group[1]:x" not in effect_names, (
        f"Reference 'group[1]:x' must not appear; got {effect_names}"
    )


# ---------------------------------------------------------------------------
# Two-factor interaction: Cartesian product of non-reference levels
# ---------------------------------------------------------------------------


def test_two_factor_interaction_cartesian_product():
    """Two labeled factors 'a' and 'b' in an interaction must produce the full
    Cartesian product of their non-reference levels."""
    reg = VariableRegistry("y ~ a:b")
    reg.set_variable_type(
        "a", "factor", n_levels=2, labels=["lo", "hi"], reference="lo"
    )
    reg.set_variable_type(
        "b", "factor", n_levels=2, labels=["ctrl", "treat"], reference="ctrl"
    )
    reg.expand_factors()
    effect_names = reg.effect_names
    # Only one non-ref level per factor → one product term
    assert "a[hi]:b[treat]" in effect_names, f"Expected 'a[hi]:b[treat]' in {effect_names}"
    # Reference-level combos must not appear
    assert "a[lo]:b[treat]" not in effect_names
    assert "a[hi]:b[ctrl]" not in effect_names
    assert "a[lo]:b[ctrl]" not in effect_names
