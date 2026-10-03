"""Factor-level expansion edge cases for VariableRegistry.expand_factors().

Covers the empty / random-only formula rejection and factor interaction
expansion with and without level_labels.
"""

import pytest

from mcpower.spec.variables import VariableRegistry


# ---------------------------------------------------------------------------
# Formula with only random effects / empty RHS
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
# Factor interaction expansion — with level_labels
# ---------------------------------------------------------------------------


def _interaction_names(reg):
    return [n for n in reg.effect_names if ":" in n]


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
    exactly 'cyl[6]:x' and 'cyl[8]:x' — the two non-reference dummy
    interactions; the reference 'cyl[4]:x' and the raw 'cyl:x' are gone."""
    reg = _make_registry_with_labeled_factor_interaction()
    assert _interaction_names(reg) == ["cyl[6]:x", "cyl[8]:x"]


# ---------------------------------------------------------------------------
# Factor interaction expansion — WITHOUT level_labels
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
    exactly 'group[2]:x' and 'group[3]:x' (reference 'group[1]:x' excluded)."""
    reg = _make_registry_without_labeled_factor_interaction()
    assert _interaction_names(reg) == ["group[2]:x", "group[3]:x"]


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
    # Only one non-ref level per factor → one product term; no reference combos.
    assert _interaction_names(reg) == ["a[hi]:b[treat]"]
