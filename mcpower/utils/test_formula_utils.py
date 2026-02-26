"""Utilities for parsing and resolving test_formula parameters.

Provides helpers that map a test formula string to registry effect names
and column indices, enabling column-subsetting for model misspecification
testing.
"""

import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np

from .parsers import _parse_equation


def _extract_test_formula_effects(
    test_formula: str,
    registry,
) -> Tuple[List[str], List[Dict]]:
    """Extract effect names from a test formula, matched against the registry.

    Parses the test formula, expands factor variables to their dummies,
    and returns the list of effect names (in registry order) that belong
    to the test formula.

    Args:
        test_formula: Formula string (e.g. ``"y ~ x1 + x2"``).
        registry: ``VariableRegistry`` instance.

    Returns:
        Tuple of ``(effect_names, random_effects)`` where *effect_names*
        are the registry effect names present in the test formula (in
        registry order), and *random_effects* is the list of parsed
        random-effect dicts from the test formula.
    """
    _dep_var, fixed_formula, random_effects = _parse_equation(test_formula)

    # Parse fixed effects into a set of term names
    test_terms = _parse_fixed_terms(fixed_formula)

    # Determine which registry effects belong to the test formula
    cluster_effects = set(registry.cluster_effect_names)
    test_effects: List[str] = []

    for effect_name in registry._effects:
        if effect_name in cluster_effects:
            continue

        effect = registry._effects[effect_name]

        if effect.effect_type == "main":
            # Direct match (continuous or interaction-less variable)
            if effect_name in test_terms:
                test_effects.append(effect_name)
            elif effect_name in registry._factor_dummies:
                # Factor dummy -- include if parent factor is in test terms
                parent_factor = registry._factor_dummies[effect_name]["factor_name"]
                if parent_factor in test_terms:
                    test_effects.append(effect_name)
        else:
            # Interaction -- check if the interaction term is in test terms
            if effect_name in test_terms:
                test_effects.append(effect_name)

    return test_effects, random_effects


def _parse_fixed_terms(fixed_formula: str) -> Set[str]:
    """Parse a fixed-effect formula string into a set of term names.

    Handles ``+`` for additive terms, ``:`` for specific interactions,
    and ``*`` for full factorial expansion (main effects plus all
    two-way through n-way interactions).

    Args:
        fixed_formula: Right-hand side of the equation, spaces already
            stripped by ``_parse_equation`` (e.g. ``"x1+x2+x1:x2"``).

    Returns:
        Set of term names (variable names and interaction terms like
        ``"x1:x2"``).
    """
    if not fixed_formula.strip():
        return set()

    terms: Set[str] = set()
    raw_terms = re.split(r"\+", fixed_formula)

    for raw in raw_terms:
        raw = raw.strip()
        if not raw:
            continue

        if "*" in raw:
            # Full factorial: x1*x2 -> x1, x2, x1:x2
            vars_in_star = [v.strip() for v in raw.split("*") if v.strip()]
            for v in vars_in_star:
                terms.add(v)
            for r in range(2, len(vars_in_star) + 1):
                for combo in combinations(vars_in_star, r):
                    terms.add(":".join(combo))
        else:
            # Plain term (may contain ":" for explicit interaction)
            terms.add(raw)

    return terms


def _compute_test_column_indices(
    all_effect_names: List[str],
    test_effect_names: List[str],
) -> np.ndarray:
    """Compute column indices in X_expanded for test formula effects.

    Args:
        all_effect_names: All non-cluster effect names in registry order.
        test_effect_names: Effect names present in the test formula
            (a subset of *all_effect_names*).

    Returns:
        Integer array of column indices into X_expanded.
    """
    test_set = set(test_effect_names)
    indices = [i for i, name in enumerate(all_effect_names) if name in test_set]
    return np.array(indices, dtype=np.int64)


def _remap_target_indices(
    original_target_indices: np.ndarray,
    test_column_indices: np.ndarray,
) -> np.ndarray:
    """Remap target indices from full X_expanded space to X_test space.

    Args:
        original_target_indices: Indices in X_expanded being tested.
        test_column_indices: Columns of X_expanded included in X_test.

    Returns:
        Indices remapped to positions within X_test.
    """
    # Build mapping: full_index -> position in X_test
    index_map = {int(full_idx): test_idx for test_idx, full_idx in enumerate(test_column_indices)}
    return np.array(
        [index_map[int(idx)] for idx in original_target_indices],
        dtype=np.int64,
    )
