"""Test-selection DSL parser (``target_test`` string -> wire dict).

Extracted free-function form of the selection logic; ``MCPower._resolve_tests``
is a thin wrapper that calls ``resolve_tests(target_test, self._registry)``.
"""

import re
from typing import Any, Dict, List, Tuple


def overall_test_available(estimator: str, registry) -> bool:
    """Whether the omnibus / overall test is defined for this fit.

    The overall test is the OLS F-test / GLM likelihood-ratio test. It is
    exposed for an *unclustered* OLS or GLM fit only. Mixed-effects fits have
    no exposed omnibus: LME (``estimator == "mle"``) and the clustered-logistic
    GLMM (``estimator == "glm"`` with a ``(1|group)`` random effect) suppress
    the overall channel — the engine's omnibus path for them is parked for a
    future joint-Wald omnibus, not wired up. An OLS fit on clustered data
    (``family='lme', estimator='ols'``) keeps the F-test: the naive omnibus
    matches the naive fit.
    """
    if estimator == "ols":
        return True
    if estimator == "glm":
        return not registry._random_effects_parsed
    return False  # mle (LME)


def resolve_tests(raw: str, registry, overall_available: bool = True) -> Dict[str, Any]:
    """Parse a ``target_test`` DSL string and return the wire dict.

    Mirrors the v1 DSL. ``target_test=None`` is *not* handled here — the
    caller (``_to_linear_spec_dict``) maps ``None`` to the family default
    before reaching this method.

    Token grammar (comma-separated):
        ``"all"``              — omnibus + every fixed-effect β (no contrasts).
        ``"all-contrasts"``    — all-pairwise post-hoc for every factor via
                                 posthoc_requests (no omnibus, no individual
                                 betas). ``"all-posthoc"`` is an alias.
        ``"overall"``          — the omnibus F-test / LRT.
        ``"<name>"``           — a named effect (e.g. ``"x1"``).
        ``"y"`` / dep_var name — alias for ``"overall"``.
        ``"<f>[<a>] vs <f>[<b>]"`` — pairwise contrast (same factor).
        ``"-<token>"``         — exclusion: remove the expanded token.

    Multiple keywords may be combined (e.g. ``"all, all-contrasts"`` gives
    omnibus + every β + posthoc_requests for every factor).

    Returns a dict with keys:
        targets: List[str]            — effect names.
        contrast_pairs: List[Tuple[str,str]]
        report_overall: bool
        posthoc_factors: List[str]    — factor names for posthoc_requests wire
                                        field; empty when no keyword used.

    Raises:
        ValueError: Unknown name, unknown exclusion, all tests excluded,
            or duplicate tokens.
    """
    # ── Phase 1: tokenize ──────────────────────────────────────────────
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    # ── Phase 2: classify ──────────────────────────────────────────────
    # "all-posthoc" is an alias for "all-contrasts"; normalise early.
    _POSTHOC_KEYWORDS = {"all-contrasts", "all-posthoc"}
    keywords: List[str] = []
    exclusions: List[str] = []
    explicit_tests: List[str] = []
    posthoc_keyword_seen: bool = False

    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower == "all":
            keywords.append("all")
        elif tok_lower in _POSTHOC_KEYWORDS:
            posthoc_keyword_seen = True
        elif tok.startswith("-"):
            exclusions.append(tok[1:].strip())
        else:
            explicit_tests.append(tok)

    # ── Phase 3: expand "all" keyword ─────────────────────────────────
    cluster_effects = set(registry.cluster_effect_names)
    keyword_expansion: List[str] = []

    if "all" in keywords:
        fixed_effects = [
            e for e in registry.effect_names if e not in cluster_effects
        ]
        # The omnibus rides along with "all" only where it is defined (OLS /
        # unclustered GLM). For mixed-effects fits "all" means every
        # fixed-effect β with no omnibus — silently dropped, not an error.
        omnibus = ["overall"] if overall_available else []
        keyword_expansion += omnibus + fixed_effects

    # ── Phase 3b: resolve posthoc keyword → posthoc_factors ───────────
    # The all-contrasts/all-posthoc keyword does NOT expand into
    # contrast_pairs strings; it produces a posthoc_factors list that is
    # emitted as posthoc_requests on the wire.  Explicit "f[a] vs f[b]"
    # syntax still uses contrast_pairs unchanged.
    posthoc_factors: List[str] = []
    if posthoc_keyword_seen:
        posthoc_factors = [
            name
            for name in registry.factor_names
            if registry._factors[name]["n_levels"] > 0
        ]
        if not posthoc_factors and not keyword_expansion and not explicit_tests:
            raise ValueError(
                "'all-contrasts'/'all-posthoc' was specified but the model "
                "has no factor variables. Post-hoc contrasts require "
                "at least one factor."
            )

    # ── Phase 4: merge ─────────────────────────────────────────────────
    expanded = keyword_expansion + explicit_tests

    # ── Phase 5: dependent-variable alias ─────────────────────────────
    dep_var_name = registry.dependent
    alias_set = {dep_var_name, "y"}
    expanded = ["overall" if t in alias_set else t for t in expanded]
    exclusions = ["overall" if e in alias_set else e for e in exclusions]

    # ── Phase 6: apply exclusions ─────────────────────────────────────
    for excl in exclusions:
        if excl not in expanded:
            raise ValueError(
                f"Exclusion '-{excl}' does not match any test in the "
                f"expanded set. Available: {', '.join(expanded)}"
            )
        expanded.remove(excl)

    if not expanded and not posthoc_factors:
        raise ValueError("All tests were excluded — nothing left to analyse.")

    # ── Phase 7: validate uniqueness ──────────────────────────────────
    seen: Dict[str, int] = {}
    for t in expanded:
        seen[t] = seen.get(t, 0) + 1
    duplicates = [t for t, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate target test(s): {', '.join(duplicates)}. Each "
            "test may appear only once. If using keyword expansion "
            "(e.g. 'all'), do not also list tests that are already included."
        )

    # ── Phase 8: parse + validate contrasts and effect names ──────────
    vs_pattern = re.compile(
        r"^([A-Za-z_][\w]*)\[([^\]]+)\]\s+vs\s+([A-Za-z_][\w]*)\[([^\]]+)\]$"
    )
    targets: List[str] = []
    contrast_pairs: List[Tuple[str, str]] = []
    report_overall: bool = False

    # Build the set of valid non-cluster effect names + "overall".
    valid_effect_names = set(
        e for e in registry.effect_names if e not in cluster_effects
    )

    for t in expanded:
        if t == "overall":
            if not overall_available:
                raise ValueError(
                    "The overall/omnibus test is not available for "
                    "mixed-effects models (LME / clustered GLMM). Request "
                    "specific fixed effects instead, e.g. target_test='x1'."
                )
            report_overall = True
            continue
        m = vs_pattern.match(t)
        if m:
            factor_a, level_a_str, factor_b, level_b_str = m.groups()
            if factor_a != factor_b:
                raise ValueError(
                    f"Post-hoc comparison must be between levels of the "
                    f"same factor, got '{factor_a}' vs '{factor_b}'"
                )
            factor_name = factor_a
            if factor_name not in registry._factors:
                raise ValueError(
                    f"Factor '{factor_name}' not found. Available: "
                    f"{', '.join(registry._factors.keys()) or 'none'}"
                )
            factor_info = registry._factors[factor_name]
            level_labels = factor_info.get("level_labels")
            if level_labels:
                avail = [str(lb) for lb in level_labels]
                if level_a_str not in avail:
                    raise ValueError(
                        f"Level '{level_a_str}' not found for factor "
                        f"'{factor_name}'. Available: {avail}"
                    )
                if level_b_str not in avail:
                    raise ValueError(
                        f"Level '{level_b_str}' not found for factor "
                        f"'{factor_name}'. Available: {avail}"
                    )
            else:
                n_levels = factor_info["n_levels"]
                try:
                    lvl_a = int(level_a_str)
                    lvl_b = int(level_b_str)
                except ValueError:
                    raise ValueError(
                        f"Factor '{factor_name}' has no named levels; "
                        f"use integer indices 1..{n_levels}"
                    )
                if lvl_a < 1 or lvl_a > n_levels:
                    raise ValueError(
                        f"Level {lvl_a} out of range for factor "
                        f"'{factor_name}' (valid: 1 to {n_levels})"
                    )
                if lvl_b < 1 or lvl_b > n_levels:
                    raise ValueError(
                        f"Level {lvl_b} out of range for factor "
                        f"'{factor_name}' (valid: 1 to {n_levels})"
                    )
                if lvl_a == lvl_b:
                    raise ValueError(f"Cannot compare a level to itself: {t}")
            # Store as the full "factor[level]" name strings — the engine
            # resolves them back to β-column indices from the contrast_pairs list.
            positive_name = f"{factor_name}[{level_a_str}]"
            negative_name = f"{factor_name}[{level_b_str}]"
            contrast_pairs.append((positive_name, negative_name))
        else:
            # Plain effect name — validate against the registry.
            if t not in valid_effect_names:
                raise ValueError(
                    f"Unknown test name '{t}'. Available effects: "
                    f"{', '.join(sorted(valid_effect_names))}"
                )
            targets.append(t)

    return {
        "targets": targets,
        "contrast_pairs": contrast_pairs,
        "report_overall": report_overall,
        "posthoc_factors": posthoc_factors,
    }
