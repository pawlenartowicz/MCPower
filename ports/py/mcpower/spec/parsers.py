"""Thin shim over engine-spec-builder's Rust formula/assignment parsers.

All formula grammar (random-effect syntax, `corr(a,b)=v` syntax, the unicode
identifier regex) lives in Rust (`mcpower/crates/engine-spec-builder/src/
{formula,assignments}.rs`). This module reshapes the Rust output into the
tuples/dicts that `model.py` and `variables.py` historically consumed.

The one exception is `variable_type` value parsing: the Rust assignments
parser only accepts a bare type string (e.g. `x=binary`), but the legacy
public API additionally accepts tuple syntax like `x=(binary,0.3)` and
`g=(factor,0.2,0.3,0.5)`. That value-shape extension is kept here as a
**non-formula** helper — no formula grammar literals are involved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mcpower import _engine

__all__: List[str] = []


# Unicode-aware identifier pattern, re-exported for validators.py (which uses
# it to extract variable names from a test-formula string). Not used here.
_IDENT = r"[^\W\d]\w*"


# ---------------------------------------------------------------------------
# Formula parsing (delegates to Rust `_engine.parse_formula`)
# ---------------------------------------------------------------------------


def _parse_equation(equation: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Parse an R-style formula into ``(dependent, fixed_formula, random_effects)``.

    Reshapes the Rust ``parse_formula`` output back to the legacy v1 dict
    shape that ``variables.py`` and ``model.py`` expect:

    - random intercept: ``{"type": "random_intercept", "grouping_var": <g>}``
      (plus ``"parent_var"`` when the grouping factor is nested).
    - random slope: ``{"type": "random_slope", "grouping_var": <g>,
      "slope_vars": [<v>, ...]}``.

    The fixed-formula string is the canonical Rust-produced shape:
    ``"x1+x2+x1:x2"`` (no spaces; ``+`` between terms; ``:`` for interactions).
    """
    # Tolerate `=` as a synonym for `~` (legacy v1 convenience) — the Rust
    # parser only accepts `~`. Rewriting at the boundary keeps the grammar
    # single-sourced in Rust.
    rewritten = equation.replace("=", "~", 1) if "~" not in equation and "=" in equation else equation

    out = _engine.parse_formula(rewritten)
    fixed_formula = _terms_to_fixed_string(out["terms"])
    random_effects = [_re_to_legacy_dict(re) for re in out["random_effects"]]
    return out["dependent"], fixed_formula, random_effects


def _parse_independent_variables(formula: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract predictors and effects from a fixed-effects-only RHS formula.

    Returns ``(variables_dict, effects_dict)`` keyed by auto-generated
    identifiers (``variable_1``, ``effect_1``, ...) — matches the legacy v1
    shape consumed by ``variables.ModelRegistry.__init__``.

    Each variable entry: ``{"name": <str>}``. Each effect entry:
    ``{"name": <str>, "type": "main"|"interaction", "var_names": [...],
    "column_index": int | "column_indices": [int, ...]}``.

    The Rust parser already implements the `*` factorial expansion and `:`
    specific-interaction handling; we just rename its term list into the
    v1 numbered-key dicts.
    """
    out = _engine.parse_formula(f"explained_variable ~ {formula}")
    predictors: List[str] = list(out["predictors"])

    variables: Dict[str, Dict[str, Any]] = {
        f"variable_{i}": {"name": name} for i, name in enumerate(predictors, start=1)
    }

    effects: Dict[str, Dict[str, Any]] = {}
    effect_counter = 1
    for term in out["terms"]:
        if term["kind"] == "main":
            name = term["name"]
            effects[f"effect_{effect_counter}"] = {
                "name": name,
                "type": "main",
                "column_index": predictors.index(name),
            }
            effect_counter += 1
        elif term["kind"] == "interaction":
            vars_list = list(term["vars"])
            name = ":".join(vars_list)
            effects[f"effect_{effect_counter}"] = {
                "name": name,
                "type": "interaction",
                "var_names": vars_list,
                "column_indices": [predictors.index(v) for v in vars_list],
            }
            effect_counter += 1
        else:
            raise ValueError(f"unknown term kind from engine: {term['kind']!r}")

    return variables, effects


def _terms_to_fixed_string(terms: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for t in terms:
        if t["kind"] == "main":
            parts.append(t["name"])
        elif t["kind"] == "interaction":
            parts.append(":".join(t["vars"]))
    return "+".join(parts)


def _re_to_legacy_dict(re_dict: Dict[str, Any]) -> Dict[str, Any]:
    kind = re_dict["kind"]
    if kind == "intercept":
        out: Dict[str, Any] = {
            "type": "random_intercept",
            "grouping_var": re_dict["group"],
        }
        parent = re_dict.get("parent")
        if parent is not None:
            out["parent_var"] = parent
        return out
    if kind == "slope":
        return {
            "type": "random_slope",
            "grouping_var": re_dict["group"],
            "slope_vars": list(re_dict["vars"]),
        }
    raise ValueError(f"unknown random effect kind from engine: {kind!r}")


# ---------------------------------------------------------------------------
# Assignment parsing (delegates to Rust `_engine.parse_assignments` for
# correlations/effects; keeps tuple-shape value handling for variable types).
# ---------------------------------------------------------------------------


class _AssignmentParser:
    """Legacy wrapper around the Rust assignments parser.

    Preserves the historic ``._parse(input_string, parse_type, available_items)``
    contract used by ``model.py``. Returns ``(parsed_dict, errors)`` keyed by
    variable name (or sorted tuple, for correlations).
    """

    def _parse(
        self,
        input_string: str,
        parse_type: str,
        available_items: List[str],
    ) -> Tuple[Dict[Any, Any], List[str]]:
        if parse_type not in {"variable_type", "correlation", "effect"}:
            return {}, [f"Unknown parse type: {parse_type}"]

        # All three kinds parse in Rust now — including variable_type tuple
        # syntax `(factor,0.2,0.3,0.5)` and bare-form defaults (engine-spec-
        # builder `parse_assignments`). Correlation legacy compatibility: v1
        # accepted the bare-parens form; normalise it (textual rewrite, no
        # grammar) before delegating.
        if parse_type == "correlation":
            input_string = _normalise_correlation_input(input_string)

        try:
            out = _engine.parse_assignments(
                input_string,
                parse_type,
                {"predictors": list(available_items), "interaction_terms": []},
            )
        except ValueError as e:
            return {}, [str(e)]

        parsed: Dict[Any, Any] = {}
        for it in out["items"]:
            key, value = _assignment_to_legacy(it)
            parsed[key] = value
        return parsed, list(out["errors"])


_parser = _AssignmentParser()


def _normalise_correlation_input(input_string: str) -> str:
    """Rewrite ``(a,b)=v`` to ``corr(a,b)=v`` so the Rust parser accepts it.

    v1 grammar made the ``corr`` prefix optional; the Rust parser requires
    it. This helper restores the v1 convenience without reintroducing any
    grammar regex in this module.
    """
    parts = _engine.split_assignments(input_string)
    rewritten: List[str] = []
    for part in parts:
        stripped = part.lstrip()
        if stripped.startswith("(") and not stripped.startswith("corr"):
            indent = part[: len(part) - len(stripped)]
            rewritten.append(f"{indent}corr{stripped}")
        else:
            rewritten.append(part)
    return ", ".join(rewritten)


def _assignment_to_legacy(item: Dict[str, Any]) -> Tuple[Any, Any]:
    """Reshape a single Rust assignment item to ``(legacy_key, legacy_value)``.

    - Name keys → plain string.
    - Pair keys (correlations) → sorted ``(a, b)`` tuple (matches v1 storage).
    - Effect/Correlation values → bare ``float``.
    - VariableType values → the engine's legacy info dict
      (``{"type": ..., ["proportion"] | ["n_levels", "proportions"]}``), consumed
      directly by ``ModelRegistry.set_variable_type``.
    """
    key_d = item["key"]
    val_d = item["value"]
    if "name" in key_d:
        legacy_key: Any = key_d["name"]
    elif "pair" in key_d:
        a, b = key_d["pair"]
        legacy_key = tuple(sorted([a, b]))
    else:
        raise ValueError(f"unknown assignment key shape from engine: {key_d!r}")

    if "effect" in val_d:
        legacy_value: Any = val_d["effect"]
    elif "correlation" in val_d:
        legacy_value = val_d["correlation"]
    elif "variable_type" in val_d:
        legacy_value = val_d["variable_type"]
    else:
        raise ValueError(f"unknown assignment value shape from engine: {val_d!r}")
    return legacy_key, legacy_value
