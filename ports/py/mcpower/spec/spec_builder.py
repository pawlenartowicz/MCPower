"""Spec builders: project MCPower state into the Rust ``LinearSpec`` JSON.

Extracted free-function form of the scenario/linear-spec serialisers;
``MCPower._to_linear_spec_dict`` is a thin wrapper that passes the needed
``self.*`` fields in as arguments.
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

from .test_selector import overall_test_available, resolve_tests


def build_scenario_dict(
    scenario_name: str, scenario_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the ``ScenarioPerturbations`` dict for ``scenario_name``."""
    configs = scenario_configs
    if scenario_name not in configs:
        raise ValueError(
            f"Unknown scenario {scenario_name!r}; configured: "
            f"{sorted(configs)}"
        )
    cfg = configs[scenario_name]

    # Map distribution names to integer codes; unknown names are dropped
    # with a clear message rather than silently producing garbage.
    from ..config import get_dist_codes

    def _encode_dist_list(names: List[str]) -> List[int]:
        dist_codes = get_dist_codes()
        out: List[int] = []
        for n in names:
            if n not in dist_codes:
                raise ValueError(
                    f"scenario {scenario_name!r}: unknown distribution "
                    f"name {n!r}; valid: {sorted(dist_codes)}"
                )
            out.append(dist_codes[n])
        return out

    def _encode_re_dist(name: str) -> int:
        # RE distribution has its own vocabulary: normal=0, heavy_tailed=1,
        # right_skewed=2. Uses re_dist_codes, not residual_codes — separate spaces.
        from ..config import get_re_dist_codes
        re_codes = get_re_dist_codes()
        if name not in re_codes:
            raise ValueError(
                f"scenario {scenario_name!r}: unknown random_effect_dist "
                f"{name!r}; valid: {sorted(re_codes)}"
            )
        return re_codes[name]

    new_dists = _encode_dist_list(list(cfg.get("new_distributions", [])))

    return {
        "name": scenario_name,
        "heterogeneity": float(cfg.get("heterogeneity", 0.0)),
        "heteroskedasticity_ratio": float(cfg.get("heteroskedasticity_ratio", 1.0)),
        "correlation_noise_sd": float(cfg.get("correlation_noise_sd", 0.0)),
        "distribution_change_prob": float(
            cfg.get("distribution_change_prob", 0.0)
        ),
        "new_distributions": new_dists,
        "sampled_factor_proportions": bool(
            cfg.get("sampled_factor_proportions", False)
        ),
        "truth_start": bool(cfg.get("truth_start", False)),
        "random_effect_dist": _encode_re_dist(
            str(cfg.get("random_effect_dist", "normal"))
        ),
        "random_effect_df": float(cfg.get("random_effect_df", 0.0)),
        "icc_noise_sd": float(cfg.get("icc_noise_sd", 0.0)),
    }


def build_linear_spec(
    registry,
    scenario_names: List[str],
    *,
    heteroskedasticity: Dict[str, Any],
    residual_dist_name: str,
    alpha: float,
    correction: Optional[str],
    wald_se: str,
    nagq: int = 1,
    df_method: str,
    target_test: Optional[str],
    test_formula: Optional[str],
    equation: str,
    scenario_configs: Dict[str, Dict[str, Any]],
    max_failed_simulations: float,
    estimator: str = "ols",
    cluster_level_vars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Project current state into the Rust ``LinearSpec`` JSON contract.

    Shape mirrors ``crates/engine-spec-builder/src/input.rs::LinearSpec``.
    The Rust builder consumes the JSON serialisation of this dict and
    returns one ``SimulationSpec`` per name in ``scenario_names`` (the
    builder seeds default scenarios when the list is empty, so we always
    pass at least one). The ``upload`` block is not in the dict:
    ``linear_spec_json`` splices it in when serialising.
    """
    reg = registry

    # Predictors — non-factors first (in formula order), then factors.
    # The Rust builder reorders to formula order internally; we emit the
    # union of the two lists here so every predictor mentioned in the
    # formula has a matching PredictorSpec.
    from ..config import get_dist_codes

    dist_codes = get_dist_codes()
    predictors: List[Dict[str, Any]] = []
    non_factor_names = reg.non_factor_names
    for name in non_factor_names:
        pred = reg.get_predictor(name)
        if pred is None:
            raise RuntimeError(f"missing predictor metadata for {name!r}")
        var_type = pred.var_type
        kind_name = var_type if var_type in dist_codes else None
        if kind_name is None:
            raise RuntimeError(
                f"predictor {name!r} has var_type {var_type!r}, which is "
                "not supported by the Rust spec builder"
            )
        entry: Dict[str, Any] = {"name": name, "kind": kind_name}
        if kind_name == "binary":
            entry["proportion"] = float(pred.proportion)
        # Omit when False to keep neutral wires byte-stable.
        if pred.pinned:
            entry["pinned"] = True
        predictors.append(entry)

    for name in reg.factor_names:
        info = reg._factors[name]
        n_levels = int(info["n_levels"])
        proportions = info.get("proportions")
        if proportions is None:
            proportions = [1.0 / n_levels] * n_levels
        level_labels = info.get("level_labels")
        reference_level = info.get("reference_level")
        if level_labels is None:
            # Parser-driven path uses integer labels 1..n_levels with
            # reference=1; mirror that as strings for the Rust contract.
            levels = [str(i) for i in range(1, n_levels + 1)]
            reference = (
                str(reference_level) if reference_level is not None else "1"
            )
        else:
            levels = [str(lb) for lb in level_labels]
            reference = (
                str(reference_level)
                if reference_level is not None
                else levels[0]
            )
        factor_entry: Dict[str, Any] = {
            "name": name,
            "kind": "factor",
            "levels": levels,
            "proportions": [float(p) for p in proportions],
            "reference": reference,
        }
        sampled = info.get("sampled_proportions")
        if sampled is not None:
            # Omit when None so the field serde-defaults to inherit; emit a bool
            # only for an explicit per-factor override.
            factor_entry["sampled_proportions"] = bool(sampled)
        predictors.append(factor_entry)

    # Effects — skip cluster effects (random, not tested). The Rust
    # builder rebuilds them from the formula and matches by name.
    cluster_effect_names = set(reg.cluster_effect_names)
    effects: List[Dict[str, Any]] = []
    for eff in reg._effects.values():
        # Cluster effect entries always carry the synthetic id_effect
        # name as their first var_name; the more direct check is on
        # `eff.name` to avoid relying on var_names being populated.
        if eff.name in cluster_effect_names:
            continue
        effects.append({"name": eff.name, "size": float(eff.effect_size)})

    # Correlations — only emit non-zero off-diagonal entries; the Rust
    # builder fills missing pairs with 0.0 and synthesises the identity
    # when no pairs are supplied.
    correlations: List[Dict[str, Any]] = []
    corr_matrix = reg.get_correlation_matrix()
    if corr_matrix is not None:
        n_nf = len(non_factor_names)
        for i in range(n_nf):
            for j in range(i + 1, n_nf):
                value = float(corr_matrix[i][j])
                if value != 0.0:
                    correlations.append(
                        {
                            "a": non_factor_names[i],
                            "b": non_factor_names[j],
                            "value": value,
                        }
                    )

    # Correction name → the snake_case strings the engine's `Correction`
    # enum deserializes. The `find_*` entry points gate the value via
    # `_validate_correction_arg` upstream, so the alias map only normalises
    # known inputs ("bh"/"fdr" → "benjamini_hochberg", "tukey" → "tukey_hsd");
    # canonical names fall through unchanged (engine rejects anything else).
    from ..config import get_correction_aliases

    correction_key = (correction or "none").lower().replace("-", "_").replace(" ", "_")
    correction_wire = get_correction_aliases().get(correction_key, correction_key)

    # wald_se: the caller (`MCPower._to_linear_spec_dict`) has already resolved
    # None to the config `estimation.wald_se` default. "hessian" and "rx"
    # (opt-in speed knob) are the valid wire values, normalised to canonical
    # snake_case. Validation already ran in _validate_wald_se_arg; this mirrors
    # the correction normalisation pattern.
    wald_se_wire = wald_se.lower().replace("-", "_").replace(" ", "_")

    # df_method: same normalisation pattern as wald_se (the caller resolves the
    # `estimation.df_method` default). "auto" (the default), "between_within",
    # "satterthwaite" and "asymptotic" are the valid wire values; validation
    # already ran in _validate_df_method_arg.
    df_method_wire = df_method.lower().replace("-", "_").replace(" ", "_")

    # Scenarios — pre-encode through `build_scenario_dict` so the integer
    # code translation matches the existing Python projection exactly.
    scenarios = [build_scenario_dict(name, scenario_configs) for name in scenario_names]
    # RE knobs (random_effect_dist/df, icc_noise_sd) perturb the random
    # effects, so they apply to every clustered model; the engine contract
    # rejects lme: Some(...) without a cluster. Zero them out for a model
    # without random effects so the spec-builder produces lme: None there.
    if not registry._random_effects_parsed:
        for sc in scenarios:
            sc["random_effect_dist"] = 0
            sc["random_effect_df"] = 0.0
            sc["icc_noise_sd"] = 0.0

    # The Rust spec builder rejects random-effects syntax (engine-spec-
    # builder/src/formula.rs::parse_formula). Strip the `(1|group)` terms
    # before handing the formula to the builder — the cluster spec is
    # injected separately into the SimulationSpec dict in
    # ``to_simulation_spec``.
    formula_for_builder = equation
    if registry._random_effects_parsed:
        from .parsers import _parse_equation

        dep, formula_part, _ = _parse_equation(equation)
        formula_for_builder = f"{dep} ~ {formula_part}"

    # Project the per-call ``target_test`` into the three wire fields:
    # targets, contrast_pairs, report_overall.
    # When target_test is None (no selection passed): emit the family
    # default → targets=["overall"] (magic for "every β"), contrast_pairs=[],
    # report_overall only where the omnibus is defined (see
    # overall_test_available — OLS / unclustered GLM, not mixed-effects fits).
    overall_available = overall_test_available(estimator, registry)
    if target_test is not None:
        tests_resolved = resolve_tests(
            target_test, registry, overall_available=overall_available
        )
    else:
        tests_resolved = None  # will use family default below

    if tests_resolved is None:
        # Family default: every β. The omnibus is added only where it is
        # defined; mixed-effects fits report the marginals without it.
        wire_targets = ["overall"]
        wire_contrast_pairs: List[Tuple[str, str]] = []
        wire_report_overall = overall_available
        wire_posthoc_factors: List[str] = []
    else:
        user_targets = tests_resolved["targets"]
        user_contrast_pairs = tests_resolved["contrast_pairs"]
        report_overall = tests_resolved["report_overall"]
        wire_posthoc_factors = tests_resolved.get("posthoc_factors", [])

        if user_targets:
            # Explicit effect names requested → pass them; magic "overall"
            # is handled separately via report_overall.
            wire_targets = list(user_targets)
        else:
            # No explicit betas, only contrasts / posthoc.
            wire_targets = []

        wire_contrast_pairs = list(user_contrast_pairs)
        wire_report_overall = report_overall

    # Tukey HSD is a family-wide correction for factor-level contrasts. If the
    # user also requested explicit marginal β targets (named effect targets,
    # NOT the magic ["overall"] family-default shorthand), warn and proceed:
    # factor levels get Tukey-corrected power, every other target reports its
    # corrected power as n/a (the engine flags it `corrected_not_applicable`).
    # The magic ["overall"] default (target_test=None path) is not subject
    # to this warning because it signifies "all betas" without a specific
    # Tukey family intent from the user.
    _has_named_beta_targets = bool(
        wire_targets and wire_targets != ["overall"]
    )
    if correction_wire == "tukey_hsd" and _has_named_beta_targets:
        warnings.warn(
            "Tukey HSD corrects factor-level contrasts only: factor levels get "
            "Tukey-corrected power, and continuous, binary and interaction targets "
            "report corrected power as n/a.",
            UserWarning,
            stacklevel=4,
        )

    payload: Dict[str, Any] = {
        "formula": formula_for_builder,
        "predictors": predictors,
        "effects": effects,
        "correlations": correlations,
        "alpha": float(alpha),
        "correction": correction_wire,
        "wald_se": wald_se_wire,
        "nagq": int(nagq),
        "df_method": df_method_wire,
        "targets": wire_targets,
        "report_overall": wire_report_overall,
        "contrast_pairs": [list(pair) for pair in wire_contrast_pairs],
        "posthoc_requests": [{"factor": f} for f in wire_posthoc_factors],
        # heteroskedasticity carries only driver_var_index — no ratio key.
        "heteroskedasticity": dict(heteroskedasticity),
        "residual": {"distribution": residual_dist_name},
        "max_failed_fraction": float(max_failed_simulations),
        "scenarios": scenarios,
    }
    if cluster_level_vars:
        payload["cluster_level_vars"] = list(cluster_level_vars)
    if test_formula is not None:
        payload["test_formula"] = test_formula

    return payload


def encode_upload_block(
    columns_typed: List[Tuple[str, str, List[Any], List[str]]], mode: str, n_rows: int
) -> str:
    """JSON for the ``LinearSpec`` ``upload`` block, built once per
    ``upload_data`` and spliced into every payload by ``linear_spec_json``."""
    from ..data.upload import value_to_label

    upload_columns: List[Dict[str, Any]] = []
    for col_name, col_type, raw_vals, col_labels in columns_typed:
        if col_type == "factor":
            # Encode raw values as integer level codes 0..k-1 using the
            # single-sourced label renderer (keeps cyl[6]/origin[Japan]
            # aligned with detect_column_types and the dummy names).
            label_to_code = {lbl: idx for idx, lbl in enumerate(col_labels)}
            values = [float(label_to_code[value_to_label(v)]) for v in raw_vals]
        elif col_type == "binary":
            # The engine reads a binary column's proportion as the mean of its
            # values, so any two-value coding is sent as 0/1 (lower value -> 0).
            low = min(float(v) for v in raw_vals)
            values = [0.0 if float(v) == low else 1.0 for v in raw_vals]
        else:
            values = [float(v) for v in raw_vals]
        upload_columns.append({
            "name": col_name,
            "col_type": col_type,
            "values": values,
            "labels": col_labels if col_type == "factor" else [],
        })
    return json.dumps({"mode": mode, "n_rows": n_rows, "columns": upload_columns})


def linear_spec_json(payload: Dict[str, Any], upload_json: Optional[str]) -> str:
    """Serialise a ``build_linear_spec`` payload, splicing in the pre-encoded
    ``upload`` block when there is one. Re-dumping a large upload's columns
    costs seconds per ``find_*`` call, so they are encoded once at upload."""
    body = json.dumps(payload)
    if upload_json is None:
        return body
    return f'{body[:-1]}, "upload": {upload_json}}}'
