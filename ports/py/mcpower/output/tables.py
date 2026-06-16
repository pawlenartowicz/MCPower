"""Text-table and plot-label rendering for result objects.

Pure display layer (formatting, row layout, the short-form summary): formats
power / CI / required-N cells, builds the per-test and post-hoc rows from the
engine skeleton + the port label store, and renders the repr() short form.
Imports nothing from ``results`` — the dependency runs results -> tables only.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sim_fallback(key: str) -> float:
    from ..config import get_simulation_defaults
    return get_simulation_defaults()[key]


# ---------------------------------------------------------------------------
# Pure display helpers (no engine / no I/O)
# ---------------------------------------------------------------------------


def fmt_pct(x: float, decimals: int) -> str:
    """Percentage with `decimals` places, e.g. fmt_pct(0.925, 1) == '92.5%'.
    Exactly 100% drops the fractional part ('100%' not '100.0%') — the trailing
    '.0' carries no information, and omitting it lets the column reserve only two
    integer digits (a 100% value then takes one leading space, not two)."""
    pct = x * 100
    if round(pct, decimals) == 100:
        return "100%"
    return f"{pct:.{decimals}f}%"


# Main per-test table: label | power | target (no CI column — reader compares
# power and target columns directly; CI goes to the §5.1 long-form section).
# Column LABELS are resolved from config at render time (see main_power_tables).
PER_TEST_ALIGNS = ["l", "r", "r"]


def fmt_ci(ci, decimals: int) -> str:
    """CI cell with both bounds padded so the percent signs and decimals stack,
    e.g. '[99.0%,  100%]'. Bounds go through ``fmt_pct``, so exactly-100% drops
    its decimals; the column then reserves two integer digits and a 100% bound
    takes one leading space, not the two a '100.0%' would force. Empty string
    when ``ci`` is falsy.

    ``ci`` is a (lo, hi) pair — positional indexing, NOT dict keys."""
    if not ci:
        return ""
    w = 2 + (1 + decimals if decimals > 0 else 0) + 1  # 2 int digits + '.dec' + '%'
    return f"[{fmt_pct(ci[0], decimals).rjust(w)}, {fmt_pct(ci[1], decimals).rjust(w)}]"


def _search_ceiling(inner) -> "int | None":
    """Largest N actually evaluated in a find_sample_size grid (the search
    ceiling). A required N reported as unreached lies above this, so it renders
    as '≥ <ceiling>' rather than a bare '—'."""
    ss = inner.get("sample_sizes") or []
    return max(ss) if ss else None


def _required_n_headline(inner, pos) -> Tuple[str, Optional[int]]:
    """The headline display string and numeric value for one required-N cell.

    Returns (display_str, numeric_or_None) following the model-based crossing
    fallback chain:
      fitted       → (str(n_achievable), n_achievable)
      at_or_below_min → (f"≤ {n_min}", n_min)    — numeric used for footer max
      not_reached  → (f"≥ {ceiling}", None)       — no numeric: above grid
      non_monotone → first_achieved fallback       — grid value or ≥ ceiling
      missing fit  → first_achieved fallback       — older payload, silent

    Callers wanting only the display string use fmt_required_n; callers needing
    the numeric (footer max, header N≥) use this directly."""
    # int keys from the engine; the `fitted` dict is keyed by position int.
    f = (inner.get("fitted") or {}).get(pos)
    if f is not None:
        status = f.get("status")
        if status == "fitted":
            return str(f["n_achievable"]), f["n_achievable"]
        if status == "at_or_below_min":
            return f"≤ {f['n_min']}", f["n_min"]
        if status == "not_reached":
            ceiling = _search_ceiling(inner)
            return (f"≥ {ceiling}" if ceiling is not None else "—"), None
        # non_monotone: fall through to first_achieved below (grid value shown)

    # Fallback: first_achieved (also covers non_monotone and missing fitted).
    v = inner["first_achieved"].get(pos)
    if v is not None:
        return str(v), v
    ceiling = _search_ceiling(inner)
    return (f"≥ {ceiling}" if ceiling is not None else "—"), None


def fmt_required_n(inner, pos) -> str:
    """Required-N cell for one tested effect.

    Implements the model-based crossing fallback chain (fitted → at_or_below_min
    → not_reached → non_monotone/empty → first_achieved). Returns a display
    string; never a bare '—' when the search ceiling is known."""
    display, _ = _required_n_headline(inner, pos)
    return display


def fmt_target(target: float, decimals: int) -> str:
    """Target cell, e.g. '80%'. No pass/fail glyph — the reader
    compares the power and target columns directly."""
    return fmt_pct(target, decimals)


def _power_row(label, power, target, dec, tdec):
    """Build a minimal_table ('row', cells) tuple for one label|power|target line."""
    return ("row", [label, fmt_pct(power, dec), fmt_target(target, tdec)])


def minimal_table(title, columns, rows, *, name_min: int = 18, name_max: int = 44) -> str:
    """Render a minimal-rules (booktabs-style) text table.

    title:   section heading printed above the table; pass ``None`` to omit it
             (e.g. the short form, where the analysis header acts as the title).
    columns: list of ``(header, align)`` where align is ``'l'`` or ``'r'``; the
             first column is the left-aligned label column.
    rows:    list of either ``("row", [cell, ...])`` for a data row or
             ``("span", text)`` for a full-width line (e.g. a factor header)
             printed verbatim. Span rows still widen the label column so data
             lines up beneath them.

    Numeric columns are right-aligned, so per-column padding aligns the decimal
    points of values that share a decimal count. The label column auto-sizes to
    its content, clamped to ``[name_min, name_max]``."""
    headers = [h for h, _ in columns]
    aligns = [a for _, a in columns]
    widths = [max(len(h), 1) for h in headers]
    for kind, payload in rows:
        if kind == "row":
            for i, cell in enumerate(payload):
                widths[i] = max(widths[i], len(cell))
        else:
            widths[0] = max(widths[0], len(payload))
    widths[0] = max(name_min, min(widths[0], name_max))
    gap = " " * 3

    def render(cells):
        return gap.join(
            cell.ljust(widths[i]) if aligns[i] == "l" else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        )

    header_line = render(headers)
    rule = "─" * len(header_line)
    lines = ([title] if title is not None else []) + [rule, header_line, rule]
    for kind, payload in rows:
        lines.append(render(payload) if kind == "row" else payload)
    lines.append(rule)
    return "\n".join(lines)


def _text_cfg():
    from ..config import get_report_config
    return get_report_config()["text"]


def _corr_on(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("correction") and meta["correction"] != "none")


def main_power_tables(scenarios, meta, *, dec, tdec, target, caption):
    """Render the main result: a list of 1 or 2 minimal_table strings.

    Correction-or-scenarios is the single extension axis;
    only when BOTH are on does it split into two tables.
      - neither            -> [Test | Power | Target]                         (1)
      - correction only    -> [Test | uncorrected | corrected | Target]       (1)
      - scenarios only     -> [Test | <scenario cols> | Target]               (1)
      - both               -> two tables (Uncorrected / Corrected),           (2)
                              each [Test | <scenario cols> | Target]

    No CI column, no ✓/✗, no Δ column (drop -> §5.3). caption is the base
    section title (pass None for the short form to omit it)."""
    cols = _text_cfg()["columns"]
    corr = _corr_on(meta)
    multi = len(scenarios) > 1
    inner0 = scenarios[0][1]
    rows = build_rows(inner0["target_indices"], meta, inner0.get("contrast_pairs") or [])
    ph = posthoc_rows(meta) if inner0.get("posthoc") else []

    def factor_span(r):
        return ("span", f"{r['label']}  (baseline: {r['baseline']})")

    def posthoc_span(r):
        return ("span", f"{r['label']}  (pairwise)")

    def label_of(r):
        return ("  " if r["kind"] == "factor_level" else "") + r["label"]

    if not multi:
        if not corr:
            columns = [(cols["test"], "l"), (cols["power"], "r"), (cols["target"], "r")]
            table = []
            if inner0.get("overall_significant_rate") is not None:
                table.append(_power_row(_overall_label(inner0),
                                        inner0["overall_significant_rate"], target, dec, tdec))
            for r in rows:
                if r["kind"] == "factor_header":
                    table.append(factor_span(r)); continue
                table.append(_power_row(label_of(r), inner0["power_uncorrected"][0][r["pos"]],
                                        target, dec, tdec))
            for r in ph:
                if r["kind"] == "posthoc_header":
                    table.append(posthoc_span(r)); continue
                val = inner0["posthoc"][r["block"]]["power_uncorrected"][r["contrast"]]
                table.append(_power_row("  " + r["label"], val, target, dec, tdec))
            return [minimal_table(caption, columns, table)]
        # correction only: Test | uncorrected | corrected | Target
        columns = [(cols["test"], "l"), (cols["uncorrected"], "r"),
                   (cols["corrected"], "r"), (cols["target"], "r")]
        table = []

        def corr_row(label, pos_or_overall):
            if pos_or_overall == "overall":
                # The omnibus test is a single test, so multiplicity correction
                # does not apply. Show "(same)" in the corrected cell — not "—",
                # which already means "no value / target not reached" elsewhere.
                u = inner0.get("overall_significant_rate")
                return ("row", [label, fmt_pct(u, dec), "(same)", fmt_target(target, tdec)])
            u = inner0["power_uncorrected"][0][pos_or_overall]
            c = inner0["power_corrected"][0][pos_or_overall]
            return ("row", [label, fmt_pct(u, dec), fmt_pct(c, dec), fmt_target(target, tdec)])

        if inner0.get("overall_significant_rate") is not None:
            table.append(corr_row(_overall_label(inner0), "overall"))
        for r in rows:
            if r["kind"] == "factor_header":
                table.append(factor_span(r)); continue
            table.append(corr_row(label_of(r), r["pos"]))
        for r in ph:
            if r["kind"] == "posthoc_header":
                table.append(posthoc_span(r)); continue
            blk = inner0["posthoc"][r["block"]]
            table.append(("row", ["  " + r["label"],
                                  fmt_pct(blk["power_uncorrected"][r["contrast"]], dec),
                                  fmt_pct(blk["power_corrected"][r["contrast"]], dec),
                                  fmt_target(target, tdec)]))
        return [minimal_table(caption, columns, table)]

    # multi-scenario: one table per active correction state.
    names = [nm for nm, _ in scenarios]

    def build_scen_table(pkey):
        columns = [(cols["test"], "l")] + [(nm, "r") for nm in names] + [(cols["target"], "r")]
        table = []

        def scen_row(label, pos_or_overall):
            vals = []
            for _, s in scenarios:
                if pos_or_overall == "overall":
                    vals.append(s.get("overall_significant_rate"))
                else:
                    vals.append(s[pkey][0][pos_or_overall])
            if any(v is None for v in vals):
                return None
            return ("row", [label] + [fmt_pct(v, dec) for v in vals] + [fmt_target(target, tdec)])

        if inner0.get("overall_significant_rate") is not None:
            r0 = scen_row(_overall_label(inner0), "overall")
            if r0:
                table.append(r0)
        for r in rows:
            if r["kind"] == "factor_header":
                table.append(factor_span(r)); continue
            rr = scen_row(label_of(r), r["pos"])
            if rr:
                table.append(rr)
        for r in ph:
            if r["kind"] == "posthoc_header":
                table.append(posthoc_span(r)); continue
            vals, ok = [], True
            for _, s in scenarios:
                blocks = s.get("posthoc") or []
                if r["block"] >= len(blocks):
                    ok = False; break
                vals.append(blocks[r["block"]][pkey][r["contrast"]])
            if ok:
                table.append(("row", ["  " + r["label"]]
                              + [fmt_pct(v, dec) for v in vals]
                              + [fmt_target(target, tdec)]))
        return columns, table

    text = _text_cfg()
    if not corr:
        columns, table = build_scen_table("power_uncorrected")
        return [minimal_table(caption, columns, table)]
    cap_u = (caption or "") + text["uncorrected_suffix"]
    cap_c = (caption or "") + text["corrected_suffix"]
    cu, tu = build_scen_table("power_uncorrected")
    cc, tc = build_scen_table("power_corrected")
    return [minimal_table(cap_u.strip() or None, cu, tu),
            minimal_table(cap_c.strip() or None, cc, tc)]


def _factor_label(factors: Dict[str, Any], fname: str, level: int) -> str:
    """Render a factor dummy's display label from the port label store:
    ``factors[fname]['levels'][level]``. ``level`` indexes the factor's FULL
    ordered label list (reference included). Falls back to ``str(level + 1)``
    when no labels are stored (an unnamed factor renders ``1..k``)."""
    levels = (factors.get(fname) or {}).get("levels") or []
    return levels[level] if 0 <= level < len(levels) else str(level + 1)


def _render_descriptor(desc: Dict[str, Any], factors: Dict[str, Any]) -> str:
    """Render one EffectDescriptor (from the engine skeleton) to a display name
    using the port's label store: continuous → the formula identifier, factor
    dummy → ``factor[label]``, interaction → its components joined by ``:``."""
    kind = desc["kind"]
    if kind == "continuous":
        return desc["predictor"]
    if kind == "factor_level":
        return f"{desc['factor']}[{_factor_label(factors, desc['factor'], desc['level'])}]"
    if kind == "interaction":
        return ":".join(_render_descriptor(c, factors) for c in desc["components"])
    return "(Intercept)"


def _contrast_label(skeleton, factors, p: int, n: int) -> str:
    """Display label for the pairwise contrast β_p − β_n: both sides rendered
    from the skeleton, joined by the configured vs token (mirrors posthoc_rows'
    "B vs A" shape but with full effect names — the sides may belong to any
    factor)."""
    vs = _text_cfg().get("vs_token", "vs")
    return (f"{_render_descriptor(skeleton[p], factors)} {vs} "
            f"{_render_descriptor(skeleton[n], factors)}")


def build_rows(target_indices: List[int], meta: Dict[str, Any],
               contrast_pairs: List[List[int]] = ()) -> List[Dict[str, Any]]:
    """Ordered display rows: continuous predictors (and interactions) as one
    row; factor dummies as a value-less header (factor (baseline: X)) + one flat
    row per non-baseline level; then one ``contrast`` row per requested pairwise
    contrast ("B vs A"). ``pos`` indexes the result's per-target arrays — the
    engine appends contrast entries after the marginals, so contrast rows get
    ``pos = len(target_indices) + j``.

    Names are rendered from the engine's index-only ``effect_skeleton`` (β-column
    aligned, intercept at 0) plus the port's label store — no factor-expansion
    layout is re-derived here, and no effect-name strings are parsed."""
    skeleton = meta["effect_skeleton"]
    factors = meta.get("factors", {})
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for pos, idx in enumerate(target_indices):
        if idx < 0 or idx >= len(skeleton):
            raise ValueError(
                f"target_indices entry {idx} is out of range for the effect "
                f"skeleton (0..{len(skeleton) - 1}). target_indices are β̂-column "
                f"indices (intercept at 0, targets ≥ 1)."
            )
        desc = skeleton[idx]
        if desc["kind"] == "factor_level":
            fname = desc["factor"]
            if fname not in seen:
                rows.append({"kind": "factor_header", "label": fname,
                             "baseline": (factors.get(fname) or {}).get("baseline")})
                seen.add(fname)
            rows.append({"kind": "factor_level",
                         "label": _factor_label(factors, fname, desc["level"]),
                         "factor": fname, "pos": pos})
        else:
            rows.append({"kind": "continuous",
                         "label": _render_descriptor(desc, factors), "pos": pos})
    n_marginals = len(target_indices)
    for j, pair in enumerate(contrast_pairs or ()):
        p, n = pair[0], pair[1]
        if not (0 <= p < len(skeleton) and 0 <= n < len(skeleton)):
            raise ValueError(
                f"contrast_pairs entry ({p}, {n}) is out of range for the "
                f"effect skeleton (0..{len(skeleton) - 1})."
            )
        rows.append({"kind": "contrast",
                     "label": _contrast_label(skeleton, factors, p, n),
                     "pos": n_marginals + j})
    return rows


def target_label_map(result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, str]:
    """Map each plot ``target_{idx}`` token to the display label the per-test
    table shows for that target, so charts read real effect names (e.g.
    ``treatment``, ``cyl[6]``) instead of the engine's generic tokens. Reuses
    ``_render_descriptor`` — the table's own label source — so plot and table
    never diverge. Returns ``{}`` when the skeleton is unavailable (relabel then
    no-ops). ``result`` may be the single-scenario dict or the multi-scenario
    envelope; ``target_indices`` are identical across scenarios."""
    skeleton = meta.get("effect_skeleton")
    if not skeleton:
        return {}
    factors = meta.get("factors", {})
    if "target_indices" in result:
        inner = result
    else:
        scenarios = result.get("scenarios") or {}
        first = next(iter(scenarios.values()), {})
        inner = first if isinstance(first, dict) else {}
    target_indices = inner.get("target_indices", [])
    contrast_pairs = inner.get("contrast_pairs") or []
    labels: Dict[str, str] = {}
    for idx in target_indices:
        if 0 <= idx < len(skeleton):
            labels[f"target_{idx}"] = _render_descriptor(skeleton[idx], factors)
    # Contrast entries carry `target_{p}_vs_{n}` tokens (see the engine's
    # plot entry_label); label them like the table's contrast rows.
    for pair in contrast_pairs:
        p, n = pair[0], pair[1]
        if 0 <= p < len(skeleton) and 0 <= n < len(skeleton):
            labels[f"target_{p}_vs_{n}"] = _contrast_label(skeleton, factors, p, n)
    # The sample-size curve's overall-test series carries an `"overall"` token;
    # label it F-test / LRT per the estimator (from meta — the sample-size host
    # dict has no top-level estimator).
    labels["overall"] = _overall_label_by_estimator(meta.get("estimator") or "ols")
    return labels


def posthoc_rows(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered post-hoc rows for the main per-test table: one ``posthoc_header``
    span per factor followed by its canonical pairwise ``posthoc_contrast`` rows.
    ``block`` indexes ``inner['posthoc']``; ``contrast`` indexes that block's
    per-pair value arrays. ``[]`` when meta carries no posthoc_factors.

    The shape mirrors build_rows' factor-header + nested-level rows so the
    contrasts render inline as a nested group, exactly like factor levels."""
    posthoc_factors = meta.get("posthoc_factors") or []
    if not posthoc_factors:
        return []
    vs = _text_cfg().get("vs_token", "vs")
    rows: List[Dict[str, Any]] = []
    for bi, fmeta in enumerate(posthoc_factors):
        levels = fmeta["levels"]
        k = len(levels)
        rows.append({"kind": "posthoc_header", "label": fmeta["name"]})
        ci = 0
        for a in range(k):
            for b in range(a + 1, k):
                rows.append({"kind": "posthoc_contrast",
                             "label": f"{levels[b]} {vs} {levels[a]}",
                             "block": bi, "contrast": ci})
                ci += 1
    return rows


def joint_distribution(
    histogram: List[int], n_sims_used: int
) -> Optional[Dict[str, List[float]]]:
    """Derive 'exactly k' and 'at least k' from the joint-significance histogram.

    The histogram has length n_targets+1, where bucket k counts simulations in
    which exactly k targets were significant. Returns None when the histogram
    is empty or n==0 (defensive guard)."""
    if n_sims_used == 0 or not histogram:
        return None
    n = float(n_sims_used)
    exactly = [h / n for h in histogram]
    total = sum(histogram)
    at_least = []
    running = total
    for h in histogram:
        at_least.append(running / n)
        running -= h
    return {"exactly": exactly, "at_least": at_least}


def _scenarios(result: Dict[str, Any]):
    """Normalise to an ordered list of (name, inner_result). Single-scenario
    results (no 'scenarios' key) become a one-element list."""
    if "scenarios" in result and isinstance(result["scenarios"], dict):
        return list(result["scenarios"].items())
    return [(result.get("scenario", "default"), result)]


def _overall_label_by_estimator(est: str) -> str:
    """Overall/omnibus row label for an estimator token (F-test for OLS, LRT for
    GLM, Wald for MLE) from config. Used by both find_power (estimator read off
    the result) and find_sample_size (estimator read off meta — the sample-size
    host dict carries no top-level estimator)."""
    from ..config import get_report_config
    return get_report_config()["overall_label_by_estimator"].get(est or "ols", "Overall")


def _overall_label(inner: Dict[str, Any]) -> str:
    est = inner.get("estimator_extras", {}).get("estimator", "ols")
    return _overall_label_by_estimator(est)


def _overall_required_n_headline(inner: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Headline display + numeric for the overall-test required-N cell — the same
    model-based crossing fallback chain as `_required_n_headline`, but for the
    single `fitted_overall` CrossingFit (no `pos`) with `first_overall_achieved`
    as the grid-empirical fallback. Callers gate the row on presence first."""
    fo = inner.get("fitted_overall") or {}
    f = fo.get(0)
    if f is None:
        f = fo.get("0")  # str-keyed (JSON) payloads
    if f is not None:
        status = f.get("status")
        if status == "fitted":
            return str(f["n_achievable"]), f["n_achievable"]
        if status == "at_or_below_min":
            return f"≤ {f['n_min']}", f["n_min"]
        if status == "not_reached":
            ceiling = _search_ceiling(inner)
            return (f"≥ {ceiling}" if ceiling is not None else "—"), None
        # non_monotone: fall through to first_overall_achieved
    v = inner.get("first_overall_achieved")
    if v is not None:
        return str(v), v
    ceiling = _search_ceiling(inner)
    return (f"≥ {ceiling}" if ceiling is not None else "—"), None


def _has_overall_required_n(inner: Dict[str, Any]) -> bool:
    """True when the sample-size result carried an overall test (OLS F /
    unclustered GLM LRT): `fitted_overall` is non-empty, or the grid-empirical
    `first_overall_achieved` is set. Both absent ⇒ mixed/GLMM suppressed it."""
    return bool(inner.get("fitted_overall")) or inner.get("first_overall_achieved") is not None


def _model_summary_block(meta, *, est_label, n, n_sims, alpha, target, tdec, scenarios):
    """Compact header for the short form: one analysis line +
    formula line + correction/scenarios lines when those features are on."""
    lines = [f"Power Analysis — {est_label}  N={n}  sims={n_sims}  "
             f"α={alpha}  target={fmt_pct(target, tdec)}",
             f"formula: {meta.get('formula', '')}"]
    if _corr_on(meta):
        lines.append(f"correction: {meta['correction']}")
    if len(scenarios) > 1:
        lines.append("scenarios: " + ", ".join(nm for nm, _ in scenarios))
    return "\n".join(lines)


def render_short(result: Dict[str, Any], meta: Dict[str, Any], *, kind: str) -> str:
    from ..config import get_report_config
    cfg = get_report_config()
    if kind == "find_sample_size":
        return _render_sample_size_short(result, meta, cfg)
    dec = cfg["format"]["power_decimals_short"]
    tdec = cfg["format"]["target_decimals"]
    scenarios = _scenarios(result)
    inner0 = scenarios[0][1]
    target = meta.get("target_power", _sim_fallback("target_power"))
    est_label = (meta.get("estimator") or inner0.get("estimator_extras", {}).get("estimator", "ols")).upper()
    n = inner0["sample_sizes"][0] if inner0.get("sample_sizes") else inner0.get("n", "?")
    n_sims = inner0.get("n_sims", "?")
    alpha = meta.get("alpha", _sim_fallback("alpha"))

    header = _model_summary_block(meta, est_label=est_label, n=n, n_sims=n_sims,
                                  alpha=alpha, target=target, tdec=tdec, scenarios=scenarios)
    tables = main_power_tables(scenarios, meta, dec=dec, tdec=tdec, target=target, caption=None)
    out = header + "\n\n" + "\n\n".join(tables)
    factor_names = list(meta.get("factors", {}))
    baseline_req = meta.get("baseline_prob_requested")
    min_cluster_size = meta.get("min_cluster_size")
    # Diagnose every scenario, not just scenario 0 — a degraded sweep scenario is
    # the whole point of running one. Prefix each message with its scenario name
    # when >1 (mirrors report.py:_diagnostics; change together).
    multi = len(scenarios) > 1
    warns = []
    for nm, scen in scenarios:
        for w in diagnostic_warnings(scen, factor_names=factor_names,
                                     baseline_prob_requested=baseline_req,
                                     min_cluster_size=min_cluster_size):
            warns.append(f"! {f'{nm}: {w}' if multi else w} — see summary()")
    return out + ("\n" + "\n".join(warns) if warns else "")


def _baseline_index(names: List[str], cfg: Dict[str, Any]) -> int:
    prefer = cfg["baseline_scenario"]["prefer_label"]
    if prefer in names:
        return names.index(prefer)
    return 0  # fallback_to_first


def _max_exclusion_rates(inner: Dict[str, Any], key: str) -> List[float]:
    """Worst-case-over-the-sweep per-factor exclusion rate (diagnostics charter:
    per-N signals reduce to the single worst point).

    find_power: counts is a flat list (one vec) — shape [n_factors].
    find_sample_size: counts is a list of lists (one vec per grid point) —
    shape [n_grid_points][n_factors]. Both normalised by n_sims (scalar)."""
    counts = inner.get(key) or []
    n_sims = inner.get("n_sims") or 0
    if not counts or not n_sims:
        return []
    per_n = counts if isinstance(counts[0], (list, tuple)) else [counts]
    return [max(row[f] for row in per_n) / n_sims for f in range(len(per_n[0]))]


def diagnostic_warnings(
    inner: Dict[str, Any],
    factor_names: Optional[List[str]] = None,
    *,
    baseline_prob_requested: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
) -> List[str]:
    """Core diagnostic message per configured threshold that trips; empty when
    clean. Callers add their own prefix/suffix (short form: '! … — see
    summary()'; long form: '! …' under a '⚠ Diagnostics' heading).

    baseline_prob_requested / min_cluster_size are meta-level (one per run): the
    requested GLM event probability (from set_baseline_probability) and the
    smallest cluster size at the evaluated N. Both are None unless the run is a
    binary-outcome GLM / GLMM; they drive the GLM-drift and Laplace-bias gates,
    comparing against the realized values the per-scenario `inner` carries.

    Faithful mirror of .diagnostic_warnings in ports/r/R/output-report.R — the
    gate set, message wording, and order must change together across both."""
    from ..config import get_report_config
    th = get_report_config()["thresholds"]
    warns: List[str] = []
    conv = inner.get("convergence_rate", 1.0)
    # convergence_rate is a list (one per N) in real results; use the worst.
    conv_scalar = min(conv) if isinstance(conv, (list, tuple)) else conv
    if conv_scalar < th["convergence_min"]:
        warns.append(f"convergence {fmt_pct(conv_scalar, 1)}")
    # Boundary-hit gates on high-τ̂ (boundary_hit==2: τ̂ pinned implausibly large,
    # or GLMM optimizer/Schur failure) ONLY. Benign τ̂=0 ("singular fit", common
    # at small ICC) is expected, not a red flag — it stays informational, already
    # surfaced as singular_fit_rate in the estimator-extras block.
    bh_ht = inner.get("boundary_hit_rate_high_tau")
    if bh_ht is None:
        raise AssertionError(
            "diagnostic_warnings reached the raw-boundary_hit fallback branch, "
            "which should be unreachable when add_boundary_hit_rates() has been called. "
            "Call add_boundary_hit_rates(result) before diagnostic_warnings(inner)."
        )
    hit_rate = max(float(ht) for ht in bh_ht) if bh_ht else 0.0
    if hit_rate > th["lme_boundary_hit_max"]:
        warns.append(f"high-τ̂ boundary {fmt_pct(hit_rate, 1)}")
    extras = inner.get("estimator_extras", {})
    if (baseline_prob_requested is not None
            and extras.get("estimator") == "glm"
            and "baseline_prob_realized" in extras):
        drift = abs(extras["baseline_prob_realized"] - baseline_prob_requested)
        if drift > th["glm_baseline_drift_max"]:
            warns.append(f"GLM baseline drift {drift:.3f}")
    th_excl = th["factor_exclusion_max"]
    for key, verb in (("factor_exclusion_counts", "excluded"),
                      ("factor_separation_counts", "separation-dropped")):
        for f, rate in enumerate(_max_exclusion_rates(inner, key)):
            if rate > th_excl:
                name = (factor_names[f] if factor_names and f < len(factor_names)
                        else f"factor {f + 1}")
                warns.append(f"{name} {verb} {rate:.1%} of sims")
    # Laplace-approximation bias: GLMM (Glm + small clusters) with large τ̂².
    # Reuses the canonical message helper so this persistent report line and the
    # transient fit-time warnings.warn (model.py) never diverge. min_cluster_size
    # is None for non-GLMM runs → check skipped.
    if min_cluster_size is not None:
        from ..config import get_config
        from ..model import _glmm_laplace_bias_warning
        lap = _glmm_laplace_bias_warning(extras, min_cluster_size, get_config())
        if lap is not None:
            warns.append(lap)
    return warns


def _render_sample_size_short(result: Dict[str, Any], meta: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Short form for find_sample_size: Required N column(s).
    Correction is a search parameter, shown in the header only; scenarios are
    the sole main-table axis."""
    cols = cfg["text"]["columns"]
    scenarios = _scenarios(result)
    name0, inner0 = scenarios[0]
    rows = build_rows(inner0["target_indices"], meta, inner0.get("contrast_pairs") or [])
    tdec = cfg["format"]["target_decimals"]
    target = meta.get("target_power", _sim_fallback("target_power"))
    est_label = (meta.get("estimator") or "ols").upper()
    alpha = meta.get("alpha", _sim_fallback("alpha"))
    head = (f"Power Analysis (sample size) — {est_label}  "
            f"target={fmt_pct(target, tdec)}  α={alpha}")
    if _corr_on(meta):
        head += f"\ncorrection: {meta['correction']}"
    if len(scenarios) > 1:
        head += "\nscenarios: " + ", ".join(nm for nm, _ in scenarios)

    # Overall (omnibus) required-N row first, mirroring the find_power short form
    # and the long-form report. Estimator label from meta (the sample-size host
    # dict carries no top-level estimator).
    overall_label = _overall_label_by_estimator(meta.get("estimator") or "ols")
    table = []
    if len(scenarios) == 1:
        columns = [(cols["test"], "l"), (cols["required_n"], "r")]
        if _has_overall_required_n(inner0):
            table.append(("row", [overall_label, _overall_required_n_headline(inner0)[0]]))
        for row in rows:
            if row["kind"] == "factor_header":
                table.append(("span", f"{row['label']}  (baseline: {row['baseline']})")); continue
            label = ("  " if row["kind"] == "factor_level" else "") + row["label"]
            table.append(("row", [label, fmt_required_n(inner0, row["pos"])]))
    else:
        names = [nm for nm, _ in scenarios]
        columns = [(cols["test"], "l")] + [(nm, "r") for nm in names]
        if any(_has_overall_required_n(s) for _, s in scenarios):
            table.append(("row", [overall_label] + [_overall_required_n_headline(s)[0] for _, s in scenarios]))
        for row in rows:
            if row["kind"] == "factor_header":
                table.append(("span", f"{row['label']}  (baseline: {row['baseline']})")); continue
            label = ("  " if row["kind"] == "factor_level" else "") + row["label"]
            table.append(("row", [label] + [fmt_required_n(s, row["pos"]) for _, s in scenarios]))
    footers = []
    # Collect non_monotone warnings across scenarios (label → max_violation).
    non_monotone_items: List[Tuple[str, float]] = []
    for nm, inner in scenarios:
        # Compute per-row numeric headlines to find the footer max.
        numerics: List[Optional[int]] = []
        has_not_reached = False
        for row in rows:
            if row["kind"] == "factor_header":
                continue
            _, num = _required_n_headline(inner, row["pos"])
            numerics.append(num)
            if num is None:
                has_not_reached = True
        if not has_not_reached and numerics and all(n is not None for n in numerics):
            footers.append(str(max(numerics)))  # type: ignore[arg-type]
        else:
            ceiling = _search_ceiling(inner)
            footers.append(f"≥ {ceiling}" if ceiling is not None else "—")

        # Collect non_monotone warnings for this scenario's targets.
        fitted_map = inner.get("fitted") or {}
        for row in rows:
            if row["kind"] == "factor_header":
                continue
            f = fitted_map.get(row["pos"])
            if f is not None and f.get("status") == "non_monotone":
                label = ("  " if row["kind"] == "factor_level" else "") + row["label"]
                non_monotone_items.append((label.strip(), f["max_violation"]))

    text = _text_cfg()
    body = (head + "\n\n" + minimal_table(None, columns, table)
            + "\n\nFirst N achieving all targets: " + " / ".join(footers))
    if non_monotone_items:
        warn_lines = [
            text["non_monotone_warning"].format(label=label, drop=f"{drop:.3f}")
            for label, drop in non_monotone_items
        ]
        body += "\n" + "\n".join(warn_lines)
    return body
