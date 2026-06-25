"""Long-form report (`result.summary()`): plain text in any terminal, text + inline Vega-Lite plots in Jupyter."""

import math
from typing import Any, Dict

from ..config import get_report_config
from .tables import (
    build_rows, diagnostic_warnings, fmt_ci, fmt_pct, fmt_required_n,
    joint_distribution, main_power_tables, minimal_table, target_label_map,
    _has_overall_required_n, _overall_label, _overall_label_by_estimator,
    _overall_required_n_headline, _required_n_headline, _scenarios,
    _search_ceiling, _text_cfg,
)


def _sim_fallback(key: str) -> float:
    from ..config import get_simulation_defaults
    return get_simulation_defaults()[key]


class Report:
    def __init__(self, result: Dict[str, Any], meta: Dict[str, Any], *, kind: str):
        self._result = result
        self._meta = meta
        self._kind = kind

    # ---- text (works in any terminal) ----
    def __str__(self) -> str:
        cfg = get_report_config()
        parts = [self._header(cfg)]
        parts.append(self._per_test_power(cfg))
        ci = self._ci_section(cfg)
        if ci:
            parts.append(ci)
        ci_n = self._required_n_ci_table(cfg)
        if ci_n:
            parts.append(ci_n)
        joint = self._joint_section(cfg)
        if joint:
            parts.append(joint)
        robustness = self._robustness(cfg)
        if robustness:
            parts.append(robustness)
        extras = self._estimator_extras(cfg)
        if extras:
            parts.append(extras)
        diagnostics = self._diagnostics(cfg)
        if diagnostics:
            parts.append(diagnostics)
        parts.append(self._plot_footer())
        return "\n\n".join(parts)

    __repr__ = __str__

    def _header(self, cfg) -> str:
        m = self._meta
        text = cfg["text"]
        scen = _scenarios(self._result)[0][1]
        if self._kind == "find_sample_size":
            # Use the headline-column max (same semantics as the short footer)
            # so the boxed header always matches what the Required N table shows.
            rows = build_rows(scen.get("target_indices", []), self._meta,
                              scen.get("contrast_pairs") or [])
            numerics = []
            has_not_reached = False
            for r in rows:
                if r["kind"] == "factor_header":
                    continue
                _, num = _required_n_headline(scen, r["pos"])
                numerics.append(num)
                if num is None:
                    has_not_reached = True
            ceiling = _search_ceiling(scen)
            if not has_not_reached and numerics and all(n is not None for n in numerics):
                n_label = f"N≥{max(numerics)}"  # type: ignore[arg-type]
            elif ceiling is not None:
                n_label = f"N≥{ceiling} (not all reached)"
            else:
                n_label = "N=— (target not reached)"
        else:
            n_raw = scen["sample_sizes"][0] if scen.get("sample_sizes") else scen.get("n", "?")
            n_label = f"N={n_raw}"
        title = text["long_title"]
        box = "=" * max(len(title) + 4, 50)
        target = m.get("target_power", _sim_fallback("target_power"))
        tdec = cfg["format"]["target_decimals"]
        lines = [box, f"  {title}", box, f"formula: {m.get('formula', '')}",
                 f"estimator: {(m.get('estimator') or 'ols').upper()}  {n_label}  "
                 f"sims={scen.get('n_sims', '?')}  α={m.get('alpha', _sim_fallback('alpha'))}  "
                 f"target={fmt_pct(target, tdec)}"]
        names = m.get("effect_names", [])
        sizes = m.get("effect_sizes", [])
        if names and sizes:
            # Logit-outcome models echo the OR = exp(β) beside each β; β is the
            # wire truth, OR a display-only readout (see tables._fmt_or).
            if m.get("outcome_kind") == "binary":
                pairs = (f"{n}={s:.2f} (OR {math.exp(s):.2f})" for n, s in zip(names, sizes))
            else:
                pairs = (f"{n}={s:.2f}" for n, s in zip(names, sizes))
            lines.append("effects: " + ", ".join(pairs))
        if m.get("correction") and m["correction"] != "none":
            lines.append(f"correction: {m['correction']}")
        if m.get("residual") and m["residual"] != "normal":
            lines.append(f"residual: {m['residual']}")
        return "\n".join(lines)

    def _per_test_power(self, cfg) -> str:
        if self._kind == "find_sample_size":
            return self._required_n_table(cfg)
        from .tables import main_power_tables
        dec = cfg["format"]["power_decimals_long"]
        tdec = cfg["format"]["target_decimals"]
        scenarios = _scenarios(self._result)
        target = self._meta.get("target_power", _sim_fallback("target_power"))
        tables = main_power_tables(scenarios, self._meta, dec=dec, tdec=tdec,
                                   target=target, caption=cfg["text"]["main_caption"])
        return "\n\n".join(tables)

    def _ci_section(self, cfg) -> str:
        """One Power & 95% CI table per scenario (find_power only).
        Uses corrected values when correction is on."""
        if self._kind == "find_sample_size":
            return ""
        text = cfg["text"]; cols = text["columns"]
        dec = cfg["format"]["power_decimals_long"]
        scenarios = _scenarios(self._result)
        corr = bool(self._meta.get("correction") and self._meta["correction"] != "none")
        pkey = "power_corrected" if corr else "power_uncorrected"
        ckey = "ci_corrected" if corr else "ci_uncorrected"
        columns = [(cols["test"], "l"), (cols["power"], "r"), (cols["ci"], "r")]
        blocks = []
        for nm, scen in scenarios:
            rows = build_rows(scen["target_indices"], self._meta,
                              scen.get("contrast_pairs") or [])
            table = []
            if scen.get("overall_significant_rate") is not None:
                table.append(("row", [_overall_label(scen),
                                      fmt_pct(scen["overall_significant_rate"], dec),
                                      fmt_ci(scen.get("overall_significant_ci"), dec)]))
            for r in rows:
                if r["kind"] == "factor_header":
                    table.append(("span", f"{r['label']}  (baseline: {r['baseline']})")); continue
                pos = r["pos"]
                label = ("  " if r["kind"] == "factor_level" else "") + r["label"]
                table.append(("row", [label, fmt_pct(scen[pkey][0][pos], dec),
                                      fmt_ci(scen[ckey][0][pos], dec)]))
            caption = text["ci_caption"] + (f" — {nm}" if len(scenarios) > 1 else "")
            footnote = text["ci_footnote"].format(n_sims=scen.get("n_sims", "?"))
            blocks.append(minimal_table(caption, columns, table) + "\n" + footnote)
        return "\n\n".join(blocks)

    def _required_n_ci_table(self, cfg) -> str:
        """Required N & 95% CI table from the model-based crossing fit.

        Rendered only for find_sample_size and only when `fitted` data is
        present (non-empty). Mirrors the position of _ci_section for find_power:
        one table per scenario when >1 scenarios, caption appended with scenario
        name. CI bounds are rounded outward (floor/ceil) to integers — only
        integers are shown (n_star is never printed)."""
        if self._kind != "find_sample_size":
            return ""
        text = _text_cfg()
        cols = text["columns"]
        scenarios = _scenarios(self._result)
        rows = build_rows(scenarios[0][1]["target_indices"], self._meta,
                          scenarios[0][1].get("contrast_pairs") or [])
        ceiling = _search_ceiling(scenarios[0][1])
        floor_n = min(scenarios[0][1].get("sample_sizes") or [0]) if scenarios[0][1].get("sample_sizes") else 0
        # Skip the section entirely when no scenario has fitted data.
        if not any(inner.get("fitted") for _, inner in scenarios):
            return ""

        columns = [(cols["test"], "l"), (cols["required_n"], "r"), (cols["ci"], "r")]
        overall_label = _overall_label_by_estimator(self._meta.get("estimator") or "ols")
        blocks = []
        for nm, scen in scenarios:
            fitted_map = scen.get("fitted") or {}
            if not fitted_map:
                continue
            table = []
            has_appr = False
            non_monotone_labels = []
            has_floor = False
            # Overall (omnibus) row first — same status dispatch as the per-target
            # loop below, but on the single fitted_overall CrossingFit. Folds its
            # footnote flag (appr / floor / non_monotone) into the section flags.
            if _has_overall_required_n(scen):
                fo = scen.get("fitted_overall") or {}
                of = fo.get(0)
                if of is None:
                    of = fo.get("0")
                ostatus = of.get("status") if of is not None else None
                if of is None or ostatus == "non_monotone":
                    v, _n = _overall_required_n_headline(scen)
                    table.append(("row", [overall_label, v, "—"]))
                    if ostatus == "non_monotone":
                        non_monotone_labels.append(overall_label)
                elif ostatus == "fitted":
                    o_lo = of.get("ci_lo")
                    o_hi = of.get("ci_hi")
                    if o_lo is None and o_hi is None:
                        o_cell = f"[≤ {floor_n}, ≥ {ceiling}]" if ceiling else "—"
                        has_floor = True
                    elif o_lo is None:
                        o_cell = f"[≤ {floor_n}, {math.ceil(o_hi)}]"
                        has_floor = True
                    elif o_hi is None:
                        o_cell = f"[{math.floor(o_lo)}, ≥ {ceiling}]" if ceiling else f"[{math.floor(o_lo)}, —]"
                    else:
                        o_cell = f"[{math.floor(o_lo)}, {math.ceil(o_hi)}]"
                    table.append(("row", [overall_label, str(of["n_achievable"]), o_cell]))
                elif ostatus == "at_or_below_min":
                    table.append(("row", [overall_label, f"≤ {of['n_min']}", "—"]))
                    has_floor = True
                elif ostatus == "not_reached":
                    o_appr = of.get("n_approx")
                    o_cell = f"appr. {o_appr}" if o_appr else "—"
                    if o_appr:
                        has_appr = True
                    table.append(("row", [overall_label, f"≥ {ceiling}" if ceiling else "—", o_cell]))
            for r in rows:
                if r["kind"] == "factor_header":
                    table.append(("span", f"{r['label']}  (baseline: {r['baseline']})")); continue
                label = ("  " if r["kind"] == "factor_level" else "") + r["label"]
                pos = r["pos"]
                f = fitted_map.get(pos)
                if f is None:
                    # Older payload or target not fit — use first_achieved fallback.
                    v, _ = _required_n_headline(scen, pos)
                    table.append(("row", [label, v, "—"]))
                    continue
                status = f.get("status")
                if status == "fitted":
                    headline = str(f["n_achievable"])
                    ci_lo = f.get("ci_lo")
                    ci_hi = f.get("ci_hi")
                    if ci_lo is None and ci_hi is None:
                        ci_cell = f"[≤ {floor_n}, ≥ {ceiling}]" if ceiling else "—"
                        has_floor = True
                    elif ci_lo is None:
                        ci_cell = f"[≤ {floor_n}, {math.ceil(ci_hi)}]"
                        has_floor = True
                    elif ci_hi is None:
                        ci_cell = f"[{math.floor(ci_lo)}, ≥ {ceiling}]" if ceiling else f"[{math.floor(ci_lo)}, —]"
                    else:
                        ci_cell = f"[{math.floor(ci_lo)}, {math.ceil(ci_hi)}]"
                    table.append(("row", [label, headline, ci_cell]))
                elif status == "at_or_below_min":
                    table.append(("row", [label, f"≤ {f['n_min']}", "—"]))
                    has_floor = True
                elif status == "not_reached":
                    n_appr = f.get("n_approx")
                    ci_cell = f"appr. {n_appr}" if n_appr else "—"
                    if n_appr:
                        has_appr = True
                    table.append(("row", [label, f"≥ {ceiling}" if ceiling else "—", ci_cell]))
                elif status == "non_monotone":
                    # Use first_achieved fallback for headline.
                    v, _ = _required_n_headline(scen, pos)
                    table.append(("row", [label, v, "—"]))
                    non_monotone_labels.append(label.strip())

            caption = text["required_n_ci_caption"] + (f" — {nm}" if len(scenarios) > 1 else "")
            footnote = text["required_n_ci_footnote"]
            if has_appr:
                footnote += "  " + text["required_n_ci_footnote_appr"]
            if non_monotone_labels:
                footnote += "  " + text["required_n_ci_footnote_suppressed"].format(
                    labels=", ".join(non_monotone_labels)
                )
            if has_floor:
                footnote += "  " + text["required_n_ci_footnote_floor"]
            blocks.append(minimal_table(caption, columns, table) + "\n" + footnote)
        return "\n\n".join(blocks)

    def _estimator_extras(self, cfg) -> str:
        """Surface GLM/MLE numerics whenever present (not only on a
        threshold trip). OLS carries only {'estimator': ...} -> nothing shown.
        One block per scenario (mirroring the per-scenario power table) so a doomer scenario hitting
        separation isn't hidden behind the first scenario's numbers."""
        def fmt_val(v):
            return f"{v:.4g}" if isinstance(v, float) else str(v)
        scenarios = _scenarios(self._result)
        caption = cfg["text"]["estimator_extras_caption"]
        blocks = []
        for nm, scen in scenarios:
            extras = dict(scen.get("estimator_extras", {}))
            extras.pop("estimator", None)
            if not extras:
                continue
            head = caption + (f" — {nm}" if len(scenarios) > 1 else "")
            lines = [f"  {k}: {fmt_val(v)}" for k, v in extras.items()]
            blocks.append(head + "\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def _required_n_table(self, cfg) -> str:
        cols = cfg["text"]["columns"]
        scenarios = _scenarios(self._result)
        rows = build_rows(scenarios[0][1]["target_indices"], self._meta,
                          scenarios[0][1].get("contrast_pairs") or [])
        # Overall (omnibus) required-N row first, positioned like the find_power
        # overall row. The estimator label comes from meta (the sample-size host
        # dict carries no top-level estimator).
        overall_label = _overall_label_by_estimator(self._meta.get("estimator") or "ols")
        if len(scenarios) == 1:
            scen = scenarios[0][1]
            table = []
            if _has_overall_required_n(scen):
                table.append(("row", [overall_label, _overall_required_n_headline(scen)[0]]))
            for r in rows:
                if r["kind"] == "factor_header":
                    table.append(("span", f"{r['label']}  (baseline: {r['baseline']})")); continue
                label = ("  " if r["kind"] == "factor_level" else "") + r["label"]
                table.append(("row", [label, fmt_required_n(scen, r["pos"])]))
            return minimal_table(cfg["text"]["sample_size_caption"],
                                 [(cols["test"], "l"), (cols["required_n"], "r")], table)
        names = [nm for nm, _ in scenarios]
        columns = [(cols["test"], "l")] + [(nm, "r") for nm in names]
        table = []
        if any(_has_overall_required_n(s) for _, s in scenarios):
            cells = [overall_label] + [_overall_required_n_headline(s)[0] for _, s in scenarios]
            table.append(("row", cells))
        for r in rows:
            if r["kind"] == "factor_header":
                table.append(("span", f"{r['label']}  (baseline: {r['baseline']})")); continue
            label = ("  " if r["kind"] == "factor_level" else "") + r["label"]
            cells = [label] + [fmt_required_n(s, r["pos"]) for _, s in scenarios]
            table.append(("row", cells))
        return minimal_table(cfg["text"]["sample_size_caption"],
                             columns, table)

    def _joint_section(self, cfg) -> str:
        if self._kind == "find_sample_size":
            return self._joint_required_n_table(cfg)
        scen = _scenarios(self._result)[0][1]
        jd = joint_distribution(
            scen.get("success_count_histogram_uncorrected", []),
            scen.get("n_sims", 0),
        )
        if jd is None:
            return "Joint significance distribution is unavailable for this result."
        dec = cfg["format"]["joint_table_decimals"]
        table = [
            ("row", [str(k), fmt_pct(ex, dec), fmt_pct(al, dec)])
            for k, (ex, al) in enumerate(zip(jd["exactly"], jd["at_least"]))
        ]
        return minimal_table("Joint significance distribution",
                             [("k", "l"), ("Exactly", "r"), ("At least", "r")], table, name_min=3)

    def _joint_required_n_table(self, cfg) -> str:
        scen = _scenarios(self._result)[0][1]
        fja = scen.get("first_joint_achieved", {})
        if not fja:
            return ""
        target = self._meta.get("target_power", _sim_fallback("target_power"))
        tdec = cfg["format"]["target_decimals"]
        ceiling = _search_ceiling(scen)
        fitted_joint = scen.get("fitted_joint") or {}
        n_targets = len(fja)
        table = []
        for j in range(n_targets - 1, -1, -1):
            k = j + 1
            # fitted_joint is keyed by int position j (parallel to first_joint_achieved).
            fj = fitted_joint.get(j)
            if fj is None:
                # Try str key for older payloads that come through as string-keyed dicts.
                fj = fitted_joint.get(str(j))
            if fj is not None:
                status = fj.get("status")
                if status == "fitted":
                    cell = str(fj["n_achievable"])
                elif status == "at_or_below_min":
                    cell = f"≤ {fj['n_min']}"
                elif status == "not_reached":
                    cell = f"≥ {ceiling}" if ceiling is not None else "—"
                else:
                    # non_monotone: fall back to first_joint_achieved.
                    n_req = fja.get(j)
                    if n_req is None:
                        n_req = fja.get(str(j))
                    cell = str(n_req) if n_req is not None else (f"≥ {ceiling}" if ceiling is not None else "—")
            else:
                # No fitted_joint (older payload) — use first_joint_achieved.
                n_req = fja.get(j)
                if n_req is None:
                    n_req = fja.get(str(j))
                cell = str(n_req) if n_req is not None else (f"≥ {ceiling}" if ceiling is not None else "—")
            table.append(("row", [f"≥ {k} of {n_targets} tests", cell]))
        return minimal_table(f"Joint detection → required N (target {fmt_pct(target, tdec)})",
                             [("Joint target", "l"), ("Required N", "r")], table)

    def _robustness(self, cfg) -> str:
        scenarios = _scenarios(self._result)
        if len(scenarios) < 2:
            return ""  # robustness section only shown when >=2 scenarios are present
        names = [nm for nm, _ in scenarios]
        prefer = cfg["baseline_scenario"]["prefer_label"]
        base_idx = names.index(prefer) if prefer in names else 0
        dec = cfg["format"]["drop_decimals"]
        corr = bool(self._meta.get("correction") and self._meta["correction"] != "none")
        pkey = "power_corrected" if corr else "power_uncorrected"
        rows = build_rows(scenarios[0][1]["target_indices"], self._meta,
                          scenarios[0][1].get("contrast_pairs") or [])
        other = [(nm, inner) for nm, inner in scenarios if nm != names[base_idx]]
        columns = [("Test", "l")] + [(nm, "r") for nm, _ in other]
        table = []
        for r in rows:
            if r["kind"] == "factor_header":
                table.append(("span", f"{r['label']}  (baseline: {r['baseline']})"))
                continue
            pos = r["pos"]
            base = scenarios[base_idx][1][pkey][0][pos]
            label = ("  " if r["kind"] == "factor_level" else "") + r["label"]
            cells = [label] + [f"{(inner[pkey][0][pos] - base) * 100:+.{dec}f} pp"
                               for _, inner in other]
            table.append(("row", cells))
        return minimal_table(f"Robustness  (Δ power vs baseline: {names[base_idx]})", columns, table)

    def _diagnostics(self, cfg) -> str:
        # Diagnostics surface only when a configured threshold trips; a healthy
        # run shows nothing. Informational GLM/MLE extras live in the result's
        # estimator_extras, not in the printed report. Every scenario is checked
        # (a degraded sweep scenario is the point of robustness runs); messages
        # carry a "{scenario}: " prefix when >1 (mirrors tables.render_short).
        scenarios = _scenarios(self._result)
        factor_names = list(self._meta.get("factors", {}))
        baseline_req = self._meta.get("baseline_prob_requested")
        min_cluster_size = self._meta.get("min_cluster_size")
        multi = len(scenarios) > 1
        warns = []
        for nm, scen in scenarios:
            for w in diagnostic_warnings(scen, factor_names=factor_names,
                                         baseline_prob_requested=baseline_req,
                                         min_cluster_size=min_cluster_size):
                warns.append(f"{nm}: {w}" if multi else w)
        if not warns:
            return ""
        return "⚠ Diagnostics\n" + "\n".join(f"! {w}" for w in warns)

    def _plot_footer(self) -> str:
        return ("Plots: result.plot() to view, "
                "result.plot('chart.png') to save.")

    # ---- Jupyter rich repr ----

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        from . import plotting
        spec = plotting.mimebundle_spec(
            self._result, self._meta, self._kind,
            label_map=target_label_map(self._result, self._meta),
        )
        return {
            "text/plain": str(self),
            "application/vnd.vegalite.v5+json": spec,
        }
