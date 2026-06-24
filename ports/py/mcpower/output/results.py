"""Post-processing helpers for engine results: folds LME boundary-hit rates into the result envelope and enforces the convergence-failure threshold."""

from typing import Any, Dict, List

from .tables import _scenarios, build_rows, render_short


def add_boundary_hit_rates(result: Dict[str, Any]) -> Dict[str, Any]:
    """Inject ``boundary_hit_rate_tau_zero`` and ``boundary_hit_rate_high_tau``
    into a raw engine result dict (or each scenario dict inside a
    multi-scenario envelope).

    ``boundary_hit`` arrives from msgpack as a nested list of shape
    ``(n_sims, n_sample_sizes)`` (uint8 values 0/1/2), or a 1-D list, or absent.

    Rates are returned as lists of floats (one per sample size), matching the
    shape of ``convergence_rate``. For OLS/Logit models ``boundary_hit`` is
    always zero so both lists contain only 0.0 values.

    The function mutates *and* returns the dict so callers can use it inline.
    """
    if "scenarios" in result:
        for scenario_dict in result["scenarios"].values():
            add_boundary_hit_rates(scenario_dict)
        return result

    bh = result.get("boundary_hit")
    if not bh:
        # None or empty: zero-fill against convergence_rate length.
        n_ss = len(result.get("convergence_rate", [1]))
        result["boundary_hit_rate_tau_zero"] = [0.0] * n_ss
        result["boundary_hit_rate_high_tau"] = [0.0] * n_ss
        return result

    # Normalise to list-of-rows; a 1-D list is a single sample-size column.
    if not isinstance(bh[0], (list, tuple)):
        bh = [[v] for v in bh]

    n_sims = len(bh)
    n_ss = len(bh[0])
    result["boundary_hit_rate_tau_zero"] = [
        sum(1 for row in bh if row[j] == 1) / n_sims for j in range(n_ss)
    ]
    result["boundary_hit_rate_high_tau"] = [
        sum(1 for row in bh if row[j] == 2) / n_sims for j in range(n_ss)
    ]
    return result


def unwrap_scenario_result(raw: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    """Unwrap the multi-scenario envelope when a single scenario was requested.

    The orchestrator always emits the multi-scenario envelope; callers that
    asked for a single scenario get the inner dict directly.
    Boundary-hit rates are folded in either way.
    """
    if len(names) == 1:
        inner = next(iter(raw["scenarios"].values()))
        return add_boundary_hit_rates(inner)
    return add_boundary_hit_rates(raw)


def _check_failure_threshold(
    convergence_rate: List[float],
    boundary_hit_rate_tau_zero: List[float],
    boundary_hit_rate_high_tau: List[float],
    threshold: float,
) -> None:
    """Raise RuntimeError if any per-N failure rate exceeds *threshold*.

    ``failure_rate = 1.0 - convergence_rate`` for each sample size evaluated.
    Called after the result envelope is complete so boundary-hit rates are
    available for the diagnostic message.

    Args:
        convergence_rate: Per-N convergence rates in [0, 1] (from engine).
        boundary_hit_rate_tau_zero: Per-N fraction of sims where τ̂ = 0.
        boundary_hit_rate_high_tau: Per-N fraction of sims where τ̂ too high.
        threshold: Maximum acceptable failure rate (0–1).  If 1.0, the check
            never raises.

    Raises:
        RuntimeError: if ``max(failure_rates) > threshold``.
    """
    failure_rates = [1.0 - cr for cr in convergence_rate]
    worst_idx = max(range(len(failure_rates)), key=lambda i: failure_rates[i])
    worst_rate = failure_rates[worst_idx]
    if worst_rate > threshold:
        tz = boundary_hit_rate_tau_zero[worst_idx]
        ht = boundary_hit_rate_high_tau[worst_idx]
        raise RuntimeError(
            f"LME convergence failure rate {worst_rate:.1%} exceeds the "
            f"configured threshold {threshold:.1%} "
            f"(sample-size index {worst_idx}). "
            f"Boundary-hit breakdown at that N: "
            f"tau_zero={tz:.1%} (τ̂=0, common for small ICC), "
            f"high_tau={ht:.1%} (τ̂ implausibly large, potential red flag). "
            f"Raise the threshold via set_max_failed_simulations() or increase "
            f"n_clusters / sample size."
        )


# ---------------------------------------------------------------------------
# Result subclasses + short-form renderer
# ---------------------------------------------------------------------------

_EXPORT_ROADMAP = ("not implemented yet — LaTeX/PDF export is on the roadmap; "
                   "use save_plot() for charts and to_dataframe() for tabular data.")


def _plot_impl(result, *, kind: str, path):
    """Shared plot() body. No path -> write a distinctly-named light-print-themed
    stacked HTML in cwd and auto-open. Path -> delegate to save_plot."""
    if path is not None:
        result.save_plot(str(path))
        return
    from . import plotting
    from .tables import target_label_map
    plotting.view_result_plot(
        result, result._meta, kind,
        label_map=target_label_map(result, result._meta),
    )


class PowerResult(dict):
    """find_power result. A dict (key access unchanged) that renders a short
    summary on repr and yields a long-form Report via summary()."""

    def __init__(self, data: Dict[str, Any], meta: Dict[str, Any]):
        super().__init__(data)
        self._meta = meta

    def __repr__(self) -> str:
        return render_short(self, self._meta, kind="find_power")

    __str__ = __repr__

    def summary(self):
        from .report import Report
        return Report(self, self._meta, kind="find_power")

    def to_dataframe(self):
        """Long-format (test x scenario x ...) frame. Parity with R's as_tibble."""
        import pandas as pd
        rows = []
        for scen_name, inner in _scenarios(self):
            corr = bool(self._meta.get("correction") and self._meta["correction"] != "none")
            pkey = "power_corrected" if corr else "power_uncorrected"
            ckey = "ci_corrected" if corr else "ci_uncorrected"
            for r in build_rows(inner["target_indices"], self._meta,
                                inner.get("contrast_pairs") or []):
                if r["kind"] == "factor_header":
                    continue
                pos = r["pos"]
                ci = inner[ckey][0][pos]
                rows.append({
                    "test": r["label"], "scenario": scen_name,
                    "power": inner[pkey][0][pos],
                    "ci_lo": ci[0], "ci_hi": ci[1],
                })
        return pd.DataFrame(rows)

    def save_plot(self, path: str, *, theme: str = "light-print", scale: float = 2.0, ppi=None) -> None:
        """Render this result's chart(s) to file(s) (png / svg / pdf / html,
        by suffix). Default theme is ``"light-print"``; pass ``theme=None`` for
        theme-naked output. For non-HTML formats one file is written per plot
        block with derived names. Needs the optional renderer for non-HTML:
        ``pip install mcpower[plot]``."""
        from . import plotting
        from .tables import target_label_map
        plotting.save_result_plot(
            self, self._meta, "find_power", path,
            theme=theme, scale=scale, ppi=ppi,
            label_map=target_label_map(self, self._meta),
        )

    def plot(self, path: str = None) -> None:
        """No path: write & open find_power.html (light-print-themed, stacked, CDN-backed).
        Path: delegate to save_plot (png/svg/pdf/html, optional renderer)."""
        _plot_impl(self, kind="find_power", path=path)

    def to_latex(self):
        raise NotImplementedError("to_latex(): " + _EXPORT_ROADMAP)

    def to_pdf(self, path):
        raise NotImplementedError("to_pdf(): " + _EXPORT_ROADMAP)


class SampleSizeResult(dict):
    """find_sample_size result; same dual behaviour as PowerResult."""

    def __init__(self, data: Dict[str, Any], meta: Dict[str, Any]):
        super().__init__(data)
        self._meta = meta

    def __repr__(self) -> str:
        return render_short(self, self._meta, kind="find_sample_size")

    __str__ = __repr__

    def summary(self):
        from .report import Report
        return Report(self, self._meta, kind="find_sample_size")

    def save_plot(self, path: str, *, theme: str = "light-print", scale: float = 2.0, ppi=None) -> None:
        """Render this result's chart(s) to file(s) (png / svg / pdf / html,
        by suffix). Default theme is ``"light-print"``; pass ``theme=None`` for
        theme-naked output. For non-HTML formats one file is written per plot
        block with derived names. Needs the optional renderer for non-HTML:
        ``pip install mcpower[plot]``."""
        from . import plotting
        from .tables import target_label_map
        plotting.save_result_plot(
            self, self._meta, "find_sample_size", path,
            theme=theme, scale=scale, ppi=ppi,
            label_map=target_label_map(self, self._meta),
        )

    def plot(self, path: str = None) -> None:
        """No path: write & open find_sample_size.html (light-print-themed, stacked).
        Path: delegate to save_plot."""
        _plot_impl(self, kind="find_sample_size", path=path)

    def to_latex(self):
        raise NotImplementedError("to_latex(): " + _EXPORT_ROADMAP)

    def to_pdf(self, path):
        raise NotImplementedError("to_pdf(): " + _EXPORT_ROADMAP)

    def to_dataframe(self):
        """Long-format (test x scenario x required_n x ci_lo/ci_hi) frame for find_sample_size.

        required_n: n_achievable (fitted), first_achieved value (non_monotone or
        no fitted), NA otherwise (not_reached, at_or_below_min — these render as
        ≤ / ≥ sentinels and have no meaningful integer to export).
        ci_lo/ci_hi: outward-rounded integers when status == fitted and bound
        present, NA otherwise. Both cast to pandas nullable Int64."""
        import math
        import pandas as pd
        rows = []
        for scen_name, inner in _scenarios(self):
            fitted_map = inner.get("fitted") or {}
            for r in build_rows(inner["target_indices"], self._meta,
                                inner.get("contrast_pairs") or []):
                if r["kind"] == "factor_header":
                    continue
                pos = r["pos"]
                f = fitted_map.get(pos)
                # required_n convention (see docstring above).
                if f is not None and f.get("status") == "fitted":
                    req = f["n_achievable"]
                elif f is None or f.get("status") == "non_monotone":
                    req = inner["first_achieved"].get(pos)
                else:
                    # at_or_below_min or not_reached: no single integer to export.
                    req = None
                # ci_lo/ci_hi: only when fitted and both bounds are not None.
                if f is not None and f.get("status") == "fitted":
                    ci_lo_raw = f.get("ci_lo")
                    ci_hi_raw = f.get("ci_hi")
                    ci_lo = math.floor(ci_lo_raw) if ci_lo_raw is not None else None
                    ci_hi = math.ceil(ci_hi_raw) if ci_hi_raw is not None else None
                else:
                    ci_lo = None
                    ci_hi = None
                rows.append({"test": r["label"], "scenario": scen_name,
                             "required_n": req, "ci_lo": ci_lo, "ci_hi": ci_hi})
        df = pd.DataFrame(rows)
        # Nullable integer dtype: NAs propagate correctly in downstream arithmetic.
        df["ci_lo"] = df["ci_lo"].astype("Int64")
        df["ci_hi"] = df["ci_hi"].astype("Int64")
        return df


def make_power_result(data: Dict[str, Any], meta: Dict[str, Any]) -> PowerResult:
    return PowerResult(data, meta)


def make_sample_size_result(data: Dict[str, Any], meta: Dict[str, Any]) -> SampleSizeResult:
    return SampleSizeResult(data, meta)
