from mcpower import MCPower
from mcpower.output.report import Report


def _model():
    m = MCPower("y = x1 + condition")
    m.set_effects("x1=0.5, condition=0.4")
    m.set_simulations(200)
    return m


def test_long_form_boxed_header_and_effects():
    rep = _model().find_power(sample_size=120, verbose=False).summary()
    s = str(rep)
    assert "=====" in s                       # boxed === header (charter §5)
    assert "MCPower · Power Analysis" in s
    assert "estimator: OLS" in s
    assert "effects:" in s and "x1=0.5" in s   # raw effect sizes echoed
    assert "Per-test power" in s
    assert "CI 95%" not in s.split("Per-test power")[1].split("Power & 95% CI")[0]  # not in main table

def test_long_form_both_axis_two_tables():
    rep = _model().find_power(sample_size=120, scenarios=True, correction="holm",
                              verbose=False).summary()
    s = str(rep)
    assert "Per-test power — Uncorrected" in s
    assert "Per-test power — Corrected" in s


def test_long_form_ci_section_single_scenario():
    rep = _model().find_power(sample_size=120, verbose=False).summary()
    s = str(rep)
    assert "Power & 95% CI" in s
    assert "Monte-Carlo (Wilson)" in s and "n_sims=200" in s
    # CI bracket appears in the CI section, not in the main per-test table
    assert "[" in s.split("Power & 95% CI")[1]

def test_long_form_ci_section_titled_per_scenario():
    rep = _model().find_power(sample_size=120, scenarios=True, verbose=False).summary()
    s = str(rep)
    assert "Power & 95% CI — optimistic" in s
    assert "Power & 95% CI — doomer" in s


def test_estimator_extras_shown_when_present():
    base = {"target_indices": [1], "power_uncorrected": [[0.8]], "power_corrected": [[0.8]],
            "ci_uncorrected": [[(0.75, 0.85)]], "ci_corrected": [[(0.75, 0.85)]],
            "sample_sizes": [120], "n_sims": 400, "convergence_rate": [1.0],
            "boundary_hit_rate_tau_zero": [0.0], "boundary_hit_rate_high_tau": [0.0],
            "overall_significant_rate": None,
            "estimator_extras": {"estimator": "glm", "baseline_prob_realized": 0.31,
                                 "separation_rate": 0.0}}
    meta = {"effect_names": ["x1"],
            "effect_skeleton": [{"kind": "intercept"},
                                {"kind": "continuous", "predictor": "x1"}],
            "factors": {}, "estimator": "glm", "alpha": 0.05,
            "correction": "none", "target_power": 0.8, "formula": "y ~ x1",
            "effect_sizes": [0.5]}
    s = str(Report(base, meta, kind="find_power"))
    assert "Estimator details" in s
    assert "baseline_prob_realized" in s and "0.31" in s

def test_estimator_extras_absent_for_plain_ols():
    rep = _model().find_power(sample_size=120, verbose=False).summary()
    # OLS extras carry only {"estimator": "ols"} — nothing informational to show.
    s = str(rep)
    assert "Estimator details" not in s


def _ss_overall_base(*, with_overall=True, overall_fitted=None):
    """Single-scenario find_sample_size result dict with marginal + overall fits."""
    base = {
        "target_indices": [1],
        "contrast_pairs": [],
        "power_uncorrected": [[0.40], [0.60], [0.85]],
        "power_corrected": [[0.40], [0.60], [0.85]],
        "ci_uncorrected": [[(0.3, 0.5)], [(0.5, 0.7)], [(0.8, 0.9)]],
        "ci_corrected": [[(0.3, 0.5)], [(0.5, 0.7)], [(0.8, 0.9)]],
        "sample_sizes": [50, 100, 150],
        "n_sims": 200,
        "convergence_rate": [1.0, 1.0, 1.0],
        "first_achieved": {0: 150},
        "fitted": {0: {"status": "fitted", "n_star": 140, "n_achievable": 150,
                       "ci_lo": 120, "ci_hi": 170}},
        "estimator_extras": {"estimator": "ols"},
        "boundary_hit_rate_tau_zero": [0.0, 0.0, 0.0],
        "boundary_hit_rate_high_tau": [0.0, 0.0, 0.0],
    }
    if with_overall:
        base["first_overall_achieved"] = 100
        base["fitted_overall"] = {0: overall_fitted or {
            "status": "fitted", "n_star": 92, "n_achievable": 100,
            "ci_lo": 80, "ci_hi": 120}}
    return base


def _ss_overall_meta(estimator="ols"):
    return {"target_power": 0.8, "estimator": estimator, "formula": "y ~ x1",
            "effect_names": ["x1"],
            "effect_skeleton": [{"kind": "intercept"},
                                {"kind": "continuous", "predictor": "x1"}],
            "factors": {}, "alpha": 0.05, "correction": "none", "effect_sizes": [0.5]}


def test_sample_size_overall_required_n_row_renders():
    """The overall (omnibus) test gets a first required-N row, labelled per the
    estimator (F-test for OLS), in both the Required N and Required N & CI tables."""
    s = str(Report(_ss_overall_base(), _ss_overall_meta("ols"), kind="find_sample_size"))
    assert "Overall F" in s
    main = s.split("Required N")[1]
    # The overall n_achievable (100) precedes the marginal's (150) in the table.
    assert main.index("Overall F") < main.index("x1")
    assert "100" in s  # overall fitted n_achievable
    # GLM relabels the omnibus to the LRT.
    s_glm = str(Report(_ss_overall_base(), _ss_overall_meta("glm"), kind="find_sample_size"))
    assert "LR χ²" in s_glm


def test_sample_size_overall_row_absent_when_not_requested():
    """No overall fields (mixed family / report_overall=False) ⇒ no overall row."""
    s = str(Report(_ss_overall_base(with_overall=False), _ss_overall_meta("ols"),
                   kind="find_sample_size"))
    assert "Overall F" not in s


def test_plot_footer_mentions_plot_method():
    s = str(_model().find_power(sample_size=120, verbose=False).summary())
    assert "result.plot()" in s and "result.plot('chart.png')" in s


def test_summary_returns_report():
    r = _model().find_power(sample_size=120, verbose=False)
    rep = r.summary()
    assert isinstance(rep, Report)


def test_long_form_has_all_sections():
    rep = _model().find_power(sample_size=120, verbose=False).summary()
    s = str(rep)
    assert "Per-test power" in s
    assert "Joint significance distribution" in s
    # Joint distribution renders as a k / Exactly / At least table.
    assert "Exactly" in s and "At least" in s
    assert "plot" in s.lower()
    # Diagnostics surface only when a threshold trips; this healthy OLS run shows none.
    assert "Diagnostics" not in s
    assert "convergence" not in s.lower()


def test_diagnostics_block_only_when_threshold_trips():
    from mcpower.config import get_report_config
    cfg = get_report_config()
    base = {"estimator_extras": {"estimator": "ols"},
            "boundary_hit_rate_tau_zero": [0.0], "boundary_hit_rate_high_tau": [0.0]}
    clean = Report({**base, "convergence_rate": [1.0]}, {}, kind="find_power")
    assert clean._diagnostics(cfg) == ""
    broken = Report({**base, "convergence_rate": [0.88]}, {}, kind="find_power")
    block = broken._diagnostics(cfg)
    assert "⚠ Diagnostics" in block and "convergence 88.0%" in block


def test_joint_section_neutral_text_when_histogram_empty():
    rep = _model().find_power(sample_size=120, verbose=False).summary()
    rep._result["success_count_histogram_uncorrected"] = []
    s = str(rep)
    assert "Joint significance distribution is unavailable for this result." in s
    assert "analytical" not in s.lower()


def test_joint_section_has_nonzero_probabilities_in_real_run():
    """Joint histogram must be non-trivially populated after a real multi-target run.

    Guards: (a) n_sims field name mismatch (→ n_sims_used=0 → section shows 'unavailable');
            (b) histogram field renamed (→ empty histogram → same).
    """
    import re
    rep = _model().find_power(sample_size=200, seed=2137, verbose=False).summary()
    s = str(rep)

    assert "Joint significance distribution" in s, "section header missing"
    joint_part = s.split("Joint significance distribution")[-1]

    assert "unavailable" not in joint_part, (
        "Joint section shows 'unavailable' — n_sims field or histogram field may be wrong"
    )
    # At least one non-zero percentage must appear (k>=1 row).
    pct_values = re.findall(r"(\d+(?:\.\d+)?)%", joint_part)
    non_zero = [v for v in pct_values if float(v) > 0.0]
    assert non_zero, (
        "All 'Exactly' / 'At least' entries are 0% — histogram may not be populated. "
        f"Joint section:\n{joint_part[:300]}"
    )
