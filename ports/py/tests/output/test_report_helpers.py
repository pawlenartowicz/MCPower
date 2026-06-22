from mcpower.output.tables import build_rows, fmt_ci, fmt_pct, joint_distribution, fmt_target


def test_fmt_target_has_no_glyph():
    from mcpower.output.tables import fmt_target
    assert fmt_target(0.8, 0) == "80%"
    assert fmt_target(0.805, 1) == "80.5%"
    # ✓ / ✗ are gone from the output character set
    assert "✓" not in fmt_target(0.8, 0)
    assert "✗" not in fmt_target(0.8, 0)


def test_power_row_is_three_cells_no_ci():
    from mcpower.output.tables import _power_row
    kind, cells = _power_row("age", 0.752, 0.80, 1, 0)
    assert kind == "row"
    assert cells == ["age", "75.2%", "80%"]   # label | power | target, no CI cell


def _synthetic(n_scen=1, corr=False):
    """Build a minimal result + meta the renderer accepts."""
    def inner(p_unc, p_cor):
        return {
            "target_indices": [1],
            "sample_sizes": [120], "n_sims": 400,
            "power_uncorrected": [[p_unc]], "power_corrected": [[p_cor]],
            "ci_uncorrected": [[(p_unc - 0.05, p_unc + 0.05)]],
            "ci_corrected": [[(p_cor - 0.05, p_cor + 0.05)]],
            "estimator_extras": {"estimator": "ols"},
            "overall_significant_rate": None,
        }
    meta = {"effect_names": ["x1"],
            "effect_skeleton": [{"kind": "intercept"},
                                {"kind": "continuous", "predictor": "x1"}],
            "factors": {}, "estimator": "ols",
            "alpha": 0.05, "correction": "holm" if corr else "none",
            "target_power": 0.8, "formula": "y ~ x1"}
    if n_scen == 1:
        return inner(0.75, 0.60), meta
    scen = {"optimistic": inner(0.85, 0.70), "doomer": inner(0.55, 0.40)}
    return {"scenarios": scen}, meta

def test_main_power_tables_neither():
    from mcpower.output.tables import main_power_tables, _scenarios
    result, meta = _synthetic()
    tables = main_power_tables(_scenarios(result), meta, dec=1, tdec=0,
                               target=0.8, caption="Per-test power")
    assert len(tables) == 1
    t = tables[0]
    assert "Per-test power" in t and "Power" in t and "Target" in t
    assert "CI 95%" not in t and "✓" not in t and "✗" not in t

def test_main_power_tables_correction_only_adds_two_value_columns():
    from mcpower.output.tables import main_power_tables, _scenarios
    result, meta = _synthetic(corr=True)
    tables = main_power_tables(_scenarios(result), meta, dec=1, tdec=0,
                               target=0.8, caption="Per-test power")
    assert len(tables) == 1
    assert "uncorrected" in tables[0] and "corrected" in tables[0]

def test_main_power_tables_scenarios_only_one_table_no_delta():
    from mcpower.output.tables import main_power_tables, _scenarios
    result, meta = _synthetic(n_scen=2)
    tables = main_power_tables(_scenarios(result), meta, dec=1, tdec=0,
                               target=0.8, caption="Per-test power")
    assert len(tables) == 1
    assert "optimistic" in tables[0] and "doomer" in tables[0]
    assert "Δ" not in tables[0] and "pp" not in tables[0]   # drop moved to the diagnostics section

def test_main_power_tables_both_splits_into_two():
    from mcpower.output.tables import main_power_tables, _scenarios
    result, meta = _synthetic(n_scen=2, corr=True)
    tables = main_power_tables(_scenarios(result), meta, dec=1, tdec=0,
                               target=0.8, caption="Per-test power")
    assert len(tables) == 2
    assert "Uncorrected" in tables[0] and "Corrected" in tables[1]


def test_build_rows_groups_factor_levels_under_header():
    # target_indices are β̂-column indices (intercept at 0); the skeleton is
    # β-aligned, so skeleton[idx] names the effect directly. Factor dummies
    # carry a `level` index into the port's full `levels` store (reference
    # "control" at 0 → "treatment" at 1, "active" at 2).
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "continuous", "predictor": "x1"},
            {"kind": "factor_level", "factor": "condition", "level": 1},
            {"kind": "factor_level", "factor": "condition", "level": 2},
        ],
        "factors": {"condition": {"baseline": "control",
                                  "levels": ["control", "treatment", "active"]}},
    }
    rows = build_rows([1, 2, 3], meta)
    assert rows[0] == {"kind": "continuous", "label": "x1", "pos": 0}
    assert rows[1] == {"kind": "factor_header", "label": "condition", "baseline": "control"}
    assert rows[2] == {"kind": "factor_level", "label": "treatment", "factor": "condition", "pos": 1}
    assert rows[3] == {"kind": "factor_level", "label": "active", "factor": "condition", "pos": 2}


def test_build_rows_appends_contrast_rows_after_marginals():
    # Pairwise contrast (β3 − β2): one `contrast` row after the marginals with
    # pos past target_indices (the engine appends contrast power entries).
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "continuous", "predictor": "x1"},
            {"kind": "factor_level", "factor": "condition", "level": 1},
            {"kind": "factor_level", "factor": "condition", "level": 2},
        ],
        "factors": {"condition": {"baseline": "control",
                                  "levels": ["control", "treatment", "active"]}},
    }
    rows = build_rows([1, 2, 3], meta, contrast_pairs=[[3, 2]])
    assert rows[-1] == {
        "kind": "contrast",
        "label": "condition[active] vs condition[treatment]",
        "pos": 3,
    }


def test_fmt_pct_respects_decimals():
    assert fmt_pct(0.925, 1) == "92.5%"
    assert fmt_pct(0.925, 2) == "92.50%"
    assert fmt_pct(0.8, 0) == "80%"


def test_fmt_pct_drops_decimals_at_exactly_100():
    # Exactly 100% (and anything that rounds to it at the given precision) drops
    # the ".0" — "100%" not "100.0%" — so the column reserves only two int digits.
    assert fmt_pct(1.0, 1) == "100%"
    assert fmt_pct(1.0, 2) == "100%"
    assert fmt_pct(0.9999, 1) == "100%"   # rounds to 100.0 at 1 dp
    assert fmt_pct(0.995, 1) == "99.5%"   # just below — keeps decimals


def test_fmt_ci_includes_percent_and_aligns_100():
    # CI bounds carry "%"; a 100% bound drops its decimals and takes one leading
    # space so it stacks under "99.0%" (cross-port parity).
    assert fmt_ci((0.99, 1.0), 1) == "[99.0%,  100%]"
    assert fmt_ci((0.982, 0.999), 1) == "[98.2%, 99.9%]"
    assert fmt_ci((0.072, 0.5), 1) == "[ 7.2%, 50.0%]"
    assert fmt_ci(None, 1) == ""


def test_joint_distribution_exactly_and_at_least():
    hist = [10, 30, 60]
    n = 100
    jd = joint_distribution(hist, n)
    assert jd["exactly"] == [0.10, 0.30, 0.60]
    assert jd["at_least"] == [1.0, 0.90, 0.60]


def test_joint_distribution_returns_none_on_empty_input():
    assert joint_distribution([], 0) is None
