import pytest

from mcpower import MCPower

pd = pytest.importorskip("pandas")


def _posthoc_only_model():
    """One-way ANOVA with a 3-level factor, matching test_posthoc.py's
    simple_anova_model fixture (dose_group[1] is the reference)."""
    m = MCPower("pain_reduction = dose_group")
    m.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")
    m.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")
    m.set_simulations(200)
    return m


def test_to_dataframe_long_format():
    m = MCPower("y = x1 + condition")
    m.set_effects("x1=0.5, condition=0.4")
    m.set_simulations(200)
    r = m.find_power(sample_size=120, verbose=False)
    df = r.to_dataframe()
    # One row per tested effect, values taken straight from the result.
    assert list(df.columns) == ["test", "scenario", "power", "ci_lo", "ci_hi"]
    assert list(df["test"]) == ["x1", "condition"]
    assert set(df["scenario"]) == {"optimistic"}
    assert df["power"].tolist() == r["power_uncorrected"][0]
    assert list(zip(df["ci_lo"], df["ci_hi"], strict=True)) == [tuple(ci) for ci in r["ci_uncorrected"][0]]


def test_export_stubs_raise_not_implemented():
    from mcpower import MCPower

    m = MCPower("y = x1").set_effects("x1=0.5").set_simulations(200)
    r = m.find_power(sample_size=120, verbose=False)
    import pytest

    with pytest.raises(NotImplementedError, match="roadmap"):
        r.to_latex()
    with pytest.raises(NotImplementedError, match="roadmap"):
        r.to_pdf("x.pdf")


def test_sample_size_to_dataframe_ci_columns():
    """to_dataframe() includes ci_lo/ci_hi columns (nullable Int64) from the
    fitted crossing fits; this design fits at least one target (x2)."""
    from mcpower import MCPower

    m = MCPower("y = x1 + x2").set_effects("x1=0.5, x2=0.3").set_simulations(600)
    df = m.find_sample_size(from_size=40, to_size=300, by=20, verbose=False).to_dataframe()
    # Both columns must be nullable Int64.
    assert str(df["ci_lo"].dtype) == "Int64", f"ci_lo dtype is {df['ci_lo'].dtype}"
    assert str(df["ci_hi"].dtype) == "Int64", f"ci_hi dtype is {df['ci_hi'].dtype}"
    assert df["ci_lo"].notna().any() and df["ci_hi"].notna().any()


def test_to_dataframe_all_contrasts_only_has_one_row_per_contrast():
    """target_test='all-contrasts' leaves target_indices empty; to_dataframe()
    must not crash and must export the C(3,2)=3 pairwise contrasts with
    non-null power and CI, labelled the same way as the printed table."""
    m = _posthoc_only_model()
    r = m.find_power(sample_size=200, target_test="all-contrasts", verbose=False)
    df = r.to_dataframe()
    assert set(df["test"]) == {"2 vs 1", "3 vs 1", "3 vs 2"}
    assert len(df) == 3
    assert df["power"].notna().all()
    assert (df["power"].between(0, 1)).all()
    assert df["ci_lo"].notna().all() and df["ci_hi"].notna().all()


def test_sample_size_to_dataframe_all_contrasts_does_not_crash():
    """Same target_test on find_sample_size: no contrast is computed for
    required-N at the engine level, so the frame is empty, but it must carry
    the full column set rather than raising KeyError on the Int64 cast."""
    m = _posthoc_only_model()
    r = m.find_sample_size(from_size=40, to_size=200, by=40, target_test="all-contrasts", verbose=False)
    df = r.to_dataframe()
    assert list(df.columns) == ["test", "scenario", "required_n", "ci_lo", "ci_hi"]
    assert len(df) == 0


def test_sample_size_to_dataframe_required_n_conventions():
    """required_n NA conventions: fitted→n_achievable, at_or_below_min→NA, not_reached→NA."""
    # Build a hand-crafted result to exercise each status without a real engine run.
    from mcpower.output.results import SampleSizeResult

    inner = {
        "target_indices": [1, 2, 3],
        "sample_sizes": [40, 80, 120, 160, 200],
        # x3 reached at the first grid point, but its fit status (at_or_below_min)
        # must win over this grid value and export NA.
        "first_achieved": {0: 40, 1: None, 2: 40},
        "first_joint_achieved": {},
        "scenario": "default",
        "n_sims": 200,
        "convergence_rate": [1.0] * 5,
        "boundary_hit_rate_tau_zero": [0.0] * 5,
        "boundary_hit_rate_high_tau": [0.0] * 5,
        "fitted": {
            0: {"status": "fitted", "n_star": 38.0, "n_achievable": 40, "ci_lo": 35.0, "ci_hi": 42.5},
            1: {"status": "not_reached", "n_approx": 220},
            2: {"status": "at_or_below_min", "n_min": 40},
        },
        "fitted_joint": {},
        "cluster_atom": 1,
    }
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "continuous", "predictor": "x1"},
            {"kind": "continuous", "predictor": "x2"},
            {"kind": "continuous", "predictor": "x3"},
        ],
        "factors": {},
        "estimator": "ols",
        "alpha": 0.05,
        "correction": "none",
        "target_power": 0.8,
        "formula": "y ~ x1 + x2 + x3",
    }
    res = SampleSizeResult(inner, meta)
    df = res.to_dataframe()
    fitted_row = df[df["test"] == "x1"].iloc[0]
    not_reached_row = df[df["test"] == "x2"].iloc[0]
    # Fitted: required_n == n_achievable; ci bounds are integers.
    assert fitted_row["required_n"] == 40
    assert fitted_row["ci_lo"] == 35  # floor(35.0)
    assert fitted_row["ci_hi"] == 43  # ceil(42.5)
    # Not-reached: required_n is NA; ci bounds are NA.
    assert pd.isna(not_reached_row["required_n"])
    assert pd.isna(not_reached_row["ci_lo"])
    assert pd.isna(not_reached_row["ci_hi"])
    # At-or-below-min: required_n is NA (renders as a ≤ sentinel); ci bounds NA.
    below_min_row = df[df["test"] == "x3"].iloc[0]
    assert pd.isna(below_min_row["required_n"])
    assert pd.isna(below_min_row["ci_lo"])
    assert pd.isna(below_min_row["ci_hi"])
