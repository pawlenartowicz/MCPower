import pytest
from mcpower import MCPower

pd = pytest.importorskip("pandas")


def test_to_dataframe_long_format():
    m = MCPower("y = x1 + condition")
    m.set_effects("x1=0.5, condition=0.4")
    m.set_simulations(200)
    df = m.find_power(sample_size=120, verbose=False).to_dataframe()
    assert {"test", "scenario", "power", "ci_lo", "ci_hi"}.issubset(df.columns)
    assert (df["power"].between(0, 1)).all()


def test_export_stubs_raise_not_implemented():
    from mcpower import MCPower
    m = MCPower("y = x1").set_effects("x1=0.5").set_simulations(200)
    r = m.find_power(sample_size=120, verbose=False)
    import pytest
    with pytest.raises(NotImplementedError, match="roadmap"):
        r.to_latex()
    with pytest.raises(NotImplementedError, match="roadmap"):
        r.to_pdf("x.pdf")

def test_sample_size_to_dataframe_long_format():
    from mcpower import MCPower
    m = MCPower("y = x1 + condition").set_effects("x1=0.5, condition=0.4").set_simulations(200)
    df = m.find_sample_size(from_size=40, to_size=200, by=40, verbose=False).to_dataframe()
    assert {"test", "scenario", "required_n"}.issubset(df.columns)


def test_sample_size_to_dataframe_ci_columns():
    """to_dataframe() now includes ci_lo/ci_hi columns (Int64) from fitted crossing fits."""
    from mcpower import MCPower
    m = MCPower("y = x1 + x2").set_effects("x1=0.5, x2=0.3").set_simulations(600)
    df = m.find_sample_size(from_size=40, to_size=300, by=20, verbose=False).to_dataframe()
    assert "ci_lo" in df.columns and "ci_hi" in df.columns
    # Both columns must be nullable Int64.
    assert str(df["ci_lo"].dtype) == "Int64", f"ci_lo dtype is {df['ci_lo'].dtype}"
    assert str(df["ci_hi"].dtype) == "Int64", f"ci_hi dtype is {df['ci_hi'].dtype}"
    # When a fitted bound is present the value must be a plain integer.
    fitted_lo = df["ci_lo"].dropna()
    if len(fitted_lo) > 0:
        assert (fitted_lo % 1 == 0).all(), "ci_lo must contain only whole numbers"
    fitted_hi = df["ci_hi"].dropna()
    if len(fitted_hi) > 0:
        assert (fitted_hi % 1 == 0).all(), "ci_hi must contain only whole numbers"


def test_sample_size_to_dataframe_required_n_conventions():
    """required_n NA conventions: fitted→n_achievable, at_or_below_min→NA, not_reached→NA."""
    # Build a hand-crafted result to exercise each status without a real engine run.
    from mcpower.output.results import SampleSizeResult
    from mcpower.output.tables import _scenarios, build_rows

    inner = {
        "target_indices": [1, 2],
        "sample_sizes": [40, 80, 120, 160, 200],
        "first_achieved": {0: 40, 1: None},
        "first_joint_achieved": {},
        "scenario": "default",
        "n_sims": 200,
        "convergence_rate": [1.0] * 5,
        "boundary_hit_rate_tau_zero": [0.0] * 5,
        "boundary_hit_rate_high_tau": [0.0] * 5,
        "fitted": {
            0: {"status": "fitted", "n_star": 38.0, "n_achievable": 40,
                "ci_lo": 35.0, "ci_hi": 42.5},
            1: {"status": "not_reached", "n_approx": 220},
        },
        "fitted_joint": {},
        "cluster_atom": 1,
    }
    meta = {
        "effect_skeleton": [{"kind": "intercept"},
                             {"kind": "continuous", "predictor": "x1"},
                             {"kind": "continuous", "predictor": "x2"}],
        "factors": {}, "estimator": "ols", "alpha": 0.05,
        "correction": "none", "target_power": 0.8, "formula": "y ~ x1 + x2",
    }
    res = SampleSizeResult(inner, meta)
    df = res.to_dataframe()
    fitted_row = df[df["test"] == "x1"].iloc[0]
    not_reached_row = df[df["test"] == "x2"].iloc[0]
    # Fitted: required_n == n_achievable; ci bounds are integers.
    assert fitted_row["required_n"] == 40
    assert fitted_row["ci_lo"] == 35      # floor(35.0)
    assert fitted_row["ci_hi"] == 43      # ceil(42.5)
    # Not-reached: required_n is NA; ci bounds are NA.
    assert pd.isna(not_reached_row["required_n"])
    assert pd.isna(not_reached_row["ci_lo"])
    assert pd.isna(not_reached_row["ci_hi"])
