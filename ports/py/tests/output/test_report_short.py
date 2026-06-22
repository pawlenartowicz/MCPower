from mcpower import MCPower
from mcpower.output.tables import _required_n_headline, fmt_required_n


def _ols_model():
    m = MCPower("y = x1 + condition")
    m.set_effects("x1=0.5, condition=0.4")
    m.set_simulations(200)
    return m


# ---------------------------------------------------------------------------
# Fallback-chain unit tests (hand-built result dicts, no engine call)
# ---------------------------------------------------------------------------

def _ss_inner(fitted=None, first_achieved=None, sample_sizes=None):
    """Minimal find_sample_size inner dict for display-layer tests."""
    return {
        "fitted": fitted or {},
        "first_achieved": first_achieved or {},
        "sample_sizes": sample_sizes or [40, 80, 120, 160, 200],
        "target_indices": [1],
    }


def test_fmt_required_n_fitted_shows_n_achievable():
    """When status=='fitted', the headline is n_achievable (not n_star)."""
    inner = _ss_inner(fitted={0: {"status": "fitted", "n_star": 84.7, "n_achievable": 85,
                                  "ci_lo": 78.0, "ci_hi": 91.3}},
                      first_achieved={0: 80})
    assert fmt_required_n(inner, 0) == "85"
    _, num = _required_n_headline(inner, 0)
    assert num == 85


def test_fmt_required_n_at_or_below_min_shows_leq():
    """status=='at_or_below_min' renders as '≤ n_min'."""
    inner = _ss_inner(fitted={0: {"status": "at_or_below_min", "n_min": 40}},
                      first_achieved={0: 40})
    assert fmt_required_n(inner, 0) == "≤ 40"
    _, num = _required_n_headline(inner, 0)
    assert num == 40


def test_fmt_required_n_not_reached_shows_geq_ceiling():
    """status=='not_reached' renders '≥ ceiling' and returns no numeric."""
    inner = _ss_inner(fitted={0: {"status": "not_reached", "n_approx": None}},
                      first_achieved={0: None})
    assert fmt_required_n(inner, 0) == "≥ 200"
    _, num = _required_n_headline(inner, 0)
    assert num is None


def test_fmt_required_n_non_monotone_falls_back_to_first_achieved():
    """status=='non_monotone' falls back to first_achieved (grid value shown)."""
    inner = _ss_inner(fitted={0: {"status": "non_monotone", "max_violation": 0.042}},
                      first_achieved={0: 120})
    assert fmt_required_n(inner, 0) == "120"
    _, num = _required_n_headline(inner, 0)
    assert num == 120


def test_fmt_required_n_missing_fitted_falls_back_to_first_achieved():
    """No fitted key (older payload) → first_achieved as before."""
    inner = _ss_inner(fitted={}, first_achieved={0: 80})
    assert fmt_required_n(inner, 0) == "80"


def test_non_monotone_warning_in_short_form():
    """A non_monotone fit must emit a warning line in the short form."""
    inner = _ss_inner(
        fitted={0: {"status": "non_monotone", "max_violation": 0.042}},
        first_achieved={0: 80},
    )
    # Build a minimal result + meta that render_short accepts.
    meta = {
        "effect_skeleton": [{"kind": "intercept"},
                             {"kind": "continuous", "predictor": "x1"}],
        "factors": {}, "estimator": "ols", "alpha": 0.05,
        "correction": "none", "target_power": 0.8, "formula": "y ~ x1",
    }
    # Attach required fields that _render_sample_size_short expects.
    inner.update({
        "target_indices": [1],
        "n_sims": 200,
        "convergence_rate": [1.0] * 5,
        "boundary_hit_rate_tau_zero": [0.0] * 5,
        "boundary_hit_rate_high_tau": [0.0] * 5,
        "first_joint_achieved": {},
        "estimator_extras": {"estimator": "ols"},
        "overall_significant_rate": None,
    })
    from mcpower.output.tables import render_short
    s = render_short(inner, meta, kind="find_sample_size")
    assert "non_monotone" in s or "not monotone" in s
    assert "0.042" in s


def test_find_power_returns_dict_subclass_with_keys_intact():
    m = _ols_model()
    r = m.find_power(sample_size=120, verbose=False)
    assert isinstance(r, dict)
    assert "power_uncorrected" in r
    assert isinstance(r["power_uncorrected"], list)


def test_short_form_headline_no_ci_no_glyph():
    m = _ols_model()
    s = str(m.find_power(sample_size=120, verbose=False))
    assert "Power Analysis — OLS" in s
    assert "N=120" in s and "sims=200" in s and "target=" in s
    assert "formula:" in s
    assert "Power" in s and "Target" in s
    assert "CI 95%" not in s           # CI moved to summary()
    assert "✓" not in s and "✗" not in s

def test_short_form_correction_adds_columns():
    m = _ols_model()
    s = str(m.find_power(sample_size=120, correction="holm", verbose=False))
    assert "uncorrected" in s and "corrected" in s
    assert "correction:" in s   # model-summary block line

def test_short_form_scenarios_no_delta():
    m = _ols_model()
    s = str(m.find_power(sample_size=120, scenarios=True, verbose=False))
    assert "optimistic" in s and "realistic" in s and "doomer" in s
    assert "Δ" not in s and "pp" not in s   # drop is in summary()
    assert "scenarios:" in s


def test_verbose_true_autoprints(capsys):
    m = _ols_model()
    m.find_power(sample_size=120, verbose=True)
    out = capsys.readouterr().out
    assert "Power Analysis" in out


def test_verbose_false_silent(capsys):
    m = _ols_model()
    m.find_power(sample_size=120, verbose=False)
    assert "Power Analysis" not in capsys.readouterr().out


def test_sample_size_short_form_shows_required_n():
    m = _ols_model()
    r = m.find_sample_size(from_size=40, to_size=200, by=20, verbose=False)
    s = str(r)
    assert "Power Analysis" in s
    assert "Required N" in s
    assert "First N achieving all targets" in s


def test_sample_size_short_form_shows_overall_row():
    """The overall (omnibus) required-N row appears first in the short form too,
    labelled per the estimator (Overall F for OLS) — mirrors the long form."""
    m = _ols_model()
    r = m.find_sample_size(from_size=40, to_size=200, by=20, verbose=False)
    s = str(r)
    assert "Overall F" in s
    rn = s.split("Required N")[-1]
    assert rn.index("Overall F") < rn.index("x1")


def test_sample_size_short_form_shows_specific_achieved_n():
    """Required-N cells must contain a concrete N, not just the ceiling sentinel.

    Protects fmt_required_n's first_achieved.get(pos) key lookup: if the key
    convention shifts, every cell silently renders '≥ ceiling'.
    """
    import re
    m = MCPower("y = x1 + condition")
    m.set_effects("x1=0.5, condition=0.4")
    r = m.find_sample_size(from_size=40, to_size=300, by=20,
                           target_power=0.8, n_sims=300, seed=2137,
                           verbose=False)

    # At least one target must have been reached in the grid.
    assert any(v is not None for v in r["first_achieved"].values()), (
        "first_achieved has no reached target — precondition for this test failed"
    )

    s = str(r)
    # A concrete N (a positive integer without leading '≥') must appear in the
    # Required N column section.
    required_n_section = s.split("Required N")[-1] if "Required N" in s else s
    assert re.search(r"(?<!≥ )\b[1-9][0-9]+\b", required_n_section), (
        "No concrete N found in Required N column — all cells may be showing '≥ ceiling'. "
        f"Rendered output:\n{s}"
    )
