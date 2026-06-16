"""boundary_hit_rate_tau_zero and boundary_hit_rate_high_tau in result envelope.

Tests that find_power (and find_sample_size) surface per-N boundary-hit rates
derived from the raw boundary_hit uint8 array:
  - code 1  → tau estimated as exactly zero (boundary REML)
  - code 2  → tau estimated as implausibly high (boundary REML)
"""

import pytest
from mcpower import MCPower


# ---------------------------------------------------------------------------
# LME: rates present and in [0, 1]
# ---------------------------------------------------------------------------

def test_result_envelope_includes_boundary_hit_rates():
    model = MCPower("y ~ x1 + (1|cluster)", family="lme")
    model.set_effects("x1=0.3").set_cluster("cluster", ICC=0.2, n_clusters=10)
    res = model.find_power(100, n_sims=50)
    assert "boundary_hit_rate_tau_zero" in res
    assert "boundary_hit_rate_high_tau" in res
    # For LME at ICC=0.2 / nc=10 / N=100, tau_zero rate is often non-zero
    # (boundary REML). Just check the range, not the exact value.
    val_low = res["boundary_hit_rate_tau_zero"]
    val_high = res["boundary_hit_rate_high_tau"]
    # Handle both scalar and per-N list formats:
    if hasattr(val_low, "__iter__") and not isinstance(val_low, (str, bytes)):
        for v in val_low:
            assert 0.0 <= v <= 1.0
        for v in val_high:
            assert 0.0 <= v <= 1.0
    else:
        assert 0.0 <= val_low <= 1.0
        assert 0.0 <= val_high <= 1.0


def test_lme_find_sample_size_includes_boundary_hit_rates():
    model = MCPower("y ~ x1 + (1|cluster)", family="lme")
    model.set_effects("x1=0.5").set_cluster("cluster", ICC=0.3, n_clusters=10)
    res = model.find_sample_size(from_size=100, to_size=200, by=100, n_sims=30)
    assert "boundary_hit_rate_tau_zero" in res
    assert "boundary_hit_rate_high_tau" in res
    rates_tz = res["boundary_hit_rate_tau_zero"]
    rates_ht = res["boundary_hit_rate_high_tau"]
    # Should be a list with one entry per sample size tested
    assert hasattr(rates_tz, "__iter__") and not isinstance(rates_tz, (str, bytes))
    assert len(rates_tz) == len(res["sample_sizes"])
    for v in rates_tz:
        assert 0.0 <= v <= 1.0
    for v in rates_ht:
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# OLS: both rates must be 0.0 (boundary_hit is always zero for OLS)
# ---------------------------------------------------------------------------

def test_ols_boundary_hit_rates_are_zero():
    model = MCPower("y ~ x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")
    res = model.find_power(100, n_sims=50)
    assert "boundary_hit_rate_tau_zero" in res
    assert "boundary_hit_rate_high_tau" in res
    val_tz = res["boundary_hit_rate_tau_zero"]
    val_ht = res["boundary_hit_rate_high_tau"]
    if hasattr(val_tz, "__iter__") and not isinstance(val_tz, (str, bytes)):
        assert all(v == 0.0 for v in val_tz)
        assert all(v == 0.0 for v in val_ht)
    else:
        assert val_tz == 0.0
        assert val_ht == 0.0


# ---------------------------------------------------------------------------
# Shape sanity: single-N find_power returns a list of length 1
# ---------------------------------------------------------------------------

def test_lme_boundary_hit_rates_shape_single_n():
    model = MCPower("y ~ x1 + (1|cluster)", family="lme")
    model.set_effects("x1=0.4").set_cluster("cluster", ICC=0.2, n_clusters=10)
    res = model.find_power(100, n_sims=30)
    tz = res["boundary_hit_rate_tau_zero"]
    ht = res["boundary_hit_rate_high_tau"]
    assert isinstance(tz, list), f"expected list, got {type(tz)}"
    assert len(tz) == 1, f"expected length 1 for single-N, got {len(tz)}"
    assert isinstance(ht, list)
    assert len(ht) == 1
