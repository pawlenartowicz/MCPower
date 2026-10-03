"""boundary_hit_rate_tau_zero and boundary_hit_rate_high_tau in result envelope.

Tests that find_power (and find_sample_size) surface per-N boundary-hit rates
derived from the raw boundary_hit uint8 array:
  - code 1  → tau estimated as exactly zero (boundary REML)
  - code 2  → tau estimated as implausibly high (boundary REML)
"""

from mcpower import MCPower

# ---------------------------------------------------------------------------
# LME find_sample_size: one rate per sample size
# ---------------------------------------------------------------------------


def test_lme_find_sample_size_includes_boundary_hit_rates():
    # ICC=0.05 (low) causes frequent tau_zero boundary hits — at seed=2137 gives [0.22, 0.12].
    # ICC=0.3 at seed=2137 yields [0.0, 0.0], which would let a broken extraction pass.
    model = MCPower("y ~ x1 + (1|cluster)", family="lme")
    model.set_effects("x1=0.5").set_cluster("cluster", ICC=0.05, n_clusters=10)
    res = model.find_sample_size(from_size=100, to_size=200, by=100, n_sims=50, seed=2137)
    assert "boundary_hit_rate_tau_zero" in res
    assert "boundary_hit_rate_high_tau" in res
    rates_tz = res["boundary_hit_rate_tau_zero"]
    rates_ht = res["boundary_hit_rate_high_tau"]
    # Should be a list with one entry per sample size tested.
    assert hasattr(rates_tz, "__iter__") and not isinstance(rates_tz, (str, bytes))
    assert len(rates_tz) == len(res["sample_sizes"])
    for v in rates_tz:
        assert 0.0 <= v <= 1.0
    for v in rates_ht:
        assert 0.0 <= v <= 1.0
    # At ICC=0.05, nc=10, seed=2137: tau_zero rates are [0.22, 0.12]; a broken
    # boundary-hit extraction (all zeros) is caught here.
    assert any(v > 0.0 for v in rates_tz), "expected at least one tau_zero boundary hit > 0 — boundary_hit extraction may be broken"


# ---------------------------------------------------------------------------
# Shape sanity: single-N find_power returns a list of length 1
# ---------------------------------------------------------------------------


def test_lme_boundary_hit_rates_shape_single_n():
    model = MCPower("y ~ x1 + (1|cluster)", family="lme")
    model.set_effects("x1=0.4").set_cluster("cluster", ICC=0.2, n_clusters=10)
    res = model.find_power(100, n_sims=30, seed=2137)
    tz = res["boundary_hit_rate_tau_zero"]
    ht = res["boundary_hit_rate_high_tau"]
    assert isinstance(tz, list), f"expected list, got {type(tz)}"
    assert len(tz) == 1, f"expected length 1 for single-N, got {len(tz)}"
    assert isinstance(ht, list)
    assert len(ht) == 1
    # Value range + nonzero guard (seed=2137: tz[0] ≈ 0.033; high_tau stays 0.0 at ICC=0.2).
    assert 0.0 <= tz[0] <= 1.0
    assert 0.0 <= ht[0] <= 1.0
    assert tz[0] > 0.0, "expected tau_zero hits > 0 at ICC=0.2 seed=2137 — extraction may be broken"
