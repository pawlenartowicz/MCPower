"""add_boundary_hit_rates must fold uint8-style nested lists into per-N rates without numpy."""
from mcpower.output.results import add_boundary_hit_rates


def test_2d_list_rates_per_sample_size():
    # 3 sims x 2 sample sizes. Col 0: two τ̂=0 hits; col 1: one high-τ (==2) hit.
    res = {"boundary_hit": [[1, 0], [1, 2], [0, 0]], "convergence_rate": [1.0, 1.0]}
    out = add_boundary_hit_rates(res)
    assert out["boundary_hit_rate_tau_zero"] == [2 / 3, 0.0]
    assert out["boundary_hit_rate_high_tau"] == [0.0, 1 / 3]


def test_1d_list_is_single_column():
    res = {"boundary_hit": [1, 1, 0], "convergence_rate": [1.0]}
    out = add_boundary_hit_rates(res)
    assert out["boundary_hit_rate_tau_zero"] == [2 / 3]
    assert out["boundary_hit_rate_high_tau"] == [0.0]


def test_missing_key_zero_filled():
    res = {"convergence_rate": [1.0, 1.0, 1.0]}
    out = add_boundary_hit_rates(res)
    assert out["boundary_hit_rate_tau_zero"] == [0.0, 0.0, 0.0]
    assert out["boundary_hit_rate_high_tau"] == [0.0, 0.0, 0.0]


def test_empty_sims_zero_filled():
    res = {"boundary_hit": [], "convergence_rate": [1.0]}
    out = add_boundary_hit_rates(res)
    assert out["boundary_hit_rate_tau_zero"] == [0.0]
    assert out["boundary_hit_rate_high_tau"] == [0.0]
