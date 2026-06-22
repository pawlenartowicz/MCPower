from mcpower.config import (
    get_config, get_report_config, get_simulation_defaults,
    get_benchmarks, get_limits, get_cluster_limits,
)


def test_sections_present():
    cfg = get_config()
    assert set(cfg) >= {"simulation", "benchmarks", "limits", "report", "upload"}


def test_simulation_defaults():
    sim = get_simulation_defaults()
    assert sim["seed"] == 2137
    assert sim["alpha"] == 0.05
    assert sim["target_power"] == 0.8
    assert sim["n_sims"]["ols"] == 1600
    assert sim["max_failed_fraction"] == 0.1
    assert sim["cluster_auto_count"] == 12
    assert sim["sample_size_bounds"] == {"from": 30, "to": 200, "by": "auto"}


def test_limits_and_benchmarks():
    assert get_limits()["max_alpha"] == 0.25
    assert get_limits()["factor_levels"] == [2, 20]
    assert get_benchmarks()["continuous"] == [0.1, 0.25, 0.4]
    assert get_report_config()["format"]["power_decimals_long"] == 1
    assert get_cluster_limits()["min_rows_per_cluster"] == 2
