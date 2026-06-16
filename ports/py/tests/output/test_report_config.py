from mcpower.config import get_report_config


def test_loads_format_constants_from_engine():
    cfg = get_report_config()
    assert cfg["format"]["power_decimals_short"] == 1
    assert cfg["format"]["power_decimals_long"] == 1
    assert cfg["thresholds"]["convergence_min"] == 0.95
    assert cfg["overall_label_by_estimator"]["ols"] == "Overall F"


def test_config_is_cached():
    assert get_report_config() is get_report_config()
