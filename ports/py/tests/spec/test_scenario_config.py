import copy
import pytest
from mcpower.spec.scenario_config import get_default_scenario_config
from mcpower.spec.spec_builder import build_scenario_dict


def test_default_scenarios_match_canonical():
    cfg = get_default_scenario_config()
    assert set(cfg) == {"optimistic", "realistic", "doomer"}
    opt = cfg["optimistic"]
    assert opt["residual_dists"] == ["high_kurtosis", "right_skewed"]
    assert opt["new_distributions"] == ["right_skewed", "left_skewed", "uniform"]
    assert opt["random_effect_dist"] == "normal"
    assert "lme" not in opt
    assert cfg["realistic"]["random_effect_dist"] == "heavy_tailed"


def test_loader_returns_fresh_copies():
    a = get_default_scenario_config()
    a["optimistic"]["heterogeneity"] = 999
    b = get_default_scenario_config()
    assert b["optimistic"]["heterogeneity"] == 0.0  # caller mutation must not leak


def test_sampled_factor_proportions_default_and_override():
    # Presets ride the engine-embedded configs/scenarios.json.
    defaults = get_default_scenario_config()
    assert defaults["optimistic"]["sampled_factor_proportions"] is False
    assert defaults["realistic"]["sampled_factor_proportions"] is True
    assert defaults["doomer"]["sampled_factor_proportions"] is True

    # Knob omitted in a custom scenario → exact-allocation default (False).
    assert (
        build_scenario_dict("custom", {"custom": {}})["sampled_factor_proportions"]
        is False
    )
    # Explicit override flows through.
    assert (
        build_scenario_dict("custom", {"custom": {"sampled_factor_proportions": True}})[
            "sampled_factor_proportions"
        ]
        is True
    )


def test_build_scenario_dict_emits_lme_keys_from_defaults():
    """build_scenario_dict emits random_effect_dist/df/icc_noise_sd from the
    scenario config dict (optimistic defaults: normal→0, df=5, icc_noise_sd=0.0)."""
    from mcpower.spec.scenario_config import get_default_scenario_config
    from mcpower.spec.spec_builder import build_scenario_dict

    cfg = get_default_scenario_config()
    d = build_scenario_dict("optimistic", cfg)
    # "lme": None must be gone
    assert "lme" not in d
    # RE knobs present
    assert "random_effect_dist" in d
    assert "random_effect_df" in d
    assert "icc_noise_sd" in d
    # normal → code 0 (per get_residual_codes())
    assert d["random_effect_dist"] == 0
    assert d["random_effect_df"] == pytest.approx(5.0)
    assert d["icc_noise_sd"] == pytest.approx(0.0)


def test_build_scenario_dict_emits_lme_keys_from_realistic():
    """realistic scenario: heavy_tailed RE dist → code 1, icc_noise_sd=0.15."""
    from mcpower.spec.scenario_config import get_default_scenario_config
    from mcpower.spec.spec_builder import build_scenario_dict

    cfg = get_default_scenario_config()
    d = build_scenario_dict("realistic", cfg)
    assert "lme" not in d
    # heavy_tailed → code 1 (RE dist vocabulary: normal=0, heavy_tailed=1)
    assert d["random_effect_dist"] == 1
    assert d["icc_noise_sd"] == pytest.approx(0.15)


def test_set_scenario_configs_accepts_lme_keys():
    """After removing the lme-key ValueError, set_scenario_configs accepts them."""
    from mcpower import MCPower

    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    # Must not raise
    m.set_scenario_configs({"optimistic": {"icc_noise_sd": 0.1}})
    assert m._scenario_configs["optimistic"]["icc_noise_sd"] == pytest.approx(0.1)
