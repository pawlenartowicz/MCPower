import mcpower
from mcpower.config import get_simulation_defaults


def test_init_defaults_from_config():
    m = mcpower.MCPower("y ~ x")
    sim = get_simulation_defaults()
    assert m.seed == sim["seed"]
    assert m.alpha == sim["alpha"]
    assert m.power == sim["target_power"] * 100  # 0.8 -> 80.0
    assert m.n_simulations == sim["n_sims"]["ols"]
    assert m.max_failed_simulations == sim["max_failed_fraction"]  # 0.1, tightened from 1.0


def test_init_n_sims_default_by_family():
    # The per-call simulation count defaults are family-aware: OLS and logit
    # (GLM) use the heavier `ols` budget, lme uses the cheaper `mixed` budget
    # because each mixed-model fit is more expensive.
    sim = get_simulation_defaults()
    assert mcpower.MCPower("y ~ x").n_simulations == sim["n_sims"]["ols"]
    assert mcpower.MCPower("y ~ x", family="logit").n_simulations == sim["n_sims"]["ols"]
    assert mcpower.MCPower("y = x + (1|g)", family="lme").n_simulations == sim["n_sims"]["mixed"]
