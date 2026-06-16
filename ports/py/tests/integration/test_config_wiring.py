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


def test_report_text_block_present():
    from mcpower.config import get_report_config
    text = get_report_config()["text"]
    assert text["long_title"] == "MCPower · Power Analysis"
    assert text["main_caption"] == "Per-test power"
    assert text["ci_caption"] == "Power & 95% CI"
    assert text["sample_size_caption"] == "Required sample size per effect"
    cols = text["columns"]
    assert cols["test"] == "Test" and cols["power"] == "Power" and cols["target"] == "Target"
    assert cols["ci"] == "CI 95%" and cols["required_n"] == "Required N"
    assert cols["uncorrected"] == "uncorrected" and cols["corrected"] == "corrected"
    assert "{n_sims}" in text["ci_footnote"]


def test_report_text_posthoc_keys_present():
    from mcpower.config import get_report_config
    text = get_report_config()["text"]
    # Post-hoc contrasts nest into the main per-test table (under a
    # "<factor>  (pairwise)" span), so the only post-hoc string still wired
    # through config is the contrast-label join token.
    assert text["vs_token"] == "vs"
