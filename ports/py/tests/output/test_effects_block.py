"""The effects block (input β, standardized β, Cohen's d, OR/RR, model R²):
wiring from the engine and rendering. The numbers themselves are checked in
engine-core (`test_standardized_effects.rs`)."""

import math
import re

from mcpower import MCPower


def _run(model, **kw):
    return model.find_power(sample_size=100, n_sims=20, verbose=False, progress_callback=False, **kw)


def _block(text: str) -> str:
    """The effects block: from the "Effect" header to the R² line, inclusive."""
    start = text.index("\nEffect ") + 1
    end = text.find("\n", text.index("\nR²", start) + 1)
    return text[start:] if end == -1 else text[start:end]


def test_values_reach_the_result_and_both_forms():
    # One normal predictor: std β = β / √(β² + 1), R² = β² / (β² + 1).
    r = _run(MCPower("y = x1").set_effects("x1=0.5"))
    se = r.standardized_effects
    assert math.isclose(se["effects"]["x1"]["std_beta"], 0.5 / math.sqrt(1.25), rel_tol=1e-12)
    assert se["effects"]["x1"]["cohen_d"] is None
    assert math.isclose(se["r2"], 0.25 / 1.25, rel_tol=1e-12)
    for text in (str(r), str(r.summary())):
        block = _block(text)
        assert re.search(r"^x1\s+0\.50\s+0\.447\s+—$", block, re.M)
        assert re.search(r"^R² = 0\.200$", block, re.M)


def test_binary_and_factor_rows_carry_d():
    m = MCPower("y = treat + group")
    m.set_variable_type("treat=binary, group=(factor,3)")
    m.set_effects("treat=0.5, group[2]=0.2, group[3]=0.4")
    se = _run(m).standardized_effects["effects"]
    assert se["treat"]["cohen_d"] is not None
    assert se["group[2]"]["cohen_d"] is not None


def test_test_formula_still_lists_every_simulated_effect():
    m = MCPower("y = x1 + x2").set_effects("x1=0.3, x2=0.4")
    r = _run(m, test_formula="y = x2", target_test="x2")
    assert set(r.standardized_effects["effects"]) == {"x1", "x2"}
    assert re.search(r"^x1\s+0\.30\s", _block(str(r)), re.M)


def test_or_only_on_the_logit_link():
    logit = _run(MCPower("y = x1", family="logit").set_effects("x1=0.5").set_baseline_probability(0.3))
    probit = _run(MCPower("y = x1", family="probit").set_effects("x1=0.5").set_baseline_probability(0.3))
    assert re.search(r"^Effect\s+Input\s+Std\. β\s+d\s+OR$", _block(str(logit)), re.M)
    for text in (str(probit), str(probit.summary())):
        assert "OR" not in text.split()


def test_count_outcome_prints_rr_and_dashes():
    r = _run(MCPower("y = x1", family="poisson").set_effects("x1=0.3").set_baseline_rate(2.0))
    block = _block(str(r))
    assert re.search(r"^Effect\s+Input\s+Std\. β\s+d\s+RR$", block, re.M)
    assert re.search(r"^x1\s+0\.30\s+—\s+—\s+1\.35$", block, re.M)
    assert re.search(r"^R² = —$", block, re.M)
    assert r.standardized_effects["r2"] is None


def test_clustered_run_labels_the_fixed_effects_r2():
    m = MCPower("y ~ x1 + (1|grp)").set_effects("x1=0.5").set_cluster("grp", ICC=0.2, n_clusters=10)
    block = _block(str(_run(m)))
    assert re.search(r"^R² \(fixed effects\) = 0\.\d{3}$", block, re.M)
