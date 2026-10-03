"""find_sample_size long-form summary — joint required-N table.

Contracts:
  1. "Joint detection" + "of 3 tests" present; "analytical fast-path" absent
     (the engine is MC-only; the string should never appear in output).
  2. The header N is the largest per-effect required N, or the
     "N≥<ceiling> (not all reached)" sentinel when a target is unreachable.
  3. "≥ <ceiling>" sentinel appears when the target is unreachable within the
     search range (the required N lies above the grid's top — never a bare "—").
  4. Joint cells follow the fitted_joint chain: fitted → n_achievable,
     at_or_below_min → "≤ n_min", not_reached → "≥ ceiling".
"""

import re

import mcpower
from mcpower.config import get_report_config
from mcpower.output.report import Report


def _fit_sample_size(target_power=0.8):
    model = mcpower.MCPower("y ~ a + b + c")
    model.set_effects("a=0.5, b=0.3, c=0.1")
    return model.find_sample_size(
        from_size=20,
        to_size=400,
        by=20,
        target_power=target_power,
        n_sims=500,
        verbose=False,
    )


def _per_effect_section(text):
    assert "Required sample size per effect" in text
    return text.split("Required sample size per effect")[1].split("\n\n")[0]


def test_long_summary_has_joint_required_n_table():
    res = _fit_sample_size()
    text = str(res.summary())
    assert "Joint detection" in text
    assert "of 3 tests" in text
    assert "analytical fast-path" not in text


def test_long_summary_header_uses_recommended_n_not_first_grid_point():
    # c=0.1 cannot reach 80% within N≤400, so the header shows the ceiling
    # sentinel rather than any grid point.
    res = _fit_sample_size()
    text = str(res.summary())
    assert "N≥400 (not all reached)" in text


def test_long_summary_header_shows_recommended_n_when_all_reached():
    # All-large effects so every target reaches the target power within the
    # grid -> the header must render the recommended "N≥<n>" form (the largest
    # per-effect required N), not the "not reached" sentinel.
    model = mcpower.MCPower("y ~ a + b + c")
    model.set_effects("a=0.5, b=0.5, c=0.5")
    res = model.find_sample_size(
        from_size=20,
        to_size=400,
        by=20,
        target_power=0.8,
        n_sims=500,
        verbose=False,
    )
    text = str(res.summary())
    assert "(not all reached)" not in text
    per_effect = [int(n) for n in re.findall(r"\s(\d+)$", _per_effect_section(text), flags=re.M)]
    header_n = int(re.search(r"N≥(\d+)\s+sims=", text).group(1))
    assert per_effect and header_n == max(per_effect)


def test_unreached_target_uses_geq_ceiling_sentinel():
    # target_power=0.99 is unreachable within from=20..to=400, so the required N
    # lies above the grid ceiling (400) and renders as "≥ 400" in the Required N
    # column. The CI column may legitimately show "—" for unreached rows.
    res = _fit_sample_size(target_power=0.99)
    section = _per_effect_section(str(res.summary()))
    assert "≥ 400" in section
    assert "—" not in section, f"Bare '—' found in required N section:\n{section}"


def test_joint_required_n_uses_fitted_joint_chain():
    """Each joint cell renders from its fitted_joint status, not from the
    first_joint_achieved grid value."""
    inner = {
        "sample_sizes": [40, 80, 120, 160, 200],
        # Grid fallbacks that differ from every fitted_joint rendering below.
        "first_joint_achieved": {0: 80, 1: 40, 2: 160},
        "fitted_joint": {
            0: {"status": "fitted", "n_star": 57.2, "n_achievable": 60},
            1: {"status": "at_or_below_min", "n_min": 40},
            2: {"status": "not_reached", "n_approx": None},
        },
    }
    table = Report(inner, {"target_power": 0.8}, kind="find_sample_size")._joint_required_n_table(get_report_config())
    assert re.search(r"≥ 1 of 3 tests\s+60$", table, flags=re.M)
    assert re.search(r"≥ 2 of 3 tests\s+≤ 40$", table, flags=re.M)
    assert re.search(r"≥ 3 of 3 tests\s+≥ 200$", table, flags=re.M)
