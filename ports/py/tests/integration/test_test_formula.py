"""Smoke tests for the ``test_formula`` kwarg on ``find_power`` /
``find_sample_size``.

The test formula lets a user generate data from the full model while fitting
only a strict subset of terms. The power for the kept terms therefore differs
from the full-model power because the omitted term contributes
omitted-variable bias / changes the residual variance.
"""

import pytest

from mcpower import MCPower


def test_test_formula_rejects_unknown_predictor() -> None:
    model = MCPower("y = x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")
    # The error must guide the user to add the term to the model formula
    # (effect may be 0), not just say "not found".
    with pytest.raises(ValueError, match="model formula"):
        model.find_power(sample_size=80, test_formula="y = x1 + z")


def test_test_formula_drops_term_and_changes_test_design() -> None:
    """Generate from x1 + x2 but fit only x1.

    A reduced ``test_formula`` must reshape the downstream test design: the
    full model tests both terms (target_indices [1, 2]) while the reduced
    ``y = x1`` test_formula tests only x1 (a single target). The LHS (``y =``)
    must be stripped before the unknown-variable check, so passing the
    dependent name in the test_formula does not raise. A broken test_formula
    plumb (ignored, or LHS not stripped) fails this shape assertion.

    This guards only the *shape* of the reduced design. That the reduced model
    is actually *fitted* (recovering the omitted-variable / marginal
    coefficient, not just narrowing reported targets) is guarded numerically by
    ``test_test_formula_reduced_fit_recovers_marginal_coefficient`` below.
    """
    seed = 4242
    n_sims = 200
    full = MCPower("y = x1 + x2")
    full.set_effects("x1=0.5, x2=0.3")
    res_full = full.find_power(sample_size=100, n_sims=n_sims, seed=seed)

    reduced = MCPower("y = x1 + x2")
    reduced.set_effects("x1=0.5, x2=0.3")
    res_reduced = reduced.find_power(
        sample_size=100,
        n_sims=n_sims,
        seed=seed,
        test_formula="y = x1",
    )

    # Full model tests both terms; the reduced test_formula tests strictly
    # fewer (here: just x1 → one target). The set of reduced targets must be
    # a proper subset of the full targets.
    full_targets = res_full["target_indices"]
    reduced_targets = res_reduced["target_indices"]
    assert res_full["n_targets"] == 2, full_targets
    assert res_reduced["n_targets"] == 1, reduced_targets
    assert set(reduced_targets) < set(full_targets), (
        f"reduced test_formula must drop a target; full={full_targets}, "
        f"reduced={reduced_targets}"
    )
    # The kept target's power row shrinks to one column accordingly.
    assert all(len(row) == 1 for row in res_reduced["power_uncorrected"])


def test_summary_renders_when_leading_term_dropped() -> None:
    """Rendering a reduced fit that drops the LEADING term and reports a later
    one must not crash.

    Regression for the ``test_formula`` target-index bug: the engine reported the
    kept term's *generation-kernel* column (caffeine = 2) while the effect
    skeleton was in *reduced test-design* space (length 2, valid 0..1), so the
    port report's ``skeleton[target_indices[i]]`` lookup went out of range and
    ``.summary()`` raised ``ValueError``. The power numbers were always correct;
    only the human-readable rendering crashed. Keeping the leading term (every
    other test in this file) hid the bug because then kernel == reduced position.
    """
    model = MCPower("score = study + caffeine")
    model.set_effects("study=0.3, caffeine=0")
    model.set_correlations("corr(study, caffeine)=0.6")
    result = model.find_power(
        sample_size=100,
        n_sims=200,
        seed=2137,
        target_test="caffeine",
        test_formula="score = caffeine",
        verbose=False,
    )

    # Only the kept term is reported, and its power is finite.
    assert result["n_targets"] == 1
    assert 0.0 <= result["power_uncorrected"][0][0] <= 1.0

    # Both the full report and the short repr must render without raising and
    # must name the kept predictor (the former crash site).
    summary = str(result.summary())
    assert "caffeine" in summary, summary
    assert "out of range" not in summary
    assert "caffeine" in repr(result)


def test_test_formula_reduced_fit_recovers_marginal_coefficient() -> None:
    """A reduced ``test_formula`` must REFIT the reduced model, not just narrow
    reported targets.

    DGP: ``exam = 0.1*study + 0.8*ability`` with ``corr(study, ability)=0.7``.
    Fitting the full model recovers ``study``'s *partial* coefficient (0.1 → low
    power). Dropping ``ability`` via ``test_formula="exam = study"`` recovers
    ``study``'s *marginal* coefficient (0.1 + 0.7*0.8 = 0.66 → ~1.0 power at
    N=120). If the engine only narrowed the reported targets (the historical
    bug) the two powers would be byte-identical; the power must jump instead.
    """
    seed = 2137
    n_sims = 1000

    def study_power(test_formula: str | None) -> float:
        model = MCPower("exam = study + ability")
        model.set_effects("study=0.1, ability=0.8")
        model.set_correlations("corr(study, ability)=0.7")
        res = model.find_power(
            sample_size=120,
            n_sims=n_sims,
            seed=seed,
            test_formula=test_formula,
            verbose=False,
        )
        # study is the first target in both the full and reduced designs; one
        # sample size ⇒ a single power row.
        return res["power_uncorrected"][0][0]

    p_full = study_power(None)
    p_reduced = study_power("exam = study")

    assert p_full < 0.30, f"full fit reports study's weak partial power, got {p_full}"
    assert p_reduced > 0.90, (
        f"reduced fit must recover study's strong marginal power, got {p_reduced}"
    )
    assert p_reduced - p_full > 0.50, (
        f"reduced-fit power must jump vs full: {p_reduced} vs {p_full}"
    )
