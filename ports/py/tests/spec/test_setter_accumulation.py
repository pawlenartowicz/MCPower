"""Multi-call accumulation for the three assignment-string setters.

``set_variable_type`` / ``set_effects`` / ``set_correlations`` accumulate their
fragments and replay them in call order (last-wins per key) instead of
overwriting. Chained/separate calls — declaring one predictor at a time, the
natural incremental pattern — must each survive. Before accumulation, every call
but the last was silently dropped, demoting earlier factors to continuous N(0,1)
columns. (A single combined string also parses correctly — the assignment parser
is paren-aware, so a factor's proportion commas ``(factor,0.5,0.5)`` don't split
it — but separate calls must be correct too.)

Each test here is strictly stronger than the pre-fix code: with overwrite
semantics the earlier fragment is lost, so the surviving-factor / both-effects /
both-correlations assertions fail.
"""

import pytest

from mcpower import MCPower


# ---------------------------------------------------------------------------
# Direct regression: multi-call accumulation through the public API
# ---------------------------------------------------------------------------


def _two_factor_model_separate_calls() -> MCPower:
    """``y ~ g1*g2`` with both factors declared via SEPARATE setter calls (the
    chained one-factor-at-a-time pattern that overwrite used to break)."""
    m = MCPower("y ~ g1*g2")
    m.set_variable_type("g1=(factor, 0.5, 0.5)")
    m.set_variable_type("g2=(factor, 0.6, 0.4)")
    m.set_effects("g1[2]=0.5, g2[2]=0.4, g1[2]:g2[2]=0.3")
    return m


def test_two_factors_separate_calls_both_survive_in_registry():
    """Both factors must register; pre-fix only the last (g2) survived."""
    m = _two_factor_model_separate_calls()
    m._apply()
    assert set(m._registry.factor_names) == {"g1", "g2"}


def test_two_factors_separate_calls_effect_names_both_suffixed():
    """Each factor contributes a ``[level]``-suffixed dummy; the demoted-g1 bug
    left g1 as a bare continuous name."""
    m = _two_factor_model_separate_calls()
    m._apply()
    assert m._registry.effect_names == ["g1[2]", "g2[2]", "g1[2]:g2[2]"]


def test_two_factors_separate_calls_generation_all_factor_columns():
    """The lowered contract must generate two factor columns — no stray
    ``Synthetic{Normal}`` from a demoted factor."""
    m = _two_factor_model_separate_calls()
    cols = m.to_simulation_spec()["generation"]["columns"]
    kinds = [next(iter(c)) for c in cols]
    assert kinds == ["FactorSynthetic", "FactorSynthetic"]


def test_set_effects_two_calls_no_effect_lost():
    """Two ``set_effects`` calls accumulate; pre-fix the first (x1) was dropped."""
    m = MCPower("y ~ x1 + x2")
    m.set_effects("x1=0.3")
    m.set_effects("x2=0.5")
    coeffs = m.to_simulation_spec()["outcome"]["coefficients"]
    # coefficients = [intercept placeholder, x1, x2]
    assert coeffs == [0.0, 0.3, 0.5]


def test_set_correlations_string_two_calls_both_pairs_present():
    """Two ``corr(...)`` string calls accumulate; pre-fix the first pair was
    dropped."""
    m = MCPower("y ~ x1 + x2 + x3")
    m.set_effects("x1=0.3, x2=0.5, x3=0.1")
    m.set_correlations("corr(x1,x2)=0.3")
    m.set_correlations("corr(x1,x3)=0.2")
    m._apply()
    mat = m.correlation_matrix
    assert mat[0][1] == pytest.approx(0.3)  # x1,x2 survived the first call
    assert mat[0][2] == pytest.approx(0.2)  # x1,x3 from the second call


# ---------------------------------------------------------------------------
# Call-grouping independence: contract(N separate) == contract(1 combined)
#
# The strong guard: it catches ANY future setter regressing to overwrite, not
# just today's three. Uses bare ``factor`` (3-level default) for brevity; the
# combined call is a single legal string that must equal the separate calls.
# ---------------------------------------------------------------------------


def _spec(build) -> dict:
    m = MCPower("y ~ a + b")
    build(m)
    m.set_effects("a[2]=0.3, a[3]=0.2, b[2]=0.4, b[3]=0.1")
    return m.to_simulation_spec()


def test_variable_type_combined_equals_separate():
    separate = _spec(lambda m: (m.set_variable_type("a=factor"),
                                m.set_variable_type("b=factor")))
    combined = _spec(lambda m: m.set_variable_type("a=factor, b=factor"))
    assert separate == combined


def test_variable_type_last_wins_per_key():
    """Re-declaring a key in a later call overrides only that key; the combined
    RHS is the deduplicated (surviving) declaration. A clean continuous→
    continuous transition leaves no stale factor/pin state."""
    def separate(m):
        m.set_variable_type("x1=normal")
        m.set_variable_type("x1=right_skewed")  # x1 last-wins → right_skewed
    lw = MCPower("y ~ x1 + x2"); separate(lw); lw.set_effects("x1=0.3, x2=0.5")
    dedup = MCPower("y ~ x1 + x2"); dedup.set_variable_type("x1=right_skewed")
    dedup.set_effects("x1=0.3, x2=0.5")
    assert lw.to_simulation_spec() == dedup.to_simulation_spec()


def test_effects_combined_equals_separate():
    def eff(build):
        m = MCPower("y ~ x1 + x2"); build(m); return m.to_simulation_spec()
    separate = eff(lambda m: (m.set_effects("x1=0.3"), m.set_effects("x2=0.5")))
    combined = eff(lambda m: m.set_effects("x1=0.3, x2=0.5"))
    assert separate == combined


def test_effects_last_wins_per_key():
    def eff(build):
        m = MCPower("y ~ x1 + x2"); build(m); return m.to_simulation_spec()
    lw = eff(lambda m: (m.set_effects("x1=0.3, x2=0.5"), m.set_effects("x1=0.9")))
    dedup = eff(lambda m: m.set_effects("x1=0.9, x2=0.5"))
    assert lw == dedup


def test_string_correlations_combined_equals_separate():
    def corr(build):
        m = MCPower("y ~ x1 + x2 + x3")
        m.set_effects("x1=0.3, x2=0.5, x3=0.1")
        build(m)
        return m.to_simulation_spec()
    separate = corr(lambda m: (m.set_correlations("corr(x1,x2)=0.3"),
                               m.set_correlations("corr(x1,x3)=0.2")))
    combined = corr(lambda m: m.set_correlations("corr(x1,x2)=0.3, corr(x1,x3)=0.2"))
    assert separate == combined


def test_correlation_matrix_call_resets_accumulator():
    """A full-matrix call is a complete spec: it resets the accumulator, so a
    prior string fragment is dropped and the result equals matrix-only."""
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    def corr(build):
        m = MCPower("y ~ x1 + x2 + x3")
        m.set_effects("x1=0.3, x2=0.5, x3=0.1")
        build(m)
        return m.to_simulation_spec()
    string_then_matrix = corr(lambda m: (m.set_correlations("corr(x1,x2)=0.4"),
                                         m.set_correlations(identity)))
    matrix_only = corr(lambda m: m.set_correlations(identity))
    assert string_then_matrix == matrix_only


def test_string_fragment_layers_onto_prior_matrix():
    """A string fragment after a matrix layers its pairwise entries on top of
    that matrix (matrix stays the accumulator base)."""
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    m = MCPower("y ~ x1 + x2 + x3")
    m.set_effects("x1=0.3, x2=0.5, x3=0.1")
    m.set_correlations(identity)
    m.set_correlations("corr(x1,x2)=0.4")
    m._apply()
    mat = m.correlation_matrix
    assert mat[0][1] == pytest.approx(0.4)  # layered on top of the identity base
    assert mat[0][2] == pytest.approx(0.0)


def test_set_baseline_probability_last_wins():
    """set_baseline_probability is a scalar overwrite: the second call wins.

    The intercept slot (outcome["intercept"]) must equal a single call with the
    final value, not accumulate or keep the earlier p=0.3 value. Discriminating:
    log(0.3/0.7) ≈ -0.847 != log(0.5/0.5) = 0.0.
    """
    chain = MCPower("y ~ x", family="logit")
    chain.set_effects("x=0.3")
    chain.set_baseline_probability(0.3).set_baseline_probability(0.5)
    spec_chain = chain.to_simulation_spec()

    single = MCPower("y ~ x", family="logit")
    single.set_effects("x=0.3")
    single.set_baseline_probability(0.5)
    spec_single = single.to_simulation_spec()

    assert spec_chain["outcome"]["intercept"] == spec_single["outcome"]["intercept"]
