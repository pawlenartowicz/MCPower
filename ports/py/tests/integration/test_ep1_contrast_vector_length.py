"""EP-1: power/ci/count vectors are sized len(target_indices) + len(contrast_pairs).

The engine contract guarantees that the result vectors
(power_uncorrected, ci_uncorrected, etc.) have one slot per *marginal* target
and one slot per *explicit contrast pair*:

    len(power_uncorrected[0]) == len(target_indices) + len(contrast_pairs)

Prior to this test, NO bridge test passed a contract with an explicit
contrast_pair through find_power across the FFI and verified the resulting
vector length.  The DSL→wire tests stop before calling the engine; the
posthoc tests use the posthoc-expansion path (engine-side), not the
explicit contrast_pairs field.

The target_test DSL form "group[2] vs group[3]" routes through the
contrast_pairs path (not posthoc), so this test exercises the branch
the gap report identified.
"""

from __future__ import annotations

from mcpower import MCPower


def _three_level_factor_model() -> MCPower:
    """OLS: y = x1 + group, where group is a 3-level integer-indexed factor.

    After _apply() the factor expands to dummies group[2] and group[3]
    (group[1] is the reference level, omitted from the design matrix).
    """
    m = MCPower("y = x1 + group")
    m.set_variable_type("group=(factor,0.34,0.33,0.33)")
    m.set_effects("x1=0.3")
    m._apply()
    for eff_name in list(m._registry._effects.keys()):
        if eff_name.startswith("group["):
            m._registry._effects[eff_name].effect_size = 0.4
    return m


def test_ep1_explicit_contrast_pair_vector_length():
    """EP-1: power_uncorrected and ci_uncorrected lengths == len(target_indices)
    + len(contrast_pairs) when an explicit pairwise contrast is the only target.

    Uses target_test='group[2] vs group[3]' — the integer-level contrast DSL form —
    which populates contrast_pairs and leaves target_indices empty.
    The vector length must equal 0 + 1 = 1, not 0 (absent) or 2 (spurious marginals).
    """
    m = _three_level_factor_model()
    result = m.find_power(
        150,
        target_test="group[2] vs group[3]",
        n_sims=100,
        seed=2137,
        verbose=False,
    )

    target_indices = result["target_indices"]
    contrast_pairs = result["contrast_pairs"]

    # At least one explicit contrast pair must have been sent and returned.
    assert len(contrast_pairs) >= 1, (
        f"expected at least one contrast pair; got contrast_pairs={contrast_pairs}"
    )

    expected_len = len(target_indices) + len(contrast_pairs)

    # power_uncorrected is [[...]] — one inner list per evaluated N.
    power_row = result["power_uncorrected"][0]
    assert len(power_row) == expected_len, (
        f"power_uncorrected[0] length {len(power_row)} != "
        f"len(target_indices)={len(target_indices)} + "
        f"len(contrast_pairs)={len(contrast_pairs)} = {expected_len}"
    )

    # ci_uncorrected mirrors the same shape.
    ci_row = result["ci_uncorrected"][0]
    assert len(ci_row) == expected_len, (
        f"ci_uncorrected[0] length {len(ci_row)} != expected {expected_len}"
    )


def test_ep3_find_sample_size_contrast_vector_length():
    """EP-3: find_sample_size result arrays sized len(target_indices) + len(contrast_pairs).

    EP-1 (this file's existing tests) covers find_power's vector lengths through
    the FFI with an explicit contrast pair.  EP-3 covers the dispatch twin
    find_sample_size, which could diverge silently: the Rust SampleSizeResult
    slots first_achieved and fitted are parallel to PowerResult's power vectors,
    and a twin-drift would produce wrong slot counts without any existing test
    catching it.

    Uses the same three-level factor + "group[2] vs group[3]" contrast DSL to
    exercise the contrast_pairs path, then verifies:
      - contrast_pairs has at least one entry (the DSL form was recognised).
      - first_achieved has one slot per (marginal target + contrast pair).
      - fitted has the same count (independent of whether the crossing was reached).
      - power_uncorrected[n_idx] (the per-N power vector at any grid point)
        has the same length.
    """
    m = _three_level_factor_model()
    result = m.find_sample_size(
        from_size=50,
        to_size=200,
        by=50,
        target_test="group[2] vs group[3]",
        n_sims=100,
        seed=2137,
        verbose=False,
    )

    target_indices = result["target_indices"]
    contrast_pairs = result["contrast_pairs"]

    assert len(contrast_pairs) >= 1, (
        f"expected at least one contrast pair; got contrast_pairs={contrast_pairs}"
    )

    expected_len = len(target_indices) + len(contrast_pairs)

    # first_achieved: dict keyed by integer position, one entry per target slot.
    first_achieved = result["first_achieved"]
    assert len(first_achieved) == expected_len, (
        f"first_achieved has {len(first_achieved)} slots but "
        f"len(target_indices)={len(target_indices)} + "
        f"len(contrast_pairs)={len(contrast_pairs)} = {expected_len}"
    )

    # fitted: same positional dict as first_achieved.
    fitted = result["fitted"]
    assert len(fitted) == expected_len, (
        f"fitted has {len(fitted)} slots but expected {expected_len}"
    )

    # power_uncorrected is [[...], ...] — one inner list per grid N.
    # Each inner list must have one entry per target slot.
    power_rows = result["power_uncorrected"]
    assert len(power_rows) >= 1, "power_uncorrected must have at least one grid point"
    for n_idx, power_row in enumerate(power_rows):
        assert len(power_row) == expected_len, (
            f"power_uncorrected[{n_idx}] length {len(power_row)} != expected {expected_len}"
        )


def test_ep1_mixed_marginal_and_contrast_vector_length():
    """EP-1 variant: one marginal target + one contrast → vector length == 2.

    target_test='x1, group[2] vs group[3]' produces
    target_indices=[1] (x1 column) and contrast_pairs=[[2, 3]],
    so the expected vector length is 1 + 1 = 2.
    """
    m = _three_level_factor_model()
    result = m.find_power(
        150,
        target_test="x1, group[2] vs group[3]",
        n_sims=100,
        seed=2137,
        verbose=False,
    )

    target_indices = result["target_indices"]
    contrast_pairs = result["contrast_pairs"]

    assert len(target_indices) == 1, (
        f"expected 1 marginal target; got target_indices={target_indices}"
    )
    assert len(contrast_pairs) >= 1, (
        f"expected at least one contrast pair; got contrast_pairs={contrast_pairs}"
    )

    expected_len = len(target_indices) + len(contrast_pairs)

    power_row = result["power_uncorrected"][0]
    assert len(power_row) == expected_len, (
        f"power_uncorrected[0] length {len(power_row)} != expected {expected_len}"
    )

    ci_row = result["ci_uncorrected"][0]
    assert len(ci_row) == expected_len, (
        f"ci_uncorrected[0] length {len(ci_row)} != expected {expected_len}"
    )
