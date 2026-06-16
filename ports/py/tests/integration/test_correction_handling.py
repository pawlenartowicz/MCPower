import pytest
import mcpower
from mcpower import MCPower


def _model():
    m = mcpower.MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")
    return m


def test_fdr_alias_maps_to_benjamini_hochberg():
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction="fdr")
    assert spec["correction"] == "benjamini_hochberg"


def test_benjamini_hochberg_underscore_alias_maps_through():
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction="benjamini_hochberg")
    assert spec["correction"] == "benjamini_hochberg"


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("BH", "benjamini_hochberg"),
        ("bh", "benjamini_hochberg"),
        ("Benjamini-Hochberg", "benjamini_hochberg"),
        ("benjamini hochberg", "benjamini_hochberg"),
        ("FDR", "benjamini_hochberg"),
        ("Bonferroni", "bonferroni"),
        ("Holm", "holm"),
    ],
)
def test_correction_aliases_are_case_insensitive(alias, expected):
    """Correction aliases resolve case-insensitively through the call path."""
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction=alias)
    assert spec["correction"] == expected


def test_none_correction_maps_to_none():
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction=None)
    assert spec["correction"] == "none"


def test_tukey_is_accepted_for_posthoc():
    """Tukey HSD is now supported; calling find_power with correction='tukey'
    must NOT raise — it is accepted and mapped to tukey_hsd on the wire."""
    m = _model()
    # No posthoc keyword here, just verifying it doesn't raise.
    result = m.find_power(50, correction="tukey", n_sims=50, verbose=False)
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("tukey", "tukey_hsd"),
        ("Tukey", "tukey_hsd"),
        ("TUKEY", "tukey_hsd"),
        ("tukey_hsd", "tukey_hsd"),
        ("tukey-hsd", "tukey_hsd"),
        ("Tukey HSD", "tukey_hsd"),
        ("Tukey-HSD", "tukey_hsd"),
    ],
)
def test_tukey_aliases_map_to_tukey_hsd_on_wire(alias, expected):
    """All Tukey spelling variants resolve to 'tukey_hsd' on the wire."""
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction=alias)
    assert spec["correction"] == expected


def test_correction_kwarg_to_find_power_works():
    """correction= is now a real find_power kwarg (key inversion vs v2 setter era)."""
    m = _model()
    result = m.find_power(50, correction="bh", n_sims=50)
    assert isinstance(result, dict)
    # Correction must do something measurable (not silently be a no-op).
    pairs = list(zip(result["power_uncorrected"][0], result["power_corrected"][0]))
    assert all(c <= u + 1e-9 for u, c in pairs)


def test_genuinely_unknown_kwarg_to_find_power_raises():
    m = _model()
    with pytest.raises(TypeError):
        m.find_power(50, definitely_not_a_kwarg=True)


def test_removed_set_correction_method_raises_instructive_error():
    """Calling the removed setter points at the correction= kwarg."""
    m = _model()
    with pytest.raises(AttributeError, match="correction"):
        m.set_correction_method("bh")


def test_corrected_power_is_at_most_uncorrected_bonferroni():
    """Bonferroni never increases power: corrected[i] <= uncorrected[i] for every target."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")
    result = m.find_power(200, correction="bonferroni", n_sims=200, seed=2137, verbose=False)
    for u, c in zip(result["power_uncorrected"][0], result["power_corrected"][0]):
        assert c <= u + 1e-9


def test_corrected_power_is_strictly_smaller_than_uncorrected_holm():
    """With two real-power targets at N=80, Holm must reduce at least one.

    N=80 gives intermediate power (~60–75%) so correction has room to bite.
    """
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")
    result = m.find_power(80, correction="holm", n_sims=800, seed=2137, verbose=False)
    pairs = list(zip(result["power_uncorrected"][0], result["power_corrected"][0]))
    assert all(c <= u + 1e-9 for u, c in pairs)
    assert any(c < u - 1e-9 for u, c in pairs), (
        "Holm correction expected to reduce power for at least one target "
        f"at N=80, effects=0.3; got uncorrected={[u for u,_ in pairs]}, "
        f"corrected={[c for _,c in pairs]}"
    )
