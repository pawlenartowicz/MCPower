import pytest

import mcpower
from mcpower import MCPower


def _model():
    m = mcpower.MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")
    return m


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("BH", "benjamini_hochberg"),
        ("bh", "benjamini_hochberg"),
        ("Benjamini-Hochberg", "benjamini_hochberg"),
        ("benjamini hochberg", "benjamini_hochberg"),
        ("benjamini_hochberg", "benjamini_hochberg"),
        ("fdr", "benjamini_hochberg"),
        ("FDR", "benjamini_hochberg"),
        ("Bonferroni", "bonferroni"),
        ("Holm", "holm"),
    ],
)
def test_correction_aliases_are_case_insensitive(alias, expected):  # APIC-63
    """Correction aliases resolve case-insensitively through the call path;
    a canonical name passes through unchanged."""
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction=alias)
    assert spec["correction"] == expected


def test_none_correction_maps_to_none():
    m = _model()
    spec = m._to_linear_spec_dict(["optimistic"], correction=None)
    assert spec["correction"] == "none"


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


def test_genuinely_unknown_kwarg_to_find_power_raises():
    m = _model()
    with pytest.raises(TypeError):
        m.find_power(50, definitely_not_a_kwarg=True)


def test_removed_set_correction_method_raises_instructive_error():
    """Calling the removed setter points at the correction= kwarg."""
    m = _model()
    with pytest.raises(AttributeError, match="correction"):
        m.set_correction_method("bh")


@pytest.mark.parametrize("correction", ["holm", "bonferroni", "bh"])
def test_corrected_power_is_strictly_smaller_than_uncorrected(correction):
    """The correction= kwarg reaches the engine and bites: with two real-power
    targets at N=80, corrected power never exceeds uncorrected and is strictly
    lower for at least one target.

    N=80 gives intermediate power (~60–75%) so correction has room to bite.
    """
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")
    result = m.find_power(80, correction=correction, n_sims=800, seed=2137, verbose=False)
    pairs = list(zip(result["power_uncorrected"][0], result["power_corrected"][0], strict=True))
    assert all(c <= u + 1e-9 for u, c in pairs)
    assert any(c < u - 1e-9 for u, c in pairs), (
        f"{correction} correction expected to reduce power for at least one target "
        f"at N=80, effects=0.3; got uncorrected={[u for u, _ in pairs]}, "
        f"corrected={[c for _, c in pairs]}"
    )
