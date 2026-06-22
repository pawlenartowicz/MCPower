"""Baseline-probability gate: hard reject outside ``(0, 1)``, soft warn outside
the ``limits.baseline_p_warn`` band ``[0.05, 0.95]``."""

from mcpower.spec.validators import _validate_baseline_probability


def test_baseline_p_inside_band_is_clean():
    r = _validate_baseline_probability(0.05)  # inclusive lower edge of the band
    assert r.is_valid
    assert r.warnings == []


def test_baseline_p_outside_band_warns_but_is_valid():
    r = _validate_baseline_probability(0.04)
    assert r.is_valid
    assert any("extreme" in w.lower() for w in r.warnings)


def test_baseline_p_zero_is_rejected():
    r = _validate_baseline_probability(0.0)
    assert not r.is_valid


def test_baseline_p_one_is_rejected():
    r = _validate_baseline_probability(1.0)
    assert not r.is_valid
