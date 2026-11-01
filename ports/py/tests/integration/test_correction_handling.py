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


def _tukey_model():
    m = MCPower("y = x + f")
    m.set_variable_type("f=(factor,3)")
    m.set_effects("x=0.3, f[2]=0.3, f[3]=0.4")
    return m


def _table_row(text, label):
    """The printed table row whose first cell is `label`."""
    return next(ln for ln in text.splitlines() if ln.split() and ln.split()[0] == label)


def test_tukey_non_factor_target_is_not_applicable():
    """Under Tukey a continuous target has no corrected power: NaN in the dict,
    flagged in corrected_not_applicable, 'n/a' in the table. The factor levels
    keep a corrected value."""
    import math

    r = _tukey_model().find_power(120, correction="tukey", n_sims=200, seed=15, verbose=False)
    assert r["corrected_not_applicable"] == [True, False, False]
    pc = r["power_corrected"][0]
    assert math.isnan(pc[0])
    assert all(0.0 < p < 1.0 for p in pc[1:])
    assert all(math.isnan(b) for b in r["ci_corrected"][0][0])
    assert "n/a" in _table_row(str(r), "x")
    assert "n/a" not in _table_row(str(r), "2")  # factor level f[2]


def test_tukey_non_factor_target_has_no_required_n():
    """find_sample_size under Tukey: the continuous target has no corrected
    crossing (None in the dict, 'n/a' in the table)."""
    r = _tukey_model().find_sample_size(correction="tukey", from_size=40, to_size=200, by=40,
                                       n_sims=100, seed=17, verbose=False)
    assert r["corrected_not_applicable"] == [True, False, False]
    assert r["first_achieved"][0] is None
    assert r["fitted"][0] is None
    assert r["fitted"][1] is not None
    assert _table_row(str(r), "x").split()[-1] == "n/a"
    # The corrected all-targets and joint views are not applicable as a whole.
    assert "First N achieving all targets: n/a" in str(r).splitlines()
    assert r["first_joint_achieved"] is None
    assert r["fitted_joint"] is None
    assert r["success_count_histogram_corrected"] is None
    assert r["success_count_histogram_uncorrected"] is not None
    assert ("Joint detection → required N (target 80%): n/a (a test has no corrected "
            "power under this correction)") in str(r.summary())


def test_tukey_plots_leave_the_flagged_target_out(tmp_path):
    """The plot data carries the flags: the corrected bars and curve drop the
    flagged target, and there is no corrected joint chart."""
    m = _tukey_model()
    r = m.find_power(120, correction="tukey", n_sims=100, seed=15, verbose=False)
    r.save_plot(str(tmp_path / "p.html"))
    html = (tmp_path / "p.html").read_text()
    assert '"target": "f[2]"' in html or '"target":"f[2]"' in html
    assert '"target": "x"' not in html and '"target":"x"' not in html
    s = m.find_sample_size(correction="tukey", from_size=40, to_size=200, by=40,
                           n_sims=50, seed=17, verbose=False)
    s.save_plot(str(tmp_path / "s.html"))
    html = (tmp_path / "s.html").read_text()
    assert "P(detect >= k)" not in html
    # Control: under Holm the same model draws the corrected joint chart.
    h = m.find_sample_size(correction="holm", from_size=40, to_size=200, by=40,
                           n_sims=50, seed=17, verbose=False)
    h.save_plot(str(tmp_path / "h.html"))
    assert "P(detect >= k)" in (tmp_path / "h.html").read_text()
