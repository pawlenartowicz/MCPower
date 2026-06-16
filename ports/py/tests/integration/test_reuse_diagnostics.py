"""L1 — unit tests for _reuse_fraction and _strict_reuse_warning."""


def test_reuse_fraction_closed_form():
    from mcpower.model import _reuse_fraction

    assert abs(_reuse_fraction(1000, 1000) - 26.0) < 1.0
    assert abs(_reuse_fraction(1000, 2000) - 59.0) < 1.0
    assert _reuse_fraction(1000, 999) >= 0.0


def test_reuse_fraction_edge_cases():
    from mcpower.model import _reuse_fraction

    assert _reuse_fraction(0, 100) == 0.0
    assert _reuse_fraction(-5, 100) == 0.0
    assert _reuse_fraction(1, 100) == 100.0


def test_warning_fires_above_ratio():
    from mcpower.model import _strict_reuse_warning

    assert _strict_reuse_warning(U=100, N=201, ratio=2.0) is not None
    assert _strict_reuse_warning(U=100, N=200, ratio=2.0) is None


def test_warning_message_content():
    from mcpower.model import _strict_reuse_warning

    msg = _strict_reuse_warning(U=50, N=200, ratio=2.0)
    assert msg is not None
    assert "partial" in msg or "none" in msg
    assert "200" in msg
    assert "50" in msg
