"""The internal-error 'report this' hint is appended to engine panics
(surfaced as RuntimeError) but never to validation errors (ValueError).

The panic seam ``_engine.panic_for_test`` exists only in test-bridge builds
(stripped from release wheels), mirroring engine-r's ``find_power_precancelled``.
"""

import pytest

from mcpower import _engine

REPORT_URL = "mcpower.app/report"

panic_for_test = getattr(_engine, "panic_for_test", None)


@pytest.mark.skipif(panic_for_test is None, reason="panic_for_test requires a test-bridge build")
def test_internal_panic_carries_report_hint():
    """A caught engine panic surfaces as RuntimeError with the report URL."""
    with pytest.raises(RuntimeError) as exc_info:
        panic_for_test()
    msg = str(exc_info.value)
    assert REPORT_URL in msg
    assert "port=py" in msg


def test_validation_error_has_no_report_hint():
    """An engine-layer ValueError (malformed contracts) must NOT carry the hint —
    the suffix is reserved for genuine internal failures, not user input."""
    with pytest.raises(ValueError) as exc_info:
        _engine.find_power(b"\xff\xff\xff", 100, 100, 0)
    assert REPORT_URL not in str(exc_info.value)
