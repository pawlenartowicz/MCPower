"""Exception-safety tests for the tqdm progress wrapper.

The wrapper opens a tqdm bar lazily on the first engine callback. If the
engine subsequently raises, the bar would otherwise leak to stderr until
interpreter shutdown — the model layer wraps engine calls in try/finally
to call ``cb.close()`` so the bar is always released.
"""

from __future__ import annotations

import io
import sys

import pytest

from mcpower.progress import _TqdmCallback, resolve_progress_callback


def test_close_is_idempotent_on_never_opened() -> None:
    cb = _TqdmCallback()
    # Never called → no bar opened. close() must be a safe no-op.
    cb.close()
    cb.close()
    assert cb._bar is None


def test_no_output_within_delay_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bar is constructed on the first report, but tqdm's native `delay`
    # must suppress all drawing until the window elapses — fast runs stay
    # silent. tqdm captures sys.stderr at construction, so redirect it first.
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    cb = _TqdmCallback()
    cb(10, 100)
    cb(20, 100)
    assert cb._bar is not None  # constructed...
    assert buf.getvalue() == ""  # ...but nothing drawn inside the delay window
    cb.close()


def test_bar_opens_at_current_position() -> None:
    cb = _TqdmCallback()
    # The bar is constructed on the first report at the reported position, not
    # from zero, so a bar that starts drawing late doesn't snap back from 0%.
    cb(40, 100)
    assert cb._bar is not None
    assert cb._bar.n == 40
    cb.close()


def test_close_releases_bar_after_partial_progress() -> None:
    cb = _TqdmCallback()
    # Open the bar with a non-trivial total so close() (not the in-call
    # `current >= total` path) is the only thing that can release it.
    cb(10, 100)
    assert cb._bar is not None
    cb.close()
    assert cb._bar is None


def test_close_handles_missing_tqdm_sentinel() -> None:
    cb = _TqdmCallback()
    # Simulate the import-failed path: __call__ assigns `False` to self._bar.
    cb._bar = False  # type: ignore[assignment]
    # close() must not raise AttributeError on the False sentinel.
    cb.close()
    assert cb._bar is None


def test_bar_closed_after_engine_exception() -> None:
    """The model layer's try/finally closes the bar on engine errors."""
    cb = _TqdmCallback()

    def fake_engine_call() -> None:
        # Mimic the engine calling progress mid-run, then panicking.
        cb(50, 1000)
        raise RuntimeError("engine panicked")

    with pytest.raises(RuntimeError, match="engine panicked"):
        try:
            fake_engine_call()
        finally:
            cb.close()

    assert cb._bar is None


def test_resolve_returns_tqdm_callback_for_default() -> None:
    # Sanity check: the model-layer isinstance() guard relies on this.
    assert isinstance(resolve_progress_callback(None), _TqdmCallback)
    assert isinstance(resolve_progress_callback(True), _TqdmCallback)
    assert resolve_progress_callback(False) is None
