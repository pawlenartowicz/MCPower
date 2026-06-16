"""Progress-callback wrappers for the Rust engine.

The Rust engine accepts a single callable ``progress(current, total) -> bool``.
This module translates the user-facing ``progress_callback`` argument
(accepting ``True`` / ``False`` / ``None`` / callable) into that contract.

Behaviour matrix:

    progress_callback = False       -> None  (engine runs silently)
    progress_callback = True        -> tqdm-on-stderr wrapper
    progress_callback = None        -> tqdm-on-stderr wrapper (default)
    progress_callback = callable    -> wrapped as-is (engine calls it directly)

The tqdm wrapper passes ``_PROGRESS_DELAY_SECONDS`` to tqdm's native ``delay``:
the bar is created on the first report (so its ``start_t`` ≈ run start) but
tqdm suppresses drawing until the delay elapses. Runs that finish inside the
window draw no bar at all; slower runs draw one with honest elapsed/rate. A
user-supplied callable is exempt — it is invoked from the first report.

A user callable may return ``None``; this is treated as "continue". Only an
explicit ``False`` cancels the run. Exceptions inside a user callable
propagate through the Rust boundary as ``KeyboardInterrupt`` (engine-py
contract) and are not swallowed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional

ProgressCb = Callable[[int, int], bool]

# Passed to tqdm's native ``delay``: the bar draws nothing until this many
# seconds after it is constructed (the first progress report, ≈ run start), so
# runs that finish inside the window stay silent and only longer runs draw a
# bar. tqdm keeps honest timing because its ``start_t`` is the construction
# moment, not the (later) instant drawing begins.
_PROGRESS_DELAY_SECONDS = 2.0


class _TqdmCallback:
    """Lazily-initialised tqdm wrapper that the engine calls per chunk.

    Initialisation is deferred until the first call because the total may not
    be known when the wrapper is constructed (the engine always passes the
    true total on the first report, so we use that). The bar is created there
    with tqdm's native ``delay`` so it stays silent until the run proves slow
    enough to be worth watching.

    The caller is responsible for invoking :meth:`close` in a ``finally``
    block so the bar is released if the engine raises mid-run.
    """

    def __init__(self) -> None:
        self._bar = None  # tqdm instance, None (not yet opened), or False (tqdm-missing sentinel)
        self._last = 0
        self._total = 0

    def __call__(self, current: int, total: int) -> bool:
        if self._bar is None:
            try:
                from tqdm import tqdm
            except ImportError:  # pragma: no cover — tqdm is a hard dep
                # Fall through to a silent no-op; the run still succeeds.
                self._bar = False  # type: ignore[assignment]
                return True
            # Construct on the first report so tqdm's start_t ≈ run start, then
            # let its native `delay` suppress drawing until the run proves slow.
            # Deferring construction instead (a hand-rolled delay) opens the bar
            # near 100% with a fresh start_t — tqdm then prints a garbage rate.
            self._bar = tqdm(
                total=total,
                unit="sim",
                initial=current,
                delay=_PROGRESS_DELAY_SECONDS,
                file=None,
            )
            self._total = total
            self._last = current
        if self._bar is False:  # type: ignore[comparison-overlap]
            return True
        if total != self._total:
            # Engine may report a different total per scenario; reset bar.
            self._bar.reset(total=total)  # type: ignore[union-attr]
            self._last = 0
            self._total = total
        delta = current - self._last
        if delta > 0:
            self._bar.update(delta)  # type: ignore[union-attr]
            self._last = current
        if current >= total:
            self.close()
        return True

    def close(self) -> None:
        """Idempotently release the underlying tqdm bar.

        Safe to call when the bar was never opened (``self._bar is None``)
        or when tqdm import failed (``self._bar is False``). Tqdm-internal
        exceptions during ``.close()`` are swallowed: the callback's job is
        to release the bar, not to mask the engine error that triggered
        unwinding in the first place.
        """
        bar = self._bar
        self._bar = None
        self._last = 0
        self._total = 0
        if bar is None or bar is False:
            return
        try:
            bar.close()
        except (AttributeError, RuntimeError):
            pass


def resolve_progress_callback(
    progress_callback: object,
) -> Optional[ProgressCb]:
    """Resolve a user-facing ``progress_callback`` to an engine callable.

    Args:
        progress_callback: ``False`` -> silent; ``True`` / ``None`` -> tqdm
            default; otherwise a callable ``(current, total) -> bool``.

    Returns:
        A function accepted by ``_engine.find_power`` / ``find_sample_size``
        as their ``progress=`` argument, or ``None`` for silent operation.
    """
    if progress_callback is False:
        return None
    if progress_callback is None or progress_callback is True:
        return _TqdmCallback()

    if not callable(progress_callback):
        raise TypeError(
            "progress_callback must be True, False, None, or a callable "
            f"(current, total) -> bool; got {type(progress_callback).__name__}"
        )

    user_cb = progress_callback

    def wrapped(current: int, total: int) -> bool:
        rv = user_cb(current, total)
        # `None` / truthy → continue; only an explicit False cancels.
        return rv is not False

    return wrapped


@contextmanager
def managed_progress(progress_callback: object) -> Iterator[Optional[ProgressCb]]:
    """Resolve a progress callback and release any tqdm bar on exit.

    Yields the engine-facing callback (or ``None`` for silent operation).
    tqdm-backed callbacks are closed on context exit so a mid-run engine
    exception doesn't leak the bar to stderr.
    """
    cb = resolve_progress_callback(progress_callback)
    try:
        yield cb
    finally:
        if isinstance(cb, _TqdmCallback):
            cb.close()


__all__ = ["resolve_progress_callback", "managed_progress", "ProgressCb"]
