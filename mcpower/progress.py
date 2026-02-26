"""
Progress reporting for MCPower simulations.

Provides a callback-based progress system that works from both Python scripts
and GUI applications. Progress is reported via a simple (current, total) callback.
"""

import sys
from typing import Callable, Optional


class SimulationCancelled(Exception):
    """Raised when a simulation is cancelled by the user."""

    pass


class ProgressReporter:
    """Wraps a ``(current, total)`` callback with counting and throttling.

    Tracks the number of completed simulation steps and fires the callback
    at most once every *update_every* advances, preventing excessive I/O
    when simulations complete very quickly.

    Args:
        total: Total number of simulation steps.
        callback: Function called as ``callback(current, total)`` on each
            (throttled) update.
        update_every: Fire the callback at most once per this many advances.
            Defaults to ``max(1, total // 200)`` (~200 updates total).
    """

    def __init__(
        self,
        total: int,
        callback: Callable[[int, int], None],
        update_every: Optional[int] = None,
    ):
        self.total = total
        self._callback = callback
        self._current = 0
        self.update_every = update_every if update_every is not None else max(1, total // 200)

    def start(self):
        """Signal the beginning of the run (fires an initial 0/total update)."""
        self._current = 0
        self._callback(0, self.total)

    def advance(self, n: int = 1):
        """Advance the counter by *n* steps, firing the callback when due."""
        self._current += n
        if self._current >= self.total or self._current % self.update_every == 0:
            self._callback(self._current, self.total)

    def finish(self):
        """Signal completion (fires a final total/total update if not already there)."""
        if self._current < self.total:
            self._current = self.total
            self._callback(self.total, self.total)


class PrintReporter:
    """Console progress reporter — prints ``\\rProgress: 45.2% (723/1600 simulations)``."""

    def __call__(self, current: int, total: int):
        if total <= 0:
            return
        pct = 100.0 * current / total
        sys.stderr.write(f"\rProgress: {pct:5.1f}% ({current}/{total} simulations)")
        sys.stderr.flush()
        if current >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()


class TqdmReporter:
    """Optional tqdm-based progress reporter (lazy import).

    Usage::

        from mcpower.progress import TqdmReporter
        model.find_power(100, progress_callback=TqdmReporter())
    """

    def __init__(self, **tqdm_kwargs):
        self._tqdm_kwargs = tqdm_kwargs
        self._bar = None

    def __call__(self, current: int, total: int):
        from tqdm import tqdm

        if self._bar is None:
            self._bar = tqdm(total=total, unit="sim", **self._tqdm_kwargs)

        delta = current - self._bar.n
        if delta > 0:
            self._bar.update(delta)

        if current >= total:
            self._bar.close()
            self._bar = None


def compute_total_simulations(
    n_simulations: int,
    n_sample_sizes: int = 1,
    n_scenarios: int = 1,
) -> int:
    """Return the total number of individual simulation iterations.

    Used to initialise ``ProgressReporter`` with an accurate total.

    Args:
        n_simulations: Simulations per sample-size per scenario.
        n_sample_sizes: Number of sample sizes being tested (1 for
            ``find_power``, multiple for ``find_sample_size``).
        n_scenarios: Number of scenarios (1 for standard analysis,
            more when ``scenarios=True``).

    Returns:
        The product ``n_simulations * n_sample_sizes * n_scenarios``.
    """
    return n_simulations * n_sample_sizes * n_scenarios
