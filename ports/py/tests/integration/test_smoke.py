"""Smoke tests — PyO3 boundary, set_n_threads guard, progress sink.

These tests exercise the Rust engine through its Python bindings only. They
build the scenarios payload via the engine's own contract builder
(``build_contract_from_spec``) from a hand-written ``LinearSpec`` JSON — the
same wire-format entry the frontend feeds ``find_power`` — so the boundary
stays locked independently of the ``MCPower`` frontend orchestration.

Kernel-level invariants (run_batch shapes, power monotonicity, invalid spec
rejection, non-increasing sample sizes) are covered by Rust unit tests in
``crates/engine-core/src/batch.rs`` and the orchestrator integration tests
under ``crates/engine-orchestrator/tests/``. This module focuses on the
PyO3-only behaviours: thread-pool guard and progress callback semantics.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

import pytest

from mcpower import _engine

# ---------------------------------------------------------------------------
# Fixtures: minimal scenarios payload built via the engine contract builder
# ---------------------------------------------------------------------------

_BASE_SEED = 2137  # mirrors the frontend's default seed


def _ols_scenarios_bytes(n_predictors: int = 1) -> bytes:
    """Build the ``Vec<SimulationContract>`` wire blob for a minimal OLS model.

    Goes through the engine's contract builder (``build_contract_from_spec``) —
    the same wire-format entry point the frontend feeds ``find_power`` — from a
    hand-written ``LinearSpec`` JSON, so these tests exercise the PyO3 boundary
    on a real contract payload without depending on the ``MCPower`` frontend.
    OLS only: ``n_predictors`` continuous predictors + intercept.
    """
    names = [f"x{i + 1}" for i in range(n_predictors)]
    linear_spec = {
        "formula": "y = " + " + ".join(names),
        "predictors": [{"name": n, "kind": "normal"} for n in names],
        "effects": [{"name": n, "size": 0.5} for n in names],
        "correlations": [],
        "alpha": 0.05,
        "correction": "none",
        "targets": ["overall"],
        "heteroskedasticity": {"driver_var_index": None},
        "residual": {"distribution": "normal"},
        "max_failed_fraction": 0.1,
        "scenarios": [],
    }
    _names, scenarios_bytes, _skeleton = _engine.build_contract_from_spec(
        json.dumps(linear_spec), "continuous", "canonical", "ols", 0.0, "[]"
    )
    return scenarios_bytes


# ---------------------------------------------------------------------------
# set_n_threads guard
# ---------------------------------------------------------------------------


def test_e3_set_n_threads_double_set_raises():
    """set_n_threads must be called before any engine invocation; a call once
    the pool is initialized must raise ``ValueError``.

    The rayon pool is global to the loaded extension module and is
    initialized lazily by the first engine call, so the test makes one
    find_power call itself first (it must not depend on test order) and never
    resizes the pool for the rest of the session.
    """
    from mcpower._engine import find_power, set_n_threads

    find_power(_ols_scenarios_bytes(), 50, 10, _BASE_SEED)
    with pytest.raises(ValueError, match="already initialized"):
        set_n_threads(4)


# ---------------------------------------------------------------------------
# Progress sink: cancel and exception, exercised through find_power
# ---------------------------------------------------------------------------


def test_e4_progress_callback_called_with_counts():
    from mcpower._engine import find_power

    calls = []

    def cb(current, total):
        calls.append((int(current), int(total)))
        return True

    find_power(_ols_scenarios_bytes(), 100, 100, _BASE_SEED, cb)
    assert calls, "progress callback never invoked"
    # total is n_sims on every call, and completion (current == total) is
    # reported. Parallel workers may deliver the cumulative counts out of
    # order, so no ordering is asserted.
    assert all(total == 100 for _, total in calls)
    assert max(current for current, _ in calls) == 100


def test_e4_progress_cancel_returns_false_raises_keyboard_interrupt():
    from mcpower._engine import find_power

    def cancel(current, total):
        return False

    with pytest.raises(KeyboardInterrupt):
        find_power(_ols_scenarios_bytes(), 100, 1000, _BASE_SEED, cancel)


def test_e4_progress_callback_exception_cancels():
    from mcpower._engine import find_power

    def boom(current, total):
        raise RuntimeError("user bailed")

    # An exception in the callback is treated as cancel by the engine →
    # KeyboardInterrupt at the boundary.
    with pytest.raises(KeyboardInterrupt):
        find_power(_ols_scenarios_bytes(), 100, 1000, _BASE_SEED, boom)


_SIGINT_CHILD = """
import signal, sys
# A shell starts background jobs with SIGINT ignored, and Python then installs
# no handler; restore the interactive default so the test does not depend on
# how pytest was launched.
signal.signal(signal.SIGINT, signal.default_int_handler)
from mcpower import MCPower, _engine
_engine.set_n_threads(2)
m = MCPower("y = x1 + x2 + x3").set_effects("x1=0.1, x2=0.2, x3=0.1")
print("start", flush=True)
try:
    # ~100 s of work: only a working Ctrl-C ends it inside the timeout below.
    m.find_power(2000, n_sims=2_000_000, progress_callback=sys.argv[1] == "on",
                 verbose=False)
    print("finished", flush=True)
except KeyboardInterrupt:
    print("interrupted", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT delivery")
@pytest.mark.parametrize("progress", ["off", "on"])
def test_sigint_stops_a_running_find_power(progress):
    """Ctrl-C mid-run raises KeyboardInterrupt promptly, with or without a
    progress callback (the engine runs off the main thread, where CPython
    never delivers signals)."""
    p = subprocess.Popen([sys.executable, "-c", _SIGINT_CHILD, progress],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert p.stdout.readline().strip() == "start"
        time.sleep(1.0)
        p.send_signal(signal.SIGINT)
        sent = time.monotonic()
        out, _ = p.communicate(timeout=20)
        elapsed = time.monotonic() - sent
    finally:
        p.kill()
    assert out.strip() == "interrupted"
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# boundary_hit surface: OLS find_power
# ---------------------------------------------------------------------------


def test_e5_boundary_hit_ols_find_power():
    """Each scenario payload must carry 'boundary_hit' as a nested list of uint8 codes.

    For OLS, all entries must be 0 (no boundary phenomena).
    Shape must be (n_sims, n_sample_sizes) = (n_sims, 1) for find_power.
    """
    from mcpower._engine import find_power

    n_sims = 200
    result = find_power(_ols_scenarios_bytes(), 100, n_sims, _BASE_SEED)

    scenario = result["scenarios"]["optimistic"]
    assert "boundary_hit" in scenario, "'boundary_hit' key missing from scenario payload"
    bh = scenario["boundary_hit"]
    assert isinstance(bh, list), "boundary_hit must be a nested list (not a numpy array)"
    assert len(bh) == n_sims, f"expected {n_sims} rows, got {len(bh)}"
    assert all(isinstance(row, list) and len(row) == 1 for row in bh), "each row must be a 1-element list (n_sample_sizes=1)"
    assert all(v == 0 for row in bh for v in row), "OLS boundary_hit must be all zeros"
