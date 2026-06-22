"""set_max_failed_simulations honours LME convergence failures.

Tests that:
  1. _check_failure_threshold raises RuntimeError with the right message when
     failure_rate > threshold.
  2. _check_failure_threshold is a no-op when failure_rate <= threshold.
  3. find_power raises RuntimeError when the engine returns convergence_rate
     below 1 - threshold and the user has tightened the threshold.
  4. find_power succeeds (no raise) when threshold is permissive enough.
"""

import contextlib
import io

import pytest

from mcpower import MCPower
from mcpower.output.results import _check_failure_threshold


# ---------------------------------------------------------------------------
# Unit tests for the _check_failure_threshold helper (pure logic, no engine)
# ---------------------------------------------------------------------------


def test_check_failure_threshold_raises_above_threshold():
    """failure_rate=0.2 with threshold=0.1 must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_failure_threshold(
            convergence_rate=[0.8],
            boundary_hit_rate_tau_zero=[0.15],
            boundary_hit_rate_high_tau=[0.05],
            threshold=0.1,
        )


def test_check_failure_threshold_passes_below_threshold():
    """failure_rate=0.05 with threshold=0.1 must not raise."""
    _check_failure_threshold(
        convergence_rate=[0.95],
        boundary_hit_rate_tau_zero=[0.05],
        boundary_hit_rate_high_tau=[0.0],
        threshold=0.1,
    )  # no exception


def test_check_failure_threshold_passes_exactly_at_threshold():
    """failure_rate == threshold: must NOT raise (strict >)."""
    _check_failure_threshold(
        convergence_rate=[0.9],      # failure_rate = 0.1
        boundary_hit_rate_tau_zero=[0.1],
        boundary_hit_rate_high_tau=[0.0],
        threshold=0.1,
    )  # no exception — boundary case is allowed


def test_check_failure_threshold_threshold_one_never_raises():
    """threshold=1.0 never raises, even at full failure."""
    _check_failure_threshold(
        convergence_rate=[0.0],
        boundary_hit_rate_tau_zero=[0.5],
        boundary_hit_rate_high_tau=[0.5],
        threshold=1.0,
    )  # no exception


def test_check_failure_threshold_multi_n_worst_triggers():
    """With multiple N values, the worst one triggers the error."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_failure_threshold(
            convergence_rate=[0.99, 0.99, 0.7],   # third N: failure=0.3
            boundary_hit_rate_tau_zero=[0.0, 0.0, 0.25],
            boundary_hit_rate_high_tau=[0.0, 0.0, 0.05],
            threshold=0.2,
        )


def test_check_failure_threshold_error_message_content():
    """RuntimeError message must include actual rate, threshold, and breakdown."""
    with pytest.raises(RuntimeError) as exc_info:
        _check_failure_threshold(
            convergence_rate=[0.75],
            boundary_hit_rate_tau_zero=[0.20],
            boundary_hit_rate_high_tau=[0.05],
            threshold=0.10,
        )
    msg = str(exc_info.value)
    assert "failure rate" in msg.lower()
    assert "25.0%" in msg or "25%" in msg   # actual rate = 1 - 0.75
    assert "10.0%" in msg or "10%" in msg   # threshold
    assert "tau_zero" in msg
    assert "high_tau" in msg


# ---------------------------------------------------------------------------
# Integration tests: _check_result_failure_threshold wired into MCPower
# ---------------------------------------------------------------------------


def test_check_result_failure_threshold_raises_on_mock_result():
    """_check_result_failure_threshold raises when given a mock low-convergence result."""
    model = MCPower("y ~ x1 + (1|c)", family="lme")
    model.set_effects("x1=0.5").set_cluster("c", ICC=0.2, n_clusters=10)
    model.set_max_failed_simulations(0.1)  # allow 10% failures

    # Construct a synthetic result dict as if the engine returned 25% failures.
    mock_result = {
        "convergence_rate": [0.75],
        "boundary_hit_rate_tau_zero": [0.20],
        "boundary_hit_rate_high_tau": [0.05],
    }
    with pytest.raises(RuntimeError, match="failure rate"):
        model._check_result_failure_threshold(mock_result)


def test_check_result_failure_threshold_passes_on_good_result():
    """_check_result_failure_threshold does not raise when convergence is high."""
    model = MCPower("y ~ x1 + (1|c)", family="lme")
    model.set_effects("x1=0.5").set_cluster("c", ICC=0.2, n_clusters=10)
    model.set_max_failed_simulations(0.1)

    mock_result = {
        "convergence_rate": [0.97],
        "boundary_hit_rate_tau_zero": [0.03],
        "boundary_hit_rate_high_tau": [0.0],
    }
    model._check_result_failure_threshold(mock_result)  # no exception


def test_check_result_failure_threshold_handles_scenarios_envelope():
    """Multi-scenario envelope: each scenario is checked independently."""
    model = MCPower("y ~ x1 + (1|c)", family="lme")
    model.set_effects("x1=0.5").set_cluster("c", ICC=0.2, n_clusters=10)
    model.set_max_failed_simulations(0.1)

    # One scenario OK, one scenario bad.
    mock_result = {
        "scenarios": {
            "optimistic": {
                "convergence_rate": [0.97],
                "boundary_hit_rate_tau_zero": [0.03],
                "boundary_hit_rate_high_tau": [0.0],
            },
            "pessimistic": {
                "convergence_rate": [0.7],   # 30% failure > 10% threshold
                "boundary_hit_rate_tau_zero": [0.25],
                "boundary_hit_rate_high_tau": [0.05],
            },
        }
    }
    with pytest.raises(RuntimeError, match="failure rate"):
        model._check_result_failure_threshold(mock_result)


def test_find_power_no_raise_with_permissive_threshold():
    """Default threshold=1.0 never raises regardless of convergence rate."""
    model = MCPower("y ~ x1 + (1|c)", family="lme")
    model.set_effects("x1=0.5").set_cluster("c", ICC=0.2, n_clusters=10)
    # Default max_failed_simulations=1.0 — should never raise.
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.find_power(100, n_sims=50, seed=2137)
    assert "convergence_rate" in result


def test_find_power_ols_never_raises():
    """OLS has no LME failures; any threshold is safe."""
    model = MCPower("y ~ x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")
    model.set_max_failed_simulations(0.0)   # tightest possible
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.find_power(100, n_sims=50, seed=42)
    # OLS on a well-conditioned design converges on every sim, so the rate is
    # exactly 1.0 (50/50). Fixed seed + exact equality — no sampling tolerance:
    # `pytest.approx` would mask a genuine sub-1.0 convergence regression.
    assert result["convergence_rate"][0] == 1.0
