"""Tests for distributions.py — optimizer functions and edge cases."""

import numpy as np
import pytest

from mcpower.stats.distributions import minimize_lbfgsb, minimize_scalar_brent


class TestOptimizerLBFGSB:
    """L-BFGS-B optimizer via native backend."""

    def test_finds_correct_minimum(self):
        # Simple quadratic: f(x) = (x-2)^2
        result = minimize_lbfgsb(
            lambda x: float((x[0] - 2) ** 2),
            x0=np.array([0.0]),
            bounds=[(-10.0, 10.0)],
        )
        assert abs(result.x[0] - 2.0) < 0.01
        assert result.fun < 0.01


class TestOptimizerBrent:
    """Brent scalar minimizer via native backend."""

    def test_finds_correct_minimum(self):
        # f(x) = (x - 3)^2
        result = minimize_scalar_brent(
            lambda x: (x - 3) ** 2,
            bounds=(0.0, 10.0),
        )
        assert abs(result.x - 3.0) < 0.01
        assert result.fun < 0.01

    def test_converged_flag(self):
        result = minimize_scalar_brent(
            lambda x: (x - 5) ** 2,
            bounds=(0.0, 10.0),
        )
        assert result.converged


