"""Tests for simulation.py — failure handling, Wald fallback, verbose diagnostics, ICC mismatch."""

import warnings
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcpower.core.simulation import SimulationMetadata, SimulationRunner, _warn_icc_mismatch


def _make_metadata(
    n_targets=2,
    cluster_specs=None,
    verbose=False,
    correction_method=0,
):
    """Create a minimal SimulationMetadata for testing."""
    return SimulationMetadata(
        target_indices=np.arange(n_targets),
        n_non_factor_vars=n_targets,
        correlation_matrix=np.eye(n_targets),
        var_types=np.zeros(n_targets, dtype=np.int64),
        var_params=np.zeros(n_targets, dtype=np.float64),
        factor_specs=[],
        upload_normal_values=np.zeros((2, 2), dtype=np.float64),
        upload_data_values=np.zeros((2, 2), dtype=np.float64),
        effect_sizes=np.array([0.5] * n_targets),
        correction_method=correction_method,
        cluster_specs=cluster_specs or {},
        verbose=verbose,
    )


def _noop_perturbations(corr, types, config, seed):
    return corr, types


class TestAllSimulationsFail:
    """When all simulations return None, RuntimeError should be raised."""

    def test_all_fail_raises(self):
        runner = SimulationRunner(n_simulations=5, seed=42)
        metadata = _make_metadata()

        def failing_sim(*args, **kwargs):
            return None

        with patch.object(runner, "_single_simulation", return_value=None):
            with pytest.raises(RuntimeError, match="All simulations failed"):
                runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )


class TestLMEThresholdExceeded:
    """LME failure rate exceeding threshold raises RuntimeError."""

    def test_high_failure_rate_raises(self):
        runner = SimulationRunner(n_simulations=10, seed=42, max_failed_simulations=0.05)
        metadata = _make_metadata(cluster_specs={"school": {"n_clusters": 5, "cluster_size": 10}})

        call_count = [0]

        def sometimes_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 5:
                return None  # 5 out of 10 fail = 50%
            return (np.array([1, 1, 1]), np.array([1, 1, 1]), False)

        with patch.object(runner, "_single_simulation", side_effect=sometimes_fail):
            with pytest.raises(RuntimeError, match="Too many failed simulations"):
                runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )


class TestOLSHighFailureWarns:
    """OLS high failure rate warns but doesn't raise."""

    def test_ols_warns_above_10_percent(self):
        runner = SimulationRunner(n_simulations=10, seed=42)
        metadata = _make_metadata()  # No cluster_specs = OLS

        call_count = [0]

        def sometimes_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return None  # 2 out of 10 fail = 20%
            return (np.array([1, 1, 1]), np.array([1, 1, 1]))

        with patch.object(runner, "_single_simulation", side_effect=sometimes_fail):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )
                assert any("failed" in str(warning.message).lower() for warning in w)


class TestWaldFallbackWarning:
    """Warn if >10% iterations use Wald test."""

    def test_wald_warning_above_threshold(self):
        runner = SimulationRunner(n_simulations=10, seed=42)
        metadata = _make_metadata()

        call_count = [0]

        def wald_heavy(*args, **kwargs):
            call_count[0] += 1
            # All return wald_flag=True
            return (np.array([1, 1, 1]), np.array([1, 1, 1]), True)

        with patch.object(runner, "_single_simulation", side_effect=wald_heavy):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )
                assert any("Wald test fallback" in str(warning.message) for warning in w)
                assert result["n_wald_fallbacks"] == 10


class TestVerboseDiagnostics:
    """Verbose mode collects diagnostics and failure reasons."""

    def test_verbose_success_collects_diagnostics(self):
        runner = SimulationRunner(n_simulations=3, seed=42)
        metadata = _make_metadata(verbose=True)

        def verbose_result(*args, **kwargs):
            return {
                "results": (np.array([1, 1, 1]), np.array([1, 1, 1])),
                "diagnostics": {"icc_estimated": 0.2},
                "wald_fallback": False,
            }

        with patch.object(runner, "_single_simulation", side_effect=verbose_result):
            result = runner.run_power_simulations(
                sample_size=100,
                metadata=metadata,
                generate_y_func=MagicMock(),
                analyze_func=MagicMock(),
                create_X_extended_func=MagicMock(),
                apply_perturbations_func=_noop_perturbations,
            )
            assert "diagnostics" in result
            assert len(result["diagnostics"]) == 3

    def test_verbose_failure_tracking(self):
        runner = SimulationRunner(n_simulations=5, seed=42)
        metadata = _make_metadata(verbose=True)

        call_count = [0]

        def mixed_results(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"failed": True, "failure_reason": "Convergence failed"}
            return {
                "results": (np.array([1, 1, 1]), np.array([1, 1, 1])),
                "diagnostics": {},
                "wald_fallback": False,
            }

        with patch.object(runner, "_single_simulation", side_effect=mixed_results):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )
            assert "failure_reasons" in result
            assert result["failure_reasons"]["Convergence failed"] == 2

    def test_verbose_none_tracking(self):
        """None results in verbose mode are tracked as unknown failures."""
        runner = SimulationRunner(n_simulations=3, seed=42)
        metadata = _make_metadata(verbose=True)

        call_count = [0]

        def mixed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return None
            return {
                "results": (np.array([1, 1, 1]), np.array([1, 1, 1])),
                "diagnostics": {},
                "wald_fallback": False,
            }

        with patch.object(runner, "_single_simulation", side_effect=mixed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = runner.run_power_simulations(
                    sample_size=100,
                    metadata=metadata,
                    generate_y_func=MagicMock(),
                    analyze_func=MagicMock(),
                    create_X_extended_func=MagicMock(),
                    apply_perturbations_func=_noop_perturbations,
                )
            assert "Unknown (returned None)" in result["failure_reasons"]


class TestICCMismatchWarning:
    """ICC mismatch warning when estimated ICC differs by >50%."""

    def test_large_mismatch_warns(self):
        metadata = _make_metadata(
            cluster_specs={"school": {"icc": 0.2, "n_clusters": 20, "cluster_size": 10}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_icc_mismatch(metadata, mean_estimated_icc=0.05)  # 75% deviation
            assert any("differs from specified" in str(warning.message) for warning in w)

    def test_within_tolerance_no_warning(self):
        metadata = _make_metadata(
            cluster_specs={"school": {"icc": 0.2, "n_clusters": 20, "cluster_size": 10}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_icc_mismatch(metadata, mean_estimated_icc=0.18)  # 10% deviation
            icc_warnings = [x for x in w if "differs from specified" in str(x.message)]
            assert len(icc_warnings) == 0

    def test_zero_estimated_icc_no_warning(self):
        metadata = _make_metadata(
            cluster_specs={"school": {"icc": 0.2, "n_clusters": 20, "cluster_size": 10}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_icc_mismatch(metadata, mean_estimated_icc=0.0)
            icc_warnings = [x for x in w if "differs from specified" in str(x.message)]
            assert len(icc_warnings) == 0

    def test_no_icc_in_spec_no_warning(self):
        metadata = _make_metadata(
            cluster_specs={"school": {"icc": None, "n_clusters": 20, "cluster_size": 10}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_icc_mismatch(metadata, mean_estimated_icc=0.5)
            icc_warnings = [x for x in w if "differs from specified" in str(x.message)]
            assert len(icc_warnings) == 0
