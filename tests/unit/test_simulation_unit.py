"""
Unit tests for simulation module internals.
"""

import numpy as np
import pytest

from mcpower.core.simulation import _generate_cluster_id_array


class TestGenerateClusterIdArray:
    """Test _generate_cluster_id_array helper."""

    def test_correct_repeat_pattern(self):
        specs = {"school": {"n_clusters": 4, "cluster_size": 3, "tau_squared": 0.5}}
        ids = _generate_cluster_id_array(12, specs)
        expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        assert np.array_equal(ids, expected)

    def test_computed_cluster_size(self):
        specs = {"school": {"n_clusters": 5, "cluster_size": None, "tau_squared": 0.5}}
        ids = _generate_cluster_id_array(100, specs)
        # cluster_size = 100 // 5 = 20
        assert len(ids) == 100
        assert len(np.unique(ids)) == 5
        # Each cluster should have 20 observations
        for c in range(5):
            assert np.sum(ids == c) == 20

    def test_no_cluster_specs(self):
        ids = _generate_cluster_id_array(100, {})
        assert ids is None

    def test_single_cluster(self):
        specs = {"g": {"n_clusters": 1, "cluster_size": 50, "tau_squared": 0.0}}
        ids = _generate_cluster_id_array(50, specs)
        assert np.all(ids == 0)
        assert len(ids) == 50


class TestSimulationCancellation:
    """Test cancellation via SimulationCancelled."""

    def test_cancellation_raises(self):
        from mcpower.progress import SimulationCancelled

        with pytest.raises(SimulationCancelled, match="cancelled"):
            raise SimulationCancelled("Simulation cancelled by user")

    def test_simulation_runner_cancel_check(self):
        """SimulationRunner respects cancel_check callback."""
        from unittest.mock import MagicMock

        from mcpower.core.simulation import SimulationRunner
        from mcpower.progress import SimulationCancelled

        runner = SimulationRunner(n_simulations=10, seed=42)

        # Create a cancel check that cancels immediately
        cancel_fn = MagicMock(return_value=True)

        # Create minimal metadata mock
        metadata = MagicMock()
        metadata.cluster_specs = {}
        metadata.verbose = False

        with pytest.raises(SimulationCancelled):
            runner.run_power_simulations(
                sample_size=100,
                metadata=metadata,
                generate_y_func=MagicMock(),
                analyze_func=MagicMock(),
                create_X_extended_func=MagicMock(),
                cancel_check=cancel_fn,
            )
