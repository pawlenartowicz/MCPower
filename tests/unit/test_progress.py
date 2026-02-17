"""
Tests for progress reporting module.
"""

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from mcpower.progress import (
    PrintReporter,
    ProgressReporter,
    SimulationCancelled,
    TqdmReporter,
    compute_total_simulations,
)


class TestSimulationCancelled:
    """Test SimulationCancelled exception."""

    def test_is_exception(self):
        assert issubclass(SimulationCancelled, Exception)

    def test_message(self):
        exc = SimulationCancelled("cancelled by user")
        assert str(exc) == "cancelled by user"


class TestProgressReporter:
    """Test ProgressReporter throttled callback wrapper."""

    def test_start_fires_zero(self):
        cb = MagicMock()
        pr = ProgressReporter(100, cb)
        pr.start()
        cb.assert_called_with(0, 100)

    def test_advance_throttled(self):
        cb = MagicMock()
        pr = ProgressReporter(100, cb, update_every=10)
        pr.start()
        cb.reset_mock()

        # Advance 5 — not a multiple of 10, should NOT fire
        pr.advance(5)
        assert cb.call_count == 0

        # Advance 5 more — now at 10 (multiple of 10), SHOULD fire
        pr.advance(5)
        cb.assert_called_with(10, 100)

    def test_advance_fires_at_completion(self):
        cb = MagicMock()
        pr = ProgressReporter(10, cb, update_every=100)
        pr.start()
        cb.reset_mock()

        # Even with large update_every, completion should fire
        for _ in range(10):
            pr.advance(1)

        # Should have fired at least once (at completion)
        assert cb.called
        cb.assert_called_with(10, 10)

    def test_finish_fires_final_update(self):
        cb = MagicMock()
        pr = ProgressReporter(100, cb)
        pr.start()
        pr.advance(50)
        cb.reset_mock()

        pr.finish()
        cb.assert_called_with(100, 100)

    def test_finish_no_double_fire(self):
        cb = MagicMock()
        pr = ProgressReporter(10, cb, update_every=1)
        pr.start()
        for _ in range(10):
            pr.advance(1)
        cb.reset_mock()

        # Already at total, finish should not fire again
        pr.finish()
        assert cb.call_count == 0

    def test_default_update_every(self):
        cb = MagicMock()
        pr = ProgressReporter(1000, cb)
        # Default: max(1, 1000 // 200) = 5
        assert pr.update_every == 5


class TestPrintReporter:
    """Test PrintReporter console output."""

    def test_output_format(self):
        reporter = PrintReporter()
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            reporter(50, 100)

        output = buf.getvalue()
        assert "50.0%" in output
        assert "50/100" in output

    def test_total_zero_early_return(self):
        reporter = PrintReporter()
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            reporter(0, 0)

        assert buf.getvalue() == ""

    def test_completion_newline(self):
        reporter = PrintReporter()
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            reporter(100, 100)

        assert buf.getvalue().endswith("\n")


class TestTqdmReporter:
    """Test TqdmReporter with mock tqdm."""

    def test_tqdm_missing_raises(self):
        reporter = TqdmReporter()
        with patch.dict("sys.modules", {"tqdm": None}):
            with pytest.raises(ImportError, match="tqdm"):
                reporter(0, 100)

    def test_tqdm_basic_flow(self):
        mock_bar = MagicMock()
        mock_bar.n = 0
        mock_tqdm_cls = MagicMock(return_value=mock_bar)
        mock_tqdm_module = MagicMock()
        mock_tqdm_module.tqdm = mock_tqdm_cls

        reporter = TqdmReporter()

        with patch.dict("sys.modules", {"tqdm": mock_tqdm_module}):
            reporter(0, 100)  # creates bar
            mock_tqdm_cls.assert_called_once()

            mock_bar.n = 0
            reporter(50, 100)  # updates
            mock_bar.update.assert_called_with(50)

            mock_bar.n = 50
            reporter(100, 100)  # closes
            mock_bar.close.assert_called_once()


class TestComputeTotalSimulations:
    """Test compute_total_simulations helper."""

    def test_single_sample_size(self):
        assert compute_total_simulations(1000) == 1000

    def test_multiple_sample_sizes(self):
        assert compute_total_simulations(1000, n_sample_sizes=5) == 5000

    def test_with_scenarios(self):
        assert compute_total_simulations(1000, n_sample_sizes=5, n_scenarios=3) == 15000
