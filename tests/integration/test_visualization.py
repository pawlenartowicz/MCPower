"""
Tests for visualization utilities (matplotlib mocked).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _setup_mock_plt():
    """Create a properly configured mock matplotlib.pyplot."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    # get_cmap returns a callable that returns a color array
    cmap_mock = MagicMock(return_value=np.zeros((10, 4)))
    mock_plt.get_cmap.return_value = cmap_mock
    return mock_plt, mock_fig, mock_ax


def _run_with_mock_plt(mock_plt, fn_kwargs):
    """Import and run _create_power_plot with mocked matplotlib."""
    # Build a mock 'matplotlib' parent whose .pyplot is our mock
    mock_mpl = MagicMock()
    mock_mpl.pyplot = mock_plt

    saved = {}
    for k in list(sys.modules):
        if k.startswith("matplotlib") or k == "mcpower.utils.visualization":
            saved[k] = sys.modules.pop(k)

    sys.modules["matplotlib"] = mock_mpl
    sys.modules["matplotlib.pyplot"] = mock_plt

    try:
        from mcpower.utils.visualization import _create_power_plot

        _create_power_plot(**fn_kwargs)
    finally:
        # Remove our mocks
        sys.modules.pop("matplotlib", None)
        sys.modules.pop("matplotlib.pyplot", None)
        sys.modules.pop("mcpower.utils.visualization", None)
        # Restore originals
        sys.modules.update(saved)


def _make_plot_args(*, achieved=None):
    sample_sizes = [50, 100, 150, 200]
    target_tests = ["x1", "x2"]
    powers_by_test = {
        "x1": [30.0, 60.0, 80.0, 95.0],
        "x2": [20.0, 45.0, 70.0, 85.0],
    }
    if achieved is None:
        achieved = {"x1": 150, "x2": 200}
    return {
        "sample_sizes": sample_sizes,
        "powers_by_test": powers_by_test,
        "first_achieved": achieved,
        "target_tests": target_tests,
        "target_power": 80.0,
        "title": "Test Power Plot",
    }


class TestCreatePowerPlot:
    """Test _create_power_plot with mocked matplotlib."""

    def test_plot_creation(self):
        """Plot is created with correct structure."""
        mock_plt, mock_fig, mock_ax = _setup_mock_plt()
        _run_with_mock_plt(mock_plt, _make_plot_args())

        mock_plt.subplots.assert_called_once()
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()

    def test_achievement_markers(self):
        """Achievement points are annotated when achieved > 0."""
        mock_plt, mock_fig, mock_ax = _setup_mock_plt()
        _run_with_mock_plt(mock_plt, _make_plot_args())

        # Should annotate for both x1 and x2 (both achieved > 0)
        assert mock_ax.annotate.call_count == 2

    def test_no_achievement_markers(self):
        """No annotation when first_achieved is 0 (not achieved)."""
        mock_plt, mock_fig, mock_ax = _setup_mock_plt()
        _run_with_mock_plt(mock_plt, _make_plot_args(achieved={"x1": 0, "x2": 0}))

        assert mock_ax.annotate.call_count == 0

    def test_target_power_line(self):
        """Target power horizontal line is drawn."""
        mock_plt, mock_fig, mock_ax = _setup_mock_plt()
        _run_with_mock_plt(mock_plt, _make_plot_args())

        mock_ax.axhline.assert_called_once()
        call_kwargs = mock_ax.axhline.call_args
        assert call_kwargs[1]["y"] == 80.0

    def test_matplotlib_import_error(self):
        """ImportError raised when matplotlib is not available."""
        # Save and remove matplotlib modules
        saved = {}
        for k in list(sys.modules):
            if k.startswith("matplotlib") or k == "mcpower.utils.visualization":
                saved[k] = sys.modules.pop(k)

        try:
            original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

            def mock_import(name, *args, **kwargs):
                if name == "matplotlib.pyplot" or name == "matplotlib":
                    raise ImportError("No module named 'matplotlib'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from mcpower.utils.visualization import _create_power_plot

                with pytest.raises(ImportError, match="matplotlib"):
                    _create_power_plot(**_make_plot_args())
        finally:
            sys.modules.update(saved)
