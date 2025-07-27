import pytest
import numpy as np


def test_import():
    """Test that the package imports correctly."""
    import mcpower

    assert hasattr(mcpower, "LinearRegression")


def test_basic_functionality():
    """Test basic functionality works."""
    import mcpower

    model = mcpower.LinearRegression("y = x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")

    # This should not raise an error
    result = model.find_power(sample_size=100, return_results=True)
    assert result is not None
