"""
Shared fixtures for MCPower tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def simple_model():
    """Create a simple two-predictor model."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2")
    return model


@pytest.fixture
def configured_model():
    """Create a fully configured model ready for analysis."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2")
    model.set_effects("x1=0.3, x2=0.2")
    return model


@pytest.fixture
def interaction_model():
    """Create a model with interaction term."""
    from mcpower import MCPower

    model = MCPower("y = a + b + a:b")
    model.set_effects("a=0.4, b=0.3, a:b=0.2")
    return model


@pytest.fixture
def factor_model():
    """Create a model with factor variable."""
    from mcpower import MCPower

    model = MCPower("y = group + x1")
    model.set_variable_type("group=(factor,3)")
    model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
    return model


@pytest.fixture
def correlation_matrix_2x2():
    """Create a simple 2x2 correlation matrix."""
    return np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.fixture
def correlation_matrix_3x3():
    """Create a 3x3 correlation matrix."""
    return np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])


@pytest.fixture
def sample_data():
    """Create sample empirical data."""
    np.random.seed(42)
    return {
        "x1": np.random.exponential(2, 100),
        "x2": np.random.normal(0, 1, 100),
    }


@pytest.fixture
def suppress_output(capsys):
    """Suppress print output during tests by capturing it."""
    yield
    # Output is automatically captured by capsys


BACKENDS = ["c++"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    """
    Force MCPower to run on a specific backend.

    Parametrizes tests against C++ (primary backend).
    Automatically resets backend after each test.
    """
    from mcpower.backends import reset_backend, set_backend

    set_backend(request.param)
    yield request.param
    reset_backend()


@pytest.fixture(autouse=True)
def reset_backend_after_test():
    """
    Automatically reset backend to default after every test.

    Ensures no hidden backend state leaks between tests.
    """
    yield
    from mcpower.backends import reset_backend

    reset_backend()


def _statsmodels_available():
    """Check if statsmodels is installed (required for LME tests)."""
    try:
        import statsmodels.regression.mixed_linear_model  # noqa: F401

        return True
    except ImportError:
        return False


_has_statsmodels = _statsmodels_available()


def pytest_collection_modifyitems(config, items):
    """Auto-skip LME tests when statsmodels is not installed."""
    if _has_statsmodels:
        return
    skip_lme = pytest.mark.skip(reason="statsmodels not installed")
    for item in items:
        if "lme" in item.keywords:
            item.add_marker(skip_lme)


@pytest.fixture(autouse=True)
def reset_lme_cache():
    """
    Automatically reset LME warm start cache before each test.

    The warm start cache stores parameters from previous fits to speed up
    convergence. However, it can cause failures when model structure changes
    between tests (e.g., different number of predictors).

    This fixture ensures tests start with clean state.
    """
    try:
        from mcpower.stats.mixed_models import reset_warm_start_cache

        reset_warm_start_cache()
    except ImportError:
        # mixed_models module not available, skip
        pass
    yield
