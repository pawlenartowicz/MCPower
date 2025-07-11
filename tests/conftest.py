"""
Shared pytest fixtures for MCPower tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Set random seed for reproducible tests
np.random.seed(42)

@pytest.fixture
def sample_model():
    """Basic LinearRegression model for testing."""
    import mcpower
    model = mcpower.LinearRegression("y = x1 + x2 + x1:x2")
    model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
    model.set_variable_type("x1=binary")
    return model

@pytest.fixture
def simple_model():
    """Simple model with one predictor."""
    import mcpower
    model = mcpower.LinearRegression("outcome = treatment")
    model.set_effects("treatment=0.4")
    model.set_variable_type("treatment=binary")
    return model

@pytest.fixture
def sample_data():
    """Sample dataset for upload tests - uses cars.csv."""
    # Try to find cars.csv in multiple locations
    possible_paths = [
        Path(__file__).parent / "cars.csv",           # tests/cars.csv
        Path(__file__).parent / "../examples/cars.csv", # examples/cars.csv
        Path("examples/cars.csv"),                    # from project root
        Path("cars.csv")                              # current directory
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    # Fallback to synthetic data if cars.csv not found
    np.random.seed(123)
    n = 32  # Same size as cars dataset
    data = {
        'mpg': np.random.normal(20, 6, n),
        'hp': np.random.normal(140, 60, n),
        'wt': np.random.normal(3.2, 1, n),
        'am': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(sample_data):
    """Temporary CSV file for data upload tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def cars_data():
    """Cars dataset specifically for realistic testing."""
    # Same logic as sample_data but clearer name
    possible_paths = [
        Path(__file__).parent / "cars.csv",           
        Path(__file__).parent / "../examples/cars.csv", 
        Path("examples/cars.csv"),                    
        Path("cars.csv")                              
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    # Fallback to synthetic cars-like data
    np.random.seed(123)
    n = 32
    data = {
        'mpg': np.random.normal(20, 6, n),
        'hp': np.random.normal(140, 60, n),
        'wt': np.random.normal(3.2, 1, n),
        'am': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

@pytest.fixture
def correlation_matrix_3x3():
    """Valid 3x3 correlation matrix."""
    return np.array([
        [1.0, 0.3, 0.5],
        [0.3, 1.0, 0.2],
        [0.5, 0.2, 1.0]
    ])

@pytest.fixture
def small_simulation_settings():
    """Settings for fast tests."""
    return {'n_simulations': 100, 'parallel': False}

# Test data constants
VALID_EQUATIONS = [
    "y = x1 + x2",
    "outcome ~ treatment + age", 
    "y = x1 * x2",
    "response = a + b + a:b"
]

INVALID_EQUATIONS = [
    "",
    "y = ",
    "= x1 + x2",
    "y ~ "
]

VALID_EFFECT_STRINGS = [
    "x1=0.5, x2=0.3",
    "treatment=0.4",
    "a=0.1, b=0.2, a:b=0.15"
]

INVALID_EFFECT_STRINGS = [
    "",
    "x1=invalid",
    "nonexistent=0.5",
    "x1="
]