# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-02-12

Releasing pipeline fixed

## [0.4.0b0] - 2026-02-12

The package has been almost entirely rebuilt from the ground up.

### Major changes
- **Complete code reorganization**: Monolithic `base.py` split into modular structure (`mcpower/core/`, `mcpower/backends/`, `mcpower/model.py`)
- **C++ native backend**: New PyBind11/Eigen-based compute backend for OLS and data generation (~3x speedup), removev Numba AOT
- **Multi-backend architecture**: Automatic selection between C++ native, Numba JIT, and pure Python backends with graceful fallback
- **Mixed-effects models**: Random intercept models via `(1|group)` syntax with statsmodels MixedLM
- **New CI/CD pipeline**: Separate workflows for stable releases (PyPI) and pre-releases (TestPyPI)
- **Build system migration**: Moved from setuptools to scikit-build-core with CMake for C++ compilation
- **Python 3.14 support**
- **Lookup tables reduced and reorganized**: removed lookuptables from OLS, precomputed critical values added

### Minor changes
- **Dependency restructuring**: `joblib` is now a core dependency; `statsmodels` moved to optional `[mixed]` extra
- Progress callback system for GUI/notebook integration
- Comprehensive test suite overhaul (650 tests across unit, integration, spec, and mixed-model categories)
- Pre-commit hooks with ruff for linting and formatting
- Removed legacy `setup.py`, `pytest.ini`, and `paper/` directory

## [0.3.3] - 2025-07-27
### Major changes
- Wheels for pyhon 3.13

## [0.3.2] - 2025-07-27

### Major changes
- Factors are now supported
- The workflow requires that the set_variable_types be set before the correlation and effects

## [0.3.1] - 2025-07-17

### Major changes
- Added weekly update checker

### Minor changes
- Small changes in documentation


## [0.3.0] - 2025-07-16

### Major changes
- AOT Compiled files added
- PyPI release

### Minor changes
- Small changes in documentation, added fallback to compilation


## [0.2.4] - 2025-07-16
- Test PyPI release

## [0.2.3] - 2025-07-16
- Test PyPI release

## [0.2.2] - 2025-07-11

### Major changes
- Moved seed setting from method arguments to `model.set_seed()` for better control
- Added public method to customize scenario configurations
- Improved lookup table performance with extended range and faster fallbacks
- Added support for 3-way and n-way interactions in formulas
- Fixed import error when uploading user data
- Added comprehensive test suite

### Minor changes
- Improved code documentation with clearer comments and global variables
- Clarified correlation behavior during variable transformations
- Fixed minor cache creation error
- Adjusted vulnerability thresholds from 30% to 20% power drop
- Added warning for inflated Type I error risk

## [0.2.1] - 2025-06-26

### Major changes
- Fixed import statements to use relative imports instead of absolute imports

### Minor changes
- Added changelog
- Extended docstrings

## [0.2.0] - 2025-06-26
- Initial release

## [0.1.0] - 2025-06-07
- Proof of concept release