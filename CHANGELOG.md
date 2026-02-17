# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Major changes
- **Custom LME solver**: Replaced statsmodels MixedLM with a custom profiled-deviance REML solver (Python + C++ + JIT backends), eliminating the statsmodels runtime dependency for mixed models
- **C++ LME backend**: LME analysis exposed via PyBind11/Eigen (`mcpower_native.lme_analysis`) for maximum throughput
- **Post-hoc pairwise comparisons**: `target_test="group[1] vs group[2]"` syntax for contrasts between factor levels
- **Tukey HSD correction**: New `correction="tukey"` option for pairwise comparisons using studentized range critical values

### Minor changes
- `target_test` default changed from `"overall"` to `"all"` (overall F-test + all individual fixed effects)
- `target_test` now supports keyword expansion (`"all"`, `"all-posthoc"`), exclusion prefixes (`"-overall"`), and comma-separated combinations
- Renamed optional dependency extra from `[mixed]` to `[lme]`
- Formatter handles `NaN` power values gracefully (displays `"-"` instead of crashing)
- README rewritten with clearer examples and MCPower GUI cross-link

### Technical
- New `mcpower/stats/` package: statistical routines (`ols.py`, `lme_solver.py`, `mixed_models.py`, `data_generation.py`) extracted from `utils/`
- Reclassified several `@pytest.mark.slow` tests to run in the default (non-slow) suite
- Removed old test files and `scripts/test_quick.py`; moved `LME_BENCHMARK_TESTS.md` to `docs/`

### Experimental
- **Random slopes**: Support for `(1 + x|group)` syntax with configurable slope variance and slope-intercept correlation
- **Nested random effects**: Support for `(1|A/B)` syntax with automatic expansion to parent and child grouping terms

## [0.4.1] - 2026-02-14

### Minor changes
- README badges now link to their respective pages
- Added MCPower GUI desktop application section to README

### Technical
- Reclassified several backend and mixed-model tests from `@pytest.mark.slow` to default suite
- Moved slow markers to individual test methods for finer-grained control

## [0.4.0] - 2026-02-12

Releasing pipeline fixed

## [0.4.0b0] - 2026-02-12

The package has been almost entirely rebuilt from the ground up.

### Major changes
- **C++ native backend**: New PyBind11/Eigen-based compute backend for OLS and data generation (~3x speedup), removed Numba AOT
- **Multi-backend architecture**: Automatic selection between C++ native, Numba JIT, and pure Python backends with graceful fallback
- **Mixed-effects models**: Random intercept models via `(1|group)` syntax with statsmodels MixedLM
- **Build system migration**: Moved from setuptools to scikit-build-core with CMake for C++ compilation
- **Python 3.14 support**
- **Lookup tables reduced and reorganized**: removed lookup tables from OLS, precomputed critical values added

### Minor changes
- `joblib` is now a core dependency; `statsmodels` moved to optional `[lme]` extra
- Progress callback system for GUI/notebook integration

### Technical
- Complete code reorganization: monolithic `base.py` split into modular structure (`mcpower/core/`, `mcpower/backends/`, `mcpower/model.py`)
- New CI/CD pipeline: separate workflows for stable releases (PyPI) and pre-releases (TestPyPI)
- Comprehensive test suite overhaul (650 tests across unit, integration, spec, and mixed-model categories)
- Pre-commit hooks with ruff for linting and formatting
- Removed legacy `setup.py`, `pytest.ini`, and `paper/` directory

## [0.3.3] - 2025-07-27

### Technical
- Wheels for Python 3.13

## [0.3.2] - 2025-07-27

### Major changes
- Factors are now supported
- The workflow requires that `set_variable_types` be set before correlation and effects

## [0.3.1] - 2025-07-17

### Major changes
- Added weekly update checker

### Minor changes
- Small changes in documentation

## [0.3.0] - 2025-07-16

### Major changes
- AOT compiled files added
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
- Added support for 3-way and n-way interactions in formulas

### Minor changes
- Improved lookup table performance with extended range and faster fallbacks
- Fixed import error when uploading user data
- Fixed minor cache creation error
- Adjusted vulnerability thresholds from 30% to 20% power drop
- Added warning for inflated Type I error risk

### Technical
- Added comprehensive test suite
- Improved code documentation with clearer comments and global variables
- Clarified correlation behavior during variable transformations

## [0.2.1] - 2025-06-26

### Minor changes
- Fixed import statements to use relative imports instead of absolute imports
- Added changelog
- Extended docstrings

## [0.2.0] - 2025-06-26
- Initial release

## [0.1.0] - 2025-06-07
- Proof of concept release
