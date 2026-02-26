# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-02-24

### Breaking changes
- **Removed `set_backend()`, `get_backend_info()`, `reset_backend()`** — only one backend (C++ native) exists since v0.5.0, so the multi-backend API was dead code. Use `from mcpower.backends import get_backend` if you need the backend instance directly
- **Removed `set_heterogeneity()` and `set_heteroskedasticity()`** — heterogeneity and heteroskedasticity are now controlled exclusively through scenario configurations (`set_scenario_configs()`). The optimistic scenario uses zero perturbation; realistic/doomer scenarios apply these automatically
- **Removed dead scipy fallback code** from `distributions.py` — scipy was never a runtime dependency since v0.5.0, so the fallback paths were unreachable dead code. The module now cleanly fails with an `ImportError` if the C++ native backend is missing
- **`_create_power_plot()` returns `fig`** — the function now accepts a `show=True` parameter and always returns the matplotlib figure object. Set `show=False` to suppress `plt.show()` for programmatic use
- **`apply()` made private (`_apply()`)** — the method is now `_apply()` and called automatically by `find_power()` / `find_sample_size()`. Direct calls should use `model._apply()` instead
- **`[all]` extra no longer includes `statsmodels`** — use `pip install mcpower[lme]` to get statsmodels for mixed-effects models

### Added
- **`test_formula` parameter** on `find_power()` and `find_sample_size()` — test a reduced model against data generated from the full model to evaluate power under model misspecification. For example, generate data with `y = x1 + x2 + x3` but test with `test_formula="y ~ x1 + x2"` to see power when `x3` is omitted. Supports interactions, factors, and mixed models.
- **C++ non-normal residual generation** — scenario perturbations now generate heavy-tailed (Student-t) and skewed (chi-squared) residuals directly in C++ via `residual_dist`/`residual_df` parameters in `generate_y()`, replacing the Python-side post-hoc perturbation approach. Applies to all model types (OLS and LME)
- **`optimistic` scenario** is now a first-class entry in `DEFAULT_SCENARIO_CONFIG` with all-zero perturbation values, eliminating the special `scenario_config=None` code path. Custom scenarios inherit from the optimistic baseline, ensuring all required keys exist

### Fixed
- **`set_variable_type()` docstring listed wrong distribution types** — documented non-existent `"skewed"` type; now lists all supported types: `right_skewed`, `left_skewed`, `high_kurtosis`, `uniform`
- **`set_scenario_configs()` docstring referenced non-existent keys** — `"effect_size_jitter"` and `"distribution_jitter"` replaced with actual keys (`correlation_noise_sd`, `distribution_change_prob`, etc.)
- **String factor levels crash in LME variance computation** — `proportions[level - 1]` crashed when factor levels were strings (e.g. `"Japan"`). Now looks up level position in the label list
- **Division by zero on constant-variance columns** — `upload_data()` normalization produced `inf`/`NaN` when a column had zero variance. Now raises `ValueError` with the column name
- **Pending state not cleared after `_apply()`** — calling `_apply()` twice could re-apply the same effects. Pending fields are now reset after each `_apply()` call
- **Parser crash on unbalanced parentheses** — unmatched `)` caused `paren_count` to go negative, producing silent misparses. Now raises `ValueError`
- **Update checker wrote cache inside installed package** — moved cache file to `~/.cache/mcpower/update_cache.json`
- **Update checker unbounded response read** — `response.read()` now limited to 1 MB
- **`scenario_config` dict access on `None`** — added `None` guards for optional scenario configuration lookups
- **NaN values in uploaded data** — `upload_data()` now rejects data containing NaN values with a clear error message listing affected columns
- **Formula minus-sign silently dropped terms** — `y = x1 - x2` silently ignored `x2`. Now raises `ValueError` explaining that term removal with `-` is not supported
- **`_create_table` crash on empty rows** — formatter now handles empty row lists by computing column widths from headers only
- **`_create_power_plot` crash when `first_achieved` not in sample sizes** — added bounds check before `.index()` call
- **Redundant `_validate_cluster_sample_size` call** — removed duplicate validation in `find_power()` (already called per-sample-size in `find_sample_size()`)

### Changed
- **`upload_data()` returns `self`** for method chaining consistency
- **Assert statements replaced with `RuntimeError`** — internal assertions now raise proper exceptions instead of using `assert`
- **Removed "(not yet implemented)" from mixed-model docstrings** — mixed model testing has been implemented since v0.4.2
- **Thread-safe RNG in data generation** — replaced global `np.random.seed()` with local `np.random.RandomState()` for thread safety
- **Update checker runs in a background thread** — no longer blocks `import mcpower` on slow networks
- **Module-level deduplication for update checker** — prevents redundant version checks within the same Python session
- **Removed unused `cluster_column_indices` parameter** from `_lme_analysis_wrapper()` and `_lme_analysis_statsmodels()` — was explicitly marked unused and kept only for API compatibility
- **Scenario formatters iterate dynamically** — no longer hardcode scenario names, enabling custom scenario display

### Packaging
- **`tqdm` added as core dependency** (`>=4.60.0`) — used for progress bars
- **Removed stale pytest warning filter** for `"Mixed-effects models are experimental"` (warning was removed in v0.5.4)
- **NumPy minimum version relaxed** to `>=1.26.0` (was `>=2.0.0`) in both build-requires and runtime dependencies
- **`scikit-build-core` bumped** to `>=0.10` (was `>=0.5`)
- **`statsmodels` added to `[dev]` extras** for test/development convenience
- **Documentation URL** now points to the GitHub wiki
- **Changelog URL** added to project URLs
- **Removed unused pytest markers** (`unit`, `integration`) — only `lme` marker remains
- **Per-module mypy overrides** replace blanket `ignore_missing_imports`

### Documentation
- Updated README requirements section: added `tqdm`, specified `NumPy (>=1.26.0)`
- Changed `pip install mcpower[all]` → `pip install mcpower[lme]` for statsmodels installation
- Wiki documentation review and cleanup: fixed broken links, corrected API signatures (`set_scenario_configs` parameter name), removed stale `apply()` and `set_heterogeneity()` wiki pages, fixed formula redundancy in Model Specification, corrected Tukey return value docs, added mixed-model caveats

### Technical
- Removed ~150 lines of dead scipy fallback shims from `distributions.py`
- Removed `_BACKEND` sentinel variable (only one backend exists)
- C++ `generate_y()` now accepts `residual_dist` and `residual_df` parameters for non-normal error generation
- `suppress_output` test fixture now actually suppresses stdout (was a no-op)
- Removed unused `correlation_matrix_3x3` test fixture
- Removed empty `tests/mcpower/` artifact directory
- Added unit tests for `ResultsProcessor` (`test_results.py`)
- Added unit tests for `normalize_upload_input` (`test_upload_data_utils.py`)
- Added integration tests for `test_formula` feature (`test_test_formula.py`)
- Added unit tests for `test_formula_utils` (`test_test_formula_utils.py`)
- Rewrote optimizer tests to test native backend directly (removed dead scipy fallback tests)

## [0.5.4] - 2026-02-22

### Changed
- **Mixed-Effects Models no longer experimental** — removed experimental warnings and labels after [validation against R's lme4](https://github.com/pawlenartowicz/MCPower/wiki/Concept-LME-Validation) across 95 scenarios using four independent strategies (all 230 scenario-strategy combinations pass)

### Fixed
- Removed unused `predictor_names` assignment in `MCPower.__init__` (ruff F841)
- Fixed mypy `attr-defined` errors in `distributions.py` — `_Result` fallback classes now use `__slots__` for proper attribute declaration

## [0.5.3] - 2026-02-21

### Technical
- Added `lint.yml` GitHub Actions workflow — runs ruff check, ruff format check, and mypy on every push to `dev`
- Updated cibuildwheel from v2.22 to v3.3.1 — enables Python 3.14 wheel builds in CI

### Fixed

- **Factor:factor interaction expansion** — `expand_factors()` now produces Cartesian product of non-reference dummy levels (e.g. `a[2]:b[2]`, `a[3]:b[2]`) instead of incorrect partial expansion (`a:b[2]`, `a[2]:b`). Affects any model with interactions between two or more factor variables
- Factor:factor interactions with named level labels now expand correctly (e.g. `origin[Japan]:cyl[6]` instead of `origin:cyl[6]`)

## [0.5.2] - 2026-02-20

### Documentation
- Fixed incorrect dummy variable names in README upload-data example (`cyl[2]` → `cyl[6]`, `cyl[3]` → `cyl[8]`)
- Fixed dependencies section in README
- Removed scipy from `[all]` optional dependencies (unused since C++ backend became required in v0.5.0)
- Updated all example scripts to use current `from mcpower import MCPower` API

### Fixed

- Auto-tag workflow now uses `RELEASE_TOKEN` for downstream workflow triggering

## [0.5.1] - 2026-02-20

### Documentation
- Rewrote "Why MCPower?" section — replaced dense paragraphs with bullet points; added mixed models entry
- Removed inaccurate "logistic regression and ANOVA are on the way" note from feature description

## [0.5.0] - 2026-02-19

### Breaking changes
- **C++ backend required**: Removed JIT (Numba) and pure-Python fallback backends — a compiled C++ extension is now required to use MCPower
- **scipy removed from core dependencies**: `scipy` is no longer required at runtime; moved to optional `[dev]` and `[all]` extras
- **pandas removed from core dependencies**: `pandas` moved to optional `[pandas]` and `[all]` extras
- Removed `@pytest.mark.slow` marker — all tests run by default

### Major changes
- **C++ distributions module**: All distribution functions (F, t, chi2, normal, studentized range) now use Boost.Math via C++ native backend, replacing scipy
- **C++ optimizers module**: L-BFGS-B (via LBFGSPP) and Brent's method implemented in C++, replacing scipy.optimize
- **Full C++ LME solvers for all model types**: Random slopes (q>1) and nested intercepts now have complete C++ solvers, matching the q=1 path — previously experimental Python implementations
- **Named factor levels**: Factor dummies use original data values (e.g. `cyl[6]` instead of `cyl[2]`); string columns auto-detected as factors; user-selectable reference level via `data_types` parameter
- **`set_factor_levels()` method**: Define named factor levels without uploading data (e.g. `set_factor_levels("group=control,drug_a,drug_b")`)

### Minor changes
- New `distributions.py` module as single import point for all distribution functions (C++ native with scipy shim fallback)
- `preserve_factor_level_names` parameter on `upload_data()` (default `True`)
- Expanded `pyproject.toml` keywords for better discoverability
- Fixed C++ vs Python precision boundary bug in LME solver
- `upload_data()` `data_types` supports tuple syntax for reference level selection: `{"cyl": ("factor", 8)}`

### Technical
- Removed `mcpower/backends/python.py` and `mcpower/backends/jit.py` — single `NativeBackend` only
- Removed `mcpower/utils/data_generation.py` and `mcpower/utils/ols.py` re-export stubs
- `lme_solver.py` reduced from ~350 lines to ~30 lines (thin wrapper for critical value precomputation)
- `mixed_models.py` simplified: removed unused Python-side warm-start code for C++ solver paths
- New C++ source files: `distributions.cpp/hpp`, `optimizers.cpp/hpp`; expanded `lme_solver.cpp/hpp` and `bindings.cpp`
- CMake FetchContent for Boost.Math (v1.87.0) and LBFGSPP (v0.3.0) header-only libraries
- CI workflows: removed stale `--ignore=tests/backend/` flag, renamed `test-release.yml` → `ci.yml`, added `auto-tag.yml`
- Test suite overhaul: removed backend parity tests (no longer applicable), old Python LME solver/slopes/nested/vs_statsmodels tests; added comprehensive distributions tests (668 lines), C++ LME general/nested tests, precision boundary tests, LME scenario tests
- Deleted empty `tests/backend/` directory

## [0.4.2] - 2026-02-17

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
