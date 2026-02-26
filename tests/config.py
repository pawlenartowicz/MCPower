"""
Shared test configuration constants.

All test files should import from this module to ensure consistency
across the test suite.
"""

# Monte Carlo simulation parameters — 4-tier ladder
N_SIMS_CHECK = 50
"""Smoke tests — just verify no crash, structure, API contract."""

N_SIMS_ORDERING = 1000
"""Ordering tests — monotonicity, correction hierarchy, A < B checks."""

N_SIMS_STANDARD = 1600
"""Standard tests — null calibration, Type I error, general validation."""

N_SIMS_ACCURACY = 5000
"""Accuracy tests — comparison against analytical power formulas."""

N_SIMS = N_SIMS_ACCURACY
"""Backward-compat alias for accuracy-level simulations."""

SEED = 2137
"""Default random seed for reproducibility."""

# Statistical test parameters
DEFAULT_ALPHA = 0.05
"""Default significance level for hypothesis tests."""

MC_Z = 3.5
"""Z-score for Monte Carlo margin of error calculations., corresponding to ~5% of bonferonized alpha accoros 100+ tests"""

ALLOWED_BIAS = 1
"""Maximum allowed bias (in percentage points) for MC power estimates."""


LME_N_SIMS_BENCHMARK = 500
"""Benchmark accuracy tests — sufficient for tight MC margins on LME power estimates."""

# =============================================================================
# Mixed Models Test Configuration
# =============================================================================

# Simulation counts for mixed models tests
LME_N_SIMS_QUICK = 20
"""Quick smoke tests - minimal simulations for fast iteration."""

LME_N_SIMS_STANDARD = 50
"""Standard tests - balance between speed and reliability."""

LME_N_SIMS_RIGOROUS = 100
"""Rigorous tests - higher precision for validation."""

LME_N_SIMS_VALIDATION = 200
"""Validation tests - highest precision for comparing with statsmodels."""

# Failure thresholds for mixed models (proportion of allowed failed simulations)
LME_THRESHOLD_STRICT = 0.05
"""Strict threshold (5%) - for simple, well-powered designs."""

LME_THRESHOLD_MODERATE = 0.15
"""Moderate threshold (15%) - for challenging designs (high ICC, few clusters)."""

# ICC (Intraclass Correlation Coefficient) values for tests
ICC_LOW = 0.1
"""Low ICC - minimal clustering effect."""

ICC_MODERATE = 0.2
"""Moderate ICC - typical for many applications."""

ICC_MODERATE_HIGH = 0.3
"""Moderate-high ICC."""

ICC_HIGH = 0.5
"""High ICC - strong clustering effect."""

ICC_VERY_HIGH = 0.7
"""Very high ICC - extreme clustering."""

# Cluster configuration constants
N_CLUSTERS_FEW = 5
"""Few clusters - challenging for estimation."""

N_CLUSTERS_MODERATE = 20
"""Moderate number of clusters - adequate for most designs."""

N_CLUSTERS_MANY = 30
"""Many clusters - optimal for mixed models."""

N_CLUSTERS_VERY_MANY = 50
"""Very many clusters - excellent statistical properties."""

# Effect sizes for tests
EFFECT_SMALL = 0.2
"""Small effect size (Cohen's d)."""

EFFECT_MEDIUM = 0.5
"""Medium effect size (Cohen's d)."""

EFFECT_LARGE = 0.8
"""Large effect size (Cohen's d)."""

EFFECT_VERY_LARGE = 1.5
"""Very large effect size - should have near-perfect power."""

# Minimum observations per cluster (from validators)
MIN_OBS_PER_CLUSTER = 5
"""Minimum observations per cluster for LME estimation (hard error below this)."""

MIN_OBS_PER_CLUSTER_WARNING = 10
"""Warning threshold — between 5 and 10 triggers a warning."""

# ICC recovery tolerances (for validation tests comparing estimated vs target ICC)
ICC_RECOVERY_TOLERANCE_LOW = 0.03
"""Tolerance for recovering low ICC (0.1) from generated data."""

ICC_RECOVERY_TOLERANCE_HIGH = 0.05
"""Tolerance for recovering higher ICC (0.3+) from generated data."""

# Theoretical comparison tolerance (design-effect formula is approximate)
THEORETICAL_POWER_TOLERANCE = 15.0
"""Max difference (pp) between empirical and design-effect theoretical power.
Wide because the theoretical formula is approximate, not because MCPower is inaccurate."""

# Type I error control range for validation tests
TYPE1_ERROR_RANGE = 3.0
"""Half-width (pp) around alpha*100 for Type I error validation.
E.g., for alpha=0.05: expect power in [2.0, 8.0]."""
