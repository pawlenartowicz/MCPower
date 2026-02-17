"""Tests for LME solver with nested random intercepts.

Tests the two-level Woodbury solver for (1|school) + (1|school:classroom).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.lme


def _generate_nested_data(
    n_schools=10, n_classrooms_per_school=3, n_students_per_classroom=30,
    beta=None, sigma2=1.0, tau_school=0.5, tau_classroom=0.3, seed=42,
):
    """Generate data from a nested random intercepts model.

    y_ijk = beta0 + beta1*x_ijk + b_school_j + b_classroom_jk + eps_ijk
    """
    if beta is None:
        beta = np.array([1.0, 0.5])  # intercept, slope

    rng = np.random.RandomState(seed)
    K_school = n_schools
    K_classroom = n_schools * n_classrooms_per_school
    N = K_classroom * n_students_per_classroom

    # Random effects
    b_school = rng.normal(0, tau_school, K_school)
    b_classroom = rng.normal(0, tau_classroom, K_classroom)

    # Assignments
    school_ids = np.repeat(np.arange(K_school), n_classrooms_per_school * n_students_per_classroom)
    classroom_ids = np.repeat(np.arange(K_classroom), n_students_per_classroom)
    child_to_parent = np.repeat(np.arange(K_school), n_classrooms_per_school)

    # Fixed effects
    x = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])

    # Generate y
    y = X @ beta + b_school[school_ids] + b_classroom[classroom_ids]
    y += rng.normal(0, np.sqrt(sigma2), N)

    return X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent


class TestNestedSufficientStatistics:
    """Test nested sufficient statistics computation."""

    def test_nested_stats_shape(self):
        from mcpower.stats.lme_solver import compute_nested_sufficient_statistics

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )
        nstats = compute_nested_sufficient_statistics(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )

        assert nstats.K_parent == K_school
        assert nstats.K_child == K_classroom
        assert nstats.child_XtX.shape == (K_classroom, 2, 2)
        assert nstats.parent_sizes.sum() == len(y)

    def test_nested_stats_consistency(self):
        """Child sizes should sum to parent sizes."""
        from mcpower.stats.lme_solver import compute_nested_sufficient_statistics

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )
        nstats = compute_nested_sufficient_statistics(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )

        # Sum child sizes per parent should equal parent sizes
        for pj in range(K_school):
            child_mask = child_to_parent == pj
            child_total = nstats.child_sizes[child_mask].sum()
            assert child_total == nstats.parent_sizes[pj]


class TestNestedProfiledDeviance:
    """Test nested profiled deviance."""

    def test_deviance_finite(self):
        from mcpower.stats.lme_solver import (
            _profiled_deviance_nested,
            compute_nested_sufficient_statistics,
        )

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )
        nstats = compute_nested_sufficient_statistics(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )

        theta = np.array([1.0, 1.0])
        dev = _profiled_deviance_nested(theta, nstats, 1)
        assert np.isfinite(dev)

    def test_deviance_at_zero(self):
        """Zero theta should give finite deviance (OLS equivalent)."""
        from mcpower.stats.lme_solver import (
            _profiled_deviance_nested,
            compute_nested_sufficient_statistics,
        )

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )
        nstats = compute_nested_sufficient_statistics(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )

        theta = np.array([0.0, 0.0])
        dev = _profiled_deviance_nested(theta, nstats, 1)
        assert np.isfinite(dev)


class TestFitNested:
    """Test nested model fitting."""

    def test_fit_recovers_beta(self):
        """Beta should be close to true values."""
        from mcpower.stats.lme_solver import lme_fit_nested

        true_beta = np.array([1.0, 0.5])
        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data(
                n_schools=15, n_classrooms_per_school=4,
                n_students_per_classroom=40, beta=true_beta,
            )
        )

        result = lme_fit_nested(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )
        assert result.converged
        np.testing.assert_allclose(result.beta, true_beta, atol=0.15)

    def test_fit_variance_components(self):
        """Variance components should be in reasonable range."""
        from mcpower.stats.lme_solver import lme_fit_nested

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data(
                n_schools=20, n_classrooms_per_school=5,
                n_students_per_classroom=50,
                tau_school=0.5, tau_classroom=0.3,
            )
        )

        result = lme_fit_nested(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )
        assert result.converged

        # G matrix should be 2x2 diagonal with [lambda_parent^2, lambda_child^2]
        theta_parent, theta_child = result.theta[0], result.theta[1]
        tau2_parent = result.sigma2 * theta_parent**2
        tau2_child = result.sigma2 * theta_child**2

        # Should be in reasonable range (true tau_school^2=0.25, tau_classroom^2=0.09)
        assert 0.01 < tau2_parent < 2.0
        assert 0.01 < tau2_child < 2.0

    def test_fit_se_positive(self):
        """Standard errors should all be positive."""
        from mcpower.stats.lme_solver import lme_fit_nested

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )

        result = lme_fit_nested(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )
        assert result.converged
        assert np.all(result.se_beta > 0)

    def test_analysis_nested_returns_array(self):
        """lme_analysis_nested should return a results array."""
        from mcpower.stats.lme_solver import (
            compute_lme_critical_values,
            lme_analysis_nested,
        )

        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data()
        )
        X_no_int = X[:, 1:]
        target_indices = np.array([0])

        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(
            0.05, 1, 1, 0
        )
        result = lme_analysis_nested(
            X_no_int, y, school_ids, classroom_ids, K_school, K_classroom,
            child_to_parent,
            target_indices=target_indices,
            chi2_crit=chi2_crit, z_crit=z_crit,
            correction_z_crits=correction_z_crits,
            correction_method=0,
        )
        assert result is not None
        assert len(result) == 1 + 2 * 1 + 1

    def test_single_classroom_per_school_degenerates(self):
        """With 1 classroom per school, nested = single intercept."""
        from mcpower.stats.lme_solver import lme_fit, lme_fit_nested

        n_schools = 20
        n_students = 50
        X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent = (
            _generate_nested_data(
                n_schools=n_schools,
                n_classrooms_per_school=1,
                n_students_per_classroom=n_students,
                tau_school=0.5,
                tau_classroom=0.0,  # no classroom variance
            )
        )

        # Nested fit
        nested_result = lme_fit_nested(
            X, y, school_ids, classroom_ids, K_school, K_classroom, child_to_parent
        )

        # Single-level fit
        single_result = lme_fit(X, y, school_ids, K_school, q=1, reml=True)

        # Beta estimates should be very close
        np.testing.assert_allclose(nested_result.beta, single_result.beta, atol=0.1)
