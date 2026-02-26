"""Tests for stats/mixed_models.py — statsmodels convergence, corrections, native wrappers.

Uses pytest.mark.lme to skip when statsmodels is not installed.
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcpower.stats.mixed_models import (
    _ensure_lme_crits,
    _lme_analysis_wrapper,
    _wrap_native_result,
    reset_warm_start_cache,
)

pytestmark = pytest.mark.lme


class TestWrapNativeResult:
    """Test _wrap_native_result helper."""

    def test_non_empty_non_verbose(self):
        result = np.array([1.0, 0.0, 1.0])
        wrapped = _wrap_native_result(result, verbose=False, solver_name="native_q1")
        np.testing.assert_array_equal(wrapped, result)

    def test_non_empty_verbose(self):
        result = np.array([1.0, 0.0, 1.0])
        wrapped = _wrap_native_result(result, verbose=True, solver_name="native_q1")
        assert isinstance(wrapped, dict)
        assert "results" in wrapped
        assert "diagnostics" in wrapped
        assert wrapped["diagnostics"]["solver"] == "native_q1"

    def test_non_empty_verbose_with_extra_diag(self):
        result = np.array([1.0])
        wrapped = _wrap_native_result(
            result, verbose=True, solver_name="native_general",
            extra_diag={"q": 3},
        )
        assert wrapped["diagnostics"]["q"] == 3

    def test_empty_non_verbose_returns_none(self):
        result = np.array([])
        assert _wrap_native_result(result, verbose=False, solver_name="native_q1") is None

    def test_empty_verbose_returns_failure_dict(self):
        result = np.array([])
        wrapped = _wrap_native_result(result, verbose=True, solver_name="native_q1")
        assert wrapped["results"] is None
        assert "failure_reason" in wrapped
        assert "empty result" in wrapped["failure_reason"]


class TestEnsureLMECrits:
    """Test _ensure_lme_crits computes when None."""

    def test_computes_when_none(self):
        chi2, z, crits = _ensure_lme_crits(
            alpha=0.05, p=3, n_targets=2, correction_method=0,
            chi2_crit=None, z_crit=None, correction_z_crits=None,
        )
        assert np.isfinite(chi2)
        assert np.isfinite(z)
        assert len(crits) == 2

    def test_passthrough_when_provided(self):
        chi2, z, crits = _ensure_lme_crits(
            alpha=0.05, p=3, n_targets=2, correction_method=0,
            chi2_crit=7.8, z_crit=1.96, correction_z_crits=np.array([1.96, 1.96]),
        )
        assert chi2 == 7.8
        assert z == 1.96
        assert len(crits) == 2


class TestLMEAnalysisWrapperRouting:
    """Test _lme_analysis_wrapper routes to correct backend."""

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _lme_analysis_wrapper(
                np.eye(10), np.ones(10), np.array([0, 1]),
                np.zeros(10, dtype=np.int32),
                correction_method=0, alpha=0.05, backend="nonexistent",
            )


class TestStatsmodelsConvergence:
    """Test statsmodels fallback path with mocked MixedLM."""

    def _make_mock_result(self, converged=True, params=None, pvalues=None, n_params=3):
        """Create a mock MixedLM result."""
        result = MagicMock()
        result.converged = converged
        result.params = params if params is not None else np.array([1.0, 0.5, 0.3])
        result.pvalues = pvalues if pvalues is not None else np.array([0.01, 0.02, 0.04])
        result.fe_params = result.params
        result.bse = np.array([0.1, 0.1, 0.1])

        # cov_re: random effects variance (needs .iloc[0, 0])
        cov_re = MagicMock()
        cov_re.iloc.__getitem__ = MagicMock(return_value=0.5)
        result.cov_re = cov_re

        result.scale = 1.0
        result.llf = -50.0

        # Make cov_params return a proper matrix
        result.cov_params.return_value = np.eye(n_params) * 0.01

        # model attribute
        result.model = MagicMock()
        result.model.exog = MagicMock()
        result.model.exog.shape = (100, n_params)

        return result

    @patch("statsmodels.regression.mixed_linear_model.MixedLM")
    def test_warm_start_retry_chain(self, mock_mixedlm_cls):
        """First fit fails, cold start succeeds."""
        from mcpower.stats.mixed_models import _lme_analysis_statsmodels, _lme_thread_local

        _lme_thread_local.warm_start_params = np.array([1.0, 0.5, 0.3])

        mock_model = MagicMock()
        mock_mixedlm_cls.return_value = mock_model

        good_result = self._make_mock_result()
        mock_model.fit.side_effect = [
            Exception("warm start diverged"),
            good_result,
        ]
        mock_model.loglike.return_value = -50.0

        result = _lme_analysis_statsmodels(
            X_expanded=np.random.randn(100, 2),
            y=np.random.randn(100),
            target_indices=np.array([0, 1]),
            cluster_ids=np.repeat(np.arange(10), 10),

            correction_method=0,
            alpha=0.05,
        )
        assert result is not None

    @patch("statsmodels.regression.mixed_linear_model.MixedLM")
    def test_all_attempts_fail_returns_none(self, mock_mixedlm_cls):
        from mcpower.stats.mixed_models import _lme_analysis_statsmodels, _lme_thread_local

        _lme_thread_local.warm_start_params = None

        mock_model = MagicMock()
        mock_mixedlm_cls.return_value = mock_model
        mock_model.fit.side_effect = Exception("always fails")

        result = _lme_analysis_statsmodels(
            X_expanded=np.random.randn(100, 2),
            y=np.random.randn(100),
            target_indices=np.array([0, 1]),
            cluster_ids=np.repeat(np.arange(10), 10),

            correction_method=0,
            alpha=0.05,
        )
        assert result is None

    @patch("statsmodels.regression.mixed_linear_model.MixedLM")
    def test_all_attempts_fail_verbose_returns_dict(self, mock_mixedlm_cls):
        from mcpower.stats.mixed_models import _lme_analysis_statsmodels, _lme_thread_local

        _lme_thread_local.warm_start_params = None

        mock_model = MagicMock()
        mock_mixedlm_cls.return_value = mock_model
        mock_model.fit.side_effect = Exception("always fails")

        result = _lme_analysis_statsmodels(
            X_expanded=np.random.randn(100, 2),
            y=np.random.randn(100),
            target_indices=np.array([0, 1]),
            cluster_ids=np.repeat(np.arange(10), 10),

            correction_method=0,
            alpha=0.05,
            verbose=True,
        )
        assert isinstance(result, dict)
        assert result["results"] is None
        assert "failure_reason" in result

    @patch("statsmodels.regression.mixed_linear_model.MixedLM")
    def test_not_converged_returns_none(self, mock_mixedlm_cls):
        """When result.converged is False for all attempts."""
        from mcpower.stats.mixed_models import _lme_analysis_statsmodels, _lme_thread_local

        _lme_thread_local.warm_start_params = None

        mock_model = MagicMock()
        mock_mixedlm_cls.return_value = mock_model

        bad_result = self._make_mock_result(converged=False)
        mock_model.fit.return_value = bad_result

        result = _lme_analysis_statsmodels(
            X_expanded=np.random.randn(100, 2),
            y=np.random.randn(100),
            target_indices=np.array([0, 1]),
            cluster_ids=np.repeat(np.arange(10), 10),

            correction_method=0,
            alpha=0.05,
        )
        assert result is None


class TestCorrections:
    """Test statsmodels FDR, Holm, Bonferroni, no-correction paths."""

    def _make_mock_result(self):
        result = MagicMock()
        result.converged = True
        result.params = np.array([1.0, 0.5, 0.3])
        result.pvalues = np.array([0.001, 0.02, 0.04])
        result.fe_params = result.params
        result.bse = np.array([0.1, 0.1, 0.1])
        result.scale = 1.0
        result.llf = -50.0
        result.model = MagicMock()
        result.model.exog = MagicMock()
        result.model.exog.shape = (100, 3)

        cov_re_mock = MagicMock()
        cov_re_mock.iloc.__getitem__ = MagicMock(return_value=0.5)
        result.cov_re = cov_re_mock
        result.cov_params.return_value = np.eye(3) * 0.01

        return result

    def _run_with_correction(self, correction_method):
        from mcpower.stats.mixed_models import _lme_analysis_statsmodels, _lme_thread_local

        _lme_thread_local.warm_start_params = None

        mock_result = self._make_mock_result()

        with patch("statsmodels.regression.mixed_linear_model.MixedLM") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model
            mock_model.fit.return_value = mock_result
            mock_model.loglike.return_value = -50.0

            out = _lme_analysis_statsmodels(
                X_expanded=np.random.randn(100, 2),
                y=np.random.randn(100),
                target_indices=np.array([0, 1]),
                cluster_ids=np.repeat(np.arange(10), 10),
    
                correction_method=correction_method,
                alpha=0.05,
            )
        return out

    def test_no_correction(self):
        result = self._run_with_correction(0)
        assert result is not None

    def test_bonferroni(self):
        result = self._run_with_correction(1)
        assert result is not None

    def test_fdr(self):
        result = self._run_with_correction(2)
        assert result is not None

    def test_holm(self):
        result = self._run_with_correction(3)
        assert result is not None


class TestResetWarmStartCache:
    """Test reset_warm_start_cache."""

    def test_clears_params(self):
        from mcpower.stats.mixed_models import _lme_thread_local

        _lme_thread_local.warm_start_params = np.array([1.0])
        reset_warm_start_cache()
        assert _lme_thread_local.warm_start_params is None
