"""Tests for mcpower.utils.mixed_models backward-compat re-exports."""

import threading

from mcpower.utils.mixed_models import (
    _lme_analysis_wrapper,
    _lme_thread_local,
    reset_warm_start_cache,
)


class TestReExports:
    """Verify that the backward-compatibility re-exports resolve correctly."""

    def test_lme_analysis_wrapper_is_callable(self):
        assert callable(_lme_analysis_wrapper)

    def test_lme_thread_local_is_threading_local(self):
        assert isinstance(_lme_thread_local, threading.local)

    def test_reset_warm_start_cache_is_callable(self):
        assert callable(reset_warm_start_cache)

    def test_reset_warm_start_cache_clears_params(self):
        _lme_thread_local.warm_start_params = "dummy"
        reset_warm_start_cache()
        assert _lme_thread_local.warm_start_params is None
