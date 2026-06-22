"""Tests for LME cluster validators in v2.

Covers _validate_cluster_config and _validate_cluster_sample_size.
v2 simplification: no nested_child, between_vars, random_slopes paths.
"""

from __future__ import annotations

import pytest

from mcpower.spec.validators import (
    _validate_cluster_config,
    _validate_cluster_sample_size,
)


# ---------------------------------------------------------------------------
# _validate_cluster_config
# ---------------------------------------------------------------------------


def test_validate_cluster_config_valid():
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert r.is_valid


def test_validate_cluster_config_xor_n_clusters_cluster_size():
    # Both provided → error.
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=10, cluster_size=5,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid
    assert any("n_clusters" in e or "cluster_size" in e for e in r.errors)


def test_validate_cluster_config_neither_n_clusters_nor_cluster_size():
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=None, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_icc_range():
    # ICC >= 1 → error
    r = _validate_cluster_config(
        "school", icc=1.5, n_clusters=10, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid
    assert any("ICC" in e for e in r.errors)


def test_validate_cluster_config_icc_zero_allowed():
    # ICC == 0 is the degenerate "no random effect" case and must be allowed.
    r = _validate_cluster_config(
        "school", icc=0, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert r.is_valid


def test_validate_cluster_config_icc_negative():
    r = _validate_cluster_config(
        "school", icc=-0.1, n_clusters=10, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_grouping_var_not_in_formula():
    r = _validate_cluster_config(
        "classroom", icc=0.2, n_clusters=10, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid
    assert any("classroom" in e for e in r.errors)


def test_validate_cluster_config_cluster_size_used():
    # cluster_size provided, n_clusters=None → valid
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=None, cluster_size=10,
        parsed_grouping_vars=["school"],
    )
    assert r.is_valid


def test_validate_cluster_config_cluster_size_too_small():
    # cluster_size < 5 → error
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=None, cluster_size=3,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_n_clusters_too_small():
    # n_clusters < 2 → error
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=1, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_icc_extreme_low():
    # ICC below the stability band [0.05, 0.95] → error (numerical stability)
    r = _validate_cluster_config(
        "school", icc=0.04, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_icc_lower_band_boundary_ok():
    # ICC == 0.05 is the inclusive lower edge of the stability band → valid.
    r = _validate_cluster_config(
        "school", icc=0.05, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert r.is_valid


def test_validate_cluster_config_icc_extreme_high():
    # ICC above the stability band [0.05, 0.95] → error
    r = _validate_cluster_config(
        "school", icc=0.96, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_icc_upper_band_boundary_ok():
    # ICC == 0.95 is the inclusive upper edge of the stability band → valid.
    r = _validate_cluster_config(
        "school", icc=0.95, n_clusters=20, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert r.is_valid


# ---------------------------------------------------------------------------
# _validate_cluster_sample_size
# ---------------------------------------------------------------------------


def test_validate_cluster_sample_size_valid():
    r = _validate_cluster_sample_size(sample_size=200, n_clusters=10, cluster_size=None)
    # 200/10 = 20 obs/cluster → valid, no warnings
    assert r.is_valid
    assert len(r.warnings) == 0


def test_validate_cluster_sample_size_min_5():
    r = _validate_cluster_sample_size(sample_size=10, n_clusters=10, cluster_size=None)
    # 10/10 = 1 obs/cluster → error
    assert not r.is_valid


def test_validate_cluster_sample_size_exactly_5():
    r = _validate_cluster_sample_size(sample_size=50, n_clusters=10, cluster_size=None)
    # 50/10 = 5 obs/cluster → valid (just at the boundary), may warn
    # The constraint is < 5 for error; exactly 5 is valid.
    assert r.is_valid


def test_validate_cluster_sample_size_warn_below_10():
    r = _validate_cluster_sample_size(sample_size=60, n_clusters=10, cluster_size=None)
    # 60/10 = 6 obs/cluster → valid but warns
    assert r.is_valid
    assert any("low" in w.lower() for w in r.warnings)


def test_validate_cluster_sample_size_with_cluster_size():
    # cluster_size takes priority over sample_size // n_clusters
    r = _validate_cluster_sample_size(sample_size=1000, n_clusters=None, cluster_size=3)
    # effective = 3 → error
    assert not r.is_valid


def test_validate_cluster_sample_size_cluster_size_valid():
    r = _validate_cluster_sample_size(sample_size=1000, n_clusters=None, cluster_size=15)
    # effective = 15 → valid
    assert r.is_valid
    assert len(r.warnings) == 0
