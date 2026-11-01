"""Tests for LME cluster validators in v2.

Covers _validate_cluster_config. The rules
that set_cluster surfaces directly (both/neither sizing, ICC above 1, ICC 0,
unknown grouping var, valid configs) are pinned at the public boundary in
test_set_cluster.py; this file keeps the rows that boundary does not reach
(negative ICC, size floors, stability band edges, sample-size warnings).
"""

from __future__ import annotations

from mcpower.spec.validators import (
    _validate_cluster_config,
)


# ---------------------------------------------------------------------------
# _validate_cluster_config
# ---------------------------------------------------------------------------


def test_validate_cluster_config_icc_negative():
    r = _validate_cluster_config(
        "school", icc=-0.1, n_clusters=10, cluster_size=None,
        parsed_grouping_vars=["school"],
    )
    assert not r.is_valid


def test_validate_cluster_config_cluster_size_too_small():
    # cluster_size < min_rows_per_cluster → error
    r = _validate_cluster_config(
        "school", icc=0.2, n_clusters=None, cluster_size=1,
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
