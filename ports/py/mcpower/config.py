"""Shared scalar build constants, sourced from the engine-embedded
``configs/config.json`` via ``_engine.config()``. No copy lives in the Python
port."""

import json
from functools import lru_cache
from typing import Any, Dict


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    from . import _engine

    return json.loads(_engine.config())


def get_simulation_defaults() -> Dict[str, Any]:
    return get_config()["simulation"]


def get_benchmarks() -> Dict[str, Any]:
    return get_config()["benchmarks"]


def get_limits() -> Dict[str, Any]:
    return get_config()["limits"]


def get_report_config() -> Dict[str, Any]:
    return get_config()["report"]


def get_upload() -> Dict[str, Any]:
    """Upload-validator limits (row caps, factor-cardinality guards). Native
    hosts use ``max_rows``; ``min_rows`` is the shared lower bound."""
    return get_config()["upload"]


def get_correction_aliases() -> Dict[str, str]:
    """Input-alias → canonical snake_case correction name (e.g. ``bh`` →
    ``benjamini_hochberg``). Canonical names are absent: callers fall through to
    the input string, which the engine's ``Correction`` enum deserializes."""
    return get_config()["correction_aliases"]


@lru_cache(maxsize=1)
def get_dist_codes() -> Dict[str, int]:
    """Synthetic-distribution name → integer code, sourced from the engine
    (``engine-spec-builder``). Single source of truth; the Python port keeps no
    copy. Covers synthetic codes 0–5 plus the uploaded sentinels 97/98/99."""
    from . import _engine

    return json.loads(_engine.dist_codes())


@lru_cache(maxsize=1)
def get_residual_codes() -> Dict[str, int]:
    """Residual-distribution name → integer code, sourced from the engine.
    Canonical five: normal=0, right_skewed=2, left_skewed=3, high_kurtosis=4,
    uniform=5. Single source of truth; the Python port keeps no copy."""
    from . import _engine

    return json.loads(_engine.residual_codes())


@lru_cache(maxsize=1)
def get_re_dist_codes() -> Dict[str, int]:
    """Random-effect distribution name → integer code, sourced from the engine.
    Vocabulary: normal=0, heavy_tailed=1, right_skewed=2. Kept separate from
    residual codes because the RE knob uses its own name space."""
    from . import _engine

    return json.loads(_engine.re_dist_codes())


def get_cluster_limits() -> Dict[str, Any]:
    """Cluster-related limits.

    Exposes the four `*_per_cluster`/`*_clusters` keys from
    `limits` plus `auto_count_default` (from `simulation.cluster_auto_count`)."""
    cfg = get_config()
    lim = cfg["limits"]
    return {
        "min_rows_per_cluster": lim["min_rows_per_cluster"],
        "min_clusters": lim["min_clusters"],
        "reliable_rows_per_cluster": lim["reliable_rows_per_cluster"],
        "recommended_rows_per_cluster": lim["recommended_rows_per_cluster"],
        "auto_count_default": cfg["simulation"]["cluster_auto_count"],
    }
