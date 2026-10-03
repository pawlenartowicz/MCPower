from mcpower.config import get_cluster_limits, get_config


def test_cluster_limits_remap_limits_and_auto_count():
    # get_cluster_limits() is the port's own projection: four keys from
    # `limits` plus `simulation.cluster_auto_count` renamed to
    # `auto_count_default`. Literal config values are pinned by engine-contract.
    cfg = get_config()
    lim = cfg["limits"]
    assert get_cluster_limits() == {
        "min_rows_per_cluster": lim["min_rows_per_cluster"],
        "min_clusters": lim["min_clusters"],
        "reliable_rows_per_cluster": lim["reliable_rows_per_cluster"],
        "recommended_rows_per_cluster": lim["recommended_rows_per_cluster"],
        "auto_count_default": cfg["simulation"]["cluster_auto_count"],
    }
