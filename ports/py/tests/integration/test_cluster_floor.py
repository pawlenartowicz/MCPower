"""The port passes a design down to the engine's rows-per-cluster floor."""

import warnings

import pytest

from mcpower import MCPower


def _intercept(n_clusters: int) -> MCPower:
    model = MCPower("y ~ x + (1|g)", family="lme").set_effects("x=0.5")
    return model.set_cluster("g", ICC=0.2, n_clusters=n_clusters)


def test_two_observations_per_cluster_run_without_a_cluster_size_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _intercept(20).find_power(40, n_sims=20, verbose=False)


def test_one_observation_per_cluster_is_refused() -> None:
    with pytest.raises(ValueError, match="a cluster needs at least 2 observations"):
        _intercept(20).find_power(20, n_sims=20, verbose=False)


def test_a_random_slope_raises_the_floor_by_one() -> None:
    model = (
        MCPower("y ~ x + (1 + x|g)", family="lme")
        .set_effects("x=0.5")
        .set_cluster("g", ICC=0.2, n_clusters=20, random_slopes=["x"], slope_variance=0.1)
    )
    with pytest.raises(ValueError, match="a cluster needs at least 3 observations"):
        model.find_power(40, n_sims=20, verbose=False)
    model.find_power(60, n_sims=20, verbose=False)
