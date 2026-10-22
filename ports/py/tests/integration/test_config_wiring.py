import mcpower
from mcpower.config import get_simulation_defaults


def test_init_defaults_from_config():
    m = mcpower.MCPower("y ~ x")
    sim = get_simulation_defaults()
    assert m.seed == sim["seed"]
    assert m.alpha == sim["alpha"]
    assert m.power == sim["target_power"] * 100  # 0.8 -> 80.0
    assert m.n_simulations is None  # resolved per design at run time
    assert m._resolved_n_simulations() == sim["n_sims"]["ols"]
    assert m.max_failed_simulations == sim["max_failed_fraction"]  # 0.1, tightened from 1.0


def test_n_sims_default_by_design():
    # The default simulation count follows the design, resolved at run time:
    # unclustered models use `ols`, clustered continuous `mixed`, and clustered
    # binary/count (GLMM) `glmm`. set_simulations overrides all of them.
    sim = get_simulation_defaults()
    assert mcpower.MCPower("y ~ x")._resolved_n_simulations() == sim["n_sims"]["ols"]
    assert mcpower.MCPower("y ~ x", family="logit")._resolved_n_simulations() == sim["n_sims"]["ols"]

    lme = mcpower.MCPower("y = x + (1|g)", family="lme")
    lme.set_cluster("g", ICC=0.2, n_clusters=20)
    assert lme._resolved_n_simulations() == sim["n_sims"]["mixed"] == 800

    for family, extra in (("logit", {"ICC": 0.2}), ("poisson", {"tau_squared": 0.3})):
        m = mcpower.MCPower("y = x + (1|g)", family=family)
        m.set_cluster("g", n_clusters=20, **extra)
        assert m._resolved_n_simulations() == sim["n_sims"]["glmm"] == 400

    lme.set_simulations(123)
    assert lme._resolved_n_simulations() == 123
