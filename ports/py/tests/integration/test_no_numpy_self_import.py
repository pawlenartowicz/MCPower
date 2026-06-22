"""With numpy blocked at import, the package must still import and run a list-input analysis."""
import builtins
import sys

import pytest


@pytest.fixture
def numpy_blocked(monkeypatch):
    # Drop any cached numpy + mcpower so the re-import re-executes module bodies.
    for name in list(sys.modules):
        if name == "numpy" or name.startswith("numpy.") or name == "mcpower" or name.startswith("mcpower."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    yield


def test_import_and_find_power_without_numpy(numpy_blocked):
    import mcpower  # must not import numpy transitively

    m = mcpower.MCPower("y ~ x1 + x2")
    m.set_effects("x1=0.5, x2=0.0")
    res = m.find_power(sample_size=50, n_sims=50)  # list/scalar inputs only
    # A real result envelope with power for the tested target(s).
    assert res is not None
    assert "power_uncorrected" in res or "scenarios" in res
    assert res["n_sims"] > 0
