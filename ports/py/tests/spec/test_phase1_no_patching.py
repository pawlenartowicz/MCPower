"""Structural guard — `to_simulation_spec` must not post-patch family/intercept/cluster.

The builder is parametrised on family, intercept, and clusters via
`_engine.build_contract_from_spec(...)`; the Python frontend must not override
those fields after the fact. This test pins that invariant by asserting the
forbidden mutations do not appear in the function source.
"""
from __future__ import annotations

import inspect

from mcpower import model as model_mod


def test_to_simulation_spec_does_not_mutate_family_or_intercept() -> None:
    """The function body must not contain string assignments to
    spec['family'] / spec['intercept'] / spec['cluster'].
    """
    src = inspect.getsource(model_mod.MCPower.to_simulation_spec)
    forbidden = [
        'spec["family"] =',
        'spec["intercept"] =',
        'spec["cluster"] =',
    ]
    for s in forbidden:
        assert s not in src, f"forbidden patch survives: {s}"
