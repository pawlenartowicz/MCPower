"""Smoke tests for the contract-bridge entry points exposed by ``_engine``.

``build_contract_from_spec`` is the host path used by every Python
``find_power`` / ``find_sample_size`` call. It returns a
``(names, contracts_msgpack)`` pair where ``contracts_msgpack`` decodes as a
list of ``SimulationContract`` dicts.

``build_contract_from_json`` returns the raw ``SimulationContract``
msgpack — gated behind the engine-py ``test-bridge`` Cargo feature so
release wheels don't ship it; the corresponding test below skips when the
symbol is absent.
"""

import json

import msgpack
import pytest

from mcpower import _engine


def _simple_linear_spec_json() -> str:
    return json.dumps(
        {
            "formula": "y = x1 + x2",
            "predictors": [
                {"name": "x1", "kind": "normal"},
                {"name": "x2", "kind": "normal"},
            ],
            "effects": [
                {"name": "x1", "size": 0.5},
                {"name": "x2", "size": 0.3},
            ],
            "correlations": [],
            "alpha": 0.05,
            "correction": "none",
            "targets": ["overall"],
            "heteroskedasticity": {"driver_var_index": None},
            "residual": {"distribution": "normal"},
            "max_failed_fraction": 0.03,
            "scenarios": [],
        }
    )


def test_contract_specs_produces_expected_simulationspec() -> None:
    names, payload, _skeleton = _engine.build_contract_from_spec(
        _simple_linear_spec_json(),
        "continuous",
        "canonical",
        "ols",
        0.0,
        "[]",
    )
    assert names == ["optimistic"]
    contracts = msgpack.unpackb(payload, raw=False)
    assert len(contracts) == 1
    contract = contracts[0]
    # Two continuous predictors → two generation columns
    assert len(contract["generation"]["columns"]) == 2
    # coefficients: [intercept=0.0, x1=0.5, x2=0.3]
    assert contract["outcome"]["coefficients"] == [0.0, 0.5, 0.3]
    # Two marginal test targets (terms 1 and 2, i.e. x1 and x2)
    targets = contract["test"]["targets"]
    assert len(targets) == 2
    assert all(t["kind"] == "marginal" for t in targets)
    assert {t["term"] for t in targets} == {1, 2}


@pytest.mark.skipif(
    not hasattr(_engine, "build_contract_from_json"),
    reason="build_contract_from_json is gated behind the engine-py `test-bridge` "
    "Cargo feature; release wheels do not expose it.",
)
def test_build_contract_returns_msgpack() -> None:
    contracts = _engine.build_contract_from_json(_simple_linear_spec_json())
    assert len(contracts) == 1
    name, payload = contracts[0]
    assert name == "optimistic"
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) > 0
