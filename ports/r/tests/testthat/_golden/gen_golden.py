"""Generate ols_contract.msgpack — the two-predictor OLS golden fixture.

Run from the _golden directory with the workspace venv activated:
    source /path/to/.venv/bin/activate
    cd mcpower/ports/r/tests/testthat/_golden
    python gen_golden.py

The blob is the raw msgpack output of build_contract_from_spec for a simple
y ~ x1 + x2 OLS model (x1=0.5, x2=0.3, scenario=["optimistic"]).
"""
import json
import mcpower
from mcpower import _engine

m = mcpower.MCPower("y ~ x1 + x2")
m.set_effects("x1=0.5, x2=0.3")

payload = m._to_linear_spec_dict(["optimistic"])
o, e, i, c = m._encode_outcome_and_clusters()
names, blob, _skeleton_json = _engine.build_contract_from_spec(json.dumps(payload), o, e, i, c)

with open("ols_contract.msgpack", "wb") as f:
    f.write(blob)

print("names:", names)
print("blob size:", len(blob), "bytes")
