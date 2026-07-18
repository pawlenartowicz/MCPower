"""Generate two_factor_accumulation_contract.msgpack — the cross-port parity
fixture for multi-call setter accumulation.

Run from the _golden directory with the workspace venv activated:
    source /path/to/.venv/bin/activate
    cd mcpower/ports/r/tests/testthat/_golden
    python gen_two_factor_golden.py

The blob is the raw msgpack output of build_contract_from_spec for a 2x2
factor model whose two factors are declared via SEPARATE set_variable_type
calls (the chained one-factor-at-a-time pattern that overwrite used to break).
R must reproduce byte-identical bytes from the same separate calls, proving the
accumulation fix is cross-port consistent.
"""
import json
import mcpower
from mcpower import _engine

m = mcpower.MCPower("y ~ g1*g2")
m.set_variable_type("g1=(factor, 0.5, 0.5)")
m.set_variable_type("g2=(factor, 0.6, 0.4)")
m.set_effects("g1[2]=0.5, g2[2]=0.4, g1[2]:g2[2]=0.3")

payload = m._to_linear_spec_dict(["optimistic"])
o, link, e, i, c = m._encode_outcome_and_clusters()
names, blob, _skeleton_json = _engine.build_contract_from_spec(json.dumps(payload), o, link, e, i, c)

with open("two_factor_accumulation_contract.msgpack", "wb") as f:
    f.write(blob)

print("names:", names)
print("blob size:", len(blob), "bytes")
