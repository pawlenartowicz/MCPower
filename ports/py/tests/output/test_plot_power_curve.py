"""Smoke test for the find_power plot-set bridge — proves the Python ⇄ Rust
wiring lands; spec shape is locked by Rust snapshot tests."""

import json

from mcpower import MCPower
from mcpower.output import plotting

V5 = "https://vega.github.io/schema/vega-lite/v5.json"


def test_plot_power_blocks_wire_through_to_rust_emitter():
    mp = (
        MCPower("y ~ x1 + x2")
        .set_effects("x1=0.4, x2=0.3")
        .set_simulations(200)            # small for test speed
        .set_seed(2137)
    )
    result = mp.find_power(120)
    blocks = plotting._plot_blocks(result, result._meta, "find_power", label_map={})
    assert len(blocks) == 1
    key, spec = blocks[0]
    assert key == "power"
    assert spec["$schema"] == V5
    assert spec["data"]["values"], "data.values must be non-empty"
    # No config — engine emits theme-naked, post-processing doesn't add theme
    assert "config" not in spec, "engine must be theme-naked by default"
