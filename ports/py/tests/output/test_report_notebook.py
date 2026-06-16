"""Notebook rich-repr test: Report._repr_mimebundle_ returns text/plain + Vega-Lite dict."""

from mcpower import MCPower

VEGA_MIME = "application/vnd.vegalite.v5+json"


def _rep():
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    m.set_simulations(200)
    return m.find_power(sample_size=120, verbose=False).summary()


def test_mimebundle_has_text_and_vegalite():
    bundle = _rep()._repr_mimebundle_(include=None, exclude=None)
    assert "text/plain" in bundle
    assert VEGA_MIME in bundle
    assert isinstance(bundle[VEGA_MIME], dict)  # parsed spec, not a string
