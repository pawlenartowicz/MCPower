"""L2 tests for find_sample_size's `unit` argument.

The engine resolves the unit → divisor and converts bounds/results (see
`engine_orchestrator::unit`); these tests check the Python binding wires it
through correctly: both cluster-unit regimes, the two validation error paths,
that the default is unchanged, and that the sample-size plot spec picks up
the divisor.
"""

from __future__ import annotations

import pytest

from mcpower import MCPower


def test_find_sample_size_unit_clusters_fixed_size():
    """FixedSize (cluster_size=8) + unit="clusters": bounds 10..50 by 5 in
    clusters convert to N bounds 80..400 step 40; sample_sizes_unit mirrors
    sample_sizes / 8."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=8)
    res = m.find_sample_size(
        from_size=10, to_size=50, by=5, unit="clusters", n_sims=20, verbose=False
    )
    assert res["sample_sizes"] == [80, 120, 160, 200, 240, 280, 320, 360, 400]
    assert res["sample_sizes_unit"] == [10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert res["unit_divisor"] == 8


def test_find_sample_size_unit_cluster_size_fixed_clusters():
    """FixedClusters (n_clusters=10) + unit="cluster_size": bounds 6..20 by 2
    (observations per school) convert to N bounds 60..200 step 20."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, n_clusters=10)
    res = m.find_sample_size(
        from_size=6, to_size=20, by=2, unit="cluster_size", n_sims=20, verbose=False
    )
    assert res["sample_sizes"] == [60, 80, 100, 120, 140, 160, 180, 200]
    assert res["sample_sizes_unit"] == [6, 8, 10, 12, 14, 16, 18, 20]
    assert res["unit_divisor"] == 10


def test_find_sample_size_unit_regime_mismatch_raises():
    """unit="cluster_size" on a FixedSize (cluster_size=...) design does not
    fit — that regime is fixed by size, not by cluster count."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=8)
    with pytest.raises(ValueError, match="does not fit the fixed_size"):
        m.find_sample_size(
            from_size=10, to_size=50, by=5, unit="cluster_size", n_sims=20, verbose=False
        )


def test_find_sample_size_unit_on_unclustered_raises():
    """unit="clusters" on a plain (non-LME) model has no cluster to divide by."""
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="needs a clustered model"):
        m.find_sample_size(
            from_size=40, to_size=200, by=40, unit="clusters", n_sims=20, verbose=False
        )


def test_find_sample_size_unit_default_unchanged():
    """Omitting `unit` behaves exactly like unit="observations" (divisor 1,
    sample_sizes_unit mirrors sample_sizes)."""
    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=8)
    res = m.find_sample_size(from_size=80, to_size=400, by=40, n_sims=20, verbose=False)
    assert res["unit_divisor"] == 1
    assert res["sample_sizes_unit"] == res["sample_sizes"]


def test_sample_size_plot_spec_uses_unit_axis():
    """With unit="clusters" the curve spec's x field is `n_unit` and every
    layer's x encoding carries a host-set title (line + CI band)."""
    from mcpower.output.plotting import _plot_blocks
    from mcpower.output.tables import target_label_map

    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=8)
    res = m.find_sample_size(
        from_size=10, to_size=50, by=5, unit="clusters", n_sims=20, verbose=False
    )
    blocks = _plot_blocks(
        res, res._meta, "find_sample_size",
        label_map=target_label_map(res, res._meta),
    )
    curve_spec = dict(blocks)["curve"]
    # Line + CI band layers carry an x encoding; a target-power-line rule
    # layer (y-only) does not — check only the ones that do.
    x_layers = [l for l in curve_spec["layer"] if "x" in l.get("encoding", {})]
    assert len(x_layers) >= 2
    for layer in x_layers:
        x_enc = layer["encoding"]["x"]
        assert x_enc["field"] == "n_unit"
        assert "title" in x_enc and x_enc["title"]


def test_sample_size_plot_spec_default_unit_unchanged():
    """With no unit argument the curve spec keeps field `n` and the engine's
    own "Sample size (N)" title — byte-identical to before this feature."""
    from mcpower.output.plotting import _plot_blocks
    from mcpower.output.tables import target_label_map

    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.5").set_cluster("school", ICC=0.2, cluster_size=8)
    res = m.find_sample_size(from_size=80, to_size=400, by=40, n_sims=20, verbose=False)
    blocks = _plot_blocks(
        res, res._meta, "find_sample_size",
        label_map=target_label_map(res, res._meta),
    )
    curve_spec = dict(blocks)["curve"]
    x_layers = [l for l in curve_spec["layer"] if "x" in l.get("encoding", {})]
    assert len(x_layers) >= 2
    for layer in x_layers:
        x_enc = layer["encoding"]["x"]
        assert x_enc["field"] == "n"
        assert x_enc["title"] == "Sample size (N)"


def test_long_form_summary_shows_unit_columns_in_every_required_n_table():
    """summary() with scenarios and two targets: the multi-scenario required-N
    table, the per-scenario CI tables and the joint table all carry the unit
    column (header from `unit_display_label`) before the N column."""
    m = MCPower("y ~ x1 + x2 + (1|school)", family="lme")
    m.set_effects("x1=0.5, x2=0.4").set_cluster("school", ICC=0.2, cluster_size=8)
    res = m.find_sample_size(
        from_size=10, to_size=50, by=5, unit="clusters", n_sims=20, verbose=False,
        scenarios=True,
    )
    text = str(res.summary())
    assert "unit: Number of school" in text
    # multi-scenario table: unit column then "<scenario> (N)" column
    assert "optimistic   optimistic (N)" in text
    # CI table and joint table: unit column between the label and Required N
    assert "Number of school   Required N   CI 95%" in text
    assert "Joint target         Number of school   Required N" in text
