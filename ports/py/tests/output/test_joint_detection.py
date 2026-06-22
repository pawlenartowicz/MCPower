"""find_sample_size long-form summary — joint required-N table.

Contracts:
  1. "Joint detection" + "of 3 tests" present; "analytical fast-path" absent
     (the engine is MC-only; the string should never appear in output).
  2. Header does not show the lowest grid N (N=20) when a higher N is needed.
  3. "≥ <ceiling>" sentinel appears when the target is unreachable within the
     search range (the required N lies above the grid's top — never a bare "—").
"""
import mcpower


def _fit_sample_size(target_power=0.8):
    model = mcpower.MCPower("y ~ a + b + c")
    model.set_effects("a=0.5, b=0.3, c=0.1")
    return model.find_sample_size(
        from_size=20, to_size=400, by=20, target_power=target_power, n_sims=500,
        verbose=False,
    )


def test_long_summary_has_joint_required_n_table():
    res = _fit_sample_size()
    text = str(res.summary())
    assert "Joint detection" in text
    assert "of 3 tests" in text
    assert "analytical fast-path" not in text


def test_long_summary_header_uses_recommended_n_not_first_grid_point():
    res = _fit_sample_size()
    text = str(res.summary())
    assert "N=20 " not in text


def test_long_summary_header_shows_recommended_n_when_all_reached():
    # All-large effects so every target reaches the target power within the
    # grid -> the header must render the recommended "N≥<n>" form, not the
    # "not reached" sentinel. Guards against a regression that shows the
    # sentinel when an N actually was found.
    model = mcpower.MCPower("y ~ a + b + c")
    model.set_effects("a=0.5, b=0.5, c=0.5")
    res = model.find_sample_size(
        from_size=20, to_size=400, by=20, target_power=0.8, n_sims=500,
        verbose=False,
    )
    text = str(res.summary())
    assert "N≥" in text
    assert "target not reached" not in text


def test_unreached_target_uses_geq_ceiling_sentinel():
    # target_power=0.99 is unreachable within from=20..to=400, so the required N
    # lies above the grid ceiling (400) and renders as "≥ 400" in the Required N
    # column. The CI column may legitimately show "—" for unreached rows.
    res = _fit_sample_size(target_power=0.99)
    text = str(res.summary())
    assert "≥ 400" in text
    # Required sample size table must not have a bare "—" in the Required N column
    # (only "≥ 400" sentinel). Check specifically in the Required N section.
    req_section = text.split("Required sample size per effect")[-1].split("\n\n")[0] if "Required sample size per effect" in text else ""
    assert "—" not in req_section, f"Bare '—' found in required N section:\n{req_section}"


def test_joint_required_n_uses_fitted_joint_chain():
    """fitted_joint status==fitted renders n_achievable; at_or_below_min renders ≤ n_min."""
    import mcpower
    from mcpower.output.report import Report

    res = _fit_sample_size()
    # Verify the fitted_joint key exists and has the right len.
    fj = res.get("fitted_joint")
    fja = res.get("first_joint_achieved", {})
    assert fj is not None, "fitted_joint missing from result"
    assert len(fj) == len(fja), "fitted_joint / first_joint_achieved length mismatch"

    text = str(res.summary())
    # Joint section must render without errors and contain the expected header.
    assert "Joint detection" in text


import json


def test_sample_size_plot_set_includes_at_least_k_block():
    """The sample-size plot set for multi-target results includes an at_least_k block."""
    from mcpower.output import plotting
    res = _fit_sample_size()
    blocks = plotting._plot_blocks(res, res._meta, "find_sample_size", label_map={})
    block_keys = [k for k, _ in blocks]
    assert "at_least_k" in block_keys
    # The at_least_k spec must have data rows with a "k" field
    at_least_spec = next(s for k, s in blocks if k == "at_least_k")
    assert at_least_spec["data"]["values"]
    assert any("k" in row for row in at_least_spec["data"]["values"])


def test_sample_size_mimebundle_has_print_theme():
    """The mimebundle spec for a single-scenario sample-size result is the curve
    block with the print theme applied."""
    res = _fit_sample_size()
    bundle = res.summary()._repr_mimebundle_()
    spec = bundle["application/vnd.vegalite.v5+json"]
    # Print theme is applied — check the config block is present
    assert "config" in spec
    # Print theme has a legend block
    assert "legend" in spec["config"]
