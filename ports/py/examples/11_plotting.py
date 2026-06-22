"""Visualise results: plot() opens a print-themed HTML (no extra deps), save_plot() renders
to png/svg/pdf/html; tours the print default, theme override, per-block file sets, and
stacked HTML."""

from mcpower import MCPower
from mcpower.output.plotting import available_themes

# Example: Clinical trial — does the new therapy speed recovery?
# Compute a result once, then tour every way to look at it.

# 1. Define the model and effects (treatment binary; see 01_basic_power).
model = MCPower("recovery = treatment + dose + age")
model.set_effects("treatment=0.5, dose=0.3, age=0.2")
model.set_variable_type("treatment=binary")

# 2. A find_power result. target_test="all" so the chart carries every effect
#    plus the omnibus row — the same view summary() prints as a table.
result = model.find_power(sample_size=120, target_test="all", verbose=False)

# 3. plot() with no path writes a print-themed stacked HTML (find_power.html)
#    and opens it. The HTML needs a CDN connection to load the Vega-Lite runtime
#    but requires NO extra Python packages. Headless? It writes the file and
#    tells you the path.
print(">>> result.plot()  # power-at-N: print-themed HTML, opened in browser")
result.plot()

# 4. save_plot() saves to a file; format follows the suffix (.png/.svg/.pdf/.html).
#    Default theme is 'print' — white background, black axes, light-grey grid,
#    colourblind-safe palette. Needs the optional renderer for non-HTML:
#    pip install mcpower[plot]
print("\n>>> result.save_plot('power_default.png')  # print theme by default")
try:
    result.save_plot("power_default.png")
    print("  saved power_default.png")
except ImportError as e:
    print(f"  renderer not installed — {e}")

# 5. theme= overrides the default. available_themes() lists the choices.
print(f"\n>>> available_themes()  ->  {available_themes()}")
print(">>> result.save_plot('power_dark.svg', theme='dark')")
try:
    result.save_plot("power_dark.svg", theme="dark")
    print("  saved power_dark.svg")
except ImportError as e:
    print(f"  renderer not installed — {e}")

# 6. HTML is the exception: one stacked file with every block, CDN-loaded runtime,
#    no extra packages needed. It is print-themed by default (it is a saved artifact).
print("\n>>> result.save_plot('power_report.html')  # stacked HTML, all blocks")
result.save_plot("power_report.html")
print("  saved power_report.html")

# 7. Power curves. A find_sample_size result plots as power-vs-N — one line per
#    effect, with the target-power reference line — the headline planning view.
#    Same plot() / save_plot() API; same print default.
curve = model.find_sample_size(
    target_test="all", from_size=40, to_size=300, by=20, verbose=False
)
print("\n>>> curve.plot()  # power curves: print-themed HTML, opened in browser")
curve.plot()

# 8. Multi-target sample-size: save_plot writes ONE FILE PER BLOCK with derived
#    names. For a single-scenario result with 3 targets the blocks are:
#      curve           -> power_curve.png           (the main per-effect curves)
#      at_least_k      -> power_curve_at_least_k.png
#      exactly_k       -> power_curve_exactly_k.png
#    The 'exactly_k' block shows P(exactly k effects significant) vs N, including
#    k=0 — new in this release.
print("\n>>> curve.save_plot('power_curve.png')  # per-block files (curve + joint)")
try:
    curve.save_plot("power_curve.png")
    print("  saved power_curve.png, power_curve_at_least_k.png, power_curve_exactly_k.png")
except ImportError as e:
    print(f"  renderer not installed — {e}")

# HTML is one stacked file even for multi-block results.
print("\n>>> curve.save_plot('power_curve.html')  # stacked HTML, all blocks")
curve.save_plot("power_curve.html")
print("  saved power_curve.html")

# 9. Scenarios. scenarios=True runs the built-in optimistic / realistic / doomer
#    assumptions. On find_power that's grouped bars (one group per scenario)...
scen = model.find_power(
    sample_size=120, target_test="all", scenarios=True, verbose=False
)
print("\n>>> scen.plot()  # scenario bars — opens in the browser")
scen.plot()

# ...and on find_sample_size it produces per-scenario blocks + an overlay grid.
# save_plot writes one file per block:
#   power_scen_<label>.png  for each scenario
#   power_scen_overlay.png  for the 3-column overlay grid
scen_curve = model.find_sample_size(
    target_test="all", from_size=40, to_size=300, by=20,
    scenarios=True, verbose=False,
)
print(">>> scen_curve.plot()  # scenario curves — opens in the browser")
scen_curve.plot()
print(">>> scen_curve.save_plot('power_scen.html')  # stacked HTML, all scenario blocks")
scen_curve.save_plot("power_scen.html")
print("  saved power_scen.html")

print(
    "\nplot() opened charts in your browser (find_power.html, find_sample_size.html; "
    "repeat calls get collision-suffixed names like find_power_2.html). HTML files "
    "are written regardless of display availability. "
    "PNG/SVG/PDF output requires: pip install mcpower[plot]"
)
