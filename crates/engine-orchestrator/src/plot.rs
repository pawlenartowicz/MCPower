//! Vega-Lite v5 emitters for power-curve plots. Theme-naked:
//! no `config`, no `mark.color`, no font choices — every port grafts its
//! theme onto `spec["config"]` before rendering.
//!
//! # Emitters
//!
//! - [`power_at_n_spec`] — horizontal bar chart: power per (scenario × target).
//! - [`sample_size_curve_spec`] — line+band: power vs N curves.
//! - [`joint_detection_curve_spec`] — P(detect >= k targets) vs N.
//! - [`exactly_k_curve_spec`] — P(exactly k targets significant) vs N.
//!
//! # Plot-set functions
//!
//! - [`power_plot_set`] — ordered `(block_key, spec)` pairs for a find-power result.
//! - [`sample_size_plot_set`] — ordered `(block_key, spec)` pairs for a find-sample-size result.
//!
//! Only `power_plot_set` / `sample_size_plot_set` (and the option/point types)
//! are port contract; the individual `*_spec` builders are internal-stability,
//! exposed for tests, free to change at any minor version.

use serde_json::{json, Value};

use crate::grid::as_proportion;
use crate::result::Ci;

const SCHEMA: &str = "https://vega.github.io/schema/vega-lite/v5.json";

/// Bar thickness in px for the horizontal `power_at_n_spec`. Height is derived
/// from bar count via this constant (the height-from-bandwidth model).
const BAR_THICKNESS: f64 = 16.0;
/// Fixed data-rect width for the horizontal power bars and the single-scenario
/// curve panels.
const PANEL_WIDTH: f64 = 360.0;
/// Fixed height for the sample-size / joint curves.
const CURVE_HEIGHT: f64 = 240.0;
/// Per-panel width inside a faceted (multi-scenario) curve grid.
const FACET_PANEL_WIDTH: f64 = 200.0;
/// Facet column count — 3 is the common scenario preset.
const FACET_COLUMNS: u64 = 3;
/// Above this scenario count, power-at-N facets into per-scenario panels instead
/// of shading by opacity — faint shades stop being distinguishable (the rare 5+
/// case). Mirrors the curve's per-scenario faceting.
const POWER_FACET_THRESHOLD: usize = 4;

/// Knobs shared by every plot emitter; `Default` = no title, no CI band, no
/// target-power rule, x in observations.
#[derive(Debug, Clone)]
pub struct PlotOptions {
    pub title: Option<String>,
    pub show_ci: bool,
    pub target_power_line: Option<f64>,
    /// Sample-size unit divisor for the curve x-axis (`SampleSizeResult
    /// .unit_divisor`). 1 = today's spec, x field `n` titled "Sample size (N)"
    /// / "N". `> 1` switches the x field to `n_unit = n / x_divisor` and emits
    /// **no** x title — the host owns the unit's wording ("Participants",
    /// "Observations per participant") and sets `encoding.x.title` itself.
    /// The row keeps `n` too, so a host can still read N off a point. Only the
    /// sample-size curves read it; power-at-N ignores it.
    pub x_divisor: usize,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            title: None,
            show_ci: false,
            target_power_line: None,
            x_divisor: 1,
        }
    }
}

/// x-axis encoding for the sample-size curves: `n` with the given title at
/// divisor 1, else `n_unit` with no title (host-labelled). The CI band layer
/// must use the same title as the line layer or Vega concatenates them.
fn x_encoding(opts: &PlotOptions, title_at_n: &str) -> Value {
    if opts.x_divisor > 1 {
        json!({ "field": "n_unit", "type": "quantitative" })
    } else {
        json!({ "field": "n", "type": "quantitative", "title": title_at_n })
    }
}

/// Add the `n_unit = n / x_divisor` row field the unit x-axis reads. Only
/// when the divisor is above 1, so the divisor-1 spec stays byte-identical to
/// the unit-less one. Integer division — every grid N is a multiple of the
/// divisor, see `sample_size_result_to_host`.
fn add_n_unit_rows(rows: &mut [Value], opts: &PlotOptions) {
    if opts.x_divisor > 1 {
        for r in rows.iter_mut() {
            let n = r["n"].as_u64().unwrap_or(0);
            r["n_unit"] = json!(n / opts.x_divisor as u64);
        }
    }
}

/// One grid point — or the single point of a find_power result — as the plot
/// emitters consume it. Only the fields the emitters actually read; nothing is
/// fabricated. `power`/`ci` are empty for the joint-detection curve; `histogram`
/// (corrected joint-significance buckets) is empty for the power curves.
#[derive(Debug, Clone)]
pub struct PlotPoint {
    pub n: usize,
    pub target_indices: Vec<usize>,
    /// Contrast identities for the power entries past the marginals — same
    /// `(positive, negative)` β-column pairs as `PowerResult.contrast_pairs`.
    /// Empty when the result carries no contrasts.
    pub contrast_pairs: Vec<(u32, u32)>,
    pub power: Vec<f64>,
    pub ci: Vec<Ci>,
    pub histogram: Vec<u64>,
    /// Overall/omnibus test power at this grid point (`overall_significant_rate`),
    /// or `None` when no overall test was requested. The power curve renders it
    /// as an extra `"overall"` series; the joint/exactly-k curves ignore it.
    pub overall_power: Option<f64>,
    /// Wilson CI for `overall_power`; `None` whenever `overall_power` is `None`.
    pub overall_ci: Option<Ci>,
}

/// One scenario's points as the plot emitters consume them.
/// `points.len() == 1` for power-at-N.
#[derive(Debug, Clone)]
pub struct PlotScenario {
    pub label: String,
    pub points: Vec<PlotPoint>,
}

fn target_label(idx: usize) -> String {
    format!("target_{idx}")
}

/// Label for the `i`-th power/CI entry of a point. Power vectors hold the
/// marginal targets first (one per `target_indices` entry) and then the
/// contrast pairs — so entries past `target_indices.len()` get an
/// identity-bearing `target_{p}_vs_{n}` token built from the pair's β-column
/// indices, which hosts relabel from the effect skeleton exactly like a
/// `target_{idx}` token. Indexing `target_indices[i]` directly panics on
/// contrast entries. Falls back to an ordinal `contrast_{j}` when the point
/// carries no pair identities (older payloads).
fn entry_label(point: &PlotPoint, i: usize) -> String {
    let n_marginals = point.target_indices.len();
    if i < n_marginals {
        target_label(point.target_indices[i])
    } else if let Some(&(p, n)) = point.contrast_pairs.get(i - n_marginals) {
        format!("target_{p}_vs_{n}")
    } else {
        format!("contrast_{}", i - n_marginals)
    }
}

/// Per-scenario `fillOpacity` range, bold → faint, floor 0.4. Scenarios are an
/// ordered set (optimistic → realistic → doomer for the presets); shade is a
/// second *identity* channel (not magnitude). The n==2 case is widened to 0.6 so
/// two scenarios stay clearly readable. Only called for 2..=POWER_FACET_THRESHOLD
/// scenarios (more → facet, see power_at_n_spec).
fn scenario_opacity_range(n: usize) -> Vec<f64> {
    match n {
        0 | 1 => vec![1.0],
        2 => vec![1.0, 0.6],
        _ => (0..n)
            .map(|i| 1.0 - 0.6 * (i as f64) / ((n - 1) as f64))
            .collect(),
    }
}

/// Theme-naked Vega-Lite v5 horizontal bar chart, one bar per (scenario × target).
/// Bars are flush within a scenario group with a ⅔-bar gap between target groups;
/// the data-rect height is derived from the bar count (see [`BAR_THICKNESS`]).
pub fn power_at_n_spec(scenarios: &[PlotScenario], opts: &PlotOptions) -> String {
    let multi = scenarios.len() > 1;

    let mut values: Vec<Value> = Vec::new();
    for sc in scenarios {
        let label = &sc.label;
        let Some(point) = sc.points.first() else {
            continue;
        };
        for (i, &power) in point.power.iter().enumerate() {
            let ci = &point.ci[i];
            values.push(json!({
                "scenario": label,
                "target":   entry_label(point, i),
                "power":    power,
                "ci_lo":    ci.lo,
                "ci_hi":    ci.hi,
            }));
        }
        // The overall/omnibus test as one more bar under the `"overall"` token
        // (hosts relabel it F-test / LRT), appended after the marginals/contrasts
        // so it lands last on the axis — matching the results table, where the
        // overall row follows the per-effect rows. Mirrors the sample-size curve.
        if let Some(op) = point.overall_power {
            let ci = point.overall_ci.unwrap_or(Ci { lo: op, hi: op });
            values.push(json!({
                "scenario": label,
                "target":   "overall",
                "power":    op,
                "ci_lo":    ci.lo,
                "ci_hi":    ci.hi,
            }));
        }
    }

    // The overall test (when present) draws one extra bar per scenario; count it
    // in the row total so the data-rect height accounts for it.
    let has_overall = scenarios
        .first()
        .and_then(|sc| sc.points.first())
        .map(|p| p.overall_power.is_some())
        .unwrap_or(false);
    // G = target rows the emitter draws (one bar per target), S = scenarios/row.
    let g = scenarios
        .first()
        .and_then(|sc| sc.points.first())
        .map(|p| p.power.len() + usize::from(has_overall))
        .unwrap_or(0) as f64;
    let s = if multi { scenarios.len() as f64 } else { 1.0 };
    // Flush within group, ⅔-bar gap between groups; 7-unit floor (the 2-row ×
    // 3-scenario case ≈ 6.67).
    let units = (g * s + (g - 1.0) * 2.0 / 3.0).max(7.0);
    let height_px = (units * BAR_THICKNESS).round();
    // gap = ⅔t, groupband = S·t ⇒ paddingInner = (⅔)/(⅔ + S) = 2/(2 + 3S).
    let y_padding_inner = 2.0 / (2.0 + 3.0 * s);

    // Scenario order = host order (optimistic → realistic → doomer), reused for
    // yOffset stacking and the fillOpacity ramp. Replaces the old mean-power sort
    // (colour no longer keys on scenario, so power-ordering it is meaningless).
    let scenario_order: Option<Value> = if multi {
        Some(json!(scenarios
            .iter()
            .map(|sc| sc.label.as_str())
            .collect::<Vec<_>>()))
    } else {
        None
    };

    let mut bar_enc = json!({
        "x": {
            "field": "power", "type": "quantitative",
            "title": "Power", "scale": { "domain": [0, 1] },
        },
        "y": {
            "field": "target", "type": "nominal", "title": "Effect",
            // `sort: null` ⇒ axis follows data order (power-vector order, with the
            // appended `overall` bar last) instead of Vega's default alphabetical.
            // This matches the results table and keeps `overall` at the bottom. A
            // token sort list would break under host relabel (see the colour note).
            "sort": null,
            "scale": { "paddingInner": y_padding_inner, "paddingOuter": 0 },
        },
    });
    // Colour by effect (`target`), applied even for a single scenario. Deliberately
    // NO explicit `scale.domain`: hosts relabel the `target` *data* values in place
    // (`target_{idx}` → effect name) and leave the encoding alone, so a domain pinned
    // to the engine's tokens would no longer match any data value — every mark would
    // fall outside the scale and render with a null fill (invisible bars). Letting
    // Vega derive the domain from the (relabelled) data keeps colour working; the same
    // effect lands on the same palette slot across this plot and the curve because both
    // emit effects in the same power-vector order.
    bar_enc["color"] = json!({ "field": "target", "type": "nominal" });
    if let Some(order) = &scenario_order {
        bar_enc["yOffset"] = json!({
            "field": "scenario", "type": "nominal",
            "scale": { "paddingInner": 0, "paddingOuter": 0 },
            "sort": order,
        });
    }
    if multi && scenarios.len() <= POWER_FACET_THRESHOLD {
        bar_enc["fillOpacity"] = json!({
            "field": "scenario", "type": "nominal",
            "sort": scenario_order.as_ref().unwrap(),
            "scale": { "range": scenario_opacity_range(scenarios.len()) },
        });
    }
    let mut layers: Vec<Value> = vec![json!({ "mark": "bar", "encoding": bar_enc })];

    if opts.show_ci {
        // Repeat the bar layer's axis titles on the shared x/y scales: a layered
        // spec concatenates the *distinct* titles of co-scaled encodings, so a
        // title-less errorbar would render the axis as "Power, ci_lo". Matching
        // titles collapse to one.
        let mut ci_enc = json!({
            "x":  { "field": "ci_lo", "type": "quantitative", "title": "Power" },
            "x2": { "field": "ci_hi" },
            // Match the bar layer's data-order sort so the errorbars align with
            // their bars on the shared y scale (see the bar `sort: null` note).
            "y":  { "field": "target", "type": "nominal", "title": "Effect", "sort": null },
        });
        // Mirror the bar layer's effect colour (no pinned domain — see the bar block).
        ci_enc["color"] = json!({ "field": "target", "type": "nominal" });
        if let Some(ref sort_arr) = scenario_order {
            ci_enc["yOffset"] = json!({
                "field": "scenario", "type": "nominal",
                "sort": sort_arr,
            });
        }
        if multi && scenarios.len() <= POWER_FACET_THRESHOLD {
            ci_enc["fillOpacity"] = json!({
                "field": "scenario", "type": "nominal",
                "sort": scenario_order.as_ref().unwrap(),
                "scale": { "range": scenario_opacity_range(scenarios.len()) },
            });
        }
        layers.push(json!({ "mark": "errorbar", "encoding": ci_enc }));
    }

    if let Some(level) = opts.target_power_line {
        layers.push(json!({
            "mark": { "type": "rule", "strokeDash": [4, 4] },
            "encoding": { "x": { "datum": level, "type": "quantitative" } },
        }));
    }

    let mut spec = if multi && scenarios.len() > POWER_FACET_THRESHOLD {
        json!({
            "$schema": SCHEMA,
            "data": { "values": values },
            "facet": {
                "field": "scenario", "type": "nominal", "columns": FACET_COLUMNS,
                "sort": scenarios.iter().map(|sc| sc.label.as_str()).collect::<Vec<_>>(),
            },
            "spec": { "width": PANEL_WIDTH, "height": height_px, "layer": layers },
        })
    } else {
        json!({
            "$schema": SCHEMA,
            "width": PANEL_WIDTH,
            "height": height_px,
            "data": { "values": values },
            "layer": layers,
        })
    };
    if let Some(title) = &opts.title {
        spec["title"] = Value::String(title.clone());
    }
    serde_json::to_string(&spec).expect("Value serialises infallibly")
}

/// Theme-naked Vega-Lite v5 line+band chart for sample-size curves.
/// Per-series rows are sorted by ascending `n` so the line renders
/// left-to-right. `target_power_line: None` ⇒ no rule layer; the wrapper
/// layer is responsible for choosing a default if desired.
pub fn sample_size_curve_spec(scenarios: &[PlotScenario], opts: &PlotOptions) -> String {
    let multi_scenario = scenarios.len() > 1;
    // Entry count, not `target_indices.len()` — power vectors append contrast
    // entries past the marginals (see `entry_label`).
    let n_targets = scenarios
        .first()
        .and_then(|sc| sc.points.first())
        .map(|p| p.power.len())
        .unwrap_or(0);
    // The overall test (when present) renders as one extra series. Gate
    // strokeDash on the *rendered* series count, not just `n_targets`: a
    // single-marginal-target curve plus the overall would otherwise draw two
    // solid lines that nearly overlap (F = t²) and can't be told apart.
    let has_overall = scenarios
        .first()
        .and_then(|sc| sc.points.first())
        .map(|p| p.overall_power.is_some())
        .unwrap_or(false);
    let multi_target = n_targets + usize::from(has_overall) > 1;

    // Colour keys on the effect (`target`) with a pinned domain so each effect
    // gets a stable palette slot shared with `power_at_n_spec`; `strokeDash` also
    // keys on the effect as a redundant channel (overlap + colourblind/print). The
    // joint/exactly-k panels are SEPARATE specs with their own colour scale — no
    // shared-scale conflict in the engine (a host vconcat would need
    // `resolve.scale.color = independent`; see the design spec).
    let series_of = |label: &str, target: &str| -> String {
        if multi_scenario {
            target.to_string()
        } else {
            format!("{label} · {target}")
        }
    };

    let mut rows: Vec<Value> = Vec::new();
    for sc in scenarios {
        let label = &sc.label;
        let mut sorted: Vec<&PlotPoint> = sc.points.iter().collect();
        sorted.sort_by_key(|p| p.n);
        for p in &sorted {
            for (t, &power) in p.power.iter().enumerate() {
                let ci = &p.ci[t];
                let target = entry_label(p, t);
                rows.push(json!({
                    "scenario": label,
                    "target":   target,
                    "series":   series_of(label, &target),
                    "n":        p.n,
                    "power":    power,
                    "ci_lo":    ci.lo,
                    "ci_hi":    ci.hi,
                }));
            }
            // The overall/omnibus test as one more series under the `"overall"`
            // token (hosts relabel it F-test / LRT). Only the power curve carries
            // it; the joint/exactly-k curves never look at `overall_power`.
            if let Some(op) = p.overall_power {
                let ci = p.overall_ci.unwrap_or(Ci { lo: op, hi: op });
                rows.push(json!({
                    "scenario": label,
                    "target":   "overall",
                    "series":   series_of(label, "overall"),
                    "n":        p.n,
                    "power":    op,
                    "ci_lo":    ci.lo,
                    "ci_hi":    ci.hi,
                }));
            }
        }
    }

    add_n_unit_rows(&mut rows, opts);

    let mut line_enc = json!({
        "x": x_encoding(opts, "Sample size (N)"),
        "y": {
            "field": "power", "type": "quantitative",
            "title": "Power", "scale": { "domain": [0, 1] },
        },
        "detail": { "field": "series", "type": "nominal" },
    });
    if multi_target {
        line_enc["strokeDash"] = json!({ "field": "target", "type": "nominal" });
    }
    // Colour by effect (`target`), no pinned `scale.domain`: hosts relabel the `target`
    // data values in place, so a domain pinned to the engine's tokens would no longer
    // match the data and every line would lose its colour. Vega derives the domain from
    // the relabelled data; because `strokeDash` also keys on `target` with no explicit
    // domain, the two channels share one scale and one merged legend. Stable per-effect
    // hue across this curve and the power bars comes from both emitting effects in the
    // same power-vector order, not from a pinned domain.
    line_enc["color"] = json!({ "field": "target", "type": "nominal" });
    let mut layers: Vec<Value> = vec![json!({
        "mark": { "type": "line", "point": true },
        "encoding": line_enc,
    })];

    if opts.show_ci {
        // Match the line layer's axis titles on the shared n/power scales so the
        // co-scaled band field doesn't get concatenated into "Power, ci_lo".
        let band_enc = json!({
            "x":      x_encoding(opts, "Sample size (N)"),
            "y":      { "field": "ci_lo", "type": "quantitative", "title": "Power" },
            "y2":     { "field": "ci_hi" },
            "detail": { "field": "series", "type": "nominal" },
        });
        layers.push(json!({
            "mark": { "type": "errorband", "opacity": 0.2 },
            "encoding": band_enc,
        }));
    }

    if let Some(level) = opts.target_power_line {
        layers.push(json!({
            "mark": { "type": "rule", "strokeDash": [4, 4] },
            "encoding": { "y": { "datum": level, "type": "quantitative" } },
        }));
    }

    let mut spec = if multi_scenario {
        // Facet by scenario; the layered curve lives inside each panel and the
        // power y-axis is shared. `data` moves to the facet level.
        json!({
            "$schema": SCHEMA,
            "data": { "values": rows },
            "facet": {
                "field": "scenario", "type": "nominal", "columns": FACET_COLUMNS,
                // Honour the host's scenario order (optimistic → realistic → doomer
                // for the presets); without an explicit sort Vega-Lite orders facet
                // panels alphabetically (doomer, optimistic, realistic).
                "sort": scenarios.iter().map(|sc| sc.label.as_str()).collect::<Vec<_>>(),
            },
            "spec": { "width": FACET_PANEL_WIDTH, "height": CURVE_HEIGHT, "layer": layers },
        })
    } else {
        json!({
            "$schema": SCHEMA,
            "width": PANEL_WIDTH,
            "height": CURVE_HEIGHT,
            "data": { "values": rows },
            "layer": layers,
        })
    };
    if let Some(title) = &opts.title {
        spec["title"] = Value::String(title.clone());
    }
    serde_json::to_string(&spec).expect("Value serialises infallibly")
}

/// Theme-naked Vega-Lite v5 spec: P(detect >= k targets) vs N, one nested line
/// per k, plus a dashed target-power reference rule. Design A ("at least k").
/// Structural twin of [`exactly_k_curve_spec`] — changes to the facet /
/// datum-rule / series-key pattern must land in both.
///
/// Reads `success_count_histogram_corrected` from each grid point — DELIBERATELY
/// corrected (the find_power joint display uses uncorrected) so the curve matches
/// the joint required-N table, which is derived from `first_joint_achieved`
/// (also corrected). Do not "fix" this to uncorrected.
pub fn joint_detection_curve_spec(scenarios: &[PlotScenario], opts: &PlotOptions) -> String {
    let multi_scenario = scenarios.len() > 1;
    // The histogram is the authority on the joint family size: bucket k counts
    // sims with exactly k significant tests over marginals + contrasts +
    // post-hoc (len == that total + 1). `target_indices.len()` misses the
    // contrast/post-hoc entries.
    let n_targets = scenarios
        .first()
        .and_then(|sc| sc.points.iter().find(|p| !p.histogram.is_empty()))
        .map(|p| p.histogram.len() - 1)
        .unwrap_or(0);
    let mut rows: Vec<Value> = Vec::new();

    for sc in scenarios {
        let label = &sc.label;
        for point in &sc.points {
            let hist = &point.histogram;
            let n_sims: u64 = hist.iter().sum();
            if n_sims == 0 {
                continue;
            }
            for k in 1..=n_targets {
                let ge_k: u64 = hist.iter().skip(k).sum();
                let p = ge_k as f64 / n_sims as f64;
                // Under facet the scenario IS the panel, so colour keys on the
                // k-series only — drop the scenario prefix that the old single
                // panel used to disambiguate.
                let series = format!(">= {k} of {n_targets}");
                rows.push(json!({
                    "scenario": label,
                    "k":        k,
                    "series":   series,
                    "n":        point.n,
                    "p":        p,
                }));
            }
        }
    }

    add_n_unit_rows(&mut rows, opts);

    let line_encoding = json!({
        "x": x_encoding(opts, "N"),
        "y": { "field": "p", "type": "quantitative", "title": "P(detect >= k)", "scale": { "domain": [0, 1] } },
        "color": { "field": "series", "type": "nominal", "title": "Joint detection" },
    });

    let mut layers: Vec<Value> = vec![json!({
        "mark": { "type": "line", "point": true },
        "encoding": line_encoding,
    })];

    if let Some(tp) = opts.target_power_line {
        let tp = as_proportion(tp);
        // `datum`-based rule so it repeats across every facet panel instead of
        // binding to one cell (a single-row inline `data` would land in only
        // one scenario panel under faceting).
        layers.push(json!({
            "mark": { "type": "rule", "strokeDash": [4, 4] },
            "encoding": { "y": { "datum": tp, "type": "quantitative" } },
        }));
    }

    let mut spec = if multi_scenario {
        json!({
            "$schema": SCHEMA,
            "data": { "values": rows },
            "facet": {
                "field": "scenario", "type": "nominal", "columns": FACET_COLUMNS,
                // Honour the host's scenario order (optimistic → realistic → doomer
                // for the presets); without an explicit sort Vega-Lite orders facet
                // panels alphabetically (doomer, optimistic, realistic).
                "sort": scenarios.iter().map(|sc| sc.label.as_str()).collect::<Vec<_>>(),
            },
            "spec": { "width": FACET_PANEL_WIDTH, "height": CURVE_HEIGHT, "layer": layers },
        })
    } else {
        json!({
            "$schema": SCHEMA,
            "width": PANEL_WIDTH,
            "height": CURVE_HEIGHT,
            "data": { "values": rows },
            "layer": layers,
        })
    };
    if let Some(title) = &opts.title {
        spec["title"] = json!(title);
    }
    spec.to_string()
}

/// Theme-naked Vega-Lite v5 spec: P(exactly k targets significant) vs N, one
/// nested line per k (including k = 0 — fully null result), plus a dashed
/// target-power reference rule. Structural twin of [`joint_detection_curve_spec`]
/// ("at least k") — changes to the facet / datum-rule / series-key pattern
/// must land in both. The differences are:
///
/// - k iterates `0..=n_targets` (k = 0 included).
/// - P(exactly k) = `hist[k] as f64 / n_sims as f64` where n_sims = hist.sum().
/// - Series label: `"= {k} of {n_targets}"` (at-least-k uses `">= {k} of {n_targets}"`).
/// - Y-axis title: `"P(exactly k)"`.
///
/// Points where the histogram sum is 0 are skipped (no data at that N). The
/// histogram input is `PlotPoint.histogram`; callers populate it with the
/// corrected histogram (buckets sum to n_sims; non-converged sims land in
/// bucket 0 — that is handled by the math above and needs no special casing).
///
/// Single-scenario: fixed 360×240 panel, no facet.
/// Multi-scenario: faceted by scenario with 3 columns; series labels carry no
/// scenario prefix so each panel colours by k only; the target rule is
/// datum-based so it repeats across every facet panel.
pub fn exactly_k_curve_spec(scenarios: &[PlotScenario], opts: &PlotOptions) -> String {
    let multi_scenario = scenarios.len() > 1;
    // Histogram-derived joint family size — mirrors joint_detection_curve_spec
    // (structural twin), see the comment there.
    let n_targets = scenarios
        .first()
        .and_then(|sc| sc.points.iter().find(|p| !p.histogram.is_empty()))
        .map(|p| p.histogram.len() - 1)
        .unwrap_or(0);
    let mut rows: Vec<Value> = Vec::new();

    for sc in scenarios {
        let label = &sc.label;
        for point in &sc.points {
            let hist = &point.histogram;
            let n_sims: u64 = hist.iter().sum();
            if n_sims == 0 {
                continue;
            }
            for k in 0..=n_targets {
                let p = hist.get(k).copied().unwrap_or(0) as f64 / n_sims as f64;
                // Under facet the scenario IS the panel, so colour keys on the
                // k-series only — drop the scenario prefix that a single panel
                // would need to disambiguate multiple scenarios.
                let series = format!("= {k} of {n_targets}");
                rows.push(json!({
                    "scenario": label,
                    "k":        k,
                    "series":   series,
                    "n":        point.n,
                    "p":        p,
                }));
            }
        }
    }

    add_n_unit_rows(&mut rows, opts);

    let line_encoding = json!({
        "x": x_encoding(opts, "N"),
        "y": { "field": "p", "type": "quantitative", "title": "P(exactly k)", "scale": { "domain": [0, 1] } },
        "color": { "field": "series", "type": "nominal", "title": "Joint detection" },
    });

    let mut layers: Vec<Value> = vec![json!({
        "mark": { "type": "line", "point": true },
        "encoding": line_encoding,
    })];

    if let Some(tp) = opts.target_power_line {
        let tp = as_proportion(tp);
        // `datum`-based rule so it repeats across every facet panel instead of
        // binding to one cell (a single-row inline `data` would land in only
        // one scenario panel under faceting).
        layers.push(json!({
            "mark": { "type": "rule", "strokeDash": [4, 4] },
            "encoding": { "y": { "datum": tp, "type": "quantitative" } },
        }));
    }

    let mut spec = if multi_scenario {
        json!({
            "$schema": SCHEMA,
            "data": { "values": rows },
            "facet": {
                "field": "scenario", "type": "nominal", "columns": FACET_COLUMNS,
                // Honour the host's scenario order (optimistic → realistic → doomer
                // for the presets); without an explicit sort Vega-Lite orders facet
                // panels alphabetically (doomer, optimistic, realistic).
                "sort": scenarios.iter().map(|sc| sc.label.as_str()).collect::<Vec<_>>(),
            },
            "spec": { "width": FACET_PANEL_WIDTH, "height": CURVE_HEIGHT, "layer": layers },
        })
    } else {
        json!({
            "$schema": SCHEMA,
            "width": PANEL_WIDTH,
            "height": CURVE_HEIGHT,
            "data": { "values": rows },
            "layer": layers,
        })
    };
    if let Some(title) = &opts.title {
        spec["title"] = json!(title);
    }
    spec.to_string()
}

/// Ordered `(block_key, theme-naked spec JSON)` pairs for a **find_power** result.
///
/// Always emits exactly one block:
/// - `"power"` → [`power_at_n_spec`] over all scenarios.
pub fn power_plot_set(scenarios: &[PlotScenario], opts: &PlotOptions) -> Vec<(String, String)> {
    vec![("power".to_string(), power_at_n_spec(scenarios, opts))]
}

/// Ordered `(block_key, theme-naked spec JSON)` pairs for a **find_sample_size** result.
///
/// Block contents depend on the number of scenarios (S) and targets (m = target
/// count from the first scenario's first point):
///
/// | S | m | Blocks (in this order) |
/// |---|---|------------------------|
/// | 1 | 1 | `"curve"` → [`sample_size_curve_spec`] |
/// | 1 | ≥ 2 | `"curve"`, then `"at_least_k"` → [`joint_detection_curve_spec`], `"exactly_k"` → [`exactly_k_curve_spec`] |
/// | ≥ 2 | 1 | `"scenario:<label>"` per scenario (one-element slice, full-size panel), then `"overlay"` → [`sample_size_curve_spec`] (faceted grid) |
/// | ≥ 2 | ≥ 2 | `"scenario:<label>"` per scenario, `"overlay"`, then `"at_least_k"`, `"exactly_k"` over all scenarios |
///
/// m = 1 never emits `at_least_k`/`exactly_k` (at-least-1-of-1 duplicates the
/// power curve). Block keys use the scenario label verbatim; hosts sanitize for
/// filenames.
pub fn sample_size_plot_set(
    scenarios: &[PlotScenario],
    opts: &PlotOptions,
) -> Vec<(String, String)> {
    // Joint family size from the histogram (covers contrasts + post-hoc),
    // matching what the at_least_k / exactly_k emitters will draw.
    let m = scenarios
        .first()
        .and_then(|sc| sc.points.iter().find(|p| !p.histogram.is_empty()))
        .map(|p| p.histogram.len() - 1)
        .unwrap_or(0);
    let multi_scenario = scenarios.len() > 1;

    let mut blocks: Vec<(String, String)> = Vec::new();

    if multi_scenario {
        // Per-scenario single panels (one-element slice each) come first.
        for sc in scenarios {
            let key = format!("scenario:{}", sc.label);
            let spec = sample_size_curve_spec(std::slice::from_ref(sc), opts);
            blocks.push((key, spec));
        }
        // Faceted overlay of all scenarios.
        blocks.push((
            "overlay".to_string(),
            sample_size_curve_spec(scenarios, opts),
        ));
    } else {
        blocks.push(("curve".to_string(), sample_size_curve_spec(scenarios, opts)));
    }

    // Joint/exactly-k blocks only when there are multiple targets.
    if m >= 2 {
        let joint_opts = PlotOptions {
            title: None,
            show_ci: false,
            target_power_line: opts.target_power_line,
            x_divisor: opts.x_divisor,
        };
        blocks.push((
            "at_least_k".to_string(),
            joint_detection_curve_spec(scenarios, &joint_opts),
        ));
        blocks.push((
            "exactly_k".to_string(),
            exactly_k_curve_spec(scenarios, &joint_opts),
        ));
    }

    blocks
}

#[cfg(test)]
mod exactly_k_curve_tests {
    use super::*;

    fn point(n: usize, hist: Vec<u64>) -> PlotPoint {
        PlotPoint {
            n,
            target_indices: vec![1, 2],
            contrast_pairs: vec![],
            power: vec![],
            ci: vec![],
            histogram: hist,
            overall_power: None,
            overall_ci: None,
        }
    }

    #[test]
    fn math_and_k0_series_present() {
        // m=2, one grid point, hist=[10,30,60] (n_sims=100).
        // Expected rows: k=0 p=0.1, k=1 p=0.3, k=2 p=0.6.
        let scenarios = vec![PlotScenario {
            label: "s".to_string(),
            points: vec![point(50, vec![10, 30, 60])],
        }];
        let opts = PlotOptions::default();
        let spec = exactly_k_curve_spec(&scenarios, &opts);
        let v: serde_json::Value = serde_json::from_str(&spec).expect("valid JSON");
        let rows = v["data"]["values"].as_array().expect("data.values array");
        // 1 point × 3 k-values (0, 1, 2) = 3 rows.
        assert_eq!(rows.len(), 3);
        // k=0 row: series "= 0 of 2", p = 0.1.
        let k0 = rows.iter().find(|r| r["k"] == 0).expect("k=0 row present");
        assert_eq!(k0["series"], "= 0 of 2");
        let p0 = k0["p"].as_f64().unwrap();
        assert!((p0 - 0.1).abs() < 1e-12, "p for k=0 was {p0}");
        // k=1: p = 0.3, k=2: p = 0.6.
        let k1 = rows.iter().find(|r| r["k"] == 1).expect("k=1 row");
        assert!((k1["p"].as_f64().unwrap() - 0.3).abs() < 1e-12);
        let k2 = rows.iter().find(|r| r["k"] == 2).expect("k=2 row");
        assert!((k2["p"].as_f64().unwrap() - 0.6).abs() < 1e-12);
        // Series count == n_targets + 1 = 3.
        let series_vals: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| r["series"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(series_vals.len(), 3);
    }

    #[test]
    fn single_scenario_is_unfaceted() {
        let scenarios = vec![PlotScenario {
            label: "opt".to_string(),
            points: vec![
                point(50, vec![80, 15, 5]),
                point(100, vec![40, 40, 20]),
                point(200, vec![5, 35, 60]),
            ],
        }];
        let opts = PlotOptions {
            target_power_line: Some(0.8),
            ..Default::default()
        };
        let spec = exactly_k_curve_spec(&scenarios, &opts);
        let v: serde_json::Value = serde_json::from_str(&spec).expect("valid JSON");
        assert!(v["$schema"].as_str().unwrap().contains("v5"));
        assert_eq!(v["width"], json!(PANEL_WIDTH));
        assert_eq!(v["height"], json!(CURVE_HEIGHT));
        assert!(v.get("facet").is_none());
        // 3 points × 3 k-values = 9 rows.
        assert_eq!(v["data"]["values"].as_array().unwrap().len(), 9);
        // Rule layer present.
        let layers = v["layer"].as_array().unwrap();
        assert!(layers.iter().any(|l| l["mark"]["type"] == "rule"));
    }

    #[test]
    fn multi_scenario_facets_by_scenario_and_rekeys_series() {
        let mk = |label: &str| PlotScenario {
            label: label.to_string(),
            points: vec![
                point(50, vec![80, 15, 5]),
                point(100, vec![40, 40, 20]),
                point(200, vec![5, 35, 60]),
            ],
        };
        let scenarios = vec![mk("optimistic"), mk("realistic"), mk("doomer")];
        let opts = PlotOptions {
            target_power_line: Some(0.8),
            ..Default::default()
        };
        let spec = exactly_k_curve_spec(&scenarios, &opts);
        let v: serde_json::Value = serde_json::from_str(&spec).expect("valid JSON");
        // Faceted by scenario, 3 columns; layers + panel size live under `spec`.
        assert_eq!(v["facet"]["field"], json!("scenario"));
        assert_eq!(v["facet"]["columns"], json!(FACET_COLUMNS));
        assert_eq!(v["spec"]["width"], json!(FACET_PANEL_WIDTH));
        assert_eq!(v["spec"]["height"], json!(CURVE_HEIGHT));
        assert!(
            v.get("layer").is_none(),
            "layers move under spec when faceted"
        );
        // `series` carries no scenario prefix — keys on k only.
        let rows = v["data"]["values"].as_array().expect("data.values array");
        assert!(rows.iter().all(|r| {
            let s = r["series"].as_str().unwrap();
            !s.contains("optimistic") && !s.contains('·')
        }));
        // Target rule is datum-based (no inline data bound to it).
        let layers = v["spec"]["layer"].as_array().expect("spec.layer array");
        let rule = layers
            .iter()
            .find(|l| l["mark"]["type"] == "rule")
            .expect("rule layer");
        assert!(
            rule.get("data").is_none(),
            "rule must not carry inline data"
        );
        assert_eq!(rule["encoding"]["y"]["datum"], json!(0.8));
    }

    #[test]
    fn skips_zero_sum_histogram_points() {
        let scenarios = vec![PlotScenario {
            label: "s".to_string(),
            points: vec![
                point(50, vec![0, 0, 0]), // sum == 0 → skip
                point(100, vec![10, 30, 60]),
            ],
        }];
        let spec = exactly_k_curve_spec(&scenarios, &PlotOptions::default());
        let v: serde_json::Value = serde_json::from_str(&spec).unwrap();
        // Only the n=100 point contributes 3 rows; n=50 is skipped.
        assert_eq!(v["data"]["values"].as_array().unwrap().len(), 3);
        assert!(v["data"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["n"] == 100));
    }
}

#[cfg(test)]
mod sample_size_curve_tests {
    use super::*;

    fn curve_pt(n: usize, power: Vec<f64>) -> PlotPoint {
        let k = power.len();
        PlotPoint {
            n,
            target_indices: (1..=k).collect(),
            contrast_pairs: vec![],
            ci: power
                .iter()
                .map(|&p| Ci {
                    lo: p - 0.05,
                    hi: p + 0.05,
                })
                .collect(),
            power,
            histogram: vec![],
            overall_power: None,
            overall_ci: None,
        }
    }

    /// `x_divisor > 1`: x field is `n_unit = n / divisor` with no title (the
    /// host labels the unit); the CI band layer switches with it; `n` stays
    /// on the row. `x_divisor == 1` is byte-identical to the default.
    #[test]
    fn curve_x_divisor_switches_x_field_and_drops_title() {
        let scenarios = vec![PlotScenario {
            label: "s".into(),
            points: vec![curve_pt(80, vec![0.3]), curve_pt(160, vec![0.6])],
        }];
        let unit = PlotOptions {
            show_ci: true,
            x_divisor: 8,
            ..Default::default()
        };
        let v: serde_json::Value =
            serde_json::from_str(&sample_size_curve_spec(&scenarios, &unit)).unwrap();
        for layer in [&v["layer"][0], &v["layer"][1]] {
            assert_eq!(layer["encoding"]["x"]["field"], json!("n_unit"));
            assert!(layer["encoding"]["x"]["title"].is_null());
        }
        let rows = v["data"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["n"], json!(80));
        assert_eq!(rows[0]["n_unit"], json!(10));
        assert_eq!(rows[1]["n_unit"], json!(20));

        let one = PlotOptions {
            show_ci: true,
            x_divisor: 1,
            ..Default::default()
        };
        let plain = PlotOptions {
            show_ci: true,
            ..Default::default()
        };
        assert_eq!(
            sample_size_curve_spec(&scenarios, &one),
            sample_size_curve_spec(&scenarios, &plain)
        );
        let v: serde_json::Value =
            serde_json::from_str(&sample_size_curve_spec(&scenarios, &one)).unwrap();
        assert_eq!(v["layer"][0]["encoding"]["x"]["field"], json!("n"));
        assert_eq!(
            v["layer"][0]["encoding"]["x"]["title"],
            json!("Sample size (N)")
        );
        assert!(v["data"]["values"][0]["n_unit"].is_null());
    }

    /// The joint / exactly-k curves follow the same x_divisor switch, and
    /// `sample_size_plot_set` threads it into their options.
    #[test]
    fn joint_curves_follow_x_divisor_through_plot_set() {
        let pt = |n: usize| PlotPoint {
            n,
            target_indices: vec![1, 2],
            contrast_pairs: vec![],
            power: vec![0.5, 0.6],
            ci: vec![Ci { lo: 0.4, hi: 0.6 }, Ci { lo: 0.5, hi: 0.7 }],
            histogram: vec![10, 50, 40],
            overall_power: None,
            overall_ci: None,
        };
        let scenarios = vec![PlotScenario {
            label: "s".into(),
            points: vec![pt(80), pt(160)],
        }];
        let opts = PlotOptions {
            x_divisor: 8,
            ..Default::default()
        };
        let blocks = sample_size_plot_set(&scenarios, &opts);
        for key in ["at_least_k", "exactly_k"] {
            let spec = &blocks.iter().find(|(k, _)| k == key).unwrap().1;
            let v: serde_json::Value = serde_json::from_str(spec).unwrap();
            assert_eq!(
                v["layer"][0]["encoding"]["x"]["field"],
                json!("n_unit"),
                "{key}"
            );
            assert!(v["layer"][0]["encoding"]["x"]["title"].is_null(), "{key}");
            assert_eq!(v["data"]["values"][0]["n_unit"], json!(10), "{key}");
        }
    }
}

#[cfg(test)]
mod power_at_n_color_tests {
    use super::*;

    fn bars_scenario(label: &str, power: Vec<f64>) -> PlotScenario {
        let n = power.len();
        PlotScenario {
            label: label.into(),
            points: vec![PlotPoint {
                n: 100,
                target_indices: (1..=n).collect(),
                contrast_pairs: vec![],
                ci: power
                    .iter()
                    .map(|&p| Ci {
                        lo: p - 0.05,
                        hi: p + 0.05,
                    })
                    .collect(),
                power,
                histogram: vec![],
                overall_power: None,
                overall_ci: None,
            }],
        }
    }

    #[test]
    fn bars_shade_scenarios_bold_to_faint_in_host_order() {
        // Host order deliberately differs from mean-power order (doomer has the
        // lowest power but comes first), so a mean-power sort would reorder it.
        let scenarios = vec![
            bars_scenario("doomer", vec![0.4, 0.5]),
            bars_scenario("optimistic", vec![0.9, 0.95]),
            bars_scenario("realistic", vec![0.7, 0.8]),
        ];
        let v: serde_json::Value =
            serde_json::from_str(&power_at_n_spec(&scenarios, &PlotOptions::default())).unwrap();
        let bar = &v["layer"][0];
        let fo = &bar["encoding"]["fillOpacity"];
        assert_eq!(fo["field"], json!("scenario"));
        assert_eq!(fo["sort"], json!(["doomer", "optimistic", "realistic"]));
        assert_eq!(fo["scale"]["range"], json!([1.0, 0.7, 0.4]));
        assert_eq!(
            bar["encoding"]["yOffset"]["sort"],
            json!(["doomer", "optimistic", "realistic"])
        );
        // Effect colour still present.
        assert_eq!(bar["encoding"]["color"]["field"], json!("target"));
    }

    #[test]
    fn many_scenarios_facet_without_opacity() {
        let scenarios: Vec<_> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|l| bars_scenario(l, vec![0.5, 0.6]))
            .collect();
        let v: serde_json::Value =
            serde_json::from_str(&power_at_n_spec(&scenarios, &PlotOptions::default())).unwrap();
        assert_eq!(v["facet"]["field"], json!("scenario"));
        let bar = &v["spec"]["layer"][0];
        assert!(bar["encoding"].get("fillOpacity").is_none());
        assert_eq!(bar["encoding"]["color"]["field"], json!("target"));
    }

    #[test]
    fn bars_and_curve_color_by_target_without_pinned_domain() {
        // Both plots colour by the same `target` field with no explicit domain, so the
        // host's data relabelling drives both legends and the same effect lands on the
        // same palette slot (both emit effects in power-vector order). A pinned token
        // domain would survive in neither once the host rewrites the data values.
        let bars = vec![bars_scenario("s", vec![0.5, 0.9])];
        let vb: serde_json::Value =
            serde_json::from_str(&power_at_n_spec(&bars, &PlotOptions::default())).unwrap();
        let bar_color = &vb["layer"][0]["encoding"]["color"];
        assert_eq!(bar_color["field"], json!("target"));
        assert!(bar_color["scale"].is_null());

        let curve = vec![PlotScenario {
            label: "s".into(),
            points: vec![PlotPoint {
                n: 100,
                target_indices: vec![1, 2],
                contrast_pairs: vec![],
                ci: vec![Ci { lo: 0.4, hi: 0.6 }, Ci { lo: 0.8, hi: 0.95 }],
                power: vec![0.5, 0.9],
                histogram: vec![],
                overall_power: None,
                overall_ci: None,
            }],
        }];
        let vc: serde_json::Value =
            serde_json::from_str(&sample_size_curve_spec(&curve, &PlotOptions::default())).unwrap();
        let curve_color = &vc["layer"][0]["encoding"]["color"];
        assert_eq!(curve_color["field"], json!("target"));
        assert!(curve_color["scale"].is_null());
    }
}
