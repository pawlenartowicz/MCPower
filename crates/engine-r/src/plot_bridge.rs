//! Plot-set bridge for the R port.
//!
//! Mirrors `crates/engine-py/src/plot_bridge.rs` but speaks JSON strings
//! instead of PyDict: extendr has no native dict type, so the R side serialises
//! the envelope with `jsonlite::toJSON` and passes it as a `&str`.
//!
//! Envelope key contract (neutral names — correction selection is done in R):
//!   - `power`      — the displayed power array (list of per-N vectors of f64)
//!   - `ci`         — the displayed CI array (list of per-N lists of [lo, hi])
//!   - `histogram`  — `success_count_histogram_corrected` (always corrected; see comment)
//!   - `sample_sizes`, `target_indices` — unchanged from before
//!
//! Envelope shape: `scenarios` is a JSON **array** of objects, each carrying its
//! label as a `"label"` field. Array order is preserved — scenarios are rendered
//! in the order they appear in the array, matching user-supplied scenario order.
//!
//! Return type for plot-set functions: named `List` (names = block keys, values
//! = spec strings), which is the natural extendr representation of an ordered
//! named structure and matches how other bridge functions in this crate return
//! multi-element results.

use engine_orchestrator::{
    Ci,
    plot::{PlotOptions, PlotPoint, PlotScenario, power_plot_set, sample_size_plot_set},
};
use extendr_api::error::Result;
use extendr_api::prelude::*;

use crate::r_error;
use serde::Deserialize;

// ── Embedded plot themes ───────────────────────────────────────────────────────
// Themes are embedded at build time from the single configs/plot-themes.json
// (a map of name -> partial Vega-Lite `config` fragment) so all ports share the
// same values.
use serde_json::Value;

const PLOT_THEMES_JSON: &str = include_str!("../../../configs/plot-themes.json");

// Embedded CDN HTML template — single `{{SPECS}}` placeholder.
const PLOT_HTML_TEMPLATE: &str = include_str!("../../../configs/plot-html-template.html");

fn themes() -> serde_json::Map<String, Value> {
    match serde_json::from_str::<Value>(PLOT_THEMES_JSON) {
        Ok(Value::Object(m)) => m,
        _ => panic!("embedded plot-themes.json must be a JSON object"),
    }
}

/// Return the embedded theme JSON for `name`.
#[extendr]
pub fn plot_theme(name: &str) -> Result<String> {
    let m = themes();
    m.get(name)
        .map(|v| v.to_string())
        .ok_or_else(|| r_error(format!("unknown theme {name:?}")))
}

/// The embedded theme names.
#[extendr]
pub fn list_plot_themes() -> Vec<String> {
    themes().keys().cloned().collect()
}

/// The embedded CDN HTML template. Contains a `{{SPECS}}` placeholder that
/// the R host replaces with a JSON array of Vega-Lite spec objects.
#[extendr]
pub fn plot_html_template() -> String {
    PLOT_HTML_TEMPLATE.to_string()
}

// ── JSON envelope types ────────────────────────────────────────────────────────
//
// The R caller serialises the result envelope with `jsonlite::toJSON`.
// We only need to deserialise the fields the plot emitters actually read.
// Neutral envelope keys: `power`, `ci`, `histogram` (see module doc).

#[derive(Deserialize)]
struct EnvelopeJson {
    scenarios: Vec<ScenarioJson>,
}

#[derive(Deserialize)]
struct ScenarioJson {
    label: String,
    sample_sizes: Vec<usize>,
    target_indices: Vec<usize>,
    /// `[positive, negative]` β-column index pairs for the contrast entries
    /// appended past the marginals in `power`/`ci` (the envelope projection of
    /// `PowerResult.contrast_pairs`). Older envelopes omit this field.
    #[serde(default)]
    contrast_pairs: Vec<Vec<u32>>,
    /// Displayed power: list of per-N vectors. The R caller selects corrected
    /// vs uncorrected before serialising under the neutral key `power`. A
    /// `null` entry (an `NA` corrected power the correction does not define)
    /// reads as NaN, which the emitters leave off the chart.
    power: Vec<Vec<Option<f64>>>,
    /// Displayed CI: list of per-N lists of [lo, hi] pairs. Same selection
    /// logic; `null` bounds read as NaN.
    ci: Vec<Vec<Vec<Option<f64>>>>,
    /// find_sample_size only: per-N corrected histogram (bucket k = #targets jointly
    /// significant). Always the corrected histogram so the at_least_k/exactly_k
    /// curves match the corrected joint required-N table. One inner Vec<u64> per
    /// sample-size grid point; older envelopes omit this field.
    #[serde(default)]
    histogram: Vec<Vec<u64>>,
    /// Overall/omnibus test rate, one entry per N in `sample_sizes`. Absent, or
    /// `null` entries, when the family suppressed the overall test → no
    /// `overall` bar (find_power) or curve series (find_sample_size).
    #[serde(default)]
    overall_power: Vec<Option<f64>>,
    /// `[lo, hi]` CI for `overall_power`, one entry per N. An absent CI falls
    /// back to a zero-width interval at the rate.
    #[serde(default)]
    overall_ci: Vec<Option<Vec<f64>>>,
    /// find_sample_size only: divisor between total N and the requested unit
    /// (1 = observations, the same value on every scenario — the orchestrator
    /// rejects a mismatch before this envelope is built). Older envelopes omit
    /// this field.
    #[serde(default)]
    unit_divisor: Option<usize>,
    /// The engine's `corrected_not_applicable` flags for the displayed power
    /// entries (the same at every N; empty for an uncorrected display). A
    /// flagged entry is not drawn. Older envelopes omit this field.
    #[serde(default)]
    not_applicable: Vec<bool>,
    /// True when any target is flagged: the corrected joint view is not
    /// applicable, so no at-least-k / exactly-k block. Older envelopes omit it.
    #[serde(default)]
    joint_not_applicable: bool,
}

fn parse_contrast_pairs(rows: &[Vec<u32>]) -> Vec<(u32, u32)> {
    rows.iter()
        .filter_map(|r| match r[..] {
            [p, n] => Some((p, n)),
            _ => None,
        })
        .collect()
}

/// Overall rate + CI at grid point `i` (see `ScenarioJson::overall_power`).
fn overall_at(sc: &ScenarioJson, i: usize) -> (Option<f64>, Option<Ci>) {
    let rate = sc.overall_power.get(i).copied().flatten();
    let ci = sc
        .overall_ci
        .get(i)
        .and_then(Option::as_ref)
        .and_then(|v| (v.len() >= 2).then(|| Ci { lo: v[0], hi: v[1] }));
    (rate, ci)
}

fn parse_ci_row(row: &[Vec<Option<f64>>]) -> Vec<Ci> {
    row.iter()
        .map(|pair| {
            let bound = |i: usize| match pair.get(i) {
                Some(Some(x)) => *x,
                Some(None) => f64::NAN,
                None => 0.0,
            };
            Ci {
                lo: bound(0),
                hi: bound(1),
            }
        })
        .collect()
}

// ── Shared scenario reconstruction helper ─────────────────────────────────────

fn plot_scenario_from_json(sc: &ScenarioJson) -> Result<PlotScenario> {
    let label = &sc.label;
    let n_ns = sc.sample_sizes.len();
    if sc.power.len() != n_ns || sc.ci.len() != n_ns {
        return Err(r_error(format!(
            "scenario `{label}`: sample_sizes, power, ci must have equal length"
        )));
    }
    let has_hist = sc.histogram.len() == n_ns;

    let points: Vec<PlotPoint> = sc
        .sample_sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let ci = parse_ci_row(&sc.ci[i]);
            let histogram = if has_hist {
                sc.histogram[i].clone()
            } else {
                Vec::new()
            };
            let (overall_power, overall_ci) = overall_at(sc, i);
            PlotPoint {
                n,
                target_indices: sc.target_indices.clone(),
                contrast_pairs: parse_contrast_pairs(&sc.contrast_pairs),
                power: sc.power[i].iter().map(|p| p.unwrap_or(f64::NAN)).collect(),
                ci,
                histogram,
                // Drives the extra `"overall"` curve series; `None` at every N
                // when the family suppressed the overall test.
                overall_power,
                overall_ci,
                not_applicable: sc.not_applicable.clone(),
                joint_not_applicable: sc.joint_not_applicable,
            }
        })
        .collect();

    Ok(PlotScenario {
        label: label.to_string(),
        points,
    })
}

// ── power_plot_set_json ────────────────────────────────────────────────────────

/// Ordered named list of (key → spec) pairs for a find_power result.
///
/// `envelope_json` is produced by the R side via `jsonlite::toJSON(...)`.
/// The envelope `scenarios` array must contain objects with a `"label"` field
/// plus `power` and `ci` carrying the displayed arrays (R caller selects
/// corrected vs uncorrected before calling). Scenarios are rendered in array
/// order — no alphabetical sorting is applied.
///
/// Returns a named R list (names = block keys, values = Vega-Lite v5 JSON strings).
#[extendr]
pub fn power_plot_set_json(
    envelope_json: &str,
    show_ci: bool,
    target_power_line: Option<f64>,
) -> Result<List> {
    let envelope: EnvelopeJson = serde_json::from_str(envelope_json)
        .map_err(|e| r_error(format!("power_plot_set_json: invalid envelope JSON: {e}")))?;
    if envelope.scenarios.is_empty() {
        return Err(r_error("envelope has zero scenarios"));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(envelope.scenarios.len());
    for sc in &envelope.scenarios {
        let label = &sc.label;
        let n = *sc
            .sample_sizes
            .first()
            .ok_or_else(|| r_error(format!("scenario `{label}`: sample_sizes is empty")))?;
        let power = sc
            .power
            .first()
            .ok_or_else(|| r_error(format!("scenario `{label}`: power is empty")))?
            .iter()
            .map(|p| p.unwrap_or(f64::NAN))
            .collect();
        let ci_row = sc
            .ci
            .first()
            .ok_or_else(|| r_error(format!("scenario `{label}`: ci is empty")))?;
        let ci = parse_ci_row(ci_row);
        let (overall_power, overall_ci) = overall_at(sc, 0);
        out.push(PlotScenario {
            label: label.clone(),
            points: vec![PlotPoint {
                n,
                target_indices: sc.target_indices.clone(),
                contrast_pairs: parse_contrast_pairs(&sc.contrast_pairs),
                power,
                ci,
                histogram: vec![],
                // The overall/omnibus test draws one more bar (last) on the
                // power-at-N chart, matching the table. Absent when the family
                // suppressed it. A present rate with no CI degenerates to a
                // zero-width interval in the orchestrator.
                overall_power,
                overall_ci,
                not_applicable: sc.not_applicable.clone(),
                joint_not_applicable: sc.joint_not_applicable,
            }],
        });
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
        x_divisor: 1,
    };
    let pairs = power_plot_set(&out, &opts);
    let names: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
    let specs: Vec<Robj> = pairs.into_iter().map(|(_, s)| s.into_robj()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    List::from_names_and_values(name_refs, specs)
        .map_err(|e| r_error(format!("power_plot_set_json: list construction: {e}")))
}

// ── sample_size_plot_set_json ──────────────────────────────────────────────────

/// Ordered named list of (key → spec) pairs for a find_sample_size result.
///
/// `envelope_json` contains `scenarios` as an ordered array of objects (each
/// with a `"label"` field) with one row per N point in `power` / `ci`, and
/// `histogram` carrying the corrected joint-significance histogram. Scenarios
/// are rendered in array order — no alphabetical sorting is applied.
/// Block order mirrors `sample_size_plot_set` (curve, at_least_k, exactly_k, …).
///
/// Returns a named R list (names = block keys, values = Vega-Lite v5 JSON strings).
#[extendr]
pub fn sample_size_plot_set_json(
    envelope_json: &str,
    show_ci: bool,
    target_power_line: Option<f64>,
) -> Result<List> {
    let envelope: EnvelopeJson = serde_json::from_str(envelope_json).map_err(|e| {
        r_error(format!(
            "sample_size_plot_set_json: invalid envelope JSON: {e}"
        ))
    })?;
    if envelope.scenarios.is_empty() {
        return Err(r_error("envelope has zero scenarios"));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(envelope.scenarios.len());
    let mut x_divisor: usize = 1;
    for sc in &envelope.scenarios {
        // `unit_divisor` is the same across every scenario (the orchestrator
        // rejects a mismatch before this envelope is built) — read it once, off
        // the first scenario that carries it.
        if x_divisor == 1 {
            if let Some(d) = sc.unit_divisor {
                x_divisor = d;
            }
        }
        out.push(plot_scenario_from_json(sc)?);
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
        x_divisor,
    };
    let pairs = sample_size_plot_set(&out, &opts);
    let names: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
    let specs: Vec<Robj> = pairs.into_iter().map(|(_, s)| s.into_robj()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    List::from_names_and_values(name_refs, specs)
        .map_err(|e| r_error(format!("sample_size_plot_set_json: list construction: {e}")))
}

// ── module registration ────────────────────────────────────────────────────────

extendr_module! {
    mod plot_bridge;
    fn power_plot_set_json;
    fn sample_size_plot_set_json;
    fn plot_theme;
    fn list_plot_themes;
    fn plot_html_template;
}
