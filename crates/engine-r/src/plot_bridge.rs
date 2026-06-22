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
    plot::{power_plot_set, sample_size_plot_set, PlotOptions, PlotPoint, PlotScenario},
    Ci,
};
use extendr_api::error::Result;
use extendr_api::prelude::*;
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
        .ok_or_else(|| Error::Other(format!("unknown theme {name:?}")))
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
    /// vs uncorrected before serialising under the neutral key `power`.
    power: Vec<Vec<f64>>,
    /// Displayed CI: list of per-N lists of [lo, hi] pairs. Same selection logic.
    ci: Vec<Vec<Vec<f64>>>,
    /// find_sample_size only: per-N corrected histogram (bucket k = #targets jointly
    /// significant). Always the corrected histogram so the at_least_k/exactly_k
    /// curves match the corrected joint required-N table. One inner Vec<u64> per
    /// sample-size grid point; older envelopes omit this field.
    #[serde(default)]
    histogram: Vec<Vec<u64>>,
}

fn parse_contrast_pairs(rows: &[Vec<u32>]) -> Vec<(u32, u32)> {
    rows.iter()
        .filter_map(|r| match r[..] {
            [p, n] => Some((p, n)),
            _ => None,
        })
        .collect()
}

fn parse_ci_row(row: &[Vec<f64>]) -> Vec<Ci> {
    row.iter()
        .map(|pair| {
            let lo = pair.first().copied().unwrap_or(0.0);
            let hi = pair.get(1).copied().unwrap_or(0.0);
            Ci { lo, hi }
        })
        .collect()
}

// ── Shared scenario reconstruction helper ─────────────────────────────────────

fn plot_scenario_from_json(sc: &ScenarioJson) -> Result<PlotScenario> {
    let label = &sc.label;
    let n_ns = sc.sample_sizes.len();
    if sc.power.len() != n_ns || sc.ci.len() != n_ns {
        return Err(Error::Other(format!(
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
            PlotPoint {
                n,
                target_indices: sc.target_indices.clone(),
                contrast_pairs: parse_contrast_pairs(&sc.contrast_pairs),
                power: sc.power[i].clone(),
                ci,
                histogram,
                // Per-N overall rates are not surfaced in the R envelope, so the
                // overall curve series is not emitted on this host.
                overall_power: None,
                overall_ci: None,
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
        .map_err(|e| Error::Other(format!("power_plot_set_json: invalid envelope JSON: {e}")))?;
    if envelope.scenarios.is_empty() {
        return Err(Error::Other("envelope has zero scenarios".into()));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(envelope.scenarios.len());
    for sc in &envelope.scenarios {
        let label = &sc.label;
        let n = *sc
            .sample_sizes
            .first()
            .ok_or_else(|| Error::Other(format!("scenario `{label}`: sample_sizes is empty")))?;
        let power = sc
            .power
            .first()
            .ok_or_else(|| Error::Other(format!("scenario `{label}`: power is empty")))?
            .clone();
        let ci_row = sc
            .ci
            .first()
            .ok_or_else(|| Error::Other(format!("scenario `{label}`: ci is empty")))?;
        let ci = parse_ci_row(ci_row);
        out.push(PlotScenario {
            label: label.clone(),
            points: vec![PlotPoint {
                n,
                target_indices: sc.target_indices.clone(),
                contrast_pairs: parse_contrast_pairs(&sc.contrast_pairs),
                power,
                ci,
                histogram: vec![],
                // The overall series rides the sample-size curve only.
                overall_power: None,
                overall_ci: None,
            }],
        });
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
    };
    let pairs = power_plot_set(&out, &opts);
    let names: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
    let specs: Vec<Robj> = pairs.into_iter().map(|(_, s)| s.into_robj()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    List::from_names_and_values(name_refs, specs)
        .map_err(|e| Error::Other(format!("power_plot_set_json: list construction: {e}")))
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
        Error::Other(format!(
            "sample_size_plot_set_json: invalid envelope JSON: {e}"
        ))
    })?;
    if envelope.scenarios.is_empty() {
        return Err(Error::Other("envelope has zero scenarios".into()));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(envelope.scenarios.len());
    for sc in &envelope.scenarios {
        out.push(plot_scenario_from_json(sc)?);
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
    };
    let pairs = sample_size_plot_set(&out, &opts);
    let names: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
    let specs: Vec<Robj> = pairs.into_iter().map(|(_, s)| s.into_robj()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    List::from_names_and_values(name_refs, specs)
        .map_err(|e| Error::Other(format!("sample_size_plot_set_json: list construction: {e}")))
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
