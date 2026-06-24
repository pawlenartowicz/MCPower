//! Plot-set bridge for the Python port.
//!
//! Mirror: `engine-r/src/plot_bridge.rs` (JSON-string flavour) — change
//! together.
//!
//! The Python host builds a result envelope and passes it as a PyDict. We
//! extract only the fields the plot emitters need, build `PlotScenario` slices,
//! call the orchestrator plot-set functions, and return ordered `(key, spec)` pairs.
//!
//! Envelope key contract (neutral names — correction selection is done in Python):
//!   - `power`      — the displayed power array (list of per-N lists of f64)
//!   - `ci`         — the displayed CI array (list of per-N lists of [lo, hi])
//!   - `histogram`  — `success_count_histogram_corrected` (always corrected; see comment)
//!   - `sample_sizes`, `target_indices` — unchanged from before
//!
//! Envelope shape: `scenarios` is a Python **list** of dicts, each carrying its
//! label as a `"label"` key. List order is preserved — scenarios are rendered
//! in the order they appear in the list, matching user-supplied scenario order.

use engine_orchestrator::{
    plot::{power_plot_set, sample_size_plot_set, PlotOptions, PlotPoint, PlotScenario},
    Ci,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};
use serde_json::Value;

// --- Embedded plot themes ---------------------------------------------------
// Workspace rule: cross-port config is centralised and embedded at build time;
// no port keeps its own copy. We `include_str!` the single shared theme JSON
// (a map of name -> partial Vega-Lite `config` fragment) and hand each entry to
// Python verbatim.
const PLOT_THEMES_JSON: &str = include_str!("../../../configs/plot-themes.json");

// Embedded CDN HTML template — single `{{SPECS}}` placeholder for a JSON array
// of spec objects. Shared across all ports; sourced from configs/.
const PLOT_HTML_TEMPLATE: &str = include_str!("../../../configs/plot-html-template.html");

fn themes() -> serde_json::Map<String, Value> {
    match serde_json::from_str::<Value>(PLOT_THEMES_JSON) {
        Ok(Value::Object(m)) => m,
        _ => panic!("embedded plot-themes.json must be a JSON object"),
    }
}

/// Return the embedded theme JSON for `name`, or `PyValueError` listing the
/// valid names if unknown.
#[pyfunction]
pub fn plot_theme(name: &str) -> PyResult<String> {
    let m = themes();
    m.get(name).map(|v| v.to_string()).ok_or_else(|| {
        let names: Vec<&String> = m.keys().collect();
        PyValueError::new_err(format!("unknown theme {name:?}; valid themes: {names:?}"))
    })
}

/// The embedded theme names.
#[pyfunction]
pub fn list_plot_themes() -> Vec<String> {
    themes().keys().cloned().collect()
}

/// The embedded CDN HTML template. Contains a `{{SPECS}}` placeholder that
/// the Python host replaces with a JSON array of Vega-Lite spec objects.
#[pyfunction]
pub fn plot_html_template() -> String {
    PLOT_HTML_TEMPLATE.to_string()
}

// ── Shared parsing helpers ──────────────────────────────────────────────────

fn first_row<'py, T>(any: Bound<'py, PyAny>, label: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    let outer = any
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be list-of-lists")))?;
    if outer.is_empty() {
        return Err(PyValueError::new_err(format!("{label} is empty")));
    }
    outer.get_item(0)?.extract()
}

fn first_ci_row(any: Bound<'_, PyAny>, label: &str) -> PyResult<Vec<Ci>> {
    let outer = any
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be list-of-lists")))?;
    if outer.is_empty() {
        return Err(PyValueError::new_err(format!("{label} is empty")));
    }
    let row = outer.get_item(0)?;
    let row = row
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err(format!("{label} inner row must be a list")))?;
    let mut out = Vec::with_capacity(row.len());
    for pair in row.iter() {
        let seq = pair
            .cast::<PySequence>()
            .map_err(|_| PyValueError::new_err("ci entry must be (lo, hi)"))?;
        let lo: f64 = seq.get_item(0)?.extract()?;
        let hi: f64 = seq.get_item(1)?.extract()?;
        out.push(Ci { lo, hi });
    }
    Ok(out)
}

/// Build a single `PlotPoint` from a find_power scenario dict.
///
/// Neutral envelope keys: `power` (list-of-lists, row 0 used), `ci` (same).
fn plot_point_from_power_dict(d: &Bound<'_, PyDict>) -> PyResult<PlotPoint> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("scenario dict missing `{k}`")))
    };
    let sample_sizes: Vec<usize> = get("sample_sizes")?.extract()?;
    let n = *sample_sizes
        .first()
        .ok_or_else(|| PyValueError::new_err("sample_sizes is empty"))?;
    let target_indices: Vec<usize> = get("target_indices")?.extract()?;
    let contrast_pairs = contrast_pairs_of(d)?;
    let power: Vec<f64> = first_row(get("power")?, "power")?;
    let ci = first_ci_row(get("ci")?, "ci")?;
    // Optional overall/omnibus test: scalar rate + `[lo, hi]` CI. Absent (or
    // `None`) whenever the family suppressed the overall test (mixed/GLMM) or an
    // older caller didn't supply it; then no `overall` bar is drawn.
    let (overall_power, overall_ci) = optional_overall(d)?;
    Ok(PlotPoint {
        n,
        target_indices,
        contrast_pairs,
        power,
        ci,
        histogram: vec![],
        overall_power,
        overall_ci,
    })
}

/// Optional `overall_power` (scalar) + `overall_ci` (`[lo, hi]`) keys. A missing
/// key, an explicit `None`, or a missing CI all yield `(None, None)` — the bridge
/// then draws no `overall` bar. A present rate with no CI degenerates to a
/// zero-width interval at the rate (matches the orchestrator's curve fallback).
fn optional_overall(d: &Bound<'_, PyDict>) -> PyResult<(Option<f64>, Option<Ci>)> {
    let rate: Option<f64> = match d.get_item("overall_power")? {
        Some(obj) if !obj.is_none() => Some(obj.extract()?),
        _ => None,
    };
    let Some(rate) = rate else {
        return Ok((None, None));
    };
    let ci = match d.get_item("overall_ci")? {
        Some(obj) if !obj.is_none() => {
            let seq = obj
                .cast::<PySequence>()
                .map_err(|_| PyValueError::new_err("overall_ci must be (lo, hi)"))?;
            Ci {
                lo: seq.get_item(0)?.extract()?,
                hi: seq.get_item(1)?.extract()?,
            }
        }
        _ => Ci { lo: rate, hi: rate },
    };
    Ok((Some(rate), Some(ci)))
}

/// Optional `contrast_pairs` key: list of `[positive, negative]` β-column
/// index pairs (the envelope's projection of `PowerResult.contrast_pairs`).
/// Absent → no contrasts (older callers).
fn contrast_pairs_of(d: &Bound<'_, PyDict>) -> PyResult<Vec<(u32, u32)>> {
    match d.get_item("contrast_pairs")? {
        Some(obj) => {
            let rows: Vec<Vec<u32>> = obj.extract()?;
            rows.into_iter()
                .map(|r| match r[..] {
                    [p, n] => Ok((p, n)),
                    _ => Err(PyValueError::new_err(
                        "contrast_pairs entries must be [positive, negative]",
                    )),
                })
                .collect()
        }
        None => Ok(vec![]),
    }
}

/// Reconstruct a [`PlotScenario`] from a find_sample_size scenario dict.
///
/// Neutral envelope keys: `power`, `ci`, `histogram` (list-of-lists; one row per N).
fn plot_scenario_from_sample_size_dict(
    label: String,
    d: &Bound<'_, PyDict>,
) -> PyResult<PlotScenario> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("scenario dict missing `{k}`")))
    };

    let sample_sizes: Vec<usize> = get("sample_sizes")?.extract()?;
    let target_indices: Vec<usize> = get("target_indices")?.extract()?;
    let contrast_pairs = contrast_pairs_of(d)?;

    let power_rows: Vec<Vec<f64>> = get("power")?.extract()?;
    let ci_rows_raw: Vec<Vec<(f64, f64)>> = get("ci")?.extract()?;
    if power_rows.len() != sample_sizes.len() || ci_rows_raw.len() != sample_sizes.len() {
        return Err(PyValueError::new_err(
            "sample_sizes, power and ci must have one row per N",
        ));
    }

    // `histogram` carries the corrected joint-significance histogram (always
    // corrected so the at_least_k / exactly_k curves match the corrected required-N
    // table). Python caller must supply `success_count_histogram_corrected` under
    // this key regardless of which correction is active.
    let hist_rows: Vec<Vec<u64>> = match d.get_item("histogram")? {
        Some(obj) => obj.extract()?,
        None => vec![Vec::new(); sample_sizes.len()],
    };
    if hist_rows.len() != sample_sizes.len() {
        return Err(PyValueError::new_err("histogram must have one row per N"));
    }

    let points: Vec<PlotPoint> = sample_sizes
        .iter()
        .copied()
        .zip(power_rows)
        .zip(ci_rows_raw)
        .zip(hist_rows)
        .map(|(((n, power), ci_raw), histogram)| {
            let ci = ci_raw.into_iter().map(|(lo, hi)| Ci { lo, hi }).collect();
            PlotPoint {
                n,
                target_indices: target_indices.clone(),
                contrast_pairs: contrast_pairs.clone(),
                power,
                ci,
                histogram,
                // Per-N overall rates are not surfaced in the Python envelope,
                // so the overall curve series is not emitted on this host.
                overall_power: None,
                overall_ci: None,
            }
        })
        .collect();

    Ok(PlotScenario { label, points })
}

// ── Public bridge functions ─────────────────────────────────────────────────

/// Ordered `(key, spec)` pairs for a find_power result.
///
/// `result` is a dict with `scenarios: [{label, sample_sizes, target_indices,
/// power, ci}, ...]`. `scenarios` is a list of dicts, each carrying its label
/// as a `"label"` key; list order is preserved (no alphabetical sorting).
/// `power` and `ci` carry the displayed arrays (Python selects corrected vs
/// uncorrected before calling). Returns `[("power", spec_json)]`.
#[pyfunction]
#[pyo3(signature = (result, *, show_ci=false, target_power_line=None))]
pub fn power_plot_set_json(
    result: &Bound<'_, PyDict>,
    show_ci: bool,
    target_power_line: Option<f64>,
) -> PyResult<Vec<(String, String)>> {
    let scenarios_any = result
        .get_item("scenarios")?
        .ok_or_else(|| PyValueError::new_err("result dict missing `scenarios`"))?;
    let scenarios = scenarios_any
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("`scenarios` must be a list"))?;
    if scenarios.is_empty() {
        return Err(PyValueError::new_err("result has zero scenarios"));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(scenarios.len());
    for item in scenarios.iter() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each element of `scenarios` must be a dict"))?;
        let label: String = d
            .get_item("label")?
            .ok_or_else(|| PyValueError::new_err("scenario dict missing `label`"))?
            .extract()?;
        out.push(PlotScenario {
            label,
            points: vec![plot_point_from_power_dict(d)?],
        });
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
    };
    Ok(power_plot_set(&out, &opts))
}

/// Ordered `(key, spec)` pairs for a find_sample_size result.
///
/// `result` is a dict with `scenarios: [{label, sample_sizes, target_indices,
/// power, ci, histogram}, ...]`. `scenarios` is a list of dicts, each carrying
/// its label as a `"label"` key; list order is preserved (no alphabetical
/// sorting). `power` and `ci` carry the displayed arrays;
/// `histogram` is always the corrected joint-significance histogram.
/// Block order follows `sample_size_plot_set` (curve, at_least_k, exactly_k, …).
#[pyfunction]
#[pyo3(signature = (result, *, show_ci=false, target_power_line=None))]
pub fn sample_size_plot_set_json(
    result: &Bound<'_, PyDict>,
    show_ci: bool,
    target_power_line: Option<f64>,
) -> PyResult<Vec<(String, String)>> {
    let scenarios_any = result
        .get_item("scenarios")?
        .ok_or_else(|| PyValueError::new_err("result dict missing `scenarios`"))?;
    let scenarios = scenarios_any
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("`scenarios` must be a list"))?;
    if scenarios.is_empty() {
        return Err(PyValueError::new_err("result has zero scenarios"));
    }
    let mut out: Vec<PlotScenario> = Vec::with_capacity(scenarios.len());
    for item in scenarios.iter() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each element of `scenarios` must be a dict"))?;
        let label: String = d
            .get_item("label")?
            .ok_or_else(|| PyValueError::new_err("scenario dict missing `label`"))?
            .extract()?;
        out.push(plot_scenario_from_sample_size_dict(label, d)?);
    }
    let opts = PlotOptions {
        title: None,
        show_ci,
        target_power_line,
    };
    Ok(sample_size_plot_set(&out, &opts))
}
