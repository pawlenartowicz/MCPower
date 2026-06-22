//! Eager Vega-Lite spec emission for native-result hosts (Tauri, WASM).
//!
//! The app holds the engine's native `ScenarioResult<…>` in-process. Each native
//! result is converted to the orchestrator's narrow `PlotScenario`/`PlotPoint`
//! plot type, then handed to the theme-naked plot-set emitters — no envelope walker.
//! We emit one block per plot key (drives the UI tab/selector) so the frontend
//! never re-invokes the engine to re-plot.
//!
//! Invariant: `PlotPoint.histogram` always carries
//! `success_count_histogram_corrected` regardless of the `corrected` flag — the
//! joint-detection curve must match the corrected joint required-N table.

use engine_orchestrator::{
    power_plot_set, sample_size_plot_set, PlotOptions, PlotPoint, PlotScenario, PowerResult,
    SampleSizeResult, ScenarioResult,
};
use serde::Serialize;

/// A single named plot block. `key` is the block identifier (`"power"`, `"curve"`,
/// `"at_least_k"`, `"exactly_k"`, `"overlay"`, or `"scenario:<label>"`).
/// `spec` is a theme-naked Vega-Lite v5 JSON string.
#[derive(Debug, Clone, Serialize)]
pub struct PlotBlock {
    pub key: String,
    pub spec: String,
}

/// Ordered plot blocks for a find_power or find_sample_size result.
/// Serialises to `{ "blocks": [{ "key": …, "spec": … }, …] }`.
#[derive(Debug, Clone, Serialize)]
pub struct PlotSpecs {
    pub blocks: Vec<PlotBlock>,
}

fn opts(target_power: f64) -> PlotOptions {
    // Match Python's rich repr: CI bands + target-power reference line shown.
    PlotOptions {
        title: None,
        show_ci: true,
        target_power_line: Some(target_power),
    }
}

/// One find_power scenario → a single-point `PlotScenario` (power-at-N bar).
/// `corrected` selects which power/CI arrays are loaded into the plot point.
fn power_scenario(label: &str, pr: &PowerResult, corrected: bool) -> PlotScenario {
    let (power, ci) = if corrected {
        (pr.power_corrected.clone(), pr.ci_corrected.clone())
    } else {
        (pr.power_uncorrected.clone(), pr.ci_uncorrected.clone())
    };
    let point = PlotPoint {
        n: pr.n,
        target_indices: pr.target_indices.clone(),
        contrast_pairs: pr.contrast_pairs.clone(),
        power,
        ci,
        histogram: Vec::new(), // power-at-N never reads the histogram
        overall_power: None,   // the overall series rides the sample-size curve only
        overall_ci: None,
    };
    PlotScenario {
        label: label.to_string(),
        points: vec![point],
    }
}

/// One find_sample_size scenario → a multi-point `PlotScenario` (curve).
/// `SampleSizeResult` holds `grid_or_trace: Vec<PowerResult>` — one `PowerResult`
/// per grid N. Each grid `PowerResult` becomes one `PlotPoint`.
/// `corrected` selects which power/CI arrays are loaded into each plot point.
fn sample_size_scenario(label: &str, ssr: &SampleSizeResult, corrected: bool) -> PlotScenario {
    let points: Vec<PlotPoint> = ssr
        .grid_or_trace
        .iter()
        .map(|pr| {
            let (power, ci) = if corrected {
                (pr.power_corrected.clone(), pr.ci_corrected.clone())
            } else {
                (pr.power_uncorrected.clone(), pr.ci_uncorrected.clone())
            };
            PlotPoint {
                n: pr.n,
                target_indices: pr.target_indices.clone(),
                contrast_pairs: pr.contrast_pairs.clone(),
                power,
                ci,
                // The overall/omnibus test power+CI for this grid point — drives
                // the extra `"overall"` curve series. `None` whenever the family
                // suppressed the overall test (mixed/GLMM), so no series renders.
                overall_power: pr.overall_significant_rate,
                overall_ci: pr.overall_significant_ci,
                // Corrected histogram (the power line above stays corrected or
                // uncorrected per `corrected`): the joint-detection curve reads
                // `PlotPoint.histogram` and MUST use the corrected counts so
                // it matches the corrected joint required-N table. Always take
                // `success_count_histogram_corrected` regardless of `corrected`.
                histogram: pr.success_count_histogram_corrected.clone(),
            }
        })
        .collect();
    PlotScenario {
        label: label.to_string(),
        points,
    }
}

/// Block-keyed plot specs for a find_power result (Summary tab).
/// `corrected` controls whether `power_corrected`/`ci_corrected` (true) or
/// `power_uncorrected`/`ci_uncorrected` (false) are loaded into each plot point.
pub fn power_plot_specs(
    result: &ScenarioResult<PowerResult>,
    target_power: f64,
    corrected: bool,
) -> PlotSpecs {
    let o = opts(target_power);
    let all: Vec<PlotScenario> = result
        .scenarios
        .iter()
        .map(|(label, pr)| power_scenario(label, pr, corrected))
        .collect();
    let blocks = power_plot_set(&all, &o)
        .into_iter()
        .map(|(key, spec)| PlotBlock { key, spec })
        .collect();
    PlotSpecs { blocks }
}

/// Block-keyed plot specs for a find_sample_size result (Summary tab).
/// `corrected` controls which power/CI arrays drive the curve. The
/// at_least_k / exactly_k blocks always read the corrected histogram (even
/// when `corrected` is false), so the joint-detection curve matches the
/// corrected joint required-N table.
pub fn sample_size_curve_specs(
    result: &ScenarioResult<SampleSizeResult>,
    target_power: f64,
    corrected: bool,
) -> PlotSpecs {
    let o = opts(target_power);
    let all: Vec<PlotScenario> = result
        .scenarios
        .iter()
        .map(|(label, ssr)| sample_size_scenario(label, ssr, corrected))
        .collect();
    let blocks = sample_size_plot_set(&all, &o)
        .into_iter()
        .map(|(key, spec)| PlotBlock { key, spec })
        .collect();
    PlotSpecs { blocks }
}
