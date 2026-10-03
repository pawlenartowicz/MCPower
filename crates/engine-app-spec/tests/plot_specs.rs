//! Tests for the eager Vega-Lite spec emitters in engine-app-spec::plot.
use engine_app_spec::{power_plot_specs, sample_size_curve_specs, PlotSpecs};
use engine_orchestrator::{
    ByValue, GridMode, PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};
use serde_json::json;

/// Build a `PowerResult` via serde — `PowerResult` has ~20 fields (most
/// `#[serde(default)]`), so a struct literal is brittle; deserializing from the
/// non-default fields is robust to additive contract growth. Field names here
/// are the struct's snake_case names (no `rename_all`); `EstimatorExtras` is
/// tagged `{"estimator": "ols"}`.
fn power_result(power_u: Vec<f64>, power_c: Vec<f64>) -> PowerResult {
    let n = power_u.len();
    let cis_u: Vec<_> = power_u
        .iter()
        .map(|&p| json!({ "lo": p - 0.05, "hi": p + 0.05 }))
        .collect();
    let cis_c: Vec<_> = power_c
        .iter()
        .map(|&p| json!({ "lo": p - 0.05, "hi": p + 0.05 }))
        .collect();
    serde_json::from_value(json!({
        "n": 80,
        "n_sims": 100,
        "target_indices": (1..=n).collect::<Vec<usize>>(),
        "power_uncorrected": power_u,
        "power_corrected": power_c,
        "ci_uncorrected": cis_u,
        "ci_corrected": cis_c,
        "convergence_rate": 1.0,
        "boundary_hit": [],
        "estimator_extras": { "estimator": "ols" },
        // Distinct histograms: P(>=1) / P(>=2) is 0.9 / 0.6 uncorrected and
        // 0.6 / 0.2 corrected, so a block reading the wrong one is visible.
        "success_count_histogram_uncorrected": [10, 30, 60],
        "success_count_histogram_corrected": [40, 40, 20],
    }))
    .expect("PowerResult deserializes from the required fields")
}

fn power_scenarios_corrected(labels: &[&str]) -> ScenarioResult<PowerResult> {
    ScenarioResult {
        scenarios: labels
            .iter()
            // uncorrected = 0.7/0.6; corrected = 0.5/0.4 (clearly different)
            .map(|l| (l.to_string(), power_result(vec![0.7, 0.6], vec![0.5, 0.4])))
            .collect(),
    }
}

fn ss_scenarios(labels: &[&str]) -> ScenarioResult<SampleSizeResult> {
    let grid = vec![power_result(vec![0.5, 0.4], vec![0.3, 0.2]), {
        let mut p = power_result(vec![0.85, 0.8], vec![0.65, 0.6]);
        p.n = 160;
        p
    }];
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    ScenarioResult {
        scenarios: labels
            .iter()
            .map(|l| {
                (
                    l.to_string(),
                    SampleSizeResult {
                        grid_or_trace: grid.clone(),
                        first_achieved: vec![Some(160), Some(160)],
                        first_joint_achieved: vec![Some(160), Some(160)],
                        fitted: vec![],
                        fitted_joint: vec![],
                        first_overall_achieved: None,
                        fitted_overall: None,
                        cluster_atom: 1,
                        unit_divisor: 1,
                        target_power: 0.8,
                        method,
                        grid_warnings: vec![],
                    },
                )
            })
            .collect(),
    }
}

/// The `field` column of a spec's inline data rows, in row order.
fn data_column(spec: &str, field: &str) -> Vec<f64> {
    let v: serde_json::Value = serde_json::from_str(spec).expect("valid JSON");
    v.pointer("/data/values")
        .and_then(serde_json::Value::as_array)
        .expect("inline data rows")
        .iter()
        .filter_map(|row| row.get(field).and_then(serde_json::Value::as_f64))
        .collect()
}

// ── power_plot_specs block shape ───────────────────────────────────────────

#[test]
fn power_specs_two_scenarios_single_power_block() {
    let result = power_scenarios_corrected(&["default", "optimistic"]);
    let PlotSpecs { blocks } = power_plot_specs(&result, 0.8, false);
    // power_plot_set always emits exactly ["power"]
    let keys: Vec<&str> = blocks.iter().map(|b| b.key.as_str()).collect();
    assert_eq!(keys, vec!["power"], "exactly one block keyed 'power'");
    // spec is valid JSON, theme-naked
    let v: serde_json::Value = serde_json::from_str(&blocks[0].spec).expect("valid JSON");
    assert!(v.get("config").is_none(), "spec must be theme-naked");
    assert!(
        blocks[0].spec.contains("\"bar\"") || blocks[0].spec.contains("\"mark\":\"bar\""),
        "bar mark present"
    );
    // multi-scenario → color encoding present
    assert!(
        blocks[0].spec.contains("\"color\""),
        "multi-scenario spec has color encoding"
    );
    // target_power_line → rule layer
    assert!(
        blocks[0].spec.contains("\"rule\""),
        "target-power rule layer present"
    );
}

// ── sample_size_curve_specs block shape ────────────────────────────────────

#[test]
fn sample_size_two_scenarios_two_targets_block_order() {
    // S=2, m=2 → ["scenario:default", "scenario:optimistic", "overlay", "at_least_k", "exactly_k"]
    let result = ss_scenarios(&["default", "optimistic"]);
    let PlotSpecs { blocks } = sample_size_curve_specs(&result, 0.8, false);
    let keys: Vec<&str> = blocks.iter().map(|b| b.key.as_str()).collect();
    assert_eq!(keys[0], "scenario:default");
    assert_eq!(keys[1], "scenario:optimistic");
    assert_eq!(keys[2], "overlay");
    assert_eq!(keys[3], "at_least_k");
    assert_eq!(keys[4], "exactly_k");
    for b in &blocks {
        let v: serde_json::Value = serde_json::from_str(&b.spec).expect("valid JSON");
        assert!(
            v.get("config").is_none(),
            "block '{}' must be theme-naked",
            b.key
        );
    }
}

// ── corrected=true changes the plotted values ──────────────────────────────

#[test]
fn corrected_flag_changes_power_values_in_spec() {
    // power_result: uncorrected=[0.7,0.6], corrected=[0.5,0.4]
    let result = power_scenarios_corrected(&["s"]);

    let spec_u = power_plot_specs(&result, 0.8, false).blocks.remove(0).spec;
    let spec_c = power_plot_specs(&result, 0.8, true).blocks.remove(0).spec;

    assert_eq!(data_column(&spec_u, "power"), [0.7, 0.6]);
    assert_eq!(data_column(&spec_c, "power"), [0.5, 0.4]);
}

#[test]
fn corrected_flag_changes_curve_values_in_spec() {
    // grid: uncorrected=[0.5,0.4]/[0.85,0.8]; corrected=[0.3,0.2]/[0.65,0.6]
    let result = ss_scenarios(&["s"]);

    let blocks_u = sample_size_curve_specs(&result, 0.8, false).blocks;
    let blocks_c = sample_size_curve_specs(&result, 0.8, true).blocks;

    // Both produce the same block keys (curve, at_least_k, exactly_k)
    let keys_u: Vec<&str> = blocks_u.iter().map(|b| b.key.as_str()).collect();
    let keys_c: Vec<&str> = blocks_c.iter().map(|b| b.key.as_str()).collect();
    assert_eq!(keys_u, keys_c);

    // The curve rows (N=80 then N=160, two targets each) follow the flag.
    let block = |blocks: &[engine_app_spec::PlotBlock], key: &str| {
        blocks.iter().find(|b| b.key == key).unwrap().spec.clone()
    };
    assert_eq!(
        data_column(&block(&blocks_u, "curve"), "power"),
        [0.5, 0.4, 0.85, 0.8]
    );
    assert_eq!(
        data_column(&block(&blocks_c, "curve"), "power"),
        [0.3, 0.2, 0.65, 0.6]
    );

    // The joint-detection block always reads the corrected histogram
    // ([40, 40, 20] → P(>=1)=0.6, P(>=2)=0.2), whatever the flag says.
    for blocks in [&blocks_u, &blocks_c] {
        assert_eq!(
            data_column(&block(blocks, "at_least_k"), "p"),
            [0.6, 0.2, 0.6, 0.2]
        );
    }
}

// ── sample-size unit: x field switches to n_unit ───────────────────────────

/// With a unit divisor the curve's x field becomes `n_unit` (= n / divisor) and
/// the engine emits no x title — the host names the unit. The `n` field stays on
/// every row so a host can still read total observations. Without one (divisor
/// 1, observations) the x field stays `n` and no row carries `n_unit`.
#[test]
fn sample_size_curve_uses_n_unit_when_the_result_carries_a_divisor() {
    let result = ss_scenarios(&["default"]);
    let PlotSpecs { blocks } = sample_size_curve_specs(&result, 0.8, false);
    let curve = blocks.iter().find(|b| b.key == "curve").unwrap();
    assert!(curve.spec.contains("\"field\":\"n\""), "{}", curve.spec);
    assert!(!curve.spec.contains("n_unit"), "{}", curve.spec);

    let mut result = ss_scenarios(&["default"]);
    for (_, ss) in result.scenarios.iter_mut() {
        ss.unit_divisor = 8;
    }
    let PlotSpecs { blocks } = sample_size_curve_specs(&result, 0.8, false);
    let curve = blocks
        .iter()
        .find(|b| b.key == "curve")
        .expect("curve block present");
    assert!(
        curve.spec.contains("\"field\":\"n_unit\""),
        "x field must be n_unit: {}",
        curve.spec
    );
    assert!(
        !curve.spec.contains("Sample size (N)"),
        "the engine must leave the axis title to the host: {}",
        curve.spec
    );
    // n / 8 on every row: two targets at n = 80, then two at n = 160.
    assert_eq!(data_column(&curve.spec, "n_unit"), [10.0, 10.0, 20.0, 20.0]);
}
