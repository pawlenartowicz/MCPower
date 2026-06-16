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
        "success_count_histogram_uncorrected": [0, 0, 100],
        "success_count_histogram_corrected": [0, 0, 100],
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
    let grid = vec![
        power_result(vec![0.5, 0.4], vec![0.3, 0.2]),
        {
            let mut p = power_result(vec![0.85, 0.8], vec![0.65, 0.6]);
            p.n = 160;
            p
        },
    ];
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
                        target_power: 0.8,
                        method,
                        grid_warnings: vec![],
                    },
                )
            })
            .collect(),
    }
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
fn sample_size_single_scenario_two_targets_block_order() {
    // S=1, m=2 → ["curve", "at_least_k", "exactly_k"]
    let result = ss_scenarios(&["default"]);
    let PlotSpecs { blocks } = sample_size_curve_specs(&result, 0.8, false);
    let keys: Vec<&str> = blocks.iter().map(|b| b.key.as_str()).collect();
    assert_eq!(
        keys,
        vec!["curve", "at_least_k", "exactly_k"],
        "S=1, m=2 block order must be curve, at_least_k, exactly_k"
    );
    for b in &blocks {
        let v: serde_json::Value = serde_json::from_str(&b.spec).expect("valid JSON");
        assert!(v.get("config").is_none(), "block '{}' must be theme-naked", b.key);
    }
}

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
        assert!(v.get("config").is_none(), "block '{}' must be theme-naked", b.key);
    }
}

// ── corrected=true changes the plotted values ──────────────────────────────

#[test]
fn corrected_flag_changes_power_values_in_spec() {
    // power_result: uncorrected=[0.7,0.6], corrected=[0.5,0.4]
    // The power spec embeds the data as inline JSON values; we check that the
    // corrected numbers appear in the spec when corrected=true, and the uncorrected
    // numbers appear when corrected=false.
    let result = power_scenarios_corrected(&["s"]);

    let spec_u = power_plot_specs(&result, 0.8, false).blocks.remove(0).spec;
    let spec_c = power_plot_specs(&result, 0.8, true).blocks.remove(0).spec;

    // uncorrected spec must contain 0.7 and NOT 0.5
    assert!(
        spec_u.contains("0.7"),
        "uncorrected spec should contain 0.7; got: {spec_u}"
    );
    assert!(
        spec_u.contains("0.6"),
        "uncorrected spec should contain 0.6; got: {spec_u}"
    );
    // corrected spec must contain 0.5 and NOT 0.7 (at the data layer)
    assert!(
        spec_c.contains("0.5"),
        "corrected spec should contain 0.5; got: {spec_c}"
    );
    assert!(
        spec_c.contains("0.4"),
        "corrected spec should contain 0.4; got: {spec_c}"
    );
    // The two specs must differ because the data is different
    assert_ne!(
        spec_u, spec_c,
        "corrected=true must produce a different spec than corrected=false"
    );
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

    // curve block differs because power values differ
    let curve_u = &blocks_u.iter().find(|b| b.key == "curve").unwrap().spec;
    let curve_c = &blocks_c.iter().find(|b| b.key == "curve").unwrap().spec;
    assert_ne!(curve_u, curve_c, "curve spec must differ when corrected changes");

    // corrected spec must contain 0.3 (corrected uncorrected is 0.5)
    assert!(
        curve_c.contains("0.3"),
        "corrected curve must contain corrected power 0.3"
    );
    assert!(
        curve_u.contains("0.5"),
        "uncorrected curve must contain uncorrected power 0.5"
    );
}
