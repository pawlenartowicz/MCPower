use engine_orchestrator::{
    ByValue, Ci, EstimatorExtras, PowerResult, SampleSizeMethod, ScenarioResult,
};

#[test]
fn power_result_serde_roundtrip() {
    let pr = PowerResult {
        n: 100,
        n_sims: 1600,
        target_indices: vec![1, 2],
        contrast_pairs: vec![],
        power_uncorrected: vec![0.80, 0.65],
        power_corrected: vec![0.78, 0.60],
        ci_uncorrected: vec![Ci { lo: 0.78, hi: 0.82 }, Ci { lo: 0.62, hi: 0.68 }],
        ci_corrected: vec![Ci { lo: 0.76, hi: 0.80 }, Ci { lo: 0.57, hi: 0.63 }],
        convergence_rate: 0.99,
        boundary_hit: vec![],
        estimator_extras: EstimatorExtras::Ols {},
        overall_significant_rate: None,
        success_counts_uncorrected: vec![1280, 1040],
        success_counts_corrected: vec![1248, 960],
        convergence_count: 1584,
        overall_significant_count: 0,
        overall_significant_ci: None,
        success_count_histogram_uncorrected: vec![],
        success_count_histogram_corrected: vec![],
        grid_warnings: vec![],
        posthoc: vec![],
        factor_exclusion_counts: vec![],
        factor_separation_counts: vec![],
    };
    let bytes = rmp_serde::to_vec(&pr).expect("encode");
    let back: PowerResult = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back.n, 100);
    assert_eq!(back.target_indices, vec![1, 2]);
    assert!(matches!(back.estimator_extras, EstimatorExtras::Ols {}));
}

/// The WASM pool serializes per-worker results to **JSON** and `merge_power_results`
/// re-parses them. `serde_json` renders `f64::NAN` as `null` and then rejects `null`
/// where it expects an `f64`, so any Glm/Mle result — whose `baseline_prob_realized`
/// / `tau_estimate` / (no-converged) `tau_squared_hat_mean` carry NaN placeholders —
/// crashed the merge. The existing extras round-trip above uses msgpack, which carries
/// NaN natively and hid the gap; this asserts the *JSON* path survives a NaN.
#[test]
fn glm_mle_extras_survive_json_roundtrip_with_nan() {
    let nan_glm = EstimatorExtras::Glm {
        baseline_prob_realized: f64::NAN,
        baseline_prob_sum: 0.0,
        baseline_prob_n: 0,
        singular_fit_rate: 0.0,
        singular_count: 0,
        singular_n: 0,
        tau_squared_hat_mean: f64::NAN,
        tau_squared_hat_sum: 0.0,
        tau_squared_hat_n: 0,
    };
    let json = serde_json::to_string(&nan_glm).expect("glm encodes");
    assert!(json.contains("null"), "NaN must render as JSON null: {json}");
    let back: EstimatorExtras = serde_json::from_str(&json).expect("glm decodes (null->NaN)");
    let EstimatorExtras::Glm {
        baseline_prob_realized,
        tau_squared_hat_mean,
        ..
    } = back
    else {
        panic!("expected Glm");
    };
    assert!(baseline_prob_realized.is_nan() && tau_squared_hat_mean.is_nan());

    let nan_mle = EstimatorExtras::Mle {
        tau_estimate: f64::NAN,
        boundary_hits: 0,
        joint_uncorrected_rate: 0.0,
        joint_corrected_rate: 0.0,
        tau_sum: 0.0,
        tau_n: 0,
        joint_uncorrected_count: 0,
        joint_corrected_count: 0,
        singular_fit_rate: 0.0,
        singular_count: 0,
        singular_n: 0,
        boundary_rate_per_component: vec![],
        boundary_component_counts: vec![],
    };
    let json = serde_json::to_string(&nan_mle).expect("mle encodes");
    let back: EstimatorExtras = serde_json::from_str(&json).expect("mle decodes (null->NaN)");
    let EstimatorExtras::Mle { tau_estimate, .. } = back else {
        panic!("expected Mle");
    };
    assert!(tau_estimate.is_nan());
}

#[test]
fn sample_size_method_grid_default_serde() {
    let m = SampleSizeMethod::Grid {
        by: ByValue::Fixed(10),
        mode: engine_orchestrator::GridMode::Linear,
    };
    let bytes = rmp_serde::to_vec(&m).expect("encode");
    let back: SampleSizeMethod = rmp_serde::from_slice(&bytes).expect("decode");
    assert!(matches!(
        back,
        SampleSizeMethod::Grid {
            by: ByValue::Fixed(10),
            ..
        }
    ));
}

#[test]
fn scenario_result_preserves_order() {
    let scenarios = ScenarioResult {
        scenarios: vec![
            ("optimistic".to_string(), 1u32),
            ("realistic".to_string(), 2u32),
            ("pessimistic".to_string(), 3u32),
        ],
    };
    let labels: Vec<&str> = scenarios
        .scenarios
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(labels, vec!["optimistic", "realistic", "pessimistic"]);
}

#[test]
fn estimator_extras_carries_three_variants_with_snake_case_tags() {
    let ols = EstimatorExtras::Ols {};
    let glm = EstimatorExtras::Glm {
        baseline_prob_realized: 0.42,
        baseline_prob_sum: 42.0,
        baseline_prob_n: 100,
        singular_fit_rate: 0.0,
        singular_count: 0,
        singular_n: 0,
        tau_squared_hat_mean: f64::NAN,
        tau_squared_hat_sum: 0.0,
        tau_squared_hat_n: 0,
    };
    let mle = EstimatorExtras::Mle {
        tau_estimate: 0.31,
        boundary_hits: 5,
        joint_uncorrected_rate: 0.6,
        joint_corrected_rate: 0.55,
        tau_sum: 31.0,
        tau_n: 100,
        joint_uncorrected_count: 60,
        joint_corrected_count: 55,
        singular_fit_rate: 0.1,
        singular_count: 3,
        singular_n: 30,
        boundary_rate_per_component: vec![0.1, 0.05],
        boundary_component_counts: vec![3, 1],
    };
    for (fx, tag) in [(ols, "ols"), (glm, "glm"), (mle, "mle")] {
        let bytes = rmp_serde::to_vec_named(&fx).unwrap();
        let back: EstimatorExtras = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(format!("{fx:?}"), format!("{back:?}"));
        let found = bytes.windows(tag.len()).any(|w| w == tag.as_bytes());
        assert!(found, "msgpack must contain {tag:?}; bytes={bytes:?}");
    }
}
