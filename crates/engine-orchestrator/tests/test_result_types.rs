use engine_orchestrator::{
    ByValue, Ci, EstimatorExtras, GridMode, PowerResult, SampleSizeMethod, SampleSizeResult,
    ScenarioResult,
};

/// The WASM pool serializes per-worker results to **JSON** and `merge_power_results`
/// re-parses them. `serde_json` renders `f64::NAN` as `null` and then rejects `null`
/// where it expects an `f64`, so any Glm/Mle result — whose `baseline_prob_realized`
/// / `tau_estimate` / (no-converged) `tau_squared_hat_mean` carry NaN placeholders —
/// crashed the merge. msgpack carries NaN natively and would hide the gap; this
/// asserts the *JSON* path survives a NaN.
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
    assert!(
        json.contains("null"),
        "NaN must render as JSON null: {json}"
    );
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

/// The WASM main thread parses `ScenarioResult<SampleSizeResult>` back from each
/// worker, and a NaN buried in an embedded `grid_or_trace` `PowerResult`'s
/// `estimator_extras` is exactly where the `serde_json` null→NaN crash strikes —
/// three containers deep (scenario result, sample-size result, grid point).
/// msgpack would not catch it (NaN→null→reject only bites JSON). Removing a
/// `nan_tolerant` guard on an embedded field would fail the round-trip here.
#[test]
fn sample_size_result_survives_json_roundtrip_with_embedded_nan() {
    let grid_pr = PowerResult {
        n: 100,
        n_sims: 800,
        target_indices: vec![1],
        contrast_pairs: vec![],
        power_uncorrected: vec![0.5],
        power_corrected: vec![0.5],
        ci_uncorrected: vec![Ci { lo: 0.4, hi: 0.6 }],
        ci_corrected: vec![Ci { lo: 0.4, hi: 0.6 }],
        convergence_rate: 1.0,
        boundary_hit: vec![],
        estimator_extras: EstimatorExtras::Mle {
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
        },
        overall_significant_rate: None,
        success_counts_uncorrected: vec![400],
        success_counts_corrected: vec![400],
        convergence_count: 800,
        overall_significant_count: 0,
        overall_significant_ci: None,
        success_count_histogram_uncorrected: vec![],
        success_count_histogram_corrected: vec![],
        grid_warnings: vec![],
        posthoc: vec![],
        factor_exclusion_counts: vec![],
        factor_separation_counts: vec![],
    };
    let ssr = SampleSizeResult {
        grid_or_trace: vec![grid_pr],
        first_achieved: vec![Some(100)],
        first_joint_achieved: vec![],
        fitted: vec![],
        fitted_joint: vec![],
        first_overall_achieved: None,
        fitted_overall: None,
        cluster_atom: 1,
        unit_divisor: 1,
        target_power: 0.8,
        method: SampleSizeMethod::Grid {
            by: ByValue::Fixed(10),
            mode: GridMode::Linear,
        },
        grid_warnings: vec![],
    };
    let wrapped = ScenarioResult {
        scenarios: vec![("optimistic".into(), ssr)],
    };
    let json = serde_json::to_string(&wrapped).expect("encode");
    assert!(
        json.contains("null"),
        "embedded NaN must render as JSON null: {json}"
    );
    let back: ScenarioResult<SampleSizeResult> =
        serde_json::from_str(&json).expect("decode (null->NaN)");
    // NaN != NaN, so assert the affected field via is_nan and the surrounding
    // structure via the finite metadata.
    let pr = &back.scenarios[0].1.grid_or_trace[0];
    let EstimatorExtras::Mle { tau_estimate, .. } = &pr.estimator_extras else {
        panic!("expected Mle extras");
    };
    assert!(
        tau_estimate.is_nan(),
        "embedded tau_estimate must survive as NaN"
    );
    assert_eq!(back.scenarios[0].1.first_achieved, vec![Some(100)]);
    assert_eq!(pr.n, 100);
}

/// The `estimator` tag is a wire contract: the app reads
/// `estimator_extras.estimator === 'glm'` from the JSON result.
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
        let v = serde_json::to_value(&fx).unwrap();
        assert_eq!(v["estimator"], tag, "{v}");
    }
}
