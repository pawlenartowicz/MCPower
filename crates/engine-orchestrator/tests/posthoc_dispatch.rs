//! Post-hoc end-to-end through both dispatch twins.
//!
//! `merge_sums_posthoc_success_counts` (merge_unit.rs) covers post-hoc pooling as
//! pure math on hand-built `PowerResult`s. This drives a real 3-level-factor
//! contract with a post-hoc request through multi-core `find_power` AND the
//! `single_core_find_power` + `merge_power_results` twin, so a fold that dropped
//! the post-hoc block — or mis-sized the joint-significance histogram — fails.

use engine_contract::OutcomeKind;
use engine_orchestrator::{
    ByValue, CancellationToken, CrossingFit, GridMode, SampleSizeMethod, SampleSizeResult,
    find_power, find_sample_size, merge_power_results, merge_sample_size_results,
    single_core_find_power, single_core_find_sample_size,
};
use engine_spec_builder::build_contract;
use engine_spec_builder::input::{
    Correction, EffectAssignment, HeteroskedasticityInput, LinearSpec, PosthocRequest,
    PredictorSpec, ResidualSpec, VarKind,
};

/// One-way ANOVA `y ~ dose_group` (3 levels) with an all-pairwise post-hoc
/// request on the factor. Mirrors `anova_3level_spec` in engine-spec-builder.
/// Targets expand to the two non-reference dummies (n_targets == 2); the
/// post-hoc family is C(3,2) == 3 pairwise contrasts.
fn anova_3level_posthoc_contract() -> Vec<engine_contract::SimulationContract> {
    let spec = LinearSpec {
        formula: "y = dose_group".into(),
        predictors: vec![PredictorSpec {
            name: "dose_group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["low".into(), "mid".into(), "high".into()],
                proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                reference: "low".into(),
                sampled_proportions: None,
            },
        }],
        effects: vec![
            EffectAssignment {
                name: "dose_group[mid]".into(),
                size: 0.3,
            },
            EffectAssignment {
                name: "dose_group[high]".into(),
                size: 0.5,
            },
        ],
        correlations: vec![],
        alpha: 0.05,
        correction: Correction::None,
        targets: vec!["overall".into()],
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: ResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![PosthocRequest {
            factor: "dose_group".into(),
            posthoc_alpha: None,
        }],
        upload: None,
        cluster_level_vars: vec![],
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    build_contract(&spec, OutcomeKind::Continuous, None, None, 0.0, vec![])
        .expect("build anova posthoc contract")
}

/// Post-hoc block survives multi-core dispatch with the documented shape:
/// one block, C(3,2) == 3 pairwise contrasts, and a joint-significance histogram
/// of length n_targets + Σ(post-hoc contrasts) + 1.
#[test]
fn find_power_surfaces_posthoc_family() {
    let contracts = anova_3level_posthoc_contract();
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, 120, 200, 2137, None, &cancel).expect("find_power ok");
    let pr = &r.scenarios[0].1;

    assert_eq!(
        pr.posthoc.len(),
        1,
        "one post-hoc block per requested factor"
    );
    let n_contrasts = pr.posthoc[0].power_uncorrected.len();
    assert_eq!(
        n_contrasts, 3,
        "3-level factor ⇒ C(3,2) == 3 pairwise contrasts"
    );
    assert!(
        !pr.posthoc[0].power_uncorrected.is_empty(),
        "post-hoc power vector must be populated"
    );
    // Histogram spans the main targets AND every post-hoc contrast, plus the
    // zero bucket: len == n_targets + Σ(post-hoc contrasts) + 1.
    assert_eq!(
        pr.success_count_histogram_uncorrected.len(),
        pr.target_indices.len() + n_contrasts + 1,
        "joint histogram length must span targets + post-hoc contrasts + 1"
    );
}

/// The same post-hoc family survives the single-core + merge twin with the
/// identical shape — proves the WASM worker path carries post-hoc through merge.
#[test]
fn single_core_merge_surfaces_posthoc_family() {
    let contracts = anova_3level_posthoc_contract();
    let cancel = CancellationToken::new();
    let mut parts = Vec::new();
    for i in 0..2u64 {
        let p = single_core_find_power(&contracts, 120, 100, 2137 + i, None, &cancel)
            .expect("single_core ok");
        parts.push(p);
    }
    let merged = merge_power_results(&parts).expect("merge ok");
    let pr = &merged.scenarios[0].1;

    assert_eq!(pr.posthoc.len(), 1, "merge must keep the post-hoc block");
    let n_contrasts = pr.posthoc[0].power_uncorrected.len();
    assert_eq!(n_contrasts, 3, "merged post-hoc family ⇒ 3 contrasts");
    assert!(
        !pr.posthoc[0].power_uncorrected.is_empty(),
        "merged post-hoc power vector must be populated"
    );
    assert_eq!(
        pr.success_count_histogram_uncorrected.len(),
        pr.target_indices.len() + n_contrasts + 1,
        "merged joint histogram length must span targets + post-hoc contrasts + 1"
    );
}

/// Required N per post-hoc contrast from a sample-size result, checked against
/// what the per-N post-hoc power implies: the grid-empirical entry is the first
/// grid N whose corrected power reaches the target, and the fit is a
/// per-contrast `CrossingFit` whose `Fitted` crossing sits inside the grid.
fn assert_posthoc_required_n(ssr: &SampleSizeResult) {
    assert_eq!(ssr.posthoc_first_achieved.len(), 1, "one block");
    assert_eq!(ssr.posthoc_fitted.len(), 1, "one block");
    assert_eq!(ssr.posthoc_first_achieved[0].len(), 3, "C(3,2) contrasts");
    assert_eq!(ssr.posthoc_fitted[0].len(), 3, "C(3,2) contrasts");
    for c in 0..3 {
        let expected = ssr
            .grid_or_trace
            .iter()
            .find(|pr| pr.posthoc[0].power_corrected[c] >= ssr.target_power)
            .map(|pr| pr.n);
        assert_eq!(ssr.posthoc_first_achieved[0][c], expected, "contrast {c}");
    }
    // The largest effect gap (low vs high) must be reachable, the smallest
    // (low vs mid) must not be, so both Some and None are exercised.
    assert!(ssr.posthoc_first_achieved[0].iter().any(Option::is_some));
    assert!(ssr.posthoc_first_achieved[0].iter().any(Option::is_none));
    let n_min = ssr.grid_or_trace.first().unwrap().n;
    let n_max = ssr.grid_or_trace.last().unwrap().n;
    for (c, fit) in ssr.posthoc_fitted[0].iter().enumerate() {
        match fit {
            CrossingFit::Fitted { n_star, .. } => {
                assert!(
                    (n_min as f64..=n_max as f64).contains(n_star),
                    "contrast {c}: n_star {n_star} outside grid"
                );
                assert!(ssr.posthoc_first_achieved[0][c].is_some(), "contrast {c}");
            }
            CrossingFit::NotReached { .. } => {
                assert!(ssr.posthoc_first_achieved[0][c].is_none(), "contrast {c}");
            }
            other => panic!("contrast {c}: unexpected fit {other:?}"),
        }
    }
}

/// Sample-size search with a post-hoc request gives one required-N entry and
/// one fit per contrast, on both dispatch twins and after merge.
#[test]
fn sample_size_surfaces_posthoc_required_n() {
    let contracts = anova_3level_posthoc_contract();
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(90),
        mode: GridMode::Linear,
    };
    let native = find_sample_size(
        &contracts,
        0.8,
        (90, 540),
        300,
        method,
        1,
        2137,
        None,
        &cancel,
    )
    .expect("find_sample_size ok");
    assert_posthoc_required_n(&native.scenarios[0].1);

    let parts: Vec<_> = (0..2u64)
        .map(|i| {
            single_core_find_sample_size(
                &contracts,
                0.8,
                (90, 540),
                150,
                method,
                1,
                2137 + i,
                None,
                &cancel,
            )
            .expect("single_core ok")
        })
        .collect();
    assert_posthoc_required_n(&parts[0].scenarios[0].1);
    let merged = merge_sample_size_results(&parts).expect("merge ok");
    assert_posthoc_required_n(&merged.scenarios[0].1);
}
