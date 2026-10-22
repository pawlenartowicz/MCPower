//! Hardcoded LinearSpec → SimulationSpec pipeline tests.
//!
//! Migrated from `engine-spec-builder/src/project.rs` when the legacy
//! `build_linear_spec` shim was removed; exercises the full contract path
//! (`build_contract` + `contract_to_simulation_spec`) end-to-end.

use engine_core::spec::{Distribution, EstimatorSpec};
use engine_spec_builder::input::{
    Correction, EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec, ResidualSpec,
    ScenarioInput, VarKind,
};

mod common;
use common::build_linear_spec;

fn simple_spec() -> LinearSpec {
    LinearSpec {
        formula: "y = x1 + x2".into(),
        predictors: vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x2".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
        ],
        effects: vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.5,
            },
            EffectAssignment {
                name: "x2".into(),
                size: 0.3,
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
        posthoc_requests: vec![],
        upload: None,
        cluster_level_vars: vec![],
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    }
}

/// Correlation is continuous-only by design: a user correlation pair that names
/// a binary predictor is rejected with `CorrelationNonContinuous`. Binary/factor
/// variables are generated from their marginals; their joint dependence is
/// preserved only by strict-mode data upload, not the synthetic copula path.
#[test]
fn synthetic_binary_correlation_pair_is_rejected() {
    use engine_spec_builder::SpecError;
    use engine_spec_builder::input::CorrelationPair;

    let mut spec = simple_spec();
    spec.predictors[1].kind = VarKind::Binary { proportion: 0.4 };
    spec.correlations = vec![CorrelationPair {
        a: "x1".into(),
        b: "x2".into(),
        value: 0.3,
    }];

    let err = build_linear_spec(&spec).expect_err("binary correlation pair must be rejected");
    match err {
        SpecError::CorrelationNonContinuous { name } => assert_eq!(name, "x2"),
        other => panic!("expected CorrelationNonContinuous, got {other:?}"),
    }
}

/// Symmetric counterpart to `synthetic_binary_correlation_pair_is_rejected`:
/// a model with a binary predictor AND correlations among the *continuous*
/// predictors builds successfully — the binary column is simply generated
/// independently (it occupies an identity row in the matrix). Guards the
/// decision to leave `invariant_06` unchanged (binary may sit in
/// `continuous_columns` with a zero off-diagonal).
#[test]
fn synthetic_binary_predictor_with_continuous_correlation_succeeds() {
    use engine_spec_builder::input::CorrelationPair;

    let mut spec = simple_spec();
    spec.formula = "y = x1 + x2 + x3".into();
    spec.predictors = vec![
        PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        },
        PredictorSpec {
            name: "x2".into(),
            pinned: false,
            kind: VarKind::Normal,
        },
        PredictorSpec {
            name: "x3".into(),
            pinned: false,
            kind: VarKind::Binary { proportion: 0.4 },
        },
    ];
    spec.effects = vec![
        EffectAssignment {
            name: "x1".into(),
            size: 0.5,
        },
        EffectAssignment {
            name: "x2".into(),
            size: 0.3,
        },
        EffectAssignment {
            name: "x3".into(),
            size: 0.2,
        },
    ];
    // Correlate only the two continuous predictors; x3 (binary) is uncorrelated.
    spec.correlations = vec![CorrelationPair {
        a: "x1".into(),
        b: "x2".into(),
        value: 0.5,
    }];

    let specs = build_linear_spec(&spec)
        .expect("binary predictor + continuous-only correlation must build");
    assert_eq!(specs.len(), 1);
    // Guardrail: correlation is continuous-only. The binary x3 (index 2) sits in the
    // 3×3 matrix with ZERO off-diagonals; only the x1–x2 pair carries r=0.5. (Was a
    // shape-only `specs.len()==1` that said nothing about the binary column.)
    assert_eq!(
        specs[0].correlation,
        vec![1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn projects_simple_two_predictor_spec() {
    let mut specs = build_linear_spec(&simple_spec()).unwrap();
    assert_eq!(specs.len(), 1);
    let s = specs.pop().unwrap();
    assert_eq!(s.scenario.name, "optimistic");
    assert_eq!(s.n_non_factor, 2);
    assert_eq!(s.n_factor_dummies, 0);
    assert_eq!(
        s.var_types,
        vec![Distribution::Normal, Distribution::Normal]
    );
    assert_eq!(s.effect_sizes, vec![0.0, 0.5, 0.3]);
    assert_eq!(s.target_indices, vec![1, 2]);
    assert!(
        s.contrast_pairs.is_empty(),
        "no contrasts requested -> empty contrast_pairs"
    );
    assert_eq!(s.correlation, vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(s.estimator, EstimatorSpec::Ols);
}

#[test]
fn effect_sizes_indexed_by_design_matrix_column_not_formula_order() {
    let mut spec = simple_spec();
    spec.formula = "y = group + x1".into();
    spec.predictors = vec![
        PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["1".into(), "2".into(), "3".into()],
                proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                reference: "1".into(),
                sampled_proportions: None,
            },
        },
        PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        },
    ];
    spec.effects = vec![
        EffectAssignment {
            name: "group[2]".into(),
            size: 0.3,
        },
        EffectAssignment {
            name: "group[3]".into(),
            size: 0.5,
        },
        EffectAssignment {
            name: "x1".into(),
            size: 0.4,
        },
    ];
    let s = build_linear_spec(&spec).unwrap().pop().unwrap();
    assert_eq!(s.effect_sizes, vec![0.0, 0.4, 0.3, 0.5]);
}

/// A reference-level contrast (`group[A] − group[B]`, A reference) collapses to a
/// Marginal on `group[B]`; if a coefficient test already requested that Marginal,
/// the two must dedup rather than reach the contract validator as a "duplicate
/// Marginal target" error. ANOVA hits this routinely: it auto-populates every
/// pairwise contrast while the user ticks per-coefficient tests.
#[test]
fn reference_contrast_dedups_against_coefficient_test_marginal() {
    let mut spec = simple_spec();
    spec.formula = "y = x1 + group".into();
    spec.predictors = vec![
        PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        },
        PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: None,
            },
        },
    ];
    spec.effects = vec![
        EffectAssignment {
            name: "x1".into(),
            size: 0.5,
        },
        EffectAssignment {
            name: "group[B]".into(),
            size: 0.2,
        },
        EffectAssignment {
            name: "group[C]".into(),
            size: 0.4,
        },
    ];
    // Coefficient test on group[B] → Marginal{group[B]}.
    spec.targets = vec!["group[B]".into()];
    // Reference pair collapses to the SAME Marginal{group[B]}; the genuine
    // non-reference pair stays a Contrast.
    spec.contrast_pairs = vec![
        ("group[A]".into(), "group[B]".into()),
        ("group[B]".into(), "group[C]".into()),
    ];

    let s = build_linear_spec(&spec).expect("dedup must not error on duplicate marginal");
    let s = s.first().expect("one scenario");
    // group[B] survives once as a marginal (not twice), and the B-vs-C contrast
    // is preserved.
    assert_eq!(
        s.target_indices.len(),
        1,
        "group[B] marginal deduped to one"
    );
    // Design matrix: col 0 = intercept, col 1 = x1, col 2 = group[B], col 3 = group[C].
    // B-vs-C contrast: positive term group[B] → col 2, negative term group[C] → col 3.
    assert_eq!(
        s.contrast_pairs,
        vec![(2u32, 3u32)],
        "B-vs-C contrast must carry (positive=2, negative=3) column indices"
    );
    // EP-1 invariant: power/ci/count vectors are sized target_indices.len() + contrast_pairs.len().
    assert_eq!(
        s.target_indices.len() + s.contrast_pairs.len(),
        2,
        "EP-1: one marginal + one contrast = 2 power-vector slots"
    );
}

/// Zero-perturbation `ScenarioInput` baseline for the validation tests below.
fn scenario_input(name: &str) -> ScenarioInput {
    ScenarioInput {
        name: name.into(),
        heterogeneity: 0.0,
        heteroskedasticity_ratio: 1.0,
        correlation_noise_sd: 0.0,
        distribution_change_prob: 0.0,
        new_distributions: vec![],
        residual_change_prob: 0.0,
        residual_dists: vec![],
        residual_df: 0.0,
        sampled_factor_proportions: false,
        truth_start: false,
        random_effect_dist: 0,
        random_effect_df: 0.0,
        icc_noise_sd: 0.0,
    }
}

#[test]
fn scenario_binary_in_new_distributions_is_rejected() {
    use engine_spec_builder::SpecError;

    let mut spec = simple_spec();
    let mut sc = scenario_input("custom");
    sc.distribution_change_prob = 0.5;
    sc.new_distributions = vec![1, 2]; // binary, right_skewed
    spec.scenarios = vec![sc];
    let err = build_linear_spec(&spec).expect_err("binary pool entry must be rejected");
    match err {
        SpecError::ScenarioBinarySwapUnsupported { name } => assert_eq!(name, "custom"),
        other => panic!("expected ScenarioBinarySwapUnsupported, got {other:?}"),
    }
}

#[test]
fn scenario_normal_in_new_distributions_is_accepted() {
    // Swap-to-normal is the identity — a legitimate way to dilute the
    // effective swap probability of the pool.
    let mut spec = simple_spec();
    let mut sc = scenario_input("custom");
    sc.distribution_change_prob = 0.5;
    sc.new_distributions = vec![0, 2]; // normal, right_skewed
    spec.scenarios = vec![sc];
    let specs = build_linear_spec(&spec).unwrap();
    assert_eq!(
        specs[0].scenario.new_distributions,
        vec![Distribution::Normal, Distribution::RightSkewed]
    );
}

#[test]
fn scenario_residual_df_unset_with_df_consuming_pool_is_rejected() {
    use engine_spec_builder::SpecError;

    let mut spec = simple_spec();
    let mut sc = scenario_input("custom");
    sc.residual_change_prob = 0.5;
    sc.residual_dists = vec![4, 2]; // high_kurtosis, right_skewed — both consume df
    sc.residual_df = 0.0; // unset: the kernel floor would silently give t(3)/χ²(3)
    spec.scenarios = vec![sc];
    let err = build_linear_spec(&spec).expect_err("df-consuming pool without df must be rejected");
    match err {
        SpecError::ScenarioResidualDfTooLow { name, got } => {
            assert_eq!(name, "custom");
            assert_eq!(got, 0.0);
        }
        other => panic!("expected ScenarioResidualDfTooLow, got {other:?}"),
    }
}

#[test]
fn scenario_normal_only_residual_pool_without_df_is_accepted() {
    // `normal` consumes no df — the df bound only arms for the t/χ² shapes.
    use engine_core::spec::ResidualDist;

    let mut spec = simple_spec();
    let mut sc = scenario_input("custom");
    sc.residual_change_prob = 0.5;
    sc.residual_dists = vec![0]; // normal only
    spec.scenarios = vec![sc];
    let specs = build_linear_spec(&spec).unwrap();
    assert_eq!(specs[0].scenario.residual_dists, vec![ResidualDist::Normal]);
}

#[test]
fn builtin_preset_scenario_payloads_still_build() {
    // Every shipped preset in configs/scenarios.json, encoded the way hosts
    // encode it (names → integer codes through the engine's code tables), must
    // pass the scenario guards (df bound, binary-swap reject) and build. The RE
    // knobs are mixed-model-only; hosts zero them for a model without random
    // effects, as here.
    use engine_spec_builder::project_contract::{DIST_CODES, RESIDUAL_CODES};
    use serde_json::Value;

    let presets: Value =
        serde_json::from_str(include_str!("../../../configs/scenarios.json")).unwrap();
    let presets = presets.as_object().expect("scenarios.json is an object");
    assert!(!presets.is_empty());
    let f64_of = |v: &Value| v.as_f64().unwrap();
    let encode = |table: &[(&str, i32)], names: &Value| -> Vec<i32> {
        names
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                let n = n.as_str().unwrap();
                table
                    .iter()
                    .find(|(name, _)| *name == n)
                    .unwrap_or_else(|| panic!("no code for {n:?}"))
                    .1
            })
            .collect()
    };

    let mut spec = simple_spec();
    spec.scenarios = presets
        .iter()
        .map(|(name, p)| ScenarioInput {
            name: name.clone(),
            heterogeneity: f64_of(&p["heterogeneity"]),
            heteroskedasticity_ratio: f64_of(&p["heteroskedasticity_ratio"]),
            correlation_noise_sd: f64_of(&p["correlation_noise_sd"]),
            distribution_change_prob: f64_of(&p["distribution_change_prob"]),
            new_distributions: encode(DIST_CODES, &p["new_distributions"]),
            residual_change_prob: f64_of(&p["residual_change_prob"]),
            residual_dists: encode(RESIDUAL_CODES, &p["residual_dists"]),
            residual_df: f64_of(&p["residual_df"]),
            sampled_factor_proportions: p["sampled_factor_proportions"].as_bool().unwrap(),
            truth_start: p["truth_start"].as_bool().unwrap(),
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        })
        .collect();
    let specs = build_linear_spec(&spec).expect("every shipped preset must build");
    assert_eq!(specs.len(), presets.len());
    for (s, (name, p)) in specs.iter().zip(presets) {
        assert_eq!(&s.scenario.name, name);
        assert_eq!(s.scenario.residual_df, f64_of(&p["residual_df"]), "{name}");
        assert_eq!(
            s.scenario.truth_start,
            p["truth_start"].as_bool().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn explicit_scenarios_produce_one_spec_each() {
    let mut spec = simple_spec();
    spec.scenarios = vec![
        ScenarioInput {
            name: "optimistic".into(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        },
        ScenarioInput {
            name: "realistic".into(),
            heterogeneity: 0.1,
            heteroskedasticity_ratio: 2.0,
            correlation_noise_sd: 0.15,
            distribution_change_prob: 0.2,
            new_distributions: vec![2],
            residual_change_prob: 0.5,
            residual_dists: vec![4],
            residual_df: 8.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        },
    ];
    let specs = build_linear_spec(&spec).unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].scenario.name, "optimistic");
    assert_eq!(specs[1].scenario.name, "realistic");
    assert_eq!(specs[1].scenario.heterogeneity, 0.1);
    assert_eq!(
        specs[1].scenario.new_distributions,
        vec![Distribution::RightSkewed]
    );
}
