// Tests for F1 (CsvData real fields + assemble wiring) and F2 (get_effects_from_data driver).

use engine_app_spec::{
    AppSpec, ClusterDim, CsvData, EffectSize, LinearSpec, LogitSpec, MixedOutcome, MixedSpec,
    ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_spec_builder::input::{UploadColumn, UploadColumnType, UploadMode};

// ─── helpers ────────────────────────────────────────────────────────────────

/// Build a minimal LinearSpec with one continuous predictor and a matching CSV
/// column, leaving effects zeroed (placeholder, as in the Python reference).
fn linear_spec_with_csv(col_values: Vec<f64>, y_values: Vec<f64>) -> AppSpec {
    let n = col_values.len() as u32;
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric { name: "x".into(), distribution: Default::default(), pinned: false }],
        effects: vec![EffectSize {
            name: "x".into(),
            value: 0.0,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        tests: TestSelection::Effects {
            names: vec!["x".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: Some(CsvData {
            mode: UploadMode::Partial,
            n_rows: n,
            columns: vec![
                UploadColumn {
                    name: "x".into(),
                    col_type: UploadColumnType::Continuous,
                    values: col_values,
                    labels: vec![],
                },
                UploadColumn {
                    name: "y".into(),
                    col_type: UploadColumnType::Continuous,
                    values: y_values,
                    labels: vec![],
                },
            ],
        }),
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    })
}

// ─── F1 tests ───────────────────────────────────────────────────────────────

/// F1: assembling an AppSpec with a CSV column produces a SimulationContract
/// whose generation spec has an `uploaded_frame` that is `Some`.
#[test]
fn f1_assemble_with_csv_sets_uploaded_frame() {
    use engine_app_spec::assemble::assemble_spec;

    let x_vals: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = (0..20).map(|i| (i as f64) * 2.0 + 1.0).collect();
    let spec = linear_spec_with_csv(x_vals, y_vals);

    let contracts = assemble_spec(&spec).expect("assemble succeeds");
    assert_eq!(contracts.len(), 1);
    let contract = &contracts[0];
    assert!(
        contract.generation.uploaded_frame.is_some(),
        "uploaded_frame should be Some when csv is present"
    );
}

/// F1: assembling an AppSpec without csv keeps uploaded_frame as None (no regression).
#[test]
fn f1_assemble_without_csv_leaves_uploaded_frame_none() {
    use engine_app_spec::assemble::assemble_spec;

    let spec = AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric { name: "x".into(), distribution: Default::default(), pinned: false }],
        effects: vec![EffectSize {
            name: "x".into(),
            value: 0.3,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    });

    let contracts = assemble_spec(&spec).expect("assemble succeeds");
    assert!(contracts[0].generation.uploaded_frame.is_none());
}

// ─── F2 tests ───────────────────────────────────────────────────────────────

/// F2: get_effects_from_data on an OLS AppSpec with known data recovers the
/// known standardized slope within tolerance ±0.1.
///
/// Construction: x = 0..N-1 (continuous), y = 2*x_std + noise_free, so the
/// standardized OLS β_x ≈ 1.0 (the regression of z-scored y on z-scored x
/// returns slope ≈ Corr(x,y) ≈ 1 when y = 2*x exactly).
///
/// With exact y = 2*x (no noise) the empirical R² → 1, so β ≈ 1.0 after
/// standardization.  We gate loosely at ±0.1.
#[test]
fn f2_get_effects_recovers_known_slope() {
    use engine_app_spec::driver::get_effects_from_data;

    let n = 40usize;
    // x values: 0..39
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // y = 2*x exactly → after z-scoring both, slope = 1.0
    let y_vals: Vec<f64> = x_vals.iter().map(|&x| 2.0 * x).collect();

    let spec = linear_spec_with_csv(x_vals, y_vals);
    let effects = get_effects_from_data(&spec).expect("get_effects_from_data succeeds");

    // Should return exactly one effect ("x"), intercept dropped.
    assert_eq!(effects.len(), 1, "one effect returned (intercept dropped)");
    assert_eq!(effects[0].name, "x");
    let beta_x = effects[0].value;
    // y = 2*x exactly (noiseless) → standardized β is exactly 1.0; tighten to
    // machine precision so standardization regressions are caught immediately.
    assert!(
        (beta_x - 1.0).abs() < 1e-9,
        "expected β_x == 1.0 (noiseless), got {beta_x}"
    );
}

/// F2: get_effects_from_data on a continuous + binary predictor spec.
/// x_cont is 0..39, x_bin alternates 0/1.  After standardization:
/// – x_cont z-scored → slope with y ≈ known
/// – x_bin centred  → slope with y ≈ 0 (y depends only on x_cont)
#[test]
fn f2_get_effects_with_binary_predictor() {
    use engine_app_spec::driver::get_effects_from_data;

    let n = 40usize;
    let x_cont: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x_bin: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    // y depends ONLY on x_cont; x_bin should have β ≈ 0
    let y_vals: Vec<f64> = x_cont.iter().map(|&x| 3.0 * x).collect();

    let spec = AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x_cont".into(), "x_bin".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric { name: "x_cont".into(), distribution: Default::default(), pinned: false },
            VarType::Binary {
                name: "x_bin".into(),
                binary_proportion: 0.5,
            },
        ],
        effects: vec![
            EffectSize {
                name: "x_cont".into(),
                value: 0.0,
            },
            EffectSize {
                name: "x_bin".into(),
                value: 0.0,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        tests: TestSelection::Effects {
            names: vec!["x_cont".into(), "x_bin".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: Some(CsvData {
            mode: UploadMode::Partial,
            n_rows: n as u32,
            columns: vec![
                UploadColumn {
                    name: "x_cont".into(),
                    col_type: UploadColumnType::Continuous,
                    values: x_cont,
                    labels: vec![],
                },
                UploadColumn {
                    name: "x_bin".into(),
                    col_type: UploadColumnType::Binary,
                    values: x_bin,
                    labels: vec![],
                },
                UploadColumn {
                    name: "y".into(),
                    col_type: UploadColumnType::Continuous,
                    values: y_vals,
                    labels: vec![],
                },
            ],
        }),
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    });

    let effects = get_effects_from_data(&spec).expect("get_effects_from_data succeeds");
    assert_eq!(effects.len(), 2, "two effects returned");
    let x_cont_effect = effects.iter().find(|e| e.name == "x_cont").expect("x_cont");
    let x_bin_effect = effects.iter().find(|e| e.name == "x_bin").expect("x_bin");
    // x_cont drives y; x_bin is orthogonal → β_bin ≈ 0
    assert!(
        x_cont_effect.value.abs() > 0.5,
        "x_cont should have non-trivial slope, got {}",
        x_cont_effect.value
    );
    assert!(
        x_bin_effect.value.abs() < 0.3,
        "x_bin slope should be near 0, got {}",
        x_bin_effect.value
    );
}

/// F2 (GLM): get_effects_from_data on a Logit AppSpec recovers the log-odds
/// coefficient of a binary predictor on its native scale. Construction is a
/// saturated 2×2 table with odds ratio 4, so the logistic MLE slope is exactly
/// ln(4) (centering the binary predictor in recovery leaves the slope
/// unchanged). The outcome is fed to the GLM fitter as raw 0/1, NOT z-scored.
#[test]
fn f2_get_effects_glm_recovers_log_odds() {
    use engine_app_spec::driver::get_effects_from_data;

    // x=1 group (30 rows): 20 successes, 10 failures → odds 2.0
    // x=0 group (30 rows): 10 successes, 20 failures → odds 0.5
    // OR = 4 → logistic β_x = ln(4).
    let mut x_vals: Vec<f64> = Vec::new();
    let mut y_vals: Vec<f64> = Vec::new();
    x_vals.extend(std::iter::repeat(1.0).take(30));
    y_vals.extend(std::iter::repeat(1.0).take(20));
    y_vals.extend(std::iter::repeat(0.0).take(10));
    x_vals.extend(std::iter::repeat(0.0).take(30));
    y_vals.extend(std::iter::repeat(1.0).take(10));
    y_vals.extend(std::iter::repeat(0.0).take(20));
    let n = x_vals.len() as u32;

    let spec = AppSpec::Logit(LogitSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Binary {
            name: "x".into(),
            binary_proportion: 0.5,
        }],
        effects: vec![EffectSize {
            name: "x".into(),
            value: 0.0,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        tests: TestSelection::Effects {
            names: vec!["x".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: Some(CsvData {
            mode: UploadMode::Partial,
            n_rows: n,
            columns: vec![
                UploadColumn {
                    name: "x".into(),
                    col_type: UploadColumnType::Binary,
                    values: x_vals,
                    labels: vec![],
                },
                UploadColumn {
                    name: "y".into(),
                    col_type: UploadColumnType::Binary,
                    values: y_vals,
                    labels: vec![],
                },
            ],
        }),
        baseline_probability: 0.5,
        test_formula: None,
        outcome_options: None,
    });

    let effects = get_effects_from_data(&spec).expect("get_effects_from_data succeeds for logit");
    assert_eq!(effects.len(), 1, "one effect returned (intercept dropped)");
    assert_eq!(effects[0].name, "x");
    let beta_x = effects[0].value;
    let expected = 4.0_f64.ln();
    assert!(
        (beta_x - expected).abs() < 1e-6,
        "expected logistic β_x ≈ ln(4) = {expected}, got {beta_x}"
    );
}

/// F2 (MLE): get_effects_from_data on a Mixed AppSpec recovers the fixed-effect
/// slope from clustered data, fitting on the native outcome scale and threading
/// the uploaded grouping column as cluster IDs. Construction: 8 balanced
/// clusters of 25 rows; `x` cycles {-2,-1,0,1,2} (population sd = √2, balanced
/// within each cluster), `y = 1·x + u_cluster + small ε`. The fixed effect is
/// recovered on the z-scored-x scale, so β_x ≈ 1·sd(x) = √2.
#[test]
fn f2_get_effects_mle_recovers_fixed_effect() {
    use engine_app_spec::driver::get_effects_from_data;

    let n = 200usize;
    let x_vals: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let group_codes: Vec<f64> = (0..n).map(|i| (i / 25) as f64).collect(); // 0..7
                                                                           // y = 1·x + u_cluster + small deterministic ε; u_cluster spreads ±~1.
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i % 5) as f64 - 2.0;
            let c = (i / 25) as f64;
            let u = (c - 3.5) * 0.3;
            let eps = ((i % 7) as f64 - 3.0) * 0.05;
            x + u + eps
        })
        .collect();

    let spec = AppSpec::Mixed(MixedSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric { name: "x".into(), distribution: Default::default(), pinned: false }],
        effects: vec![EffectSize {
            name: "x".into(),
            value: 0.0,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        tests: TestSelection::Effects {
            names: vec!["x".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: Some(CsvData {
            mode: UploadMode::Partial,
            n_rows: n as u32,
            columns: vec![
                UploadColumn {
                    name: "x".into(),
                    col_type: UploadColumnType::Continuous,
                    values: x_vals,
                    labels: vec![],
                },
                UploadColumn {
                    name: "group".into(),
                    col_type: UploadColumnType::Continuous,
                    values: group_codes,
                    labels: vec![],
                },
                UploadColumn {
                    name: "y".into(),
                    col_type: UploadColumnType::Continuous,
                    values: y_vals,
                    labels: vec![],
                },
            ],
        }),
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
        cluster_name: "group".into(),
        icc: 0.2,
        cluster_dim: ClusterDim::NClusters { value: 8 },
        cluster_level_vars: vec![],
        extra_groupings: vec![],
        slopes: vec![],
        outcome: MixedOutcome::Gaussian,
    });

    let effects = get_effects_from_data(&spec).expect("get_effects_from_data succeeds for mixed");
    assert_eq!(
        effects.len(),
        1,
        "one fixed effect returned (intercept dropped)"
    );
    assert_eq!(effects[0].name, "x");
    let beta_x = effects[0].value;
    let expected = 2.0_f64.sqrt(); // 1·sd(x), sd(x) = √2
    assert!(
        (beta_x - expected).abs() < 0.1,
        "expected fixed-effect β_x ≈ √2 = {expected}, got {beta_x}"
    );
}

/// F2: get_effects_from_data with a factor predictor recovers known dummy
/// effects and emits no EffectSize for the reference level.
///
/// Construction: 3-level factor `g` (labels ["ctrl", "treatA", "treatB"]).
/// Level codes: 0 = ctrl (reference, excluded from dummies), 1 = treatA, 2 = treatB.
/// y = 1.5 * g[treatA] + 0 * g[treatB] + noise_free.
///
/// After z-scoring y and building 0/1 dummies, OLS should recover β_treatA > 0
/// and β_treatB ≈ 0 (within tolerance), and no effect for "ctrl" (reference).
#[test]
fn f2_get_effects_with_factor_predictor() {
    use engine_app_spec::driver::get_effects_from_data;

    // 30 rows: 10 ctrl (code 0), 10 treatA (code 1), 10 treatB (code 2).
    let n = 30usize;
    let g_codes: Vec<f64> = (0..n).map(|i| (i / 10) as f64).collect(); // 0,0,...,1,1,...,2,2,...
    let labels = vec![
        "ctrl".to_string(),
        "treatA".to_string(),
        "treatB".to_string(),
    ];

    // y = 3.0 for treatA rows, 0.0 otherwise (large effect, no noise).
    let y_vals: Vec<f64> = g_codes
        .iter()
        .map(|&c| if c == 1.0 { 3.0 } else { 0.0 })
        .collect();

    let spec = AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["g".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Factor {
            name: "g".into(),
            factor_n_levels: 3,
            factor_proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
        }],
        effects: vec![
            EffectSize {
                name: "g[2]".into(),
                value: 0.0,
            },
            EffectSize {
                name: "g[3]".into(),
                value: 0.0,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 32,
        seed: 42,
        // Use numeric level names for TestSelection — assemble_spec builds factor
        // levels as "1".."k"; the driver uses real label names for beta mapping.
        tests: TestSelection::Effects {
            names: vec!["g[2]".into(), "g[3]".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: Some(CsvData {
            mode: UploadMode::Partial,
            n_rows: n as u32,
            columns: vec![
                UploadColumn {
                    name: "g".into(),
                    col_type: UploadColumnType::Factor,
                    values: g_codes,
                    labels: labels.clone(),
                },
                UploadColumn {
                    name: "y".into(),
                    col_type: UploadColumnType::Continuous,
                    values: y_vals,
                    labels: vec![],
                },
            ],
        }),
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    });

    let effects = get_effects_from_data(&spec).expect("get_effects_from_data succeeds");

    // Expect exactly 2 effects: g[treatA] and g[treatB]. Reference (ctrl) must be absent.
    assert_eq!(
        effects.len(),
        2,
        "two dummies returned (reference excluded)"
    );
    assert!(
        !effects.iter().any(|e| e.name == "g[ctrl]"),
        "reference level g[ctrl] must not appear in returned effects"
    );

    let treat_a = effects
        .iter()
        .find(|e| e.name == "g[treatA]")
        .expect("g[treatA] effect present");
    let treat_b = effects
        .iter()
        .find(|e| e.name == "g[treatB]")
        .expect("g[treatB] effect present");

    // treatA drives y; β_treatA should be substantially positive.
    assert!(
        treat_a.value > 0.5,
        "g[treatA] should have a large positive effect, got {}",
        treat_a.value
    );
    // treatB has zero effect; β_treatB should be near zero.
    assert!(
        treat_b.value.abs() < 0.1,
        "g[treatB] should be near zero, got {}",
        treat_b.value
    );
}
