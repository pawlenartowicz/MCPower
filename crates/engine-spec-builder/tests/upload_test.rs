use engine_spec_builder::{
    LinearSpec, SpecError, UploadColumn, UploadColumnType, UploadInput, UploadMode,
};

#[test]
fn linear_spec_without_upload_deserializes() {
    let json = r#"{"formula":"y = x1","predictors":[{"name":"x1","kind":"normal"}],
        "effects":[{"name":"x1","size":0.5}],"correlations":[],"alpha":0.05,
        "correction":"bonferroni","targets":["overall"],"heterogeneity":0.0,
        "heteroskedasticity":{},"residual":{"distribution":"normal","df":0.0},
        "max_failed_fraction":0.1,"scenarios":[]}"#;
    let spec: LinearSpec = serde_json::from_str(json).unwrap();
    assert!(spec.upload.is_none());
}

#[test]
fn upload_input_round_trips() {
    let up = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 3,
        columns: vec![UploadColumn {
            name: "x1".into(),
            col_type: UploadColumnType::Continuous,
            values: vec![1.0, 2.0, 3.0],
            labels: vec![],
        }],
    };
    let s = serde_json::to_string(&up).unwrap();
    let back: UploadInput = serde_json::from_str(&s).unwrap();
    assert_eq!(up, back);
}

#[test]
fn standardize_continuous_is_zscore_ddof0() {
    let col = UploadColumn {
        name: "x".into(),
        col_type: UploadColumnType::Continuous,
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        labels: vec![],
    };
    let z = engine_spec_builder::upload::standardize_column(&col);
    let sd = 2f64.sqrt();
    assert!((z[0] - (-2.0 / sd)).abs() < 1e-12);
    assert!((z[2] - 0.0).abs() < 1e-12);
    assert!((z[4] - (2.0 / sd)).abs() < 1e-12);
    let m: f64 = z.iter().sum::<f64>() / z.len() as f64;
    assert!(m.abs() < 1e-12);
}

#[test]
fn standardize_binary_centers_at_proportion() {
    let col = UploadColumn {
        name: "b".into(),
        col_type: UploadColumnType::Binary,
        values: vec![0.0, 0.0, 1.0, 1.0, 1.0],
        labels: vec![],
    };
    let z = engine_spec_builder::upload::standardize_column(&col);
    assert!((z[0] - (-0.6)).abs() < 1e-12);
    assert!((z[2] - (0.4)).abs() < 1e-12);
}

// ── A3 tests ─────────────────────────────────────────────────────────────────

fn make_two_predictor_spec_with_upload(upload: UploadInput) -> LinearSpec {
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    }
}

#[test]
fn matched_predictor_becomes_resampled_unmatched_stays_synthetic() {
    use engine_contract::{ColumnSpec, EstimatorSpec, OutcomeKind, SyntheticKind};

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            // extra column z is NOT in the model — should be ignored
            UploadColumn {
                name: "z".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![10.0, 20.0, 30.0, 40.0, 50.0],
                labels: vec![],
            },
        ],
    };
    let spec = make_two_predictor_spec_with_upload(upload);
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");
    let c = &contracts[0];

    // x1 is matched → Resampled
    assert!(
        matches!(&c.generation.columns[0], ColumnSpec::Resampled { .. }),
        "x1 should be Resampled, got: {:?}",
        c.generation.columns[0]
    );
    // x2 is unmatched → Synthetic Normal
    assert!(
        matches!(
            &c.generation.columns[1],
            ColumnSpec::Synthetic {
                kind: SyntheticKind::Normal,
                ..
            }
        ),
        "x2 should be Synthetic Normal, got: {:?}",
        c.generation.columns[1]
    );
    // uploaded_frame: only x1 is matched → n_cols == 1
    let frame = c
        .generation
        .uploaded_frame
        .as_ref()
        .expect("uploaded_frame Some");
    assert_eq!(frame.n_cols, 1, "only 1 matched column");
    assert_eq!(frame.n_rows, 5, "n_rows from upload.n_rows");
    // validate still passes
    c.validate().expect("contract validates with upload");
}

#[test]
fn binary_matched_column_becomes_resampled_binary_with_proportion() {
    use engine_contract::{ColumnSpec, EstimatorSpec, OutcomeKind};

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![UploadColumn {
            name: "x1".into(),
            col_type: UploadColumnType::Binary,
            // 3 ones out of 5 → proportion = 0.6
            values: vec![0.0, 1.0, 1.0, 0.0, 1.0],
            labels: vec![],
        }],
    };
    // Use a single-predictor spec for simplicity
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };
    let spec = LinearSpec {
        formula: "y = x1".into(),
        predictors: vec![PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        }],
        effects: vec![EffectAssignment {
            name: "x1".into(),
            size: 0.5,
        }],
        correlations: vec![],
        alpha: 0.05,
        correction: Correction::None,
        targets: vec!["x1".into()],
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: ResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![],
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");
    let c = &contracts[0];

    match &c.generation.columns[0] {
        ColumnSpec::ResampledBinary { proportion, .. } => {
            assert!(
                (*proportion - 0.6).abs() < 1e-12,
                "proportion should be 0.6, got {proportion}"
            );
        }
        other => panic!("expected ResampledBinary, got: {other:?}"),
    }
    c.validate().expect("contract validates");
}

// ── A4 tests ─────────────────────────────────────────────────────────────────

fn measured_offdiag(contracts: &[engine_contract::SimulationContract]) -> f64 {
    match &contracts[0].generation.correlations {
        engine_contract::Correlations::Matrix { values, .. } => {
            // 2x2 column-major: values = [m00, m10, m01, m11]
            // off-diagonal at [1] (m10) or [2] (m01) — both equal for symmetric
            values[1]
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

/// Correlation is continuous-only by design: a binary uploaded column is NOT
/// included in the measured correlation matrix, even when it is strongly
/// associated with a continuous column in the data. Its off-diagonal entry
/// stays at the identity default (0) — the binary column is generated from its
/// marginal, independent of the continuous predictor.
#[test]
fn partial_mode_does_not_correlate_binary_column() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    // x1 continuous and x2 binary are strongly associated in the data
    // (x2 = 1 exactly when x1 is large), yet binary must not be correlated.
    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Binary,
                values: vec![0.0, 0.0, 0.0, 1.0, 1.0],
                labels: vec![],
            },
        ],
    };
    let spec = LinearSpec {
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
                kind: VarKind::Binary { proportion: 0.4 },
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let offdiag = measured_offdiag(&contracts);
    assert!(
        offdiag.abs() < 1e-9,
        "binary column must not be correlated (continuous-only by design), got offdiag={offdiag}"
    );
}

#[test]
fn partial_mode_measures_empirical_correlation() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    // Build two perfectly monotone columns: Spearman ρ_S = 1.0, a fixed point of
    // the latent conversion (2·sin(π·1/6) = 1.0), so the installed value is 1.0.
    // x1 = [1, 2, 3, 4, 5], x2 = [2, 4, 6, 8, 10] → ρ_S = 1.0
    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![2.0, 4.0, 6.0, 8.0, 10.0],
                labels: vec![],
            },
        ],
    };
    let spec = LinearSpec {
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let offdiag = measured_offdiag(&contracts);
    assert!(
        (offdiag - 1.0).abs() < 0.05,
        "expected r≈1.0 for perfectly correlated columns, got {offdiag}"
    );
}

/// The installed off-diagonal is the **Spearman** rank correlation converted to
/// the latent-Gaussian scale (`2·sin(π·ρ_S/6)`), not the raw Pearson r.
///
/// x1=[1,2,3,4,5], x2=[1,2,3,5,4] (last two swapped) → ρ_S = 0.9, and the raw
/// Pearson of these same-valued columns is *also* 0.9. The latent conversion
/// inflates it to `2·sin(π·0.9/6) = 0.9079809994790935` — so a result of 0.9
/// would mean we measured Pearson (or skipped the conversion), and only 0.90798
/// proves both Spearman measurement and the latent inversion.
#[test]
fn partial_mode_measures_spearman_converted() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 5.0, 4.0], // ρ_S = 0.9 (last two swapped)
                labels: vec![],
            },
        ],
    };
    let spec = LinearSpec {
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let offdiag = measured_offdiag(&contracts);
    let expected = 2.0 * (std::f64::consts::PI * 0.9 / 6.0).sin(); // 0.9079809994790935
    assert!(
        (offdiag - expected).abs() < 1e-9,
        "expected latent-converted Spearman {expected}, got {offdiag} (0.9 would mean raw Pearson)"
    );
}

#[test]
fn explicit_correlation_overrides_measured() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, CorrelationPair, EffectAssignment, HeteroskedasticityInput, PredictorSpec,
        ResidualSpec, VarKind,
    };

    // Same perfectly-correlated columns, but user explicitly sets r=0.1
    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![2.0, 4.0, 6.0, 8.0, 10.0],
                labels: vec![],
            },
        ],
    };
    let spec = LinearSpec {
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
        // User explicitly overrides to 0.1
        correlations: vec![CorrelationPair {
            a: "x1".into(),
            b: "x2".into(),
            value: 0.1,
        }],
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let offdiag = measured_offdiag(&contracts);
    assert!(
        (offdiag - 0.1).abs() < 1e-9,
        "explicit correlation=0.1 must override measured r≈1.0, got {offdiag}"
    );
}

// ── A5 tests ─────────────────────────────────────────────────────────────────

#[test]
fn matched_factor_column_becomes_factor_from_frame() {
    use engine_contract::{ColumnSpec, EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    // Factor predictor "g" with levels ["A","B","C"], reference "A".
    // Upload column "g": 9 rows cycling through codes 0,1,2 → equal proportions 1/3 each.
    let labels = vec!["A".into(), "B".into(), "C".into()];
    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 9,
        columns: vec![UploadColumn {
            name: "g".into(),
            col_type: UploadColumnType::Factor,
            values: vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            labels: labels.clone(),
        }],
    };
    let spec = LinearSpec {
        formula: "y = g".into(),
        predictors: vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: labels.clone(),
                proportions: vec![0.5, 0.3, 0.2],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }],
        // Factor with reference "A" expands to dummies g[B] and g[C].
        effects: vec![
            EffectAssignment {
                name: "g[B]".into(),
                size: 0.5,
            },
            EffectAssignment {
                name: "g[C]".into(),
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");
    let c = &contracts[0];

    // Factor columns come after all non-factor columns.
    // This spec has zero non-factor predictors, so the factor is at columns[0].
    match &c.generation.columns[0] {
        ColumnSpec::FactorFromFrame {
            frame_column,
            n_levels,
            proportions,
            ..
        } => {
            assert_eq!(*frame_column, 0u32, "first (and only) frame column");
            assert_eq!(*n_levels, 3u32, "three labels → three levels");
            // Empirical proportions from values [0,1,2,0,1,2,0,1,2]: each code 3/9 = 1/3.
            assert_eq!(proportions.len(), 3, "one proportion per level");
            let sum: f64 = proportions.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "proportions must sum to 1.0, got {sum}"
            );
            for (i, &p) in proportions.iter().enumerate() {
                assert!(
                    (p - 1.0 / 3.0).abs() < 1e-12,
                    "level {i} proportion should be 1/3, got {p}"
                );
            }
        }
        other => panic!("expected FactorFromFrame, got: {other:?}"),
    }

    // uploaded_frame must be Some (the factor column was matched).
    let frame = c
        .generation
        .uploaded_frame
        .as_ref()
        .expect("uploaded_frame Some");
    assert_eq!(frame.n_cols, 1, "one matched column in frame");
    assert_eq!(frame.n_rows, 9, "nine rows");

    c.validate().expect("contract validates");
}

// ── I3 test ───────────────────────────────────────────────────────────────────

/// A partial upload with 4 matched continuous columns where the user-specified
/// pairs alone are PSD (the initial build_contract_correlations check passes),
/// but the combined measured+overlay matrix is not PSD.
///
/// Setup:
///   - x1 and x2 are perfectly correlated in the upload → measured r(x1,x2)=1.0
///   - x3 is independent (measured r(x1,x3)≈0, r(x2,x3)≈0)
///   - x4 is a filler predictor, fully unmatched
///   - User specifies only r(x1,x3)=0.6 and r(x2,x3)=-0.6
///
/// Initial check (user pairs only, r(x1,x2) defaults to 0):
///   3×3 block [[1,0,0.6],[0,1,-0.6],[0.6,-0.6,1]] → det=0.28 → PSD, passes.
///
/// Combined matrix after measure_correlations (measured r(x1,x2)=1.0):
///   3×3 block [[1,1,0.6],[1,1,-0.6],[0.6,-0.6,1]] → det=-1.44 → not PSD.
#[test]
fn partial_upload_nonpsd_combined_matrix_is_rejected() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, CorrelationPair, EffectAssignment, HeteroskedasticityInput, PredictorSpec,
        ResidualSpec, VarKind,
    };

    // x1 and x2 are perfectly correlated; x3 is independent; x4 is unmatched.
    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![2.0, 4.0, 6.0, 8.0, 10.0], // perfect r(x1,x2)=1.0
                labels: vec![],
            },
            UploadColumn {
                name: "x3".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![5.0, 1.0, 3.0, 2.0, 4.0], // uncorrelated with x1,x2
                labels: vec![],
            },
            // x4 is NOT uploaded — stays synthetic (filler for 4-predictor model)
        ],
    };
    let spec = LinearSpec {
        formula: "y = x1 + x2 + x3 + x4".into(),
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
            PredictorSpec {
                name: "x3".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x4".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
        ],
        effects: vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.4,
            },
            EffectAssignment {
                name: "x2".into(),
                size: 0.3,
            },
            EffectAssignment {
                name: "x3".into(),
                size: 0.2,
            },
            EffectAssignment {
                name: "x4".into(),
                size: 0.1,
            },
        ],
        // User pairs that are PSD with r(x1,x2)=0 (initial check with defaults)
        // but non-PSD with the measured r(x1,x2)=1.0 (combined matrix).
        correlations: vec![
            CorrelationPair {
                a: "x1".into(),
                b: "x3".into(),
                value: 0.6,
            },
            CorrelationPair {
                a: "x2".into(),
                b: "x3".into(),
                value: -0.6,
            },
        ],
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let err = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect_err("non-PSD combined partial-upload correlation matrix must be rejected");
    assert!(
        matches!(err, SpecError::CorrelationNotPsd),
        "expected CorrelationNotPsd, got: {err:?}"
    );
}

/// Extract the full k×k correlation matrix from the installed `Correlations::Matrix`
/// as a flat row-major Vec (m[i][j] at index i*k+j). Panics if not Matrix.
fn extract_matrix(contracts: &[engine_contract::SimulationContract]) -> (Vec<f64>, usize) {
    match &contracts[0].generation.correlations {
        engine_contract::Correlations::Matrix { values, continuous_columns } => {
            let k = continuous_columns.len();
            // values is stored with values[j*k+i] = values[i*k+j] = r for i<j
            // Return as-is (flat col-major == flat row-major for symmetric matrices).
            (values.clone(), k)
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

/// Compute Spearman rank correlation by ranking then calling Pearson on centered ranks.
/// Mirrors the `spearman_to_latent` implementation in upload.rs (rank-then-Pearson).
fn spearman_ref(a: &[f64], b: &[f64]) -> f64 {
    fn rank(v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j + 1 < n && v[idx[j]] == v[idx[j + 1]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    }
    fn pearson_centered(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma: f64 = a.iter().sum::<f64>() / n;
        let mb: f64 = b.iter().sum::<f64>() / n;
        let ca: Vec<f64> = a.iter().map(|x| x - ma).collect();
        let cb: Vec<f64> = b.iter().map(|x| x - mb).collect();
        let num: f64 = ca.iter().zip(cb.iter()).map(|(x, y)| x * y).sum();
        let da: f64 = ca.iter().map(|x| x * x).sum::<f64>().sqrt();
        let db: f64 = cb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if da * db < 1e-15 {
            return 0.0;
        }
        num / (da * db)
    }
    let ra = rank(a);
    let rb = rank(b);
    pearson_centered(&ra, &rb)
}

/// Non-trivial Spearman correlation (x1,x2) ≈ 0.85 installed as latent Gaussian,
/// plus binary column b has zero off-diagonal slots in the installed matrix.
///
/// Validates that NORTA installs the frame's empirical Spearman r (converted to
/// the latent Gaussian scale via `2·sin(π·ρ_S/6)`) for continuous pairs, while
/// binary columns have 0 in their off-diagonal slots — generated from their
/// marginal only, independent of all other predictors.
#[test]
fn partial_mode_measures_nontrivial_and_excludes_binary() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    let x1 = vec![-1.2, -0.5, 0.1, 0.4, 0.9, 1.3, -0.8, 0.2, 0.6, -0.1];
    let x2 = vec![-0.9, -0.7, 0.3, 0.1, 1.1, 0.8, -0.2, 0.5, 0.2, -0.4];
    let b = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 10,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: x1.clone(),
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: x2.clone(),
                labels: vec![],
            },
            UploadColumn {
                name: "b".into(),
                col_type: UploadColumnType::Binary,
                values: b,
                labels: vec![],
            },
        ],
    };
    let spec = LinearSpec {
        formula: "y = x1 + x2 + b".into(),
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
            PredictorSpec {
                name: "b".into(),
                pinned: false,
                kind: VarKind::Binary { proportion: 0.5 },
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
            EffectAssignment {
                name: "b".into(),
                size: 0.2,
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let (vals, k) = extract_matrix(&contracts);
    assert_eq!(k, 3, "3 non-factor predictors → 3×3 matrix");

    // Column ordering: x1=0, x2=1, b=2 (non-factor declaration order).
    // values is symmetric; (i,j) entry at vals[i*k+j] == vals[j*k+i].
    let r_installed = vals[0 * k + 1]; // (x1, x2) off-diagonal

    let rho_s = spearman_ref(&x1, &x2);
    let r_expected = 2.0 * (std::f64::consts::PI * rho_s / 6.0).sin();
    assert!(
        (r_installed - r_expected).abs() < 1e-9,
        "installed latent r {r_installed} != Spearman-to-latent {r_expected} (rho_S={rho_s})"
    );

    // Binary column b occupies index 2 — its off-diagonal slots must be 0.
    let r_x1b = vals[0 * k + 2]; // (x1, b)
    let r_x2b = vals[1 * k + 2]; // (x2, b)
    assert!(
        r_x1b.abs() < 1e-9,
        "binary column slot (x1,b) must be 0, got {r_x1b}"
    );
    assert!(
        r_x2b.abs() < 1e-9,
        "binary column slot (x2,b) must be 0, got {r_x2b}"
    );
}

// ── P2-3 strict (bootstrap) signal tests ──────────────────────────────────────

/// Strict mode must now BUILD a contract (no longer rejected) and the produced
/// contract's `uploaded_frame.bootstrap` must be `true`.
#[test]
fn strict_mode_sets_bootstrap_flag_on_uploaded_frame() {
    use engine_contract::{EstimatorSpec, OutcomeKind};

    let upload = UploadInput {
        mode: UploadMode::Strict,
        n_rows: 5,
        columns: vec![UploadColumn {
            name: "x1".into(),
            col_type: UploadColumnType::Continuous,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            labels: vec![],
        }],
    };
    let spec = make_two_predictor_spec_with_upload(upload);
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("strict mode must build a contract, not error");
    let c = &contracts[0];
    let frame = c
        .generation
        .uploaded_frame
        .as_ref()
        .expect("uploaded_frame Some");
    assert!(frame.bootstrap, "strict mode must set bootstrap = true");
    c.validate().expect("strict contract validates");
}

/// Partial mode must leave `uploaded_frame.bootstrap == false`.
#[test]
fn partial_mode_leaves_bootstrap_flag_false() {
    use engine_contract::{EstimatorSpec, OutcomeKind};

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![UploadColumn {
            name: "x1".into(),
            col_type: UploadColumnType::Continuous,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            labels: vec![],
        }],
    };
    let spec = make_two_predictor_spec_with_upload(upload);
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");
    let c = &contracts[0];
    let frame = c
        .generation
        .uploaded_frame
        .as_ref()
        .expect("uploaded_frame Some");
    assert!(
        !frame.bootstrap,
        "partial mode must leave bootstrap = false"
    );
}

#[test]
fn upload_mode_none_does_not_measure_correlations() {
    use engine_contract::{Correlations, EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    // Two perfectly-correlated continuous columns (same data pattern as the
    // partial-mode correlation test, but with UploadMode::None).
    let upload = UploadInput {
        mode: UploadMode::None,
        n_rows: 5,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![2.0, 4.0, 6.0, 8.0, 10.0],
                labels: vec![],
            },
        ],
    };
    // No explicit spec.correlations — none mode must not install any measured r.
    let spec = LinearSpec {
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
        upload: Some(upload),
        cluster_level_vars: vec![],
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");
    let c = &contracts[0];

    // With UploadMode::None and no explicit correlations, the builder must
    // leave correlations as Identity — the empirical r≈1.0 must NOT be installed.
    assert!(
        matches!(c.generation.correlations, Correlations::Identity),
        "UploadMode::None must yield Correlations::Identity, got: {:?}",
        c.generation.correlations
    );
    // mode=None must still match and store the uploaded frame (frame matching
    // runs for both None and Partial; only correlation measurement is skipped).
    let frame = c
        .generation
        .uploaded_frame
        .as_ref()
        .expect("UploadMode::None with matched columns must produce uploaded_frame");
    assert_eq!(frame.n_cols, 2, "both matched continuous columns in frame");
    // None mode is not a bootstrap source — only Strict whole-row resampling sets it.
    assert!(!frame.bootstrap, "UploadMode::None must leave bootstrap = false");
}
