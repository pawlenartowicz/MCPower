use engine_spec_builder::{
    LinearSpec, SpecError, UploadColumn, UploadColumnType, UploadInput, UploadMode,
};

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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    }
}

/// A predictor bound to an uploaded column cannot also be cluster-level: the
/// engine resamples it from the frame. A synthesized cluster-level predictor
/// next to the upload still builds.
#[test]
fn uploaded_predictor_cannot_be_cluster_level() {
    use engine_contract::{EstimatorSpec, OutcomeKind};

    let upload = || UploadInput {
        mode: UploadMode::Partial,
        n_rows: 5,
        columns: vec![UploadColumn {
            name: "x1".into(),
            col_type: UploadColumnType::Continuous,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            labels: vec![],
        }],
    };
    let build = |spec: &LinearSpec| {
        engine_spec_builder::build_contract(
            spec,
            OutcomeKind::Continuous,
            None,
            Some(EstimatorSpec::Ols),
            0.0,
            vec![],
        )
    };

    let mut spec = make_two_predictor_spec_with_upload(upload());
    spec.cluster_level_vars = vec!["x1".into()];
    let err = build(&spec).unwrap_err();
    assert!(
        matches!(err, SpecError::ClusterLevelUploaded { ref name } if name == "x1"),
        "got {err:?}"
    );

    let mut spec = make_two_predictor_spec_with_upload(upload());
    spec.cluster_level_vars = vec!["x2".into()];
    build(&spec).expect("a synthesized cluster-level predictor builds");
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
        None,
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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

/// The installed off-diagonal is the **Spearman** rank correlation converted to
/// the latent-Gaussian scale (`2·sin(π·ρ_S/6)`), not the raw Pearson r.
///
/// x1=[1,2,3,4,5], x2=[1,2,3,5,4] (last two swapped) → ρ_S = 0.9, and the raw
/// Pearson of these same-valued columns is *also* 0.9. The latent conversion
/// inflates it to `2·sin(π·0.9/6) = 0.9079809994790935`, so a result of 0.9
/// would mean the conversion was skipped. Rank-based measurement itself is
/// pinned by `partial_mode_measures_nontrivial_and_excludes_binary`, whose data
/// has Pearson r ≠ ρ_S.
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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

    // Perfectly correlated columns (measured ρ = 1), but the user explicitly sets r=0.1
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let err = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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
        engine_contract::Correlations::Matrix {
            values,
            continuous_columns,
        } => {
            let k = continuous_columns.len();
            // values is stored with values[j*k+i] = values[i*k+j] = r for i<j
            // Return as-is (flat col-major == flat row-major for symmetric matrices).
            (values.clone(), k)
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

/// Non-trivial Spearman correlation (x1,x2) installed as latent Gaussian, plus a
/// binary column b with zero off-diagonal slots in the installed matrix.
///
/// Validates that NORTA installs the frame's empirical Spearman ρ_S (converted to
/// the latent Gaussian scale via `2·sin(π·ρ_S/6)`) for continuous pairs, while
/// binary columns have 0 in their off-diagonal slots: correlation is
/// continuous-only by design, so b is generated from its marginal only, even
/// though it tracks x1 in the data (b = 1 exactly when x1 > 0, ρ_S(x1,b) ≈ 0.85).
#[test]
fn partial_mode_measures_nontrivial_and_excludes_binary() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    let x1 = vec![-1.2, -0.5, 0.1, 0.4, 0.9, 1.3, -0.8, 0.2, 0.6, -0.1];
    let x2 = vec![-0.9, -0.7, 0.3, 0.1, 1.1, 0.8, -0.2, 0.5, 0.2, -0.4];
    let b = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let upload = UploadInput {
        mode: UploadMode::Partial,
        n_rows: 10,
        columns: vec![
            UploadColumn {
                name: "x1".into(),
                col_type: UploadColumnType::Continuous,
                values: x1,
                labels: vec![],
            },
            UploadColumn {
                name: "x2".into(),
                col_type: UploadColumnType::Continuous,
                values: x2,
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
        Some(EstimatorSpec::Ols),
        0.0,
        vec![],
    )
    .expect("build_contract");

    let (vals, k) = extract_matrix(&contracts);
    assert_eq!(k, 3, "3 non-factor predictors → 3×3 matrix");

    // Column ordering: x1=0, x2=1, b=2 (non-factor declaration order).
    // ρ_S(x1,x2) = 47/55 (R `cor(x1, x2, method = "spearman")`), so the latent
    // entry is 2·sin(π·(47/55)/6) = 0.8653160468930163 (computed in R). The
    // binary column's slots stay 0, and the PSD block is installed unshrunk.
    let r = 0.8653160468930163;
    #[rustfmt::skip]
    let expected = [
        1.0, r,   0.0,
        r,   1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    for (i, (got, want)) in vals.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "entry ({}, {}) = {got}, expected {want}; matrix {vals:?}",
            i / k,
            i % k
        );
    }
}

// ── P2-3 strict (bootstrap) signal tests ──────────────────────────────────────

/// `uploaded_frame.bootstrap` follows the upload mode: Strict resamples
/// (bootstrap = true), Partial does not (bootstrap = false). Both modes must
/// BUILD a contract — Strict is no longer rejected. Parametrized over the two
/// modes: identical setup, only the mode and expected flag differ.
#[test]
fn upload_mode_sets_bootstrap_flag() {
    use engine_contract::{EstimatorSpec, OutcomeKind};

    for (mode, expect_bootstrap) in [(UploadMode::Strict, true), (UploadMode::Partial, false)] {
        let upload = UploadInput {
            mode,
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
            None,
            Some(EstimatorSpec::Ols),
            0.0,
            vec![],
        )
        .expect("both modes must build a contract, not error");
        let c = &contracts[0];
        let frame = c
            .generation
            .uploaded_frame
            .as_ref()
            .expect("uploaded_frame Some");
        assert_eq!(
            frame.bootstrap, expect_bootstrap,
            "bootstrap flag must be {expect_bootstrap} for this mode"
        );
        c.validate().expect("contract validates");
    }
}

#[test]
fn upload_mode_none_does_not_measure_correlations() {
    use engine_contract::{Correlations, EstimatorSpec, OutcomeKind};
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, PredictorSpec, ResidualSpec, VarKind,
    };

    // Two perfectly-correlated continuous columns, uploaded with UploadMode::None.
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
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    let contracts = engine_spec_builder::build_contract(
        &spec,
        OutcomeKind::Continuous,
        None,
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
    assert!(
        !frame.bootstrap,
        "UploadMode::None must leave bootstrap = false"
    );
}
