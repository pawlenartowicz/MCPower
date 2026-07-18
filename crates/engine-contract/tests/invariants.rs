//! Integration tests for `SimulationContract::validate`. One test per
//! numbered invariant enforced by the contract.

use engine_contract::fixtures::example1_simple_ols;
use engine_contract::*;

fn baseline() -> SimulationContract {
    example1_simple_ols()
}

fn some_valid_cluster_spec() -> ClusterSpec {
    ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 10 },
        tau_squared: 0.1,
        slopes: vec![],
        extra_groupings: vec![],
    }
}

#[test]
fn invariant_1_coefficient_length() {
    let mut c = baseline();
    c.outcome.coefficients.pop();
    assert!(matches!(
        c.validate(),
        Err(ContractError::CoefficientLengthMismatch {
            coeffs: 2,
            terms: 3
        })
    ));
    assert!(baseline().validate().is_ok());
}

#[test]
fn invariant_2_test_target_indices_in_range() {
    let mut c = baseline();
    c.test.targets.push(TestTarget::Marginal { term: 99 });
    assert!(matches!(
        c.validate(),
        Err(ContractError::TestTargetTermOutOfRange { term: 99, .. })
    ));
}

#[test]
fn invariant_3_test_target_well_formed_rejects_empty_targets() {
    let mut c = baseline();
    c.test.targets.clear();
    assert!(matches!(
        c.validate(),
        Err(ContractError::InvalidTestSpec(_))
    ));
}

#[test]
fn invariant_3_test_target_well_formed_rejects_duplicate_marginal() {
    let mut c = baseline();
    c.test.targets = vec![
        TestTarget::Marginal { term: 1 },
        TestTarget::Marginal { term: 1 },
    ];
    assert!(matches!(
        c.validate(),
        Err(ContractError::InvalidTestSpec(_))
    ));
}

#[test]
fn invariant_3_test_target_well_formed_rejects_short_joint() {
    let mut c = baseline();
    c.test.targets = vec![TestTarget::Joint { terms: vec![1] }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::InvalidTestSpec(_))
    ));
}

#[test]
fn invariant_3_contrast_rejects_equal_positive_negative() {
    let mut c = baseline();
    // positive == negative must be rejected (positive: 0, negative: 0)
    c.test.targets = vec![TestTarget::Contrast {
        positive: 0,
        negative: 0,
    }];
    assert!(
        matches!(c.validate(), Err(ContractError::InvalidTestSpec(_))),
        "Contrast with positive == negative must fail validation"
    );
}

#[test]
fn invariant_2_contrast_indices_in_range() {
    let mut c = baseline();
    let n_terms = c.design_generation.terms.len() as u32;
    // positive out of range
    c.test.targets = vec![TestTarget::Contrast {
        positive: n_terms,
        negative: 0,
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::TestTargetTermOutOfRange { .. })
    ));
    // negative out of range
    c.test.targets = vec![TestTarget::Contrast {
        positive: 0,
        negative: n_terms,
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::TestTargetTermOutOfRange { .. })
    ));
}

#[test]
fn invariant_3_contrast_duplicate_reversed_pair_rejected() {
    let mut c = baseline();
    // Contrast{1,2} and Contrast{2,1} are the same canonical pair (min=1, max=2).
    // The second one must be rejected as a duplicate.
    c.test.targets = vec![
        TestTarget::Contrast {
            positive: 1,
            negative: 2,
        },
        TestTarget::Contrast {
            positive: 2,
            negative: 1,
        },
    ];
    assert!(
        matches!(c.validate(), Err(ContractError::InvalidTestSpec(_))),
        "reversed duplicate contrast pair must fail validation"
    );
}

#[test]
fn invariant_3_contrast_valid_pair_validates_ok() {
    let mut c = baseline();
    // example1_simple_ols has design_generation: [Const(0), Direct{0}(1), Direct{1}(2)]
    // design_test is None so invariant_02 uses design_generation
    c.test.targets = vec![TestTarget::Contrast {
        positive: 1,
        negative: 2,
    }];
    assert!(c.validate().is_ok(), "valid contrast must pass validation");
}

#[test]
fn invariant_4_column_id_out_of_range() {
    let mut c = baseline();
    c.design_generation.terms.push(DesignTerm::Direct {
        column: ColumnId(99),
    });
    c.outcome.coefficients.push(0.1); // keep invariant 1 happy
    assert!(matches!(
        c.validate(),
        Err(ContractError::ColumnIdOutOfRange { id: 99, .. })
    ));
}

#[test]
fn invariant_4_cluster_level_column_out_of_range() {
    let mut c = baseline();
    // baseline() (example1_simple_ols) has 2 columns (indices 0 and 1).
    // ColumnId(99) is out of range; validate() must catch it before
    // expand_cluster_level_columns can panic at the unchecked index in
    // contract_adapter.rs:expand_cluster_level_columns.
    c.generation.cluster_level_columns = vec![ColumnId(99)];
    assert!(matches!(
        c.validate(),
        Err(ContractError::ColumnIdOutOfRange { id: 99, .. })
    ));
}

#[test]
fn invariant_5_correlation_dimension_mismatch() {
    let mut c = baseline();
    c.generation.correlations = Correlations::Matrix {
        continuous_columns: vec![ColumnId(0), ColumnId(1)],
        values: vec![1.0, 0.3, 0.3],
    };
    assert!(matches!(
        c.validate(),
        Err(ContractError::CorrelationDimensionMismatch { .. })
    ));
}

#[test]
fn invariant_6_correlation_only_continuous() {
    let mut c = baseline();
    // Make column 1 a factor.
    c.generation.columns[1] = ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: vec![0.5, 0.5],
        sampled_proportions: None,
    };
    // The original design has `Direct { ColumnId(1) }` which would also fail
    // invariant 9; replace it with a DummyOf so we isolate invariant 6.
    c.design_generation.terms[2] = DesignTerm::DummyOf {
        column: ColumnId(1),
        level_index: 1,
    };
    c.generation.correlations = Correlations::Matrix {
        continuous_columns: vec![ColumnId(0), ColumnId(1)],
        values: vec![1.0, 0.3, 0.3, 1.0],
    };
    assert!(matches!(
        c.validate(),
        Err(ContractError::CorrelationOnNonContinuous { id: 1 })
    ));
}

#[test]
fn invariant_7_factor_proportions_length_mismatch() {
    let mut c = baseline();
    c.generation.columns.push(ColumnSpec::FactorSynthetic {
        n_levels: 3,
        proportions: vec![0.5, 0.5],
        sampled_proportions: None,
    });
    assert!(matches!(
        c.validate(),
        Err(ContractError::InvalidFactorProportions(_))
    ));
}

#[test]
fn invariant_7_factor_proportions_must_sum_to_one() {
    let mut c = baseline();
    // proportions = [0.3, 0.3] → sum = 0.6 ≠ 1.0; n_levels matches len.
    c.generation.columns.push(ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: vec![0.3, 0.3],
        sampled_proportions: None,
    });
    assert!(
        matches!(
            c.validate(),
            Err(ContractError::InvalidFactorProportions(_))
        ),
        "proportions summing to 0.6 must be rejected"
    );
}

#[test]
fn invariant_8_dummy_level_out_of_range() {
    let mut c = baseline();
    c.generation.columns[1] = ColumnSpec::FactorSynthetic {
        n_levels: 3,
        proportions: vec![0.5, 0.3, 0.2],
        sampled_proportions: None,
    };
    c.design_generation.terms[2] = DesignTerm::DummyOf {
        column: ColumnId(1),
        level_index: 3,
    };
    assert!(matches!(
        c.validate(),
        Err(ContractError::DummyLevelOutOfRange { level_index: 3, .. })
    ));
}

#[test]
fn invariant_9_direct_on_factor_rejected() {
    let mut c = baseline();
    c.generation.columns[1] = ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: vec![0.5, 0.5],
        sampled_proportions: None,
    };
    // design_generation.terms[2] is `Direct { column: 1 }` — now illegal.
    assert!(matches!(
        c.validate(),
        Err(ContractError::DirectOnFactor { id: 1 })
    ));
}

#[test]
fn invariant_10_heteroskedasticity_on_factor_rejected() {
    let mut c = baseline();
    c.generation.columns[0] = ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: vec![0.5, 0.5],
        sampled_proportions: None,
    };
    c.design_generation.terms[1] = DesignTerm::DummyOf {
        column: ColumnId(0),
        level_index: 1,
    };
    c.outcome.heteroskedasticity_driver = Some(ColumnId(0));
    assert!(matches!(
        c.validate(),
        Err(ContractError::HeteroskedasticityOnFactor { id: 0 })
    ));
}

#[test]
fn invariant_11_resampled_without_uploaded_frame() {
    let mut c = baseline();
    c.generation.columns[0] = ColumnSpec::Resampled { frame_column: 0 };
    assert!(matches!(
        c.validate(),
        Err(ContractError::UploadedFrameMissing { frame_column: 0 })
    ));
}

// Second branch of invariant 11: the frame is PRESENT but too narrow — a Resampled
// column references frame_column 1 while the uploaded frame has only 1 column (valid
// index 0). The frame-absent test above never exercises the `max_col >= n_cols` arm.
#[test]
fn invariant_11_resampled_frame_column_out_of_range() {
    let mut c = baseline();
    c.generation.columns[0] = ColumnSpec::Resampled { frame_column: 1 };
    c.generation.uploaded_frame = Some(UploadedFrame {
        data: vec![1.0, 2.0],
        n_rows: 2,
        n_cols: 1,
        bootstrap: false,
    });
    assert!(matches!(
        c.validate(),
        Err(ContractError::UploadedFrameMissing { frame_column: 1 })
    ));
}

#[test]
fn invariant_13_lme_scenario_requires_mle() {
    let mut c = baseline();
    // Add a minimal LmeScenarioPerturbations while keeping estimator as Ols (non-Mle).
    c.scenario.lme = Some(LmeScenarioPerturbations {
        random_effect_dist: ResidualDist::Normal,
        random_effect_df: 5.0,
        icc_noise_sd: 0.05,
    });
    assert!(matches!(
        c.validate(),
        Err(ContractError::LmeScenarioRequiresMle)
    ));
}

#[test]
fn validate_template_skips_invariant_13_but_full_validate_fires_it() {
    let mut c = baseline();
    c.scenario.lme = Some(LmeScenarioPerturbations {
        random_effect_dist: ResidualDist::Normal,
        random_effect_df: 5.0,
        icc_noise_sd: 0.05,
    });
    // Full validate() must still reject Ols + lme.
    assert!(matches!(
        c.validate(),
        Err(ContractError::LmeScenarioRequiresMle)
    ));
    // validate_template() must accept the same contract (invariant_13 skipped).
    assert!(
        c.validate_template().is_ok(),
        "validate_template() must not fire invariant_13 for an Ols+lme template"
    );
}

#[test]
fn invariant_13_admits_glmm_scenario() {
    // Glm + Binary + cluster = Some is the GLMM pairing (M4). scenario.lme
    // must be accepted because the GLMM kernel draws per-block τ² from it,
    // exactly like LMM (Mle). The old guard only allowed Mle; this test
    // pins the new two-armed guard (mle || glmm).
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    c.generation.cluster = Some(some_valid_cluster_spec());
    c.scenario.lme = Some(LmeScenarioPerturbations {
        random_effect_dist: ResidualDist::Normal,
        random_effect_df: 5.0,
        icc_noise_sd: 0.05,
    });
    assert!(
        c.validate().is_ok(),
        "Glm + Binary + cluster + scenario.lme must be accepted (GLMM path)"
    );
}

#[test]
fn invariant_14_max_failed_fraction_out_of_range() {
    // Reject every value outside the closed [0.0, 1.0] interval, plus NaN
    // (the catalogue names the NaN branch explicitly).
    for bad in [1.5_f64, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut c = baseline();
        c.max_failed_fraction = bad;
        assert!(
            matches!(
                c.validate(),
                Err(ContractError::MaxFailedFractionOutOfRange { .. })
            ),
            "max_failed_fraction = {bad} must be rejected"
        );
    }
    // Boundaries are inclusive: 0.0 and 1.0 must pass.
    for ok in [0.0_f64, 1.0] {
        let mut c = baseline();
        c.max_failed_fraction = ok;
        assert!(
            c.validate().is_ok(),
            "max_failed_fraction = {ok} must be accepted"
        );
    }
}

#[test]
fn invariant_15_alpha_out_of_range() {
    // alpha must be strictly inside the open interval (0.0, 1.0): both ends
    // rejected, plus values beyond each end and NaN.
    for bad in [0.0_f64, 1.0, -0.01, 1.01, f64::NAN] {
        let mut c = baseline();
        c.test.alpha = bad;
        assert!(
            matches!(c.validate(), Err(ContractError::AlphaOutOfRange { .. })),
            "alpha = {bad} must be rejected"
        );
    }
    // An interior value must still pass.
    let mut c = baseline();
    c.test.alpha = 0.5;
    assert!(c.validate().is_ok(), "alpha = 0.5 must be accepted");
}

#[test]
fn invariant_16_non_psd_correlation_matrix_rejected() {
    let mut c = baseline();
    // 2×2 matrix with off-diagonal 1.5: det = 1 - 2.25 = -1.25 < 0 → not PSD.
    // baseline has 2 continuous columns at indices 0 and 1.
    c.generation.correlations = Correlations::Matrix {
        continuous_columns: vec![ColumnId(0), ColumnId(1)],
        values: vec![1.0, 1.5, 1.5, 1.0],
    };
    assert!(
        matches!(c.validate(), Err(ContractError::CorrelationNotPsd)),
        "non-PSD correlation matrix must be rejected by validate()"
    );
}

#[test]
fn invariant_17_posthoc_requires_factor_column() {
    let mut c = baseline();
    c.posthoc = vec![PosthocSpec {
        factor_column: ColumnId(0), // continuous
        target_term_indices: vec![1],
        posthoc_alpha: None,
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::InvalidPosthoc(_))
    ));
}

#[test]
fn invariant_7_factor_with_fewer_than_two_levels_rejected() {
    // Exercises the `n_levels < 2` branch of invariant 07 (distinct from the
    // length-mismatch branch). proportions.len() matches n_levels so
    // the length check cannot fire first — the n_levels guard is what triggers.
    let mut c = baseline();
    c.generation.columns.push(ColumnSpec::FactorSynthetic {
        n_levels: 1,
        proportions: vec![1.0],
        sampled_proportions: None,
    });
    match c.validate() {
        Err(ContractError::InvalidFactorProportions(msg)) => {
            assert_eq!(
                msg, "n_levels must be >= 2",
                "wrong InvalidFactorProportions branch"
            );
        }
        other => panic!("expected InvalidFactorProportions(n_levels), got {other:?}"),
    }
}

#[test]
fn invariant_17_posthoc_target_must_be_dummy_of_factor() {
    // The factor_column is a valid factor (17a passes) and the estimator is Ols
    // (17c passes), but a posthoc target_term_index points at a non-DummyOf term
    // (the Const at index 0). The 17b branch must fire.
    let mut c = baseline();
    c.generation.columns[1] = ColumnSpec::FactorSynthetic {
        n_levels: 3,
        proportions: vec![0.5, 0.3, 0.2],
        sampled_proportions: None,
    };
    c.design_generation.terms = vec![
        DesignTerm::Const,
        DesignTerm::Direct {
            column: ColumnId(0),
        },
        DesignTerm::DummyOf {
            column: ColumnId(1),
            level_index: 1,
        },
        DesignTerm::DummyOf {
            column: ColumnId(1),
            level_index: 2,
        },
    ];
    c.outcome.coefficients = vec![0.0, 0.5, 0.3, 0.2];
    c.test.targets = vec![TestTarget::Marginal { term: 1 }];
    // factor_column is the factor (17a ok); target_term_indices[0] = 0 = Const (not
    // a DummyOf of column 1) → 17b fires before the Ols 17c guard.
    c.posthoc = vec![PosthocSpec {
        factor_column: ColumnId(1),
        target_term_indices: vec![0],
        posthoc_alpha: None,
    }];
    match c.validate() {
        Err(ContractError::InvalidPosthoc(msg)) => {
            assert_eq!(msg, "posthoc target must be a DummyOf of factor_column");
        }
        other => panic!("expected InvalidPosthoc(17b), got {other:?}"),
    }
}

#[test]
fn example2_logit_with_uploaded_data_validates_cleanly() {
    let c = SimulationContract {
        generation: GenerationSpec {
            columns: vec![
                ColumnSpec::Resampled { frame_column: 0 },
                ColumnSpec::FactorFromFrame {
                    frame_column: 1,
                    n_levels: 3,
                    proportions: vec![0.45, 0.35, 0.20],
                    sampled_proportions: None,
                },
            ],
            correlations: Correlations::Identity,
            cluster: None,
            uploaded_frame: Some(UploadedFrame {
                data: vec![1.0, 2.0, 3.0, 4.0],
                n_rows: 2,
                n_cols: 2,
                bootstrap: false,
            }),
            cluster_level_columns: vec![],
        },
        design_generation: DesignSpec {
            terms: vec![
                DesignTerm::Const,
                DesignTerm::Direct {
                    column: ColumnId(0),
                },
                DesignTerm::DummyOf {
                    column: ColumnId(1),
                    level_index: 1,
                },
                DesignTerm::DummyOf {
                    column: ColumnId(1),
                    level_index: 2,
                },
            ],
        },
        outcome: OutcomeSpec {
            kind: OutcomeKind::Binary,
            intercept: -1.3863,
            coefficients: vec![-1.3863, 0.4, 0.2, 0.5],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: false,
            },
            heteroskedasticity_driver: None,
            link: None,
        },
        design_test: None,
        estimator: EstimatorSpec::Glm,
        wald_se: Default::default(),
        nagq: 1,
        test: TestSpec {
            targets: vec![
                TestTarget::Marginal { term: 1 },
                TestTarget::Marginal { term: 2 },
                TestTarget::Marginal { term: 3 },
            ],
            correction: CorrectionMethod::None,
            alpha: 0.05,
        },
        posthoc: vec![],
        scenario: ScenarioPerturbations::default(),
        max_failed_fraction: 0.03,
    };
    assert!(c.validate().is_ok(), "Example 2 must validate cleanly");
    // Confirm the validated contract is the intended logit+upload shape, not an
    // accidental Ok on a degenerate build (was: is_ok only).
    assert!(
        c.generation.uploaded_frame.is_some(),
        "example 2 carries an uploaded frame"
    );
    assert_eq!(c.test.targets.len(), 3, "three marginal targets");
}

// --- estimator×outcome validity matrix tests ---

#[test]
fn glm_requires_binary_outcome() {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Glm; // outcome.kind stays Continuous
    assert!(matches!(
        c.validate(),
        Err(ContractError::GlmRequiresBinaryOrCount)
    ));
}

#[test]
fn glm_with_binary_is_ok() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    assert!(c.validate().is_ok());
}

#[test]
fn glm_with_count_is_ok() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Count;
    c.estimator = EstimatorSpec::Glm;
    assert!(c.validate().is_ok());
}

// --- invariant 24: link matches kind ---

#[test]
fn probit_on_binary_is_ok() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.outcome.link = Some(LinkKind::Probit);
    c.estimator = EstimatorSpec::Glm;
    assert!(c.validate().is_ok());
}

#[test]
fn probit_on_non_binary_rejected() {
    for kind in [OutcomeKind::Continuous, OutcomeKind::Count] {
        let mut c = example1_simple_ols();
        c.outcome.kind = kind;
        c.outcome.link = Some(LinkKind::Probit);
        c.estimator = if kind == OutcomeKind::Count {
            EstimatorSpec::Glm
        } else {
            EstimatorSpec::Ols
        };
        assert!(
            matches!(c.validate(), Err(ContractError::ProbitRequiresBinary)),
            "probit on {kind:?} must be rejected"
        );
    }
}

// --- invariant 25: nagq backstop ---

#[test]
fn nagq_even_or_out_of_range_rejected() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    c.generation.cluster = Some(some_valid_cluster_spec());
    for bad in [0u8, 2, 4, 26, 100] {
        c.nagq = bad;
        assert!(
            matches!(c.validate(), Err(ContractError::NagqOutOfRange { got }) if got == bad),
            "nagq = {bad} must be rejected as out-of-range"
        );
    }
}

#[test]
fn nagq_gt1_on_eligible_binary_glmm_is_ok() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    c.generation.cluster = Some(some_valid_cluster_spec()); // single grouping, intercept-only (q=1)
    c.nagq = 7;
    assert!(
        c.validate().is_ok(),
        "eligible clustered-binary AGQ must validate"
    );
}

#[test]
fn nagq_upper_bound_25_eligible_is_ok() {
    // 25 is the top of the odd `1..=25` range; an off-by-one such as
    // `k >= 25` would wrongly reject the boundary value itself.
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    c.generation.cluster = Some(some_valid_cluster_spec());
    c.nagq = 25;
    assert!(
        c.validate().is_ok(),
        "nagq = 25 on an eligible shape must validate"
    );
}

#[test]
fn nagq_three_res_eligible_is_ok() {
    // Intercept + 2 slopes = 3 REs, the max allowed; an off-by-one such as
    // `re_count < 3` would wrongly reject this shape.
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    let mut cl = some_valid_cluster_spec();
    for (i, col) in [0u32, 1].into_iter().enumerate() {
        cl.slopes.push(SlopeTerm {
            column: ColumnId(col),
            variance: 0.2,
            corr_with_intercept: 0.0,
            corr_with: vec![0.0; i],
        });
    }
    c.generation.cluster = Some(cl);
    c.nagq = 3;
    assert!(
        c.validate().is_ok(),
        "3 REs (intercept + 2 slopes) with nagq = 3 must validate"
    );
}

#[test]
fn nagq_gt1_ineligible_axes_rejected() {
    // Unclustered: no random effects to integrate over.
    let mut unclustered = example1_simple_ols();
    unclustered.outcome.kind = OutcomeKind::Binary;
    unclustered.estimator = EstimatorSpec::Glm;
    unclustered.nagq = 7;
    assert!(matches!(
        unclustered.validate(),
        Err(ContractError::NagqIneligibleShape { got: 7 })
    ));

    // Continuous outcome (LMM): AGQ is Binomial/Poisson-only.
    let mut gaussian = example1_simple_ols();
    gaussian.estimator = EstimatorSpec::Mle;
    gaussian.generation.cluster = Some(some_valid_cluster_spec());
    gaussian.nagq = 7;
    assert!(matches!(
        gaussian.validate(),
        Err(ContractError::NagqIneligibleShape { got: 7 })
    ));

    // Crossed/nested extra grouping: single-grouping-factor rule.
    let mut crossed = example1_simple_ols();
    crossed.outcome.kind = OutcomeKind::Count;
    crossed.estimator = EstimatorSpec::Glm;
    let mut cl = some_valid_cluster_spec();
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 4 },
        tau_squared: 0.1,
        slopes: vec![],
    }];
    crossed.generation.cluster = Some(cl);
    crossed.nagq = 7;
    assert!(matches!(
        crossed.validate(),
        Err(ContractError::NagqIneligibleShape { got: 7 })
    ));
}

#[test]
fn mle_requires_cluster() {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle; // generation.cluster is None
    assert!(matches!(
        c.validate(),
        Err(ContractError::MleWithoutCluster)
    ));
}

#[test]
fn ols_accepts_clustered_generation() {
    // THE new free pairing: Continuous + cluster=Some + Ols must validate.
    let mut c = example1_simple_ols();
    c.generation.cluster = Some(some_valid_cluster_spec());
    c.estimator = EstimatorSpec::Ols;
    assert!(c.validate().is_ok());
}

#[test]
fn invariant_19_primary_slopes_now_live_capability() {
    // Base: valid Mle + cluster contract — must pass before we mutate.
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(some_valid_cluster_spec());
    assert!(
        c.validate().is_ok(),
        "base Mle+cluster must validate cleanly"
    );

    // Primary slopes are now a live capability (invariant_21 validates structurally);
    // column 1 is a continuous Direct term and tau_squared > 0.
    c.generation.cluster.as_mut().unwrap().slopes = vec![SlopeTerm {
        column: ColumnId(1),
        variance: 0.4,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    assert!(c.validate().is_ok());
}

#[test]
fn invariant_19_extra_grouping_slopes_accepted() {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(some_valid_cluster_spec());
    assert!(
        c.validate().is_ok(),
        "base Mle+cluster must validate cleanly"
    );

    // Intercept-only extra groupings are legal…
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 12 },
        tau_squared: 0.1,
        slopes: vec![],
    }];
    assert!(c.validate().is_ok(), "intercept-only extras must validate");

    // …and a random slope on an extra grouping is now SUPPORTED (gated kernel path):
    // q_g = 2 ≤ MAX_EXTRA_Q, slope on continuous col 1 which is a Direct fixed effect.
    c.generation.cluster.as_mut().unwrap().extra_groupings[0].slopes = vec![SlopeTerm {
        column: ColumnId(1),
        variance: 0.4,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    assert!(
        c.validate().is_ok(),
        "an extra-grouping random slope on a valid column must validate"
    );
}

#[test]
fn extra_grouping_accepts_one_slope() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 6 },
        tau_squared: 0.1,
        slopes: vec![slope(1, 0.10, 0.0)], // q_g = 2, on continuous Direct col 1
    }];
    assert!(spec.validate().is_ok());
}

#[test]
fn extra_grouping_rejects_too_many_slopes() {
    // q_g = 5 (intercept + 4 slopes) exceeds MAX_EXTRA_Q = 4. The q_g bound fires
    // before any per-slope check, so the empty corr_with on later slopes is irrelevant.
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 6 },
        tau_squared: 0.1,
        slopes: vec![
            slope(0, 0.10, 0.0),
            slope(1, 0.10, 0.0),
            slope(0, 0.10, 0.0),
            slope(1, 0.10, 0.0),
        ],
    }];
    assert!(matches!(
        spec.validate(),
        Err(ContractError::ExtraGroupingQTooLarge { got: 5, max: 4 })
    ));
}

#[test]
fn extra_grouping_accepts_three_slopes_at_cap() {
    // q_g = 4 (intercept + 3 slopes) is exactly MAX_EXTRA_Q after the M1 bump.
    // corr_with lower triangle: slope[k] needs exactly k entries (mirrors the primary
    // slope validator — see check_slope_terms). All zeros → diagonal D_g (PSD trivially).
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 6 },
        tau_squared: 0.1,
        slopes: vec![
            slope(0, 0.10, 0.0), // k=0: corr_with=[]
            SlopeTerm {
                column: ColumnId(1),
                variance: 0.10,
                corr_with_intercept: 0.0,
                corr_with: vec![0.0],
            }, // k=1
            SlopeTerm {
                column: ColumnId(0),
                variance: 0.10,
                corr_with_intercept: 0.0,
                corr_with: vec![0.0, 0.0],
            }, // k=2
        ],
    }];
    assert!(spec.validate().is_ok());
}

fn mle_with_cluster(sizing: ClusterSizing) -> SimulationContract {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(ClusterSpec::intercept_only(sizing, 0.25));
    c
}

#[test]
fn invariant_20_crossed_requires_fixed_clusters() {
    let mut c = mle_with_cluster(ClusterSizing::FixedSize { cluster_size: 10 });
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 12 },
        tau_squared: 0.1,
        slopes: vec![],
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::CrossedRequiresFixedClusters)
    ));
}

#[test]
fn invariant_20_at_most_one_nested() {
    let mut c = mle_with_cluster(ClusterSizing::FixedClusters { n_clusters: 10 });
    let nested = GroupingSpec {
        relation: GroupingRelation::NestedWithin { n_per_parent: 3 },
        tau_squared: 0.1,
        slopes: vec![],
    };
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![nested.clone(), nested];
    assert!(matches!(
        c.validate(),
        Err(ContractError::MultipleNestedUnsupported)
    ));
}

#[test]
fn invariant_20_nested_divisibility_in_regime_b() {
    let mut c = mle_with_cluster(ClusterSizing::FixedSize { cluster_size: 10 });
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::NestedWithin { n_per_parent: 3 },
        tau_squared: 0.1,
        slopes: vec![],
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::NestedSizeIndivisible {
            cluster_size: 10,
            n_per_parent: 3
        })
    ));
    // Divisible passes.
    c.generation.cluster.as_mut().unwrap().sizing = ClusterSizing::FixedSize { cluster_size: 9 };
    assert!(c.validate().is_ok());
}

#[test]
fn invariant_20_grouping_bounds() {
    let mut c = mle_with_cluster(ClusterSizing::FixedClusters { n_clusters: 10 });
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 1 },
        tau_squared: 0.1,
        slopes: vec![],
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::GroupingCountTooSmall { got: 1 })
    ));

    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 12 },
        tau_squared: -0.1,
        slopes: vec![],
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::GroupingTauSquaredInvalid { .. })
    ));
}

#[test]
fn invariant_20_legal_crossed_plus_nested_accepted() {
    let mut c = mle_with_cluster(ClusterSizing::FixedClusters { n_clusters: 10 });
    c.generation.cluster.as_mut().unwrap().extra_groupings = vec![
        GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 12 },
            tau_squared: 0.1,
            slopes: vec![],
        },
        GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 3 },
            tau_squared: 0.05,
            slopes: vec![],
        },
    ];
    assert!(c.validate().is_ok());
}

#[test]
fn invariant_20_rejects_too_many_extra_groupings() {
    let mut c = mle_with_cluster(ClusterSizing::FixedClusters { n_clusters: 10 });
    // MAX_EXTRA_GROUPINGS + 1 crossed factors — one past the solver's stack
    // id-buffer ceiling, which would otherwise overflow `gid` at fit time.
    c.generation.cluster.as_mut().unwrap().extra_groupings = (0..=MAX_EXTRA_GROUPINGS)
        .map(|_| GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.1,
            slopes: vec![],
        })
        .collect();
    assert!(matches!(
        c.validate(),
        Err(ContractError::TooManyExtraGroupings { got, max })
            if got as usize == MAX_EXTRA_GROUPINGS + 1 && max as usize == MAX_EXTRA_GROUPINGS
    ));
}

#[test]
fn too_many_slopes_rejected() {
    // Build an MLE cluster spec with MAX_PRIMARY_Q slopes (q_p = 1 + MAX_PRIMARY_Q
    // > MAX_PRIMARY_Q), which would overflow the solver's stack-allocated θ
    // truth-start buffer if not caught. Each slope references its own distinct
    // continuous column (all Direct fixed effects, as invariant_21 requires).
    let mut c = mle_with_cluster(ClusterSizing::FixedClusters { n_clusters: 10 });
    let n_slopes = MAX_PRIMARY_Q; // q_p = 1 + n_slopes = MAX_PRIMARY_Q + 1 > MAX_PRIMARY_Q

    // Extend the contract with n_slopes additional continuous columns + Direct terms.
    let base_col = c.generation.columns.len() as u32;
    for i in 0..n_slopes {
        c.generation.columns.push(ColumnSpec::Synthetic {
            kind: SyntheticKind::Normal,
            pinned: false,
        });
        c.design_generation.terms.push(DesignTerm::Direct {
            column: ColumnId(base_col + i as u32),
        });
        c.outcome.coefficients.push(0.1);
        c.test.targets.push(TestTarget::Marginal {
            term: (c.design_generation.terms.len() - 1) as u32,
        });
    }

    // Build n_slopes slopes (corr_with grows: slope k has k earlier-slope entries).
    c.generation.cluster.as_mut().unwrap().slopes = (0..n_slopes)
        .map(|k| SlopeTerm {
            column: ColumnId(base_col + k as u32),
            variance: 0.1,
            corr_with_intercept: 0.0,
            corr_with: vec![0.0; k],
        })
        .collect();

    assert!(matches!(
        c.validate(),
        Err(ContractError::TooManySlopes { got, max })
            if got as usize == 1 + n_slopes && max as usize == MAX_PRIMARY_Q
    ));
}

#[test]
fn posthoc_rejected_for_non_ols() {
    let mut c = example1_simple_ols();
    c.outcome.kind = OutcomeKind::Binary;
    c.estimator = EstimatorSpec::Glm;
    // Set up a valid factor column so 17a/17b pass and 17c (PosthocRequiresOls) is reached.
    c.generation.columns[1] = ColumnSpec::FactorSynthetic {
        n_levels: 3,
        proportions: vec![0.5, 0.3, 0.2],
        sampled_proportions: None,
    };
    c.design_generation.terms[2] = DesignTerm::DummyOf {
        column: ColumnId(1),
        level_index: 1,
    };
    c.design_generation.terms.push(DesignTerm::DummyOf {
        column: ColumnId(1),
        level_index: 2,
    });
    c.outcome.coefficients.push(0.2);
    c.test.targets.push(TestTarget::Marginal { term: 3 });
    c.posthoc = vec![PosthocSpec {
        factor_column: ColumnId(1),
        target_term_indices: vec![2, 3],
        posthoc_alpha: None,
    }];
    assert!(matches!(
        c.validate(),
        Err(ContractError::PosthocRequiresOls)
    ));
}

// --- serde round-trip tests ---

#[test]
fn roundtrip_each_canonical_pairing_and_design_test_omission() {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    // linear (design_test None must serialise/deserialise as None)
    let c = example1_simple_ols();
    let bytes = rmp_serde::to_vec_named(&c).unwrap();
    let back: SimulationContract = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(c, back);
    assert!(back.design_test.is_none());

    // logistic
    let mut c2 = example1_simple_ols();
    c2.outcome.kind = OutcomeKind::Binary;
    c2.estimator = EstimatorSpec::Glm;
    let back2: SimulationContract =
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&c2).unwrap()).unwrap();
    assert_eq!(c2, back2);

    // clustered-OLS (the new free pairing)
    let mut c3 = example1_simple_ols();
    c3.generation.cluster = Some(some_valid_cluster_spec());
    c3.estimator = EstimatorSpec::Ols;
    let back3: SimulationContract =
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&c3).unwrap()).unwrap();
    assert_eq!(c3, back3);

    // misspecified design_test = Some(subset)
    let mut c4 = example1_simple_ols();
    c4.design_test = Some(c4.design_generation.clone());
    let back4: SimulationContract =
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&c4).unwrap()).unwrap();
    assert_eq!(c4, back4);
}

#[test]
fn enum_wire_strings_are_snake_case() {
    // All three EstimatorSpec tags must be stable wire strings, not just Glm.
    for (variant, tag) in [
        (EstimatorSpec::Ols, "\"ols\""),
        (EstimatorSpec::Glm, "\"glm\""),
        (EstimatorSpec::Mle, "\"mle\""),
    ] {
        assert_eq!(serde_json::to_string(&variant).unwrap(), tag);
    }
    let k = serde_json::to_string(&engine_contract::OutcomeKind::Continuous).unwrap();
    assert_eq!(k, "\"continuous\"");
}

/// Every `EstimatorSpec` variant's snake_case wire tag is stable and survives
/// a msgpack round-trip. A kernel that renamed or reordered the tags would fail.
#[test]
fn estimator_spec_all_snake_case_tags() {
    for (variant, tag) in [
        (EstimatorSpec::Ols, "ols"),
        (EstimatorSpec::Glm, "glm"),
        (EstimatorSpec::Mle, "mle"),
    ] {
        let bytes = rmp_serde::to_vec_named(&variant).unwrap();
        let back: EstimatorSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(variant, back, "{tag} must round-trip");
        let found = bytes.windows(tag.len()).any(|w| w == tag.as_bytes());
        assert!(found, "msgpack must contain tag {tag:?}; bytes={bytes:?}");
    }
}

/// Invariant 02 indexes into the *resolved* design spec — `design_test` when
/// `Some`, otherwise `design_generation`. A target index valid against the
/// (longer) generation design but out of range for a shorter `design_test` must
/// be rejected; the same index passes when `design_test` is `None`.
#[test]
fn design_test_overrides_generation_for_target_range() {
    let mut c = baseline();
    // baseline design_generation has 3 terms (indices 0,1,2). Target term 2 is
    // valid against generation.
    c.test.targets = vec![TestTarget::Marginal { term: 2 }];
    assert!(
        c.validate().is_ok(),
        "term 2 must be valid when resolved against design_generation (design_test = None)"
    );

    // Supply a shorter design_test (only 2 terms: indices 0,1). Now term 2 is
    // out of range and invariant 02 must index into design_test, not generation.
    c.design_test = Some(DesignSpec {
        terms: vec![
            DesignTerm::Const,
            DesignTerm::Direct {
                column: ColumnId(0),
            },
        ],
    });
    assert!(
        matches!(
            c.validate(),
            Err(ContractError::TestTargetTermOutOfRange {
                term: 2,
                n_terms: 2
            })
        ),
        "term 2 must be rejected once design_test resolves to 2 terms"
    );
}

// --- interaction component validation ---

#[test]
fn interaction_with_too_few_components_is_rejected() {
    let mut c = example1_simple_ols();
    c.design_generation.terms.push(DesignTerm::Interaction {
        components: vec![DesignTerm::Direct {
            column: ColumnId(0),
        }],
    });
    c.outcome.coefficients.push(0.0);
    let err = c.validate().unwrap_err();
    assert!(
        matches!(err, ContractError::InteractionTooFewComponents { n: 1 }),
        "got {err:?}"
    );
}

#[test]
fn interaction_with_const_component_is_rejected() {
    let mut c = example1_simple_ols();
    c.design_generation.terms.push(DesignTerm::Interaction {
        components: vec![
            DesignTerm::Direct {
                column: ColumnId(0),
            },
            DesignTerm::Const,
        ],
    });
    c.outcome.coefficients.push(0.0);
    let err = c.validate().unwrap_err();
    assert!(
        matches!(err, ContractError::InteractionBadComponent),
        "got {err:?}"
    );
}

#[test]
fn interaction_with_out_of_range_component_column_is_rejected() {
    let mut c = example1_simple_ols();
    c.design_generation.terms.push(DesignTerm::Interaction {
        components: vec![
            DesignTerm::Direct {
                column: ColumnId(0),
            },
            DesignTerm::Direct {
                column: ColumnId(99),
            },
        ],
    });
    c.outcome.coefficients.push(0.0);
    let err = c.validate().unwrap_err();
    assert!(
        matches!(err, ContractError::ColumnIdOutOfRange { id: 99, .. }),
        "got {err:?}"
    );
}

#[test]
fn valid_continuous_interaction_passes_validation() {
    let mut c = example1_simple_ols();
    c.design_generation.terms.push(DesignTerm::Interaction {
        components: vec![
            DesignTerm::Direct {
                column: ColumnId(0),
            },
            DesignTerm::Direct {
                column: ColumnId(1),
            },
        ],
    });
    c.outcome.coefficients.push(0.3);
    assert!(c.validate().is_ok());
}

// --- invariant_21 tests: primary random slope structure ---

/// `y ~ x1 + x2 + (1 + x1 | g)` — Mle, two continuous Direct terms, one slope.
/// Builds from example1_simple_ols (cols 0,1 as Synthetic::Normal with Direct terms).
fn two_continuous_lme_spec() -> SimulationContract {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(ClusterSpec::intercept_only(
        ClusterSizing::FixedClusters { n_clusters: 10 },
        0.25,
    ));
    c
}

/// `y ~ x1 + (1 + x2 | g)` — Mle, col 1 exists but has NO Direct term in the design.
fn one_continuous_lme_spec() -> SimulationContract {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(ClusterSpec::intercept_only(
        ClusterSizing::FixedClusters { n_clusters: 10 },
        0.25,
    ));
    // Remove col 1's Direct term (index 2) so only col 0 is in the design.
    c.design_generation.terms.remove(2);
    c.outcome.coefficients.remove(2);
    c.test.targets = vec![TestTarget::Marginal { term: 1 }];
    c
}

/// `y ~ x1 + g[1] + (1 | cluster)` — Mle, col 2 is a FactorSynthetic with DummyOf terms.
fn factor_lme_spec() -> SimulationContract {
    let mut c = example1_simple_ols();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(ClusterSpec::intercept_only(
        ClusterSizing::FixedClusters { n_clusters: 10 },
        0.25,
    ));
    // Add col 2 as a 2-level factor.
    c.generation.columns.push(ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: vec![0.5, 0.5],
        sampled_proportions: None,
    });
    c.design_generation.terms.push(DesignTerm::DummyOf {
        column: ColumnId(2),
        level_index: 1,
    });
    c.outcome.coefficients.push(0.3);
    c.test.targets.push(TestTarget::Marginal { term: 3 });
    c
}

/// Helper: a SlopeTerm with no slope↔slope correlations (the common case).
fn slope(col: u32, var: f64, corr_int: f64) -> SlopeTerm {
    SlopeTerm {
        column: ColumnId(col),
        variance: var,
        corr_with_intercept: corr_int,
        corr_with: vec![],
    }
}

/// `y ~ x1 + (1 + x1 | g)` — one correlated slope; valid (back-compat).
#[test]
fn single_slope_well_formed_accepts() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![slope(0, 0.10, 0.3)];
    assert!(spec.validate().is_ok());
}

/// `y ~ x1 + x2 + (1 + x1 + x2 | g)` — two slopes, with the slope_1↔slope_0
/// correlation supplied via corr_with; PSD. Valid.
#[test]
fn multi_slope_well_formed_accepts() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![
        slope(0, 0.10, 0.3),
        SlopeTerm {
            column: ColumnId(1),
            variance: 0.08,
            corr_with_intercept: 0.1,
            corr_with: vec![0.2],
        },
    ];
    assert!(spec.validate().is_ok());
}

/// slope_1.corr_with must have length 1 (one earlier slope); length 0 rejects.
#[test]
fn corr_with_wrong_length_rejected() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![
        slope(0, 0.10, 0.3),
        slope(1, 0.08, 0.1), // corr_with = [] but index 1 needs length 1
    ];
    assert!(matches!(
        spec.validate(),
        Err(ContractError::SlopeCorrWithLenInvalid {
            slope: 1,
            expected: 1,
            got: 0
        })
    ));
}

/// An indefinite correlation structure (intercept↔both ≈ +1, slope↔slope ≈ −1)
/// makes D non-PSD.
#[test]
fn non_psd_covariance_rejected() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![
        slope(0, 0.10, 0.99),
        SlopeTerm {
            column: ColumnId(1),
            variance: 0.10,
            corr_with_intercept: 0.99,
            corr_with: vec![-0.99],
        },
    ];
    assert!(matches!(
        spec.validate(),
        Err(ContractError::SlopeCovarianceNotPSD)
    ));
}

/// `y ~ x1 + (1 + x1 | g) + (1 | item)` — a primary slope composed with an
/// intercept-only crossed extra. Valid (composition, Task 6).
#[test]
fn slope_with_intercept_only_extra_accepts() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![slope(0, 0.10, 0.3)];
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 6 },
        tau_squared: 0.1,
        slopes: vec![], // intercept-only extra
    }];
    assert!(spec.validate().is_ok());
}

/// A primary slope composed with a slope ON an extra grouping — now SUPPORTED
/// (invariant_19, gated kernel path). `y ~ x1 + (1 + x1 | g) + (1 + x2 | item)`.
#[test]
fn slopes_on_extra_grouping_accepted() {
    let mut spec = two_continuous_lme_spec();
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![slope(0, 0.10, 0.3)];
    cl.extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 6 },
        tau_squared: 0.1,
        slopes: vec![slope(1, 0.05, 0.0)],
    }];
    assert!(spec.validate().is_ok());
}

#[test]
fn slope_on_factor_column_rejected() {
    let mut spec = factor_lme_spec(); // column 2 is FactorSynthetic, with DummyOf terms
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![slope(2, 0.10, 0.0)];
    assert!(matches!(
        spec.validate(),
        Err(ContractError::SlopeColumnNotContinuous { id: 2 })
    ));
}

#[test]
fn slope_without_fixed_effect_rejected() {
    // column 1 is continuous but has no Direct term in the design.
    let mut spec = one_continuous_lme_spec(); // only column 0 is a Direct term
    let cl = spec.generation.cluster.as_mut().unwrap();
    cl.tau_squared = 0.25;
    cl.slopes = vec![slope(1, 0.10, 0.0)];
    assert!(matches!(
        spec.validate(),
        Err(ContractError::SlopeColumnNotInDesign { id: 1 })
    ));
}

#[test]
fn slope_bad_corr_and_zero_intercept_var_rejected() {
    let mut spec = two_continuous_lme_spec();
    {
        let cl = spec.generation.cluster.as_mut().unwrap();
        cl.tau_squared = 0.25;
        cl.slopes = vec![slope(0, 0.10, 1.5)];
    }
    assert!(matches!(
        spec.validate(),
        Err(ContractError::SlopeCorrInvalid { .. })
    ));

    {
        let cl = spec.generation.cluster.as_mut().unwrap();
        cl.slopes = vec![slope(0, 0.10, 0.3)];
        cl.tau_squared = 0.0;
    }
    assert!(
        matches!(spec.validate(), Err(ContractError::SlopeInterceptVarianceMissing { got }) if got == 0.0)
    );
}

/// A negative slope variance hits SlopeVarianceInvalid; a corr_with entry
/// outside [-1, 1] hits SlopeCorrInvalid via the inner slope↔slope loop
/// (distinct from the corr_with_intercept path in the test above).
#[test]
fn slope_bad_variance_and_corr_with_entry_rejected() {
    let mut spec = two_continuous_lme_spec();
    {
        let cl = spec.generation.cluster.as_mut().unwrap();
        cl.tau_squared = 0.25;
        cl.slopes = vec![slope(0, -0.1, 0.3)];
    }
    assert!(
        matches!(spec.validate(), Err(ContractError::SlopeVarianceInvalid { got }) if got == -0.1)
    );

    {
        let cl = spec.generation.cluster.as_mut().unwrap();
        cl.slopes = vec![
            slope(0, 0.10, 0.3),
            SlopeTerm {
                column: ColumnId(1),
                variance: 0.08,
                corr_with_intercept: 0.1,
                corr_with: vec![1.5],
            },
        ];
    }
    assert!(matches!(spec.validate(), Err(ContractError::SlopeCorrInvalid { got }) if got == 1.5));
}

// T3 — whole-envelope serde_json round-trip on the finite example1 fixture.
// Guards the NaN→null→reject class across all f64 fields without injecting NaN
// (finite fixture always passes; this is a structural serde symmetry guard).
#[test]
fn simulation_contract_serde_json_roundtrip_example1() {
    let c = example1_simple_ols();
    let s = serde_json::to_string(&c).unwrap();
    let back: SimulationContract = serde_json::from_str(&s).unwrap();
    assert_eq!(c, back);
}

// T4 — unknown tag → clean Err (not panic) for append-only enums.
// Each test uses the enum's actual tag field / representation with a bogus
// variant; the deserialization must return Err, never panic.

#[test]
fn estimator_spec_unknown_tag_returns_err() {
    // EstimatorSpec is rename_all="snake_case" with no `tag` field — serializes
    // as a plain JSON string. An unrecognized string must yield Err.
    let result = serde_json::from_str::<EstimatorSpec>(r#""quantile_regression""#);
    assert!(result.is_err());
}

#[test]
fn test_target_unknown_kind_returns_err() {
    // TestTarget uses #[serde(tag = "kind", rename_all = "snake_case")].
    let result = serde_json::from_str::<TestTarget>(r#"{"kind":"beta"}"#);
    assert!(result.is_err());
}

#[test]
fn residual_dist_unknown_tag_returns_err() {
    // ResidualDist is rename_all="snake_case" with no `tag` field — plain string.
    let result = serde_json::from_str::<ResidualDist>(r#""bimodal""#);
    assert!(result.is_err());
}
