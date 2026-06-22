//! Numerical equivalence test for the contrast/marginal equivalence acceptance criterion:
//!
//! "With reference A, `Contrast { B, A_index }` must equal `Marginal { B }` for OLS."
//!
//! This is a **contract-layer** test.  The two `SimulationContract` inputs
//! differ (one uses `TestTarget::Marginal`, the other `TestTarget::Contrast`
//! with `negative` pointing at the `DesignTerm::Const` reference term), but
//! after `contract_to_simulation_spec` both collapse to the same `SimulationSpec`.
//! Running `run_batch` on both with the same seed must therefore yield
//! bit-identical `uncorrected` significance vectors.
//!
//! Design used here: 3-level factor (A = reference, B, C) with a non-trivial
//! effect on B (β = 0.4) so the test runs with meaningful power, not at the
//! null.  We exercise `TestTarget::Marginal { term: 1 }` (B) vs
//! `TestTarget::Contrast { positive: 1, negative: 0 }` (B vs Const).

use engine_contract::{
    ColumnId, ColumnSpec, CorrectionMethod, DesignSpec, DesignTerm, EstimatorSpec, GenerationSpec,
    OutcomeKind, OutcomeSpec, ResidualDist, ResidualSpec, ScenarioPerturbations,
    SimulationContract, TestSpec, TestTarget,
};
use engine_core::contract_adapter::contract_to_simulation_spec;
use engine_core::run_batch_st;

/// Build a 3-level factor contract.  `targets` is set by each test case.
fn three_level_factor_contract(targets: Vec<TestTarget>) -> SimulationContract {
    // Generation: one 3-level factor (equal proportions).
    // Design (generation + test): intercept, dummy-for-level-1, dummy-for-level-2.
    //   term 0 = Const          (reference / intercept)
    //   term 1 = DummyOf{col 0, level 1}   (= B)
    //   term 2 = DummyOf{col 0, level 2}   (= C)
    let terms = vec![
        DesignTerm::Const,
        DesignTerm::DummyOf {
            column: ColumnId(0),
            level_index: 1,
        },
        DesignTerm::DummyOf {
            column: ColumnId(0),
            level_index: 2,
        },
    ];
    SimulationContract {
        generation: GenerationSpec {
            columns: vec![ColumnSpec::FactorSynthetic {
                n_levels: 3,
                proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                sampled_proportions: None,
            }],
            correlations: engine_contract::Correlations::Identity,
            cluster: None,
            uploaded_frame: None,
            cluster_level_columns: vec![],
        },
        design_generation: DesignSpec {
            terms: terms.clone(),
        },
        outcome: OutcomeSpec {
            kind: OutcomeKind::Continuous,
            intercept: 0.0,
            // Non-trivial effect on B (coeff index 1) so power > 0.
            coefficients: vec![0.0, 0.4, 0.0],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: false,
            },
            heteroskedasticity_driver: None,
        },
        design_test: Some(DesignSpec { terms }),
        estimator: EstimatorSpec::Ols,
        test: TestSpec {
            targets,
            correction: CorrectionMethod::None,
            alpha: 0.05,
        },
        posthoc: vec![],
        scenario: ScenarioPerturbations::default(),
        max_failed_fraction: 0.1,
    }
}

#[test]
fn contrast_ref_collapse_matches_marginal_power() {
    // Contract 1: Marginal { term: 1 }  (B level)
    let c_marginal = three_level_factor_contract(vec![TestTarget::Marginal { term: 1 }]);

    // Contract 2: Contrast { positive: 1, negative: 0 }
    //   positive = term 1 = DummyOf B
    //   negative = term 0 = Const  (reference-level absorber)
    // The contract adapter's reference-collapse rule in
    // `translate_targets_and_contrasts` must reduce this to
    // `target_indices = [B_col]` — identical to the Marginal spec.
    let c_contrast = three_level_factor_contract(vec![TestTarget::Contrast {
        positive: 1,
        negative: 0,
    }]);

    // Translate both through the public contract → SimulationSpec adapter.
    let spec_marginal = contract_to_simulation_spec(&c_marginal)
        .expect("Marginal contract must translate without error");
    let spec_contrast = contract_to_simulation_spec(&c_contrast)
        .expect("Contrast contract must translate without error");

    // Structural sanity: the two specs must be byte-identical at the kernel level.
    assert_eq!(
        spec_marginal.target_indices, spec_contrast.target_indices,
        "target_indices must be equal after reference-collapse"
    );
    assert!(
        spec_contrast.contrast_pairs.is_empty(),
        "reference-collapse must not populate contrast_pairs"
    );

    // End-to-end numerical equivalence: run both through `run_batch_st` with
    // the same (n, seed, n_sims).  Because the SimulationSpecs are identical,
    // the raw significance vectors must be bit-equal.
    let n_sims: u32 = 2000;
    let sample_size: u32 = 80;
    let seed: u64 = 42;

    let result_marginal = run_batch_st(&spec_marginal, &[sample_size], n_sims, seed, None)
        .expect("run_batch_st must not error for Marginal spec");
    let result_contrast = run_batch_st(&spec_contrast, &[sample_size], n_sims, seed, None)
        .expect("run_batch_st must not error for Contrast spec");

    assert_eq!(
        result_marginal.uncorrected, result_contrast.uncorrected,
        "Marginal{{B}} and Contrast{{B, Const}} must produce identical per-sim \
         significance vectors.  A mismatch means the reference-collapse in \
         translate_targets_and_contrasts is not routing correctly."
    );
}
