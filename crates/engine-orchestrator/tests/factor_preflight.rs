//! Integration tests for sparse-factor preflight warnings in `find_power`,
//! `single_core_find_power`, `find_sample_size`, and
//! `single_core_find_sample_size`.
//!
//! The preflight runs the exact fixed-allocation walk before any simulation,
//! so these tests confirm warning content without depending on stochastic
//! power values.

mod common;
use common::minimal_ols_contract;

use engine_contract::ids::ColumnId;
use engine_contract::test_spec::CorrectionMethod;
use engine_contract::{
    ColumnSpec, Correlations, DesignSpec, DesignTerm, EstimatorSpec, GenerationSpec,
    OutcomeKind, OutcomeSpec, ResidualDist, ResidualSpec,
    ScenarioPerturbations, SimulationContract, TestSpec, TestTarget,
};
use engine_orchestrator::{
    find_power, find_sample_size, single_core_find_power, single_core_find_sample_size, ByValue,
    CancellationToken, GridMode, SampleSizeMethod,
};

/// Contract with a 2-level factor whose proportions are `probs` and no other
/// predictors. The factor is the sole design term (after intercept); target is
/// its dummy. `sampled_factor_proportions` is `false` by default; callers can flip it.
fn factor_contract(probs: [f64; 2]) -> SimulationContract {
    // One factor column with the given level proportions.
    let column = ColumnSpec::FactorSynthetic {
        n_levels: 2,
        proportions: probs.to_vec(),
        sampled_proportions: None,
    };
    let generation = GenerationSpec {
        columns: vec![column],
        correlations: Correlations::Identity,
        cluster: None,
        uploaded_frame: None,
        cluster_level_columns: vec![],
    };
    // Design: intercept + dummy for level 1 of column 0 (reference = level 0).
    let design_generation = DesignSpec {
        terms: vec![
            DesignTerm::Const,
            DesignTerm::DummyOf {
                column: ColumnId(0),
                level_index: 1,
            },
        ],
    };
    let outcome = OutcomeSpec {
        kind: OutcomeKind::Continuous,
        intercept: 0.0,
        // [intercept_coeff, dummy_coeff]
        coefficients: vec![0.0, 0.5],
        residual: ResidualSpec {
            distribution: ResidualDist::Normal,
            pinned: false,
        },
        heteroskedasticity_driver: None,
    };
    let test = TestSpec {
        targets: vec![TestTarget::Marginal { term: 1 }],
        correction: CorrectionMethod::None,
        alpha: 0.05,
    };
    let scenario = ScenarioPerturbations {
        name: "optimistic".into(),
        ..ScenarioPerturbations::default()
        // sampled_factor_proportions defaults to false
    };
    SimulationContract {
        generation,
        design_generation,
        outcome,
        design_test: None,
        estimator: EstimatorSpec::Ols,
        test,
        posthoc: vec![],
        scenario,
        max_failed_fraction: 0.05,
    }
}

/// Same as `factor_contract` but with `sampled_factor_proportions = true` (sampled).
fn factor_contract_sampled(probs: [f64; 2]) -> SimulationContract {
    let mut c = factor_contract(probs);
    c.scenario.sampled_factor_proportions = true;
    c
}

// ── find_power: sparse factor at single N ────────────────────────────────────

#[test]
fn preflight_warns_on_sparse_factor_at_single_n() {
    // [0.95, 0.05] at N=40 → level 1 gets 2 obs (< 5); warning expected.
    let c = factor_contract([0.95, 0.05]);
    let cancel = CancellationToken::new();
    let res = find_power(&[c], 40, 50, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    // One warning must carry both substrings — ports substring-match on them.
    assert!(
        pr.grid_warnings
            .iter()
            .any(|w| w.contains("factor 1") && w.contains("excluded from every simulation")),
        "expected a single warning with 'factor 1' + 'excluded from every simulation', got: {:?}",
        pr.grid_warnings
    );
}

// ── Twin mandate: find_power and single_core_find_power emit identical warnings

#[test]
fn preflight_twins_emit_identical_warnings() {
    // [0.95, 0.05] at N=40 → both twins must produce the same grid_warnings.
    let c = factor_contract([0.95, 0.05]);
    let cancel = CancellationToken::new();
    let multi = find_power(&[c.clone()], 40, 50, 2137, None, &cancel).unwrap();
    let single = single_core_find_power(&[c], 40, 50, 2137, None, &cancel).unwrap();
    let multi_warns = &multi.scenarios[0].1.grid_warnings;
    let single_warns = &single.scenarios[0].1.grid_warnings;
    assert_eq!(
        multi_warns, single_warns,
        "find_power and single_core_find_power must emit identical grid_warnings"
    );
    // Sanity: the expected warning is actually present.
    assert!(
        multi_warns.iter().any(|w| w.contains("factor 1")),
        "expected 'factor 1' warning, got: {multi_warns:?}"
    );
}

// ── find_sample_size: names min_inclusion_n ───────────────────────────────────

#[test]
fn preflight_sample_size_names_min_inclusion_n() {
    // [0.9, 0.1] grid [20, 200]: level 1 is sparse at low N.
    // min_inclusion_n should return the first N where both levels clear 5 obs.
    let c = factor_contract([0.9, 0.1]);
    let cancel = CancellationToken::new();
    let res = find_sample_size(
        &[c],
        0.8,
        (20, 200),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 10 },
            mode: GridMode::Linear,
        },
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let (_, ss) = &res.scenarios[0];
    // The warning must name the exact value min_inclusion_n computes for these
    // proportions within the grid ceiling — not just any "N >= " phrasing.
    let n_inc = engine_core::min_inclusion_n(&[0.9, 0.1], 5, 200).unwrap();
    let needle = format!("N >= {n_inc}");
    assert!(
        ss.grid_warnings.iter().any(|w| w.contains(&needle)),
        "expected '{needle}' warning, got: {:?}",
        ss.grid_warnings
    );
    // A healthy balanced factor produces no factor warning.
    let healthy = factor_contract([0.5, 0.5]);
    let cancel2 = CancellationToken::new();
    let res2 = find_sample_size(
        &[healthy],
        0.8,
        (20, 200),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 10 },
            mode: GridMode::Linear,
        },
        2137,
        None,
        &cancel2,
    )
    .unwrap();
    let (_, ss2) = &res2.scenarios[0];
    assert!(
        !ss2.grid_warnings.iter().any(|w| w.contains("factor")),
        "balanced factor should produce no factor warning, got: {:?}",
        ss2.grid_warnings
    );
}

// ── find_sample_size: whole-grid sparse (min_inclusion_n = None branch) ──────

#[test]
fn preflight_sample_size_warns_when_factor_stays_sparse_across_whole_grid() {
    // [0.999, 0.001] at grid [10, 20] — level 1 never reaches 5 obs within the
    // ceiling, so the None branch of min_inclusion_n fires.
    let c = factor_contract([0.999, 0.001]);
    let cancel = CancellationToken::new();
    let res = find_sample_size(
        &[c],
        0.8,
        (10, 20),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 5 },
            mode: GridMode::Linear,
        },
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let (_, ss) = &res.scenarios[0];
    assert!(
        ss.grid_warnings
            .iter()
            .any(|w| w.contains("stays under") && w.contains("excluded everywhere")),
        "expected the whole-range sparse warning, got: {:?}",
        ss.grid_warnings
    );
}

// ── Twin mandate: sample-size twins emit identical warnings ──────────────────

#[test]
fn preflight_sample_size_twins_emit_identical_warnings() {
    let c = factor_contract([0.9, 0.1]);
    let method = SampleSizeMethod::Grid {
        by: ByValue::Auto { count: 10 },
        mode: GridMode::Linear,
    };
    let cancel = CancellationToken::new();
    let multi = find_sample_size(
        &[c.clone()],
        0.8,
        (20, 200),
        50,
        method.clone(),
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let single = single_core_find_sample_size(
        &[c],
        0.8,
        (20, 200),
        50,
        method,
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let multi_warns = &multi.scenarios[0].1.grid_warnings;
    let single_warns = &single.scenarios[0].1.grid_warnings;
    assert_eq!(
        multi_warns, single_warns,
        "find_sample_size and single_core_find_sample_size must emit identical \
         grid_warnings"
    );
    assert!(
        multi_warns.iter().any(|w| w.contains("factor 1")),
        "expected 'factor 1' warning, got: {multi_warns:?}"
    );
}

// ── Silent cases: healthy designs and sampled-allocation ─────────────────────

#[test]
fn preflight_silent_for_healthy_designs() {
    // 1. Balanced factor [0.5, 0.5] at N=100: no factor warning.
    let c = factor_contract([0.5, 0.5]);
    let cancel = CancellationToken::new();
    let res = find_power(&[c], 100, 50, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    assert!(
        !pr.grid_warnings.iter().any(|w| w.contains("factor")),
        "balanced factor at N=100 should have no factor warning, got: {:?}",
        pr.grid_warnings
    );

    // 2. Sparse [0.95, 0.05] factor with sampled (non-fixed) allocation:
    //    preflight is fixed-allocation only; no warning expected.
    let c_sampled = factor_contract_sampled([0.95, 0.05]);
    let cancel2 = CancellationToken::new();
    let res2 = find_power(&[c_sampled], 40, 50, 2137, None, &cancel2).unwrap();
    let (_, pr2) = &res2.scenarios[0];
    assert!(
        !pr2.grid_warnings.iter().any(|w| w.contains("factor")),
        "sampled-allocation sparse factor should have no preflight warning, got: {:?}",
        pr2.grid_warnings
    );

    // 3. No-factor OLS design: no factor warning.
    let no_factor = minimal_ols_contract();
    let cancel3 = CancellationToken::new();
    let res3 = find_power(&[no_factor], 100, 50, 2137, None, &cancel3).unwrap();
    let (_, pr3) = &res3.scenarios[0];
    assert!(
        !pr3.grid_warnings.iter().any(|w| w.contains("factor")),
        "factor-free design should have no factor warning, got: {:?}",
        pr3.grid_warnings
    );
}
