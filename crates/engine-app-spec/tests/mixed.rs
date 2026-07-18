use engine_app_spec::app_spec::{AppSpec, ClusterDim, MixedOutcome, MixedSpec};
use engine_contract::CorrectionMethod;
use serde::Serialize;

/// A minimal Mixed spec: `y ~ x + (1|school)`, one numeric predictor, one effect.
fn minimal_mixed(cluster_dim: ClusterDim) -> MixedSpec {
    use engine_app_spec::app_spec::{ParsedFormula, TestSelection, VarType};
    use engine_app_spec::EffectSize;
    MixedSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric {
            name: "x".into(),
            distribution: Default::default(),
            pinned: false,
        }],
        effects: vec![EffectSize {
            name: "x".into(),
            value: 0.3,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 800,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        wald_se: Default::default(),
        scenarios: vec![],
        csv: None,
        report_overall: true,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
        cluster_name: "school".into(),
        icc: 0.2,
        cluster_dim,
        cluster_level_vars: vec![],
        extra_groupings: vec![],
        slopes: vec![],
        outcome: MixedOutcome::Gaussian,
        agq: 1,
    }
}

#[test]
fn mixed_appspec_serde_roundtrips_with_tagged_cluster_dim() {
    let spec = AppSpec::Mixed(minimal_mixed(ClusterDim::NClusters { value: 20 }));
    let json = serde_json::to_string(&spec).unwrap();
    // family tag is lowercase; cluster_dim is internally tagged on "kind".
    assert!(json.contains("\"family\":\"mixed\""), "got: {json}");
    assert!(json.contains("\"kind\":\"n_clusters\""), "got: {json}");
    assert!(json.contains("\"value\":20"), "got: {json}");
    let back: AppSpec = serde_json::from_str(&json).unwrap();
    assert!(
        matches!(back, AppSpec::Mixed(_)),
        "expected Mixed after roundtrip, got: {back:?}"
    );
}

#[test]
fn cluster_dim_deserializes_both_kinds_from_ts_shape() {
    let n: ClusterDim = serde_json::from_str(r#"{"kind":"n_clusters","value":12}"#).unwrap();
    assert_eq!(n, ClusterDim::NClusters { value: 12 });
    let s: ClusterDim = serde_json::from_str(r#"{"kind":"cluster_size","value":30}"#).unwrap();
    assert_eq!(s, ClusterDim::ClusterSize { value: 30 });
}

// ── assemble tests ────────────────────────────────────────────────────────────

use engine_app_spec::assemble::assemble_spec;

// `assemble_spec` Mixed/ClusterSpec behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn assemble_mixed_emits_one_cluster_with_mle_and_tau_from_icc() {
    let spec = AppSpec::Mixed(minimal_mixed(ClusterDim::NClusters { value: 20 }));
    let contracts = assemble_spec(&spec).expect("assemble mixed ok");
    assert_eq!(contracts.len(), 1);
    let c = &contracts[0];
    assert_eq!(c.estimator, engine_contract::EstimatorSpec::Mle);
    let cl = c.generation.cluster.as_ref().expect("cluster present");
    match &cl.sizing {
        engine_contract::ClusterSizing::FixedClusters { n_clusters } => assert_eq!(*n_clusters, 20),
        other => panic!("expected FixedClusters, got {other:?}"),
    }
    // icc 0.2 -> tau^2 = 0.2 / 0.8 = 0.25
    assert!(
        (cl.tau_squared - 0.25).abs() < 1e-12,
        "tau^2 = {}",
        cl.tau_squared
    );
}

#[test]
fn assemble_mixed_cluster_size_mode_emits_fixed_size() {
    // cluster_size mode now maps directly to ClusterSizing::FixedSize.
    // tau^2 is still derived from icc at assemble time.
    let spec = AppSpec::Mixed(minimal_mixed(ClusterDim::ClusterSize { value: 30 }));
    let contracts = assemble_spec(&spec).expect("assemble mixed (cluster_size) ok");
    let cl = contracts[0].generation.cluster.as_ref().unwrap();
    match &cl.sizing {
        engine_contract::ClusterSizing::FixedSize { cluster_size } => assert_eq!(*cluster_size, 30),
        other => panic!("expected FixedSize, got {other:?}"),
    }
    assert!((cl.tau_squared - 0.25).abs() < 1e-12);
}

#[test]
fn icc_zero_yields_zero_tau() {
    let mut m = minimal_mixed(ClusterDim::NClusters { value: 5 });
    m.icc = 0.0;
    let contracts = assemble_spec(&AppSpec::Mixed(m)).unwrap();
    assert_eq!(
        contracts[0]
            .generation
            .cluster
            .as_ref()
            .unwrap()
            .tau_squared,
        0.0
    );
}

// ── AGQ eligibility (invariant_25_nagq_backstop pre-empted host-side) ─────────

use engine_app_spec::app_spec::{AppGroupingRelation, AppGroupingSpec};
use engine_app_spec::assemble::assemble_spec_with_skeleton;

/// An ineligible shape (a second grouping factor) with `agq=5` must strip to
/// Laplace (`nagq=1`) and surface a warning — not reach the engine's
/// `invariant_25_nagq_backstop` error. Mirrors `model.py`'s
/// `_resolve_estimation` warn-and-strip behavior.
#[test]
fn ineligible_second_grouping_strips_agq_to_laplace_with_warning() {
    let mut m = minimal_mixed(ClusterDim::NClusters { value: 20 });
    m.outcome = MixedOutcome::Binary {
        baseline_probability: 0.3,
        link: Default::default(),
    };
    m.agq = 5;
    m.extra_groupings = vec![AppGroupingSpec {
        tau_squared: 0.1,
        relation: AppGroupingRelation::Crossed { n_clusters: 10 },
        cluster_name: Some("teacher".into()),
        slopes: vec![],
    }];
    let (contracts, _skeleton, warnings) =
        assemble_spec_with_skeleton(&AppSpec::Mixed(m)).expect("ineligible agq must not error");
    assert_eq!(contracts[0].nagq, 1, "must strip to Laplace");
    assert!(
        warnings.iter().any(|w| w.contains("agq=5")),
        "expected an agq warning, got: {warnings:?}"
    );
    contracts[0]
        .validate()
        .expect("stripped contract must self-validate");
}

/// An eligible shape (single grouping, ≤3 REs, Binary GLMM) keeps the
/// requested `agq` untouched and emits no warning.
#[test]
fn eligible_binary_glmm_keeps_agq_with_no_warning() {
    let mut m = minimal_mixed(ClusterDim::NClusters { value: 20 });
    m.outcome = MixedOutcome::Binary {
        baseline_probability: 0.3,
        link: Default::default(),
    };
    m.agq = 5;
    let (contracts, _skeleton, warnings) =
        assemble_spec_with_skeleton(&AppSpec::Mixed(m)).expect("eligible agq must not error");
    assert_eq!(contracts[0].nagq, 5);
    assert!(
        warnings.is_empty(),
        "expected no warnings, got: {warnings:?}"
    );
}

// ── driver tests ──────────────────────────────────────────────────────────────

use engine_app_spec::{
    run_find_power, run_find_sample_size, run_single_core_find_power,
    run_single_core_find_sample_size, NullEmitter,
};
use engine_orchestrator::{ByValue, CancellationToken, GridMode, SampleSizeMethod};

fn minimal_mixed_with_effect(cluster_dim: ClusterDim) -> MixedSpec {
    let mut m = minimal_mixed(cluster_dim);
    m.n_sims = 20; // keep the engine run cheap in tests
    m
}

#[test]
fn run_find_sample_size_cluster_size_succeeds_atom_snapped() {
    // cluster_size=30 → atom=30, hard_min = min_clusters(5)*30 = 150.
    // Bounds (50, 250): grid spans [150, 240] in steps of 30.
    let spec = AppSpec::Mixed(minimal_mixed_with_effect(ClusterDim::ClusterSize {
        value: 30,
    }));
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(30),
        mode: GridMode::Linear,
    };
    let res = run_find_sample_size(&spec, (50, 250), method, &NullEmitter, &cancel)
        .expect("cluster_size + find_sample_size now supported");
    let (_, ss) = &res.scenarios[0];
    let ns: Vec<usize> = ss.grid_or_trace.iter().map(|p| p.n).collect();
    assert!(
        ns.iter().all(|&n| n % 30 == 0),
        "grid N's must be multiples of cluster_size 30: {ns:?}"
    );
}

/// EP-3 twin — GAP 2: `run_find_sample_size_cluster_size_succeeds_atom_snapped`
/// above exercises `run_find_sample_size` (native multi-core) only. The WASM
/// worker path uses `run_single_core_find_sample_size`. Divergence risk: cluster
/// atom snapping broken on the single-core code path (separate code path through
/// `single_core_find_sample_size`).
#[test]
fn run_single_core_find_sample_size_cluster_size_atom_snapped() {
    // cluster_size=30 → atom=30; grid is snapped to multiples of 30.
    // n_sims=20 keeps the run cheap; base_seed matches the native sibling.
    let spec = AppSpec::Mixed(minimal_mixed_with_effect(ClusterDim::ClusterSize {
        value: 30,
    }));
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(30),
        mode: GridMode::Linear,
    };
    let res =
        run_single_core_find_sample_size(&spec, (50, 250), method, 20, 2137, &NullEmitter, &cancel)
            .expect("cluster_size + single_core_find_sample_size supported");
    let (_, ss) = &res.scenarios[0];
    let ns: Vec<usize> = ss.grid_or_trace.iter().map(|p| p.n).collect();
    assert!(
        ns.iter().all(|&n| n % 30 == 0),
        "grid N's must be multiples of cluster_size 30 on the single-core path: {ns:?}"
    );
}

#[test]
fn run_find_power_cluster_size_runs_as_fixed_size() {
    // cluster_size=30, sample_size=180 → 180/30=6 complete clusters (≥ min_clusters=5).
    // The contract carries FixedSize{30}; find_power snaps N to 180 and lays out 6
    // blocks. Returns a well-formed single-scenario ScenarioResult<PowerResult>.
    let spec = AppSpec::Mixed(minimal_mixed_with_effect(ClusterDim::ClusterSize {
        value: 30,
    }));
    let cancel = CancellationToken::new();
    let res = run_find_power(&spec, 180, &NullEmitter, &cancel).expect("find_power ok");
    assert_eq!(res.scenarios.len(), 1);
    let (_, pr) = &res.scenarios[0];
    assert_eq!(pr.target_indices.len(), pr.power_uncorrected.len());
    assert!(!pr.target_indices.is_empty());
    assert!(pr.n_sims > 0, "n_sims must be positive, got {}", pr.n_sims);
}

/// End-to-end pin for the B3 fix: an actual `run_find_power` call (the real
/// production entry point Tauri/WASM invoke, not just the assembler) with an
/// ineligible `agq=5` must reach the user through `PowerResult.grid_warnings` —
/// the same field `ConvergenceNotice.svelte` already renders — instead of the
/// warning being silently dropped between assembly and the returned result.
#[test]
fn run_find_power_surfaces_agq_strip_warning_in_grid_warnings() {
    let mut m = minimal_mixed_with_effect(ClusterDim::NClusters { value: 20 });
    m.outcome = MixedOutcome::Binary {
        baseline_probability: 0.3,
        link: Default::default(),
    };
    m.agq = 5;
    // A second grouping factor makes this shape AGQ-ineligible.
    m.extra_groupings = vec![AppGroupingSpec {
        tau_squared: 0.1,
        relation: AppGroupingRelation::Crossed { n_clusters: 10 },
        cluster_name: Some("teacher".into()),
        slopes: vec![],
    }];
    let spec = AppSpec::Mixed(m);
    let cancel = CancellationToken::new();
    let res = run_find_power(&spec, 200, &NullEmitter, &cancel).expect("find_power ok");
    let (_, pr) = &res.scenarios[0];
    assert!(
        pr.grid_warnings.iter().any(|w| w.contains("agq=5")),
        "expected the agq strip warning in grid_warnings, got: {:?}",
        pr.grid_warnings
    );
}

/// WASM-dispatch twin of the test above: `run_single_core_find_power` is the
/// per-worker entry a browser worker actually calls; it must carry the same
/// warning (the driver-level invariant is that these two never drift).
#[test]
fn run_single_core_find_power_surfaces_agq_strip_warning_in_grid_warnings() {
    let mut m = minimal_mixed_with_effect(ClusterDim::NClusters { value: 20 });
    m.outcome = MixedOutcome::Binary {
        baseline_probability: 0.3,
        link: Default::default(),
    };
    m.agq = 5;
    m.extra_groupings = vec![AppGroupingSpec {
        tau_squared: 0.1,
        relation: AppGroupingRelation::Crossed { n_clusters: 10 },
        cluster_name: Some("teacher".into()),
        slopes: vec![],
    }];
    let spec = AppSpec::Mixed(m);
    let cancel = CancellationToken::new();
    let res = run_single_core_find_power(&spec, 200, 20, 2137, &NullEmitter, &cancel)
        .expect("single_core find_power ok");
    let (_, pr) = &res.scenarios[0];
    assert!(
        pr.grid_warnings.iter().any(|w| w.contains("agq=5")),
        "expected the agq strip warning in grid_warnings, got: {:?}",
        pr.grid_warnings
    );
}

// ── Snapshot test ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AssembleSummary {
    n_contracts: usize,
    target_count: usize,
    coefficient_count: usize,
    generation_column_count: usize,
    alpha: f64,
    correction_code: i32,
    outcome_kind_is_continuous: bool,
}

fn build_summary_mixed(contracts: &[engine_contract::SimulationContract]) -> AssembleSummary {
    let c = &contracts[0];
    AssembleSummary {
        n_contracts: contracts.len(),
        target_count: c.test.targets.len(),
        coefficient_count: c.outcome.coefficients.len(),
        generation_column_count: c.generation.columns.len(),
        alpha: c.test.alpha,
        correction_code: c.test.correction.code(),
        outcome_kind_is_continuous: c.outcome.kind == engine_contract::OutcomeKind::Continuous,
    }
}

#[test]
fn assemble_mixed_matches_golden_projection() {
    let spec = AppSpec::Mixed(minimal_mixed(ClusterDim::NClusters { value: 20 }));
    let contracts = assemble_spec(&spec).unwrap();
    let summary = build_summary_mixed(&contracts);
    insta::assert_json_snapshot!(summary);
}
