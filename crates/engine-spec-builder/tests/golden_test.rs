use engine_core::Distribution;
use engine_spec_builder::LinearSpec;
use std::fs;
use std::path::PathBuf;

mod common;
use common::build_linear_spec;

#[derive(serde::Deserialize)]
struct GoldenCase {
    input: LinearSpec,
    expected_scenarios: Vec<String>,
    expected_n_non_factor: u32,
    expected_n_factor_dummies: u32,
    expected_var_types: Vec<i32>,
    expected_effect_sizes: Vec<f64>,
    expected_target_indices: Vec<u32>,
}

/// Translates Python-side `_DIST_CODE` integers (the JSON fixtures still encode
/// distributions as the historical integers) into the typed `Distribution`
/// variants used by `SimulationSpec.var_types`.
fn int_to_distribution(code: i32) -> Distribution {
    match code {
        0 => Distribution::Normal,
        1 => Distribution::Binary,
        2 => Distribution::RightSkewed,
        3 => Distribution::LeftSkewed,
        4 => Distribution::HighKurtosis,
        5 => Distribution::Uniform,
        97 => Distribution::UploadedFactor,
        98 => Distribution::UploadedBinary,
        99 => Distribution::UploadedData,
        other => panic!("unknown _DIST_CODE integer in golden fixture: {other}"),
    }
}

fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/golden");
    p
}

fn run_case(file_name: &str) {
    let path = fixtures_dir().join(file_name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let case: GoldenCase =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {file_name}: {e}"));

    let specs = build_linear_spec(&case.input).expect("build");
    let names: Vec<String> = specs.iter().map(|s| s.scenario.name.clone()).collect();
    assert_eq!(
        names, case.expected_scenarios,
        "[{file_name}] scenario names"
    );

    let s = &specs[0];
    assert_eq!(
        s.n_non_factor, case.expected_n_non_factor,
        "[{file_name}] n_non_factor"
    );
    assert_eq!(
        s.n_factor_dummies, case.expected_n_factor_dummies,
        "[{file_name}] n_factor_dummies"
    );
    let expected_var_types: Vec<Distribution> = case
        .expected_var_types
        .iter()
        .copied()
        .map(int_to_distribution)
        .collect();
    assert_eq!(s.var_types, expected_var_types, "[{file_name}] var_types");
    assert_eq!(
        s.effect_sizes, case.expected_effect_sizes,
        "[{file_name}] effect_sizes"
    );
    assert_eq!(
        s.target_indices, case.expected_target_indices,
        "[{file_name}] target_indices"
    );
}

#[test]
fn linear_2predictors_no_corr() {
    run_case("linear_2predictors_no_corr.json");
}

#[test]
fn linear_with_factor_3lvl() {
    run_case("linear_with_factor_3lvl.json");
}

// The two interaction cases below assert that build_linear_contract accepts
// interaction terms and emits at least one DesignTerm::Interaction.
// They call build_linear_contract directly because contract_to_simulation_spec
// has todo!() stubs for the Interaction arm; the full pipeline is not yet wired.

#[test]
fn linear_interaction_xi_xj_builds_with_interaction_term() {
    // y = x1 + x2 + x1:x2 now builds; the contract carries an Interaction term.
    use engine_contract::{ids::ColumnId, DesignTerm};
    use engine_spec_builder::build_linear_contract;
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec,
        ResidualSpec, VarKind,
    };

    let spec = LinearSpec {
        formula: "y = x1 + x2 + x1:x2".into(),
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
            EffectAssignment {
                name: "x1:x2".into(),
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
        upload: None,
        cluster_level_vars: vec![],
    };

    let contracts = build_linear_contract(&spec).expect("interaction now supported");
    let dt = contracts[0]
        .design_test
        .as_ref()
        .unwrap_or(&contracts[0].design_generation);
    // x1:x2 ⇒ one Interaction over the two Direct predictor columns (not merely
    // "some Interaction exists"); mirrors build_design_appends_interaction_term at
    // the build_linear_contract level.
    assert!(
        dt.terms.iter().any(|t| matches!(t,
            DesignTerm::Interaction { components }
                if components.len() == 2
                    && matches!(components[0], DesignTerm::Direct { column } if column == ColumnId(0))
                    && matches!(components[1], DesignTerm::Direct { column } if column == ColumnId(1)))),
        "terms={:?}",
        dt.terms
    );
}

#[test]
fn linear_factor_interaction_builds_with_interaction_term() {
    // y = x1 + group + x1:group now builds; the contract carries at least one
    // Interaction term (one per non-reference level: x1:group[B], x1:group[C]).
    use engine_contract::{ids::ColumnId, DesignTerm};
    use engine_spec_builder::build_linear_contract;
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec,
        ResidualSpec, VarKind,
    };

    let spec = LinearSpec {
        formula: "y = x1 + group + x1:group".into(),
        predictors: vec![
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
                    proportions: vec![0.34, 0.33, 0.33],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ],
        effects: vec![
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
            EffectAssignment {
                name: "x1:group[B]".into(),
                size: 0.1,
            },
            EffectAssignment {
                name: "x1:group[C]".into(),
                size: 0.15,
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
    };

    let contracts = build_linear_contract(&spec).expect("interaction now supported");
    let dt = contracts[0]
        .design_test
        .as_ref()
        .unwrap_or(&contracts[0].design_generation);
    // x1:group ⇒ exactly two Interaction terms (one per non-reference level), each
    // Direct(x1) × DummyOf(group, level k) — pins the dummy level_index so a B/C
    // swap or a wrong component fails (was "some Interaction exists").
    let interactions: Vec<_> = dt
        .terms
        .iter()
        .filter_map(|t| match t {
            DesignTerm::Interaction { components } => Some(components),
            _ => None,
        })
        .collect();
    assert_eq!(interactions.len(), 2, "terms={:?}", dt.terms);
    for (k, comps) in interactions.iter().enumerate() {
        assert_eq!(comps.len(), 2, "interaction {k} arity");
        assert!(
            matches!(&comps[0], DesignTerm::Direct { column } if *column == ColumnId(0)),
            "interaction {k} comp0 = {:?}",
            comps[0]
        );
        match &comps[1] {
            DesignTerm::DummyOf {
                column,
                level_index,
            } => {
                assert_eq!(*column, ColumnId(1));
                assert_eq!(*level_index as usize, k + 1);
            }
            other => panic!("interaction {k} comp1 expected DummyOf, got {other:?}"),
        }
    }
}

#[test]
fn build_linear_contract_generation_column_kinds() {
    // A 3-level factor must become FactorSynthetic, not a Synthetic continuous
    // column — a silent demotion would pass the downstream oracle.
    use engine_contract::{ColumnSpec, SyntheticKind};
    use engine_spec_builder::build_linear_contract;
    use engine_spec_builder::input::{
        Correction, EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec,
        ResidualSpec, VarKind,
    };

    let spec = LinearSpec {
        formula: "y = x1 + group".into(),
        predictors: vec![
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
                    proportions: vec![0.34, 0.33, 0.33],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ],
        effects: vec![
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
    };

    let contracts = build_linear_contract(&spec).expect("build");
    let cols = &contracts[0].generation.columns;
    assert_eq!(cols.len(), 2);
    assert!(
        matches!(
            cols[0],
            ColumnSpec::Synthetic {
                kind: SyntheticKind::Normal,
                ..
            }
        ),
        "col[0] must be Synthetic(Normal), got {:?}",
        cols[0]
    );
    assert!(
        matches!(cols[1], ColumnSpec::FactorSynthetic { n_levels: 3, .. }),
        "col[1] must be FactorSynthetic{{n_levels:3}}, got {:?}",
        cols[1]
    );
}
