use engine_app_spec::{AppSpec, EffectSize, LinearSpec, ParsedFormula, TestSelection, VarType};
use engine_contract::CorrectionMethod;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Build a minimal AppSpec with a single 3-level factor `treatment`
/// (levels "1", "2", "3", reference "1") and no other predictors.
/// The `contrasts` and `tests` fields are supplied by each test.
fn three_level_factor_spec(tests: TestSelection, contrasts: Vec<(String, String)>) -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["treatment".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Factor {
            name: "treatment".into(),
            factor_n_levels: 3,
            factor_proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
        }],
        effects: vec![
            EffectSize {
                name: "treatment[2]".into(),
                value: 0.3,
            },
            EffectSize {
                name: "treatment[3]".into(),
                value: 0.5,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 100,
        seed: 2137,
        tests,
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts,
        test_formula: None,
        outcome_options: None,
    })
}

// ── Test 1: serde round-trip ──────────────────────────────────────────────────

#[test]
fn contrasts_field_round_trips() {
    use serde_json;

    // With a non-empty contrasts field: round-trip preserves the pairs.
    let spec = three_level_factor_spec(
        TestSelection::All,
        vec![("treatment[B]".into(), "treatment[C]".into())],
    );
    let json = serde_json::to_string(&spec).expect("serialize");
    let decoded: AppSpec = serde_json::from_str(&json).expect("deserialize");
    let AppSpec::Linear(inner) = decoded else {
        panic!("expected AppSpec::Linear");
    };
    assert_eq!(
        inner.contrasts,
        vec![("treatment[B]".to_string(), "treatment[C]".to_string())],
        "contrasts field must survive JSON round-trip"
    );

    // Without the contrasts key in JSON: serde default = empty vec.
    let json_no_contrasts = serde_json::json!({
        "family": "linear",
        "parsed_formula": { "outcome": "y", "predictors": ["x1"], "interaction_terms": [] },
        "var_types": [{ "kind": "numeric", "name": "x1" }],
        "effects": [{ "name": "x1", "value": 0.3 }],
        "correlations": null,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_sims": 100,
        "seed": 1,
        "tests": { "kind": "all" },
        "correction": "none",
        "scenarios": [],
        "csv": null,
        "report_overall": false
        // no "contrasts" key
    });
    let decoded_no_key: AppSpec =
        serde_json::from_value(json_no_contrasts).expect("deserialize without contrasts key");
    let AppSpec::Linear(inner2) = decoded_no_key else {
        panic!("expected AppSpec::Linear");
    };
    assert!(
        inner2.contrasts.is_empty(),
        "missing contrasts key must deserialize to empty vec via serde default"
    );
}

// ── Test 2: assemble routing ──────────────────────────────────────────────────

/// The assembler must forward AppSpec.contrasts → builder.contrast_pairs.
///
/// Factor `treatment`: levels ["1","2","3"], reference "1" (assemble.rs uses
/// integer string levels in this order for Factor with factor_n_levels).
/// Effect names in the contract: "treatment[2]", "treatment[3]".
///
/// Pair ("treatment[2]", "treatment[1]"):
///   "treatment[1]" is the reference → collapses to Marginal on treatment[2].
///
/// Pair ("treatment[2]", "treatment[3]"):
///   neither is the reference → emits Contrast { positive, negative }.
#[test]
fn assemble_emits_marginal_for_ref_pair_and_contrast_for_nonref_pair() {
    // Use TestSelection::Effects { names: [] } so no extra Marginals from
    // `targets` are emitted — only the contrast pairs contribute to targets.
    let spec = three_level_factor_spec(
        TestSelection::Effects { names: vec![] },
        vec![
            ("treatment[2]".into(), "treatment[1]".into()), // ref side → Marginal
            ("treatment[2]".into(), "treatment[3]".into()), // non-ref → Contrast
        ],
    );

    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble should succeed");
    assert_eq!(contracts.len(), 1);
    let targets = &contracts[0].test.targets;

    // Design terms for a sole 3-level factor (reference "1"):
    //   index 0 → Const (intercept / reference level "1")
    //   index 1 → DummyOf treatment[2]
    //   index 2 → DummyOf treatment[3]
    //
    // Pair ("treatment[2]", "treatment[1]"): "treatment[1]" is the reference,
    // so the pair collapses to Marginal { term: 1 } (treatment[2]'s position).
    //
    // Pair ("treatment[2]", "treatment[3]"): both non-reference, so a
    // Contrast { positive: 1, negative: 2 } is emitted.

    let marginal_term = targets
        .iter()
        .find_map(|t| match t {
            engine_contract::TestTarget::Marginal { term } => Some(*term),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected exactly 1 Marginal; got targets: {targets:?}"));

    let (contrast_positive, contrast_negative) = targets
        .iter()
        .find_map(|t| match t {
            engine_contract::TestTarget::Contrast { positive, negative } => {
                Some((*positive, *negative))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected exactly 1 Contrast; got targets: {targets:?}"));

    assert_eq!(
        targets.len(),
        2,
        "expected exactly 2 targets; got: {targets:?}"
    );

    assert_eq!(
        marginal_term, 1,
        "Marginal must point to term index 1 (treatment[2]); got: {targets:?}"
    );
    assert_eq!(
        contrast_positive, 1,
        "Contrast positive must be term index 1 (treatment[2]); got: {targets:?}"
    );
    assert_eq!(
        contrast_negative, 2,
        "Contrast negative must be term index 2 (treatment[3]); got: {targets:?}"
    );
}
