//! JSON-wire contracts of the WASM worker path, run natively.
//!
//! A browser worker runs `run_single_core_find_power`, serialises the part with
//! `serde_json`, posts it to the main thread, which decodes it and calls
//! `merge_power_results`; formula parsing crosses the same boundary as JSON.
//! The `engine-wasm` exports are thin serde wrappers over exactly these calls,
//! so the wire shapes are pinned here, where `cargo test` runs them.

use engine_app_spec::{
    parse_formula, run_single_core_find_power, AppSpec, EffectSize, LinearSpec, LogitSpec,
    NullEmitter, ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    merge_power_results, CancellationToken, EstimatorExtras, PowerResult, ScenarioResult,
};
use serde_json::json;

/// `FormulaParse` serialises to the key shape the app reads
/// (`ports/app/src/lib/domain/result.ts` `FormulaParse`, e.g. `parsed.dependent`).
#[test]
fn parse_formula_serializes_to_the_app_key_shape() {
    let parsed = parse_formula("y ~ x1 + x2").expect("parse ok");
    let v = serde_json::to_value(&parsed).expect("serializes");
    assert_eq!(
        v,
        json!({
            "dependent": "y",
            "predictors": ["x1", "x2"],
            "terms": [
                { "kind": "main", "name": "x1" },
                { "kind": "main", "name": "x2" }
            ],
            "random_effects": []
        })
    );
}

/// EP-1: power/ci/count vectors are sized `target_indices.len() + contrast_pairs.len()`.
///
/// This test carries a real contrast through the WASM worker path (JSON wire
/// round-trip AND `merge_power_results`), so a mis-sizing in either the adapter
/// or the merge surfaces here rather than only at the orchestrator's
/// merge-mismatch gate.
///
/// Design: 3-level factor `treatment` (levels "1"/"2"/"3", reference "1").
///   - 1 Marginal target  → `treatment[2]`
///   - 1 Contrast target  → `treatment[2]` vs `treatment[3]` (both non-reference)
///
/// Expected: `power_uncorrected.len() == 2`, `contrast_pairs.len() == 1`,
/// `success_counts_uncorrected.len() == 2`.
#[test]
fn contrast_bearing_spec_ep1_vector_lengths_survive_wire_and_merge() {
    // 3-level factor, reference level "1": dummies "2" and "3" become design
    // columns 1 and 2. The contrast pair ("treatment[2]", "treatment[3]")
    // maps to Contrast { positive: 1, negative: 2 } — both sides are
    // non-reference, so the assembler emits a true contrast, not a marginal.
    let spec = AppSpec::Linear(LinearSpec {
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
        n_sims: 128,
        seed: 2137,
        // One explicit marginal target + one contrast pair → total 2 result slots.
        tests: TestSelection::Effects {
            names: vec!["treatment[2]".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts: vec![("treatment[2]".into(), "treatment[3]".into())],
        test_formula: None,
        outcome_options: None,
    });

    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 200, 11, &NullEmitter, &cancel).expect("run ok");

    // Wire round-trip: same path the WASM worker takes handing results to the
    // main thread over the JS boundary.
    let json = serde_json::to_string(&part).expect("serializes");
    let back: ScenarioResult<PowerResult> =
        serde_json::from_str(&json).expect("round-trips through JSON wire");

    let (_, power) = &back.scenarios[0];

    // EP-1: one marginal + one contrast → both result vectors must be length 2.
    assert_eq!(
        power.contrast_pairs.len(),
        1,
        "expected exactly 1 contrast pair"
    );
    assert_eq!(
        power.power_uncorrected.len(),
        power.target_indices.len() + power.contrast_pairs.len(),
        "EP-1: power_uncorrected must be sized target_indices + contrast_pairs"
    );
    assert_eq!(
        power.success_counts_uncorrected.len(),
        power.power_uncorrected.len(),
        "success_counts_uncorrected must parallel power_uncorrected"
    );

    // Merge path: same path the WASM main thread takes when aggregating two workers.
    let merged = merge_power_results(&[back.clone(), back]).expect("merge ok");
    let (_, merged_power) = &merged.scenarios[0];

    assert_eq!(
        merged_power.contrast_pairs.len(),
        1,
        "merge must preserve contrast_pairs"
    );
    assert_eq!(
        merged_power.power_uncorrected.len(),
        merged_power.target_indices.len() + merged_power.contrast_pairs.len(),
        "EP-1 must hold in the merged result"
    );
    assert_eq!(
        merged_power.n_sims, 400,
        "merged n_sims should sum the two parts"
    );
}

/// EP-2: NaN-bearing GLM extras survive the JSON wire and `merge_power_results`.
///
/// `serde_json` renders `f64::NAN` as JSON `null` and rejects `null` back as an
/// `f64` unless the field uses the `nan_tolerant` deserializer. Any logit result
/// from a single-core (WASM worker) run carries exactly two structural NaN fields:
///   - `baseline_prob_realized` — computed as `sum/n`, but per-worker `n == 0`
///     (the accumulator is filled only at `merge_power_results` time, not in the
///     single-core path's `EstimatorExtras::from_batch`)
///   - `tau_squared_hat_mean` — `tau_sum/tau_n`, with `tau_n == 0` for any
///     unclustered logit spec (no Laplace random-effect estimates)
///
/// The production WASM path is: worker calls `run_single_core_find_power` →
/// `serde_json::to_string` → postMessage → main thread `serde_json::from_str` →
/// `merge_power_results`. This test pins that `nan_tolerant` guards those fields,
/// so removing the guard later breaks this test rather than silently crashing a
/// live browser merge.
#[test]
fn nan_bearing_extras_survive_json_wire_and_merge() {
    // A minimal unclustered logit spec — `baseline_probability` anchors the
    // GLM intercept; effects are small so the run doesn't take long.
    let spec = AppSpec::Logit(LogitSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric {
            name: "x1".into(),
            distribution: Default::default(),
            pinned: false,
        }],
        effects: vec![EffectSize {
            name: "x1".into(),
            value: 0.3,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 64,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        wald_se: Default::default(),
        link: Default::default(),
        agq: 1,
        scenarios: vec![],
        csv: None,
        baseline_probability: 0.3,
        test_formula: None,
        outcome_options: None,
    });

    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 64, 11, &NullEmitter, &cancel).expect("run ok");

    // The single-core result must carry GLM extras with NaN in both structural
    // placeholders: baseline_prob_realized (accumulator only filled at merge,
    // never in from_batch) and tau_squared_hat_mean (tau_n==0, no RE estimates).
    let (_, pr) = &part.scenarios[0];
    let EstimatorExtras::Glm {
        baseline_prob_realized,
        tau_squared_hat_mean,
        ..
    } = &pr.estimator_extras
    else {
        panic!("logit run must yield Glm extras");
    };
    assert!(
        baseline_prob_realized.is_nan(),
        "baseline_prob_realized must be NaN before merge (n==0 in from_batch)"
    );
    assert!(
        tau_squared_hat_mean.is_nan(),
        "tau_squared_hat_mean must be NaN for unclustered logit (tau_n==0)"
    );

    // JSON wire: serde_json encodes NaN as null; nan_tolerant must decode it back.
    let json = serde_json::to_string(&part).expect("serializes");
    assert!(
        json.contains("null"),
        "NaN must appear as JSON null before the wire: {json}"
    );
    let back: ScenarioResult<PowerResult> =
        serde_json::from_str(&json).expect("EP-2: JSON null→NaN decode must not fail");

    let EstimatorExtras::Glm {
        baseline_prob_realized: bp_back,
        tau_squared_hat_mean: tau_back,
        ..
    } = &back.scenarios[0].1.estimator_extras
    else {
        panic!("extras must remain Glm after JSON round-trip");
    };
    assert!(
        bp_back.is_nan(),
        "baseline_prob_realized must survive JSON null→NaN"
    );
    assert!(
        tau_back.is_nan(),
        "tau_squared_hat_mean must survive JSON null→NaN"
    );

    // Merge path: two decoded worker parts merged by the WASM main thread.
    // The merge must succeed (not panic or error on the NaN extras) and keep
    // the Glm extras variant; the merged extras' values are the orchestrator
    // merge's own contract.
    let merged = merge_power_results(&[back.clone(), back])
        .expect("EP-2: merge must not fail on NaN extras");
    let (_, merged_pr) = &merged.scenarios[0];
    assert!(
        matches!(merged_pr.estimator_extras, EstimatorExtras::Glm { .. }),
        "merged extras must remain Glm"
    );
    assert_eq!(
        merged_pr.n_sims, 128,
        "merged n_sims must sum the two parts"
    );
}
