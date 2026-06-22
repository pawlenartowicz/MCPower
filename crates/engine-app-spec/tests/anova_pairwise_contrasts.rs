//! Regression guard for the ANOVA UI's auto-generated full pairwise contrast
//! set. The app (anova-contrasts.ts::regenAutoContrasts) emits every C(k,2)
//! level pair of a k-level factor — including pairs touching the reference
//! level, which the spec builder collapses to Marginal targets. For k ≥ 4 the
//! total test-target count (marginals + dummy-vs-dummy contrasts) exceeds the
//! predictor count P, which the engine workspace must tolerate.

use engine_app_spec::{
    power_plot_specs, run_single_core_find_power, run_single_core_find_sample_size,
    sample_size_curve_specs, AppSpec, EffectSize, LinearSpec, NullEmitter, ParsedFormula,
    TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{ByValue, CancellationToken, GridMode, SampleSizeMethod};

/// One-way ANOVA with a single k-level factor, no covariate — the exact shape
/// the ANOVA entrypoint sends: tests = Effects{names: []} (no marginal
/// targets), contrasts = all C(k,2) pairwise level pairs (reference included).
fn one_way_anova_spec(k: usize, correction: CorrectionMethod) -> AppSpec {
    let mut contrasts = Vec::new();
    for i in 1..=k {
        for j in (i + 1)..=k {
            contrasts.push((format!("F1[{i}]"), format!("F1[{j}]")));
        }
    }
    let effects = (2..=k)
        .map(|lv| EffectSize {
            name: format!("F1[{lv}]"),
            value: 0.4,
        })
        .collect();
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["F1".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Factor {
            name: "F1".into(),
            factor_n_levels: k as u32,
            factor_proportions: vec![1.0 / k as f64; k],
            factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
        }],
        effects,
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 60,
        seed: 2137,
        tests: TestSelection::Effects { names: vec![] },
        correction,
        scenarios: vec![],
        csv: None,
        report_overall: true,
        contrasts,
        test_formula: None,
        outcome_options: None,
    })
}

fn assert_power_runs(k: usize, correction: CorrectionMethod) {
    let cancel = CancellationToken::default();
    let spec = one_way_anova_spec(k, correction);
    let result = run_single_core_find_power(&spec, 120, 60, 2137, &NullEmitter, &cancel)
        .unwrap_or_else(|e| panic!("k={k} pairwise-contrast power run failed: {e}"));
    let (_, pr) = &result.scenarios[0];
    // k−1 reference pairs collapse to marginals; C(k−1,2) dummy pairs stay contrasts.
    let n_marginals = k - 1;
    let n_contrasts = (k - 1) * (k - 2) / 2;
    assert_eq!(pr.target_indices.len(), n_marginals);
    assert_eq!(pr.contrast_pairs.len(), n_contrasts);
    assert_eq!(pr.power_uncorrected.len(), n_marginals + n_contrasts);
    assert!(pr
        .power_uncorrected
        .iter()
        .all(|&p| (0.0..=1.0).contains(&p)));
}

#[test]
fn pairwise_contrasts_k3_power() {
    assert_power_runs(3, CorrectionMethod::None);
}

#[test]
fn pairwise_contrasts_k4_power() {
    assert_power_runs(4, CorrectionMethod::None);
}

#[test]
fn pairwise_contrasts_k5_power() {
    assert_power_runs(5, CorrectionMethod::None);
}

#[test]
fn pairwise_contrasts_k4_tukey_power() {
    assert_power_runs(4, CorrectionMethod::TukeyHsd);
}

#[test]
fn pairwise_contrasts_k4_holm_power() {
    assert_power_runs(4, CorrectionMethod::Holm);
}

fn assert_sample_size_covers_contrasts(k: usize) {
    let cancel = CancellationToken::default();
    let spec = one_way_anova_spec(k, CorrectionMethod::None);
    let method = SampleSizeMethod::Grid {
        by: ByValue::Auto { count: 6 },
        mode: GridMode::Linear,
    };
    let ssr =
        run_single_core_find_sample_size(&spec, (20, 200), method, 60, 2137, &NullEmitter, &cancel)
            .unwrap_or_else(|e| panic!("k={k} pairwise-contrast sample-size grid run failed: {e}"));
    let (_, sres) = &ssr.scenarios[0];
    // Required-N derivations must carry one slot per power-vector entry —
    // marginals then contrasts — not just per target_indices entry.
    let expected = (k - 1) + (k - 1) * (k - 2) / 2;
    assert_eq!(sres.first_achieved.len(), expected);
    assert_eq!(sres.fitted.len(), expected);
}

#[test]
fn pairwise_contrasts_k3_sample_size_grid() {
    assert_sample_size_covers_contrasts(3);
}

/// Plot emitters must tolerate power vectors longer than `target_indices`
/// (contrast entries ride at the end with no kernel column index). k=3 → 2
/// marginals + 1 contrast: before the fix, `power_at_n_spec` /
/// `sample_size_curve_spec` panicked at `target_indices[2]` (len 2, index 2).
#[test]
fn pairwise_contrasts_k3_plot_specs() {
    let cancel = CancellationToken::default();
    let spec = one_way_anova_spec(3, CorrectionMethod::None);

    let pr = run_single_core_find_power(&spec, 120, 60, 2137, &NullEmitter, &cancel)
        .expect("k=3 power run failed");
    let blocks = power_plot_specs(&pr, 0.8, false).blocks;
    assert!(!blocks.is_empty(), "power plot specs must be emitted");
    // The bar chart must include the contrast entry under its identity-bearing
    // token (target_{p}_vs_{n} from plot.rs entry_label), built from the
    // surfaced contrast_pairs — hosts relabel it to a real name.
    let (_, pres) = &pr.scenarios[0];
    assert_eq!(pres.contrast_pairs.len(), 1);
    let (p, n) = pres.contrast_pairs[0];
    let token = format!("target_{p}_vs_{n}");
    let bar = &blocks[0].spec;
    assert!(
        bar.contains(&token),
        "contrast entry must appear in the power plot as {token}: {bar}"
    );

    let method = SampleSizeMethod::Grid {
        by: ByValue::Auto { count: 6 },
        mode: GridMode::Linear,
    };
    let ssr =
        run_single_core_find_sample_size(&spec, (20, 200), method, 60, 2137, &NullEmitter, &cancel)
            .expect("k=3 sample-size run failed");
    let blocks = sample_size_curve_specs(&ssr, 0.8, false).blocks;
    assert!(!blocks.is_empty(), "curve plot specs must be emitted");
}

#[test]
fn pairwise_contrasts_k4_sample_size_grid() {
    assert_sample_size_covers_contrasts(4);
}
