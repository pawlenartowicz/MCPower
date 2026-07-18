//! Pre-projection and post-expansion validation of `LinearSpec` fields; rejects invalid inputs before the engine sees them.

use crate::error::SpecError;
use crate::input::{LinearSpec, VarKind};

/// Validate `LinearSpec` fields that aren't checked elsewhere (formula
/// parser, predictor table, correlation builder, targets resolver each
/// cover their own slice).
///
/// # Errors
/// - [`SpecError::NonFiniteEffect`]: a non-finite effect size.
/// - [`SpecError::FactorProportionNonPositive`]: a binary/factor proportion ≤ 0, > 1, or non-finite.
/// - [`SpecError::FactorLevelCount`]: a factor level count outside the configured bounds.
/// - [`SpecError::FactorProportionLengthMismatch`]: proportions length ≠ level count.
/// - [`SpecError::FactorProportionSum`]: factor proportions don't sum to 1.
/// - [`SpecError::FactorReferenceMissing`]: the reference level is absent from `levels`.
pub fn validate_pre_projection(spec: &LinearSpec) -> Result<(), SpecError> {
    // Alpha validity (0,1) is the contract invariant `invariant_15`, enforced at
    // every dispatch via `contract_to_simulation_spec`; the `max_alpha` ceiling is
    // a host-side soft warning, not a hard reject — so neither lives here.
    // Factor-level bounds are single-sourced from the embedded `configs/config.json`.
    let limits = engine_contract::config().limits;
    let (factor_min, factor_max) = (
        limits.factor_levels[0] as usize,
        limits.factor_levels[1] as usize,
    );

    for eff in &spec.effects {
        if !eff.size.is_finite() {
            return Err(SpecError::NonFiniteEffect {
                name: eff.name.clone(),
                value: eff.size,
            });
        }
    }

    for pred in &spec.predictors {
        match &pred.kind {
            VarKind::Binary { proportion }
                if !(0.0..=1.0).contains(proportion) || !proportion.is_finite() =>
            {
                return Err(SpecError::FactorProportionNonPositive {
                    name: pred.name.clone(),
                });
            }
            VarKind::Binary { .. } => {}
            VarKind::Factor {
                levels,
                proportions,
                reference,
                ..
            } => {
                if levels.len() < factor_min || levels.len() > factor_max {
                    return Err(SpecError::FactorLevelCount {
                        name: pred.name.clone(),
                        got: levels.len(),
                        min: factor_min,
                        max: factor_max,
                    });
                }
                if proportions.len() != levels.len() {
                    return Err(SpecError::FactorProportionLengthMismatch {
                        name: pred.name.clone(),
                        expected: levels.len(),
                        got: proportions.len(),
                    });
                }
                if proportions.iter().any(|p| !p.is_finite() || *p <= 0.0) {
                    return Err(SpecError::FactorProportionNonPositive {
                        name: pred.name.clone(),
                    });
                }
                let sum: f64 = proportions.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(SpecError::FactorProportionSum {
                        name: pred.name.clone(),
                        sum,
                    });
                }
                if !levels.iter().any(|l| l == reference) {
                    return Err(SpecError::FactorReferenceMissing {
                        name: pred.name.clone(),
                        reference: reference.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Validate that every parsed effect (after factor expansion) has an entry
/// in `spec.effects`, and every entry refers to a real effect.
///
/// # Errors
/// - [`SpecError::EffectCountMismatch`]: `spec.effects` length differs from `effect_names`, or duplicate effect names reduce the set.
/// - [`SpecError::UnknownTarget`]: an effect name in `spec.effects` is not in `effect_names`.
pub fn validate_effect_assignments(
    spec: &LinearSpec,
    effect_names: &[String],
) -> Result<(), SpecError> {
    if spec.effects.len() != effect_names.len() {
        return Err(SpecError::EffectCountMismatch {
            expected: effect_names.len(),
            got: spec.effects.len(),
        });
    }
    let parsed: std::collections::BTreeSet<&str> =
        effect_names.iter().map(String::as_str).collect();
    for eff in &spec.effects {
        if !parsed.contains(eff.name.as_str()) {
            return Err(SpecError::UnknownTarget {
                name: eff.name.clone(),
            });
        }
    }
    let assigned: std::collections::BTreeSet<&str> =
        spec.effects.iter().map(|e| e.name.as_str()).collect();
    if assigned.len() != spec.effects.len() {
        return Err(SpecError::EffectCountMismatch {
            expected: effect_names.len(),
            got: assigned.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::*;

    fn base_spec() -> LinearSpec {
        LinearSpec {
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
            wald_se: Default::default(),
            nagq: 1,
        }
    }

    #[test]
    fn happy_path_passes_pre_projection() {
        assert!(validate_pre_projection(&base_spec()).is_ok());
    }

    #[test]
    fn alpha_above_warn_threshold_passes_pre_projection() {
        // The 0.25 ceiling is now a host-side soft warning, not a spec-builder
        // hard reject; only the contract invariant `(0,1)` blocks alpha. So an
        // alpha above `max_alpha` must pass pre-projection unchanged.
        let mut s = base_spec();
        s.alpha = 0.3;
        assert!(validate_pre_projection(&s).is_ok());
    }

    #[test]
    fn factor_proportions_must_sum_to_one() {
        let mut s = base_spec();
        s.predictors = vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into()],
                proportions: vec![0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        s.effects = vec![EffectAssignment {
            name: "g[B]".into(),
            size: 0.5,
        }];
        s.formula = "y = g".into();
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorProportionSum { .. })
        ));
    }

    #[test]
    fn effect_count_must_match_after_expansion() {
        // Two parsed effects but only one assignment.
        let spec = LinearSpec {
            effects: vec![EffectAssignment {
                name: "x1".into(),
                size: 0.5,
            }],
            ..base_spec()
        };
        let err = validate_effect_assignments(&spec, &["x1".into(), "x2".into()]).unwrap_err();
        assert!(matches!(err, SpecError::EffectCountMismatch { .. }));
    }

    // Non-finite effect sizes (NaN, ±Inf) are rejected before reaching the engine.
    #[test]
    fn non_finite_effect_rejected() {
        let mut s = base_spec();
        s.effects[0].size = f64::NAN;
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::NonFiniteEffect { .. })
        ));
    }

    // A binary predictor proportion outside [0, 1] is rejected.
    #[test]
    fn binary_proportion_out_of_range_rejected() {
        let mut s = base_spec();
        s.predictors = vec![PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Binary { proportion: 1.5 },
        }];
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorProportionNonPositive { .. })
        ));
    }

    // A factor with fewer than 2 or more than 20 levels is rejected.
    #[test]
    fn factor_level_count_out_of_range_rejected() {
        let mut s = base_spec();
        s.formula = "y = g".into();
        s.predictors = vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into()],
                proportions: vec![1.0],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        s.effects = vec![];
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorLevelCount { .. })
        ));
    }

    // A factor whose proportions length differs from levels length is rejected.
    #[test]
    fn factor_proportion_length_mismatch_rejected() {
        let mut s = base_spec();
        s.formula = "y = g".into();
        s.predictors = vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into()],
                proportions: vec![1.0],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        s.effects = vec![EffectAssignment {
            name: "g[B]".into(),
            size: 0.5,
        }];
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorProportionLengthMismatch { .. })
        ));
    }

    // A factor with a non-positive proportion is rejected.
    #[test]
    fn factor_non_positive_proportion_rejected() {
        let mut s = base_spec();
        s.formula = "y = g".into();
        s.predictors = vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into()],
                proportions: vec![1.2, -0.2],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        s.effects = vec![EffectAssignment {
            name: "g[B]".into(),
            size: 0.5,
        }];
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorProportionNonPositive { .. })
        ));
    }

    // A factor whose declared reference level is not among its levels is rejected.
    #[test]
    fn factor_reference_missing_rejected() {
        let mut s = base_spec();
        s.formula = "y = g".into();
        s.predictors = vec![PredictorSpec {
            name: "g".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into()],
                proportions: vec![0.5, 0.5],
                reference: "Z".into(),
                sampled_proportions: None,
            },
        }];
        s.effects = vec![EffectAssignment {
            name: "g[B]".into(),
            size: 0.5,
        }];
        assert!(matches!(
            validate_pre_projection(&s),
            Err(SpecError::FactorReferenceMissing { .. })
        ));
    }

    // An effect name not present in the expanded effect_names set is rejected
    // as UnknownTarget (distinct from the count-mismatch path).
    #[test]
    fn unknown_effect_name_rejected() {
        let spec = LinearSpec {
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "nonexistent".into(),
                    size: 0.3,
                },
            ],
            ..base_spec()
        };
        // Count matches (2 == 2) so this exercises the name-membership path,
        // not the count check.
        let err = validate_effect_assignments(&spec, &["x1".into(), "x2".into()]).unwrap_err();
        assert!(matches!(err, SpecError::UnknownTarget { .. }));
    }

    #[test]
    fn duplicate_effect_names_rejected() {
        // Two effects with the same name should be caught even when total count
        // matches — without this, the projection step would silently miss an
        // expanded effect and panic later when assembling effect_sizes.
        let spec = LinearSpec {
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.3,
                },
            ],
            ..base_spec()
        };
        let err = validate_effect_assignments(&spec, &["x1".into(), "x2".into()]).unwrap_err();
        assert!(matches!(err, SpecError::EffectCountMismatch { .. }));
    }
}
