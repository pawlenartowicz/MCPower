//! Builds the `PredictorTable` by expanding factors into dummies and mapping every effect to its design-matrix column index.
//!
//! Column-ordering invariant (owning explanation — `targets.rs`, `upload.rs`, `project_contract.rs` cite this):
//! design column 0 = intercept; non-factor predictors occupy columns 1..=n_non_factor (in formula `predictors` order);
//! factor dummies follow at n_non_factor+1.. (in factor declaration order, reference level omitted).

use crate::error::SpecError;
use crate::formula::{ParsedFormula, Term};
use crate::input::{PredictorSpec, VarKind};
use crate::skeleton::EffectDescriptor;

/// Resolved view of all predictors after factor expansion. Column order is
/// `non_factor_columns ++ dummy_columns` (the engine's expected layout).
#[derive(Debug, Clone)]
pub struct PredictorTable {
    pub non_factor_names: Vec<String>,
    pub var_types: Vec<i32>,
    /// Per-non-factor pin (aligned with `var_types`): `true` = explicitly
    /// chosen distribution, scenario swaps leave the column alone.
    pub var_pinned: Vec<bool>,
    pub var_params: Vec<f64>,
    /// Factor declaration order (matches input ordering).
    pub factor_n_levels: Vec<i32>,
    pub factor_proportions: Vec<f64>,
    /// Per-factor proportion-sampling override, aligned with `factor_n_levels`.
    /// `None` = inherit scenario `sampled_factor_proportions`; `Some(true/false)` = exact override.
    pub factor_sampled: Vec<Option<bool>>,
    /// Factor names in declaration order — parallel to `factor_n_levels`. Used
    /// by callers (e.g. `project_contract` building `design_test` from a
    /// separate `test_formula`) that need to map a factor's name back to its
    /// generation-column position.
    pub factor_names: Vec<String>,
    /// Per-factor level lists (non-reference levels in declaration order).
    /// Parallel to `factor_names`. `level_index` for a dummy "group[B]" is
    /// `factor_levels[factor_index].iter().position(|l| l == "B")? + 1` —
    /// the +1 is because contract dummy indices are 1-based (level 0 is the
    /// reference, dropped from the design matrix).
    pub factor_levels: Vec<Vec<String>>,
    /// Reference level string for each factor, in declaration order.
    /// Parallel to `factor_names` and `factor_levels`.
    pub factor_references: Vec<String>,
    /// All effect names in projection order (main effects + interactions),
    /// after factor expansion. Used by the targets resolver and the effect-
    /// size assembler.
    pub effect_names: Vec<String>,
    /// For each effect name, its column index in the design matrix
    /// (intercept = 0; first non-factor predictor = 1; ...). Interactions
    /// have one index per component.
    pub effect_columns: Vec<Vec<u32>>,
    /// Parallel to `effect_names`. For an interaction effect, its components in
    /// formula order as `(predictor_name, level)`: `level = None` for a
    /// continuous component, `Some(level_name)` for a factor cell. Empty for
    /// non-interaction effects. Names (not column indices) so both the
    /// generation design and a name-remapped test design can resolve them
    /// without inverting design-matrix positions.
    pub interaction_components: Vec<Vec<(String, Option<String>)>>,
    /// Parallel to `effect_names`. A structured, index-only descriptor for each
    /// effect — continuous main effect, factor dummy, or interaction. For a
    /// dummy, `level` is the 0-based position of its level within the factor's
    /// FULL declared level list (reference included), so the port renders
    /// `labels[factor][level]` directly. Reordered into β-column space (and
    /// prefixed with the intercept) by `build_effect_skeleton`.
    pub effect_descriptors: Vec<EffectDescriptor>,
}

pub fn build_predictor_table(
    formula: &ParsedFormula,
    predictors: &[PredictorSpec],
) -> Result<PredictorTable, SpecError> {
    use std::collections::HashMap;

    // Index predictors by name for fast lookup.
    let mut by_name: HashMap<&str, &PredictorSpec> = HashMap::new();
    for p in predictors {
        by_name.insert(p.name.as_str(), p);
    }

    // Every formula-mentioned predictor must have a PredictorSpec.
    for name in &formula.predictors {
        if !by_name.contains_key(name.as_str()) {
            return Err(SpecError::UnknownPredictor { name: name.clone() });
        }
    }

    // Split into non-factors (in formula order) and factors (in formula order).
    let mut non_factor_names: Vec<String> = Vec::new();
    let mut var_types: Vec<i32> = Vec::new();
    let mut var_pinned: Vec<bool> = Vec::new();
    let mut var_params: Vec<f64> = Vec::new();
    let mut factor_n_levels: Vec<i32> = Vec::new();
    let mut factor_proportions: Vec<f64> = Vec::new();
    let mut factor_sampled: Vec<Option<bool>> = Vec::new();
    let mut factor_levels: Vec<(String, Vec<String>, String)> = Vec::new(); // (name, levels, reference)

    for name in &formula.predictors {
        let pred = by_name[name.as_str()];
        match &pred.kind {
            VarKind::Factor {
                levels,
                proportions,
                reference,
                sampled_proportions,
            } => {
                factor_n_levels.push(levels.len() as i32);
                factor_proportions.extend(proportions.iter().copied());
                factor_sampled.push(*sampled_proportions);
                factor_levels.push((name.clone(), levels.clone(), reference.clone()));
            }
            other => {
                non_factor_names.push(name.clone());
                var_types.push(distribution_code(other));
                var_pinned.push(pred.pinned);
                var_params.push(distribution_param(other));
            }
        }
    }

    // Build effect_names + effect_columns + interaction_components by applying
    // factor expansion to every Term in the formula.
    let mut effect_names: Vec<String> = Vec::new();
    let mut effect_columns: Vec<Vec<u32>> = Vec::new();
    let mut interaction_components: Vec<Vec<(String, Option<String>)>> = Vec::new();
    let mut effect_descriptors: Vec<EffectDescriptor> = Vec::new();

    // Column index helper: continuous columns 1..=n_nf; dummy columns follow.
    let n_nf = non_factor_names.len() as u32;
    let dummy_column_for = build_dummy_column_index(&factor_levels, n_nf);

    let factor_set: std::collections::BTreeSet<&str> =
        factor_levels.iter().map(|(n, _, _)| n.as_str()).collect();

    for term in &formula.terms {
        match term {
            Term::Main { name } => {
                if factor_set.contains(name.as_str()) {
                    // Factor main effect expands to k-1 dummies.
                    let (_, levels, reference) =
                        factor_levels.iter().find(|(n, _, _)| n == name).unwrap();
                    for lvl in levels {
                        if lvl == reference {
                            continue;
                        }
                        let dummy = format!("{name}[{lvl}]");
                        let col = dummy_column_for(&dummy);
                        // `level` indexes the FULL declared level list (reference
                        // included), matching the port's `labels[factor]` store.
                        let level = levels
                            .iter()
                            .position(|l| l == lvl)
                            .expect("lvl drawn from levels")
                            as u32;
                        effect_columns.push(vec![col]);
                        effect_names.push(dummy);
                        interaction_components.push(Vec::new());
                        effect_descriptors.push(EffectDescriptor::FactorLevel {
                            factor: name.clone(),
                            level,
                        });
                    }
                } else {
                    // Continuous main effect.
                    let col = column_of_non_factor(&non_factor_names, name);
                    effect_columns.push(vec![col]);
                    effect_names.push(name.clone());
                    interaction_components.push(Vec::new());
                    effect_descriptors.push(EffectDescriptor::Continuous {
                        predictor: name.clone(),
                    });
                }
            }
            Term::Interaction { vars } => {
                // Per-variable candidate cells, carrying identity: continuous →
                // one cell (display, name, None); factor → one cell per
                // non-reference level (display "v[L]", name "v", Some("L")).
                let per_var: Vec<Vec<(String, String, Option<String>)>> = vars
                    .iter()
                    .map(|v| {
                        if let Some((_, levels, reference)) =
                            factor_levels.iter().find(|(n, _, _)| n == v)
                        {
                            levels
                                .iter()
                                .filter(|l| *l != reference)
                                .map(|l| (format!("{v}[{l}]"), v.clone(), Some(l.clone())))
                                .collect()
                        } else {
                            vec![(v.clone(), v.clone(), None)]
                        }
                    })
                    .collect();
                for combo in cartesian_components(&per_var) {
                    let cols: Vec<u32> = combo
                        .iter()
                        .map(|(display, _, _)| {
                            if display.contains('[') {
                                dummy_column_for(display)
                            } else {
                                column_of_non_factor(&non_factor_names, display)
                            }
                        })
                        .collect();
                    let name = combo
                        .iter()
                        .map(|(display, _, _)| display.as_str())
                        .collect::<Vec<_>>()
                        .join(":");
                    let comps: Vec<(String, Option<String>)> = combo
                        .iter()
                        .map(|(_, pred, level)| (pred.clone(), level.clone()))
                        .collect();
                    let comp_descriptors: Vec<EffectDescriptor> = combo
                        .iter()
                        .map(|(_, pred, level)| match level {
                            None => EffectDescriptor::Continuous {
                                predictor: pred.clone(),
                            },
                            Some(lvl) => EffectDescriptor::FactorLevel {
                                factor: pred.clone(),
                                level: full_level_position(&factor_levels, pred, lvl),
                            },
                        })
                        .collect();
                    effect_columns.push(cols);
                    effect_names.push(name);
                    interaction_components.push(comps);
                    effect_descriptors.push(EffectDescriptor::Interaction {
                        components: comp_descriptors,
                    });
                }
            }
        }
    }

    let factor_names: Vec<String> = factor_levels.iter().map(|(n, _, _)| n.clone()).collect();
    // Per-factor non-reference level lists, in declaration order. Used by
    // callers that need (factor_name, level_name) → contract `level_index`.
    let factor_level_lists: Vec<Vec<String>> = factor_levels
        .iter()
        .map(|(_, levels, reference)| {
            levels
                .iter()
                .filter(|l| *l != reference)
                .cloned()
                .collect::<Vec<String>>()
        })
        .collect();
    // Reference level string per factor, in declaration order. Used by the
    // contrast router to distinguish "factor[reference]" (reference level,
    // maps to Const/β=0) from truly unknown level names.
    let factor_references: Vec<String> = factor_levels.iter().map(|(_, _, r)| r.clone()).collect();

    Ok(PredictorTable {
        non_factor_names,
        var_types,
        var_pinned,
        var_params,
        factor_n_levels,
        factor_proportions,
        factor_sampled,
        factor_names,
        factor_levels: factor_level_lists,
        factor_references,
        effect_names,
        effect_columns,
        interaction_components,
        effect_descriptors,
    })
}

/// 0-based position of `level` within `factor`'s FULL declared level list
/// (reference included) — the index the port uses against its `labels[factor]`
/// store. Both `factor` and `level` are validated upstream.
fn full_level_position(
    factor_levels: &[(String, Vec<String>, String)],
    factor: &str,
    level: &str,
) -> u32 {
    factor_levels
        .iter()
        .find(|(n, _, _)| n == factor)
        .and_then(|(_, levels, _)| levels.iter().position(|l| l == level))
        .expect("factor/level validated upstream") as u32
}

fn distribution_code(kind: &VarKind) -> i32 {
    match kind {
        VarKind::Normal => 0,
        VarKind::Binary { .. } => 1,
        VarKind::RightSkewed => 2,
        VarKind::LeftSkewed => 3,
        VarKind::HighKurtosis => 4,
        VarKind::Uniform => 5,
        VarKind::Factor { .. } => unreachable!("factor handled separately"),
    }
}

fn distribution_param(kind: &VarKind) -> f64 {
    match kind {
        VarKind::Binary { proportion } => *proportion,
        // Non-binary types don't use the param for data generation (ignored by
        // the engine), but the Python frontend defaults `PredictorVar.proportion`
        // to 0.5 for every variable. Mirror that default here so the wire bytes
        // produced by the Rust builder are byte-identical to the Python path.
        _ => 0.5,
    }
}

fn column_of_non_factor(non_factor: &[String], name: &str) -> u32 {
    // Column 0 is the intercept; non-factor predictors start at column 1.
    (non_factor
        .iter()
        .position(|n| n == name)
        .expect("validated upstream")
        + 1) as u32
}

fn build_dummy_column_index<'a>(
    factor_levels: &'a [(String, Vec<String>, String)],
    n_non_factor: u32,
) -> impl Fn(&str) -> u32 + 'a {
    // Compute column number for each dummy in factor-declaration order,
    // non-reference levels in `levels` order.
    let mut map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut col = n_non_factor + 1; // +1 for intercept
    for (factor, levels, reference) in factor_levels {
        for lvl in levels {
            if lvl == reference {
                continue;
            }
            let dummy = format!("{factor}[{lvl}]");
            map.insert(dummy, col);
            col += 1;
        }
    }
    move |dummy: &str| -> u32 { *map.get(dummy).expect("dummy column lookup") }
}

fn cartesian_components(
    lists: &[Vec<(String, String, Option<String>)>],
) -> Vec<Vec<(String, String, Option<String>)>> {
    let mut acc: Vec<Vec<(String, String, Option<String>)>> = vec![vec![]];
    for list in lists {
        let mut next = Vec::new();
        for prefix in &acc {
            for item in list {
                let mut row = prefix.clone();
                row.push(item.clone());
                next.push(row);
            }
        }
        acc = next;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::input::PredictorSpec;

    #[test]
    fn non_factor_table_assigns_column_1_to_first_predictor() {
        let f = parse_formula("y = x1 + x2").unwrap();
        let preds = vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x2".into(),
                pinned: false,
                kind: VarKind::Binary { proportion: 0.4 },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.non_factor_names, vec!["x1", "x2"]);
        assert_eq!(t.var_types, vec![0, 1]);
        assert_eq!(t.var_params, vec![0.5, 0.4]);
        assert!(t.factor_n_levels.is_empty());
        assert_eq!(t.effect_names, vec!["x1", "x2"]);
        assert_eq!(t.effect_columns, vec![vec![1], vec![2]]);
    }

    #[test]
    fn unknown_predictor_in_formula_errors() {
        let f = parse_formula("y = x1 + x99").unwrap();
        let preds = vec![PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        }];
        let err = build_predictor_table(&f, &preds).unwrap_err();
        match err {
            SpecError::UnknownPredictor { name } => assert_eq!(name, "x99"),
            other => panic!("expected UnknownPredictor, got {other:?}"),
        }
    }

    #[test]
    fn factor_expands_to_k_minus_1_dummies() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert!(t.non_factor_names.is_empty());
        assert_eq!(t.factor_n_levels, vec![3]);
        assert_eq!(t.factor_proportions, vec![0.4, 0.3, 0.3]);
        assert_eq!(t.effect_names, vec!["group[B]", "group[C]"]);
        // Columns: intercept=0, no non-factor, so dummies at 1, 2.
        assert_eq!(t.effect_columns, vec![vec![1], vec![2]]);
    }

    #[test]
    fn factor_dummies_ordered_after_non_factors() {
        let f = parse_formula("y = x1 + group + x2").unwrap();
        let preds = vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["L1".into(), "L2".into()],
                    proportions: vec![0.5, 0.5],
                    reference: "L1".into(),
                    sampled_proportions: None,
                },
            },
            PredictorSpec {
                name: "x2".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.non_factor_names, vec!["x1", "x2"]);
        // group is the only factor → 1 dummy at column n_nf+1 = 3.
        let group_b_col = t
            .effect_columns
            .iter()
            .zip(t.effect_names.iter())
            .find_map(|(c, n)| (n == "group[L2]").then_some(c.clone()))
            .expect("group[L2] should be present");
        assert_eq!(group_b_col, vec![3]);
    }

    // Dummy effect_columns are 1-element vecs; the column index is 1-based
    // (reference level excluded), and the level ordering matches the declaration
    // order in `levels` with the reference removed.
    #[test]
    fn dummy_effect_columns_are_single_one_based_in_level_order() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                // reference is "B" (not first) to check ordering excludes it
                // wherever it sits in declaration order.
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "B".into(),
                sampled_proportions: None,
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        // Non-reference levels in declaration order: A, C.
        assert_eq!(t.effect_names, vec!["group[A]", "group[C]"]);
        // Each dummy maps to exactly one column.
        assert!(t.effect_columns.iter().all(|c| c.len() == 1));
        // 1-based, immediately after the (zero) non-factor block: cols 1, 2.
        assert_eq!(t.effect_columns, vec![vec![1], vec![2]]);
    }

    // An interaction between two k-level factors produces (k1-1)*(k2-1) effect
    // entries — the full Cartesian product of non-reference levels.
    #[test]
    fn interaction_two_factors_is_cartesian_of_non_reference_levels() {
        let f = parse_formula("y = f1 + f2 + f1:f2").unwrap();
        let preds = vec![
            PredictorSpec {
                name: "f1".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into()], // k1=3 → 2 dummies
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
            PredictorSpec {
                name: "f2".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["P".into(), "Q".into()], // k2=2 → 1 dummy
                    proportions: vec![0.5, 0.5],
                    reference: "P".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        let interaction_count = t.effect_names.iter().filter(|n| n.contains(':')).count();
        // (3-1)*(2-1) = 2 interaction entries.
        assert_eq!(interaction_count, 2);
        assert!(t.effect_names.contains(&"f1[B]:f2[Q]".to_string()));
        assert!(t.effect_names.contains(&"f1[C]:f2[Q]".to_string()));
    }

    // factor_levels holds only non-reference levels in declaration order;
    // factor_references is parallel to factor_names with the reference string for each factor.
    #[test]
    fn factor_levels_excludes_reference_and_references_parallel() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "B".into(),
                sampled_proportions: None,
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.factor_names, vec!["group"]);
        // Non-reference levels in declaration order, reference "B" dropped.
        assert_eq!(
            t.factor_levels,
            vec![vec!["A".to_string(), "C".to_string()]]
        );
        // Parallel reference vec.
        assert_eq!(t.factor_references, vec!["B"]);
    }

    // effect_names and effect_columns are parallel vecs of equal length on every successful build.
    #[test]
    fn effect_names_and_columns_are_parallel() {
        let f = parse_formula("y = x1 + group").unwrap();
        let preds = vec![
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
                    proportions: vec![0.4, 0.3, 0.3],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.effect_names.len(), t.effect_columns.len());
        assert!(!t.effect_names.is_empty());
    }

    #[test]
    fn interaction_components_captured_by_name() {
        let f = parse_formula("y = x1 + group + x1:group").unwrap();
        let preds = vec![
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
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        // effect order: x1, group[B], group[C], x1:group[B], x1:group[C]
        assert_eq!(t.interaction_components.len(), t.effect_names.len());
        // main effects carry empty component lists
        assert!(t.interaction_components[0].is_empty()); // x1
        assert!(t.interaction_components[1].is_empty()); // group[B]
                                                         // x1:group[B] → [("x1", None), ("group", Some("B"))]
        assert_eq!(
            t.interaction_components[3],
            vec![
                ("x1".to_string(), None),
                ("group".to_string(), Some("B".to_string()))
            ]
        );
        assert_eq!(
            t.interaction_components[4],
            vec![
                ("x1".to_string(), None),
                ("group".to_string(), Some("C".to_string()))
            ]
        );
    }

    // The `level` in a FactorLevel descriptor indexes the FULL declared level
    // list (reference included), NOT the reference-stripped dummy list and NOT
    // `level_index - 1`. With reference "B" sitting in the middle, the two
    // dummies A and C must carry level = 0 and level = 2 — the obvious
    // off-by-one place. (Mirrors the plan's worked example.)
    #[test]
    fn factor_level_descriptor_indexes_full_level_list_including_reference() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "B".into(),
                sampled_proportions: None,
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.effect_names, vec!["group[A]", "group[C]"]);
        assert_eq!(
            t.effect_descriptors,
            vec![
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 0
                },
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 2
                },
            ]
        );
    }

    // Descriptors are parallel to effect_names and carry the right variant for
    // continuous mains, factor dummies, and interactions (with per-component
    // level indices into each factor's full level list).
    #[test]
    fn effect_descriptors_cover_continuous_factor_and_interaction() {
        let f = parse_formula("y = x1 + group + x1:group").unwrap();
        let preds = vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    // reference "A" first → dummies B (level 1), C (level 2).
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        // effect order: x1, group[B], group[C], x1:group[B], x1:group[C]
        assert_eq!(t.effect_descriptors.len(), t.effect_names.len());
        assert_eq!(
            t.effect_descriptors[0],
            EffectDescriptor::Continuous {
                predictor: "x1".into()
            }
        );
        assert_eq!(
            t.effect_descriptors[1],
            EffectDescriptor::FactorLevel {
                factor: "group".into(),
                level: 1
            }
        );
        assert_eq!(
            t.effect_descriptors[2],
            EffectDescriptor::FactorLevel {
                factor: "group".into(),
                level: 2
            }
        );
        assert_eq!(
            t.effect_descriptors[3],
            EffectDescriptor::Interaction {
                components: vec![
                    EffectDescriptor::Continuous {
                        predictor: "x1".into()
                    },
                    EffectDescriptor::FactorLevel {
                        factor: "group".into(),
                        level: 1
                    },
                ],
            }
        );
        assert_eq!(
            t.effect_descriptors[4],
            EffectDescriptor::Interaction {
                components: vec![
                    EffectDescriptor::Continuous {
                        predictor: "x1".into()
                    },
                    EffectDescriptor::FactorLevel {
                        factor: "group".into(),
                        level: 2
                    },
                ],
            }
        );
    }

    #[test]
    fn interaction_with_factor_expands_to_cartesian_product() {
        let f = parse_formula("y = x1 + group + x1:group").unwrap();
        let preds = vec![
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
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let t = build_predictor_table(&f, &preds).unwrap();
        // Effects: x1, group[B], group[C], x1:group[B], x1:group[C]
        assert_eq!(
            t.effect_names,
            vec!["x1", "group[B]", "group[C]", "x1:group[B]", "x1:group[C]"]
        );
    }

    #[test]
    fn factor_sampled_collected_into_predictor_table() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: Some(true),
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.factor_n_levels, vec![3]);
        assert_eq!(t.factor_sampled, vec![Some(true)]);
    }

    #[test]
    fn factor_sampled_absent_collects_none() {
        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: None,
            },
        }];
        let t = build_predictor_table(&f, &preds).unwrap();
        assert_eq!(t.factor_sampled, vec![None]);
    }
}
