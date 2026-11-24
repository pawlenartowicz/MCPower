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
    /// Non-factor predictor names in formula order; parallel to `var_types`/`var_pinned`/`var_params`.
    pub non_factor_names: Vec<String>,
    /// Distribution code per non-factor predictor (parallel to `non_factor_names`).
    pub var_types: Vec<i32>,
    /// Per-non-factor pin (aligned with `var_types`): `true` = explicitly
    /// chosen distribution, scenario swaps leave the column alone.
    pub var_pinned: Vec<bool>,
    /// Distribution parameter per non-factor predictor (e.g. binary proportion).
    pub var_params: Vec<f64>,
    /// Level count per factor, in declaration order.
    pub factor_n_levels: Vec<i32>,
    /// Per-level proportions, flattened across factors in declaration order.
    /// Within a factor they follow the engine's level codes, not the declared
    /// level order: the reference first (code 0, no dummy), then the
    /// non-reference levels in declared order (codes 1.., matching
    /// `DummyOf { level_index }`).
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
    /// Per-factor FULL level list in declared order (reference included).
    /// Parallel to `factor_names`. Uploaded factor columns whose labels are
    /// not the level names map their codes to levels by position in this list.
    pub factor_declared_levels: Vec<Vec<String>>,
    /// All effect names in projection order (main effects + interactions),
    /// after factor expansion. Used by the targets resolver and the effect-
    /// size assembler.
    pub effect_names: Vec<String>,
    /// For each effect name, its column index in the design matrix
    /// (intercept = 0; first non-factor predictor = 1; ...). Interactions
    /// have one index per component, except in a `PredictorTable::fitted`
    /// table, where their entry is empty.
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
    /// How many leading entries of `effect_names` come from the formula's own
    /// terms. The rest are the terms `build_model_table` adds for R's factor
    /// coding: generated with effect 0 and fitted, but never assigned an effect
    /// or tested.
    pub n_formula_effects: usize,
}

impl PredictorTable {
    /// Whether predictor `name` has a main-effect term: a continuous main
    /// effect, or the dummies of a factor. In `y ~ x:z` neither x nor z has
    /// one: both are generated, but the fitted model leaves them out, as R's
    /// `lm(y ~ x:z)` does. In a `build_model_table` table the terms added for
    /// R's factor coding count, so x has one in `y ~ f:x`.
    pub fn has_main_term(&self, name: &str) -> bool {
        self.effect_descriptors.iter().any(|d| match d {
            EffectDescriptor::Continuous { predictor } => predictor == name,
            EffectDescriptor::FactorLevel { factor, .. } => factor == name,
            _ => false,
        })
    }

    /// The table of the fitted design: every predictor without a main-effect
    /// term is dropped, so the column layout (intercept, kept non-factors, kept
    /// dummies, interactions) is the fitted design's. Interaction effects stay,
    /// with an empty `effect_columns` entry, since a component may have no
    /// fitted column. `None` when every predictor has a main-effect term, so
    /// the table already is the fitted one.
    pub(crate) fn fitted(&self) -> Option<PredictorTable> {
        let keep_nf: Vec<bool> = self
            .non_factor_names
            .iter()
            .map(|n| self.has_main_term(n))
            .collect();
        let keep_f: Vec<bool> = self
            .factor_names
            .iter()
            .map(|n| self.has_main_term(n))
            .collect();
        if keep_nf.iter().chain(&keep_f).all(|&k| k) {
            return None;
        }
        fn pick<T: Clone>(v: &[T], keep: &[bool]) -> Vec<T> {
            v.iter()
                .zip(keep)
                .filter(|(_, k)| **k)
                .map(|(x, _)| x.clone())
                .collect()
        }
        // Proportions are flattened per factor, `factor_n_levels[f]` each.
        let mut factor_proportions = Vec::new();
        let mut offset = 0;
        for (&nl, &keep) in self.factor_n_levels.iter().zip(&keep_f) {
            let n = nl.max(0) as usize;
            if keep {
                factor_proportions.extend_from_slice(&self.factor_proportions[offset..offset + n]);
            }
            offset += n;
        }
        let non_factor_names = pick(&self.non_factor_names, &keep_nf);
        let factor_names = pick(&self.factor_names, &keep_f);
        let factor_references = pick(&self.factor_references, &keep_f);
        let factor_declared_levels = pick(&self.factor_declared_levels, &keep_f);
        let kept_factors: Vec<(String, Vec<String>, String)> = factor_names
            .iter()
            .zip(&factor_declared_levels)
            .zip(&factor_references)
            .map(|((n, l), r)| (n.clone(), l.clone(), r.clone()))
            .collect();
        let dummy_column_for =
            build_dummy_column_index(&kept_factors, non_factor_names.len() as u32);
        let effect_columns = self
            .effect_descriptors
            .iter()
            .zip(&self.effect_names)
            .map(|(d, name)| match d {
                EffectDescriptor::Continuous { predictor } => {
                    vec![column_of_non_factor(&non_factor_names, predictor)]
                }
                EffectDescriptor::FactorLevel { .. } => vec![dummy_column_for(name)],
                _ => Vec::new(),
            })
            .collect();
        Some(PredictorTable {
            var_types: pick(&self.var_types, &keep_nf),
            var_pinned: pick(&self.var_pinned, &keep_nf),
            var_params: pick(&self.var_params, &keep_nf),
            factor_n_levels: pick(&self.factor_n_levels, &keep_f),
            factor_proportions,
            factor_sampled: pick(&self.factor_sampled, &keep_f),
            factor_levels: pick(&self.factor_levels, &keep_f),
            non_factor_names,
            factor_names,
            factor_references,
            factor_declared_levels,
            effect_names: self.effect_names.clone(),
            effect_columns,
            interaction_components: self.interaction_components.clone(),
            effect_descriptors: self.effect_descriptors.clone(),
            n_formula_effects: self.n_formula_effects,
        })
    }
}

/// Expand the formula's predictors into a `PredictorTable`: non-factor columns,
/// factor dummies, and every effect's design-matrix column index.
///
/// # Errors
/// [`SpecError::UnknownPredictor`] when a formula predictor has no `PredictorSpec`.
pub fn build_predictor_table(
    formula: &ParsedFormula,
    predictors: &[PredictorSpec],
) -> Result<PredictorTable, SpecError> {
    build_table(formula, predictors, false)
}

/// As [`build_predictor_table`], plus the terms R's factor coding adds to the
/// model (see [`margin_terms`]), appended after the formula's own effects. The
/// generation and fitted designs are built from this table.
///
/// # Errors
/// As [`build_predictor_table`].
pub(crate) fn build_model_table(
    formula: &ParsedFormula,
    predictors: &[PredictorSpec],
) -> Result<PredictorTable, SpecError> {
    build_table(formula, predictors, true)
}

fn build_table(
    formula: &ParsedFormula,
    predictors: &[PredictorSpec],
    with_margins: bool,
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
                // Engine code order: the reference's proportion, then the
                // non-reference levels' in declared order.
                let by_level = levels.iter().zip(proportions);
                factor_proportions.extend(
                    by_level
                        .clone()
                        .filter(|(l, _)| *l == reference)
                        .map(|(_, &p)| p),
                );
                factor_proportions
                    .extend(by_level.filter(|(l, _)| *l != reference).map(|(_, &p)| p));
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

    let mut effect_names: Vec<String> = Vec::new();
    let mut effect_columns: Vec<Vec<u32>> = Vec::new();
    let mut interaction_components: Vec<Vec<(String, Option<String>)>> = Vec::new();
    let mut effect_descriptors: Vec<EffectDescriptor> = Vec::new();

    // Column index helper: continuous columns 1..=n_nf; dummy columns follow.
    let n_nf = non_factor_names.len() as u32;
    let dummy_column_for = build_dummy_column_index(&factor_levels, n_nf);

    let factor_set: std::collections::BTreeSet<&str> =
        factor_levels.iter().map(|(n, _, _)| n.as_str()).collect();

    let margins = if with_margins {
        margin_terms(&formula.terms, |v| {
            matches!(by_name[v].kind, VarKind::Factor { .. })
        })
    } else {
        Vec::new()
    };
    let mut n_formula_effects = None;
    for (ti, term) in formula.terms.iter().chain(&margins).enumerate() {
        if ti == formula.terms.len() {
            n_formula_effects = Some(effect_names.len());
        }
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
    let factor_declared_levels: Vec<Vec<String>> =
        factor_levels.iter().map(|(_, l, _)| l.clone()).collect();

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
        factor_declared_levels,
        n_formula_effects: n_formula_effects.unwrap_or(effect_names.len()),
        effect_names,
        effect_columns,
        interaction_components,
        effect_descriptors,
    })
}

/// The terms R adds to a model through its factor coding, as `model.matrix`
/// does. R codes a factor f inside a term T with contrasts (one column per
/// non-reference level) only when T without f is also in the model, and with
/// one column per level otherwise: `y ~ f:x` gets fa:x, fb:x, fc:x. One column
/// per level of f spans the same columns as `T without f` plus T with f
/// contrast-coded, so each missing `T without f` is returned here (`x` for
/// `y ~ f:x`) and then checked the same way. The intercept is always in the
/// model, so a main effect adds nothing, and an interaction of continuous
/// predictors only (`x:z`) adds nothing either.
fn margin_terms(terms: &[Term], is_factor: impl Fn(&str) -> bool) -> Vec<Term> {
    let vars_of = |t: &Term| match t {
        Term::Main { name } => vec![name.clone()],
        Term::Interaction { vars } => vars.clone(),
    };
    let sorted = |v: &[String]| {
        let mut k = v.to_vec();
        k.sort();
        k
    };
    let mut queue: Vec<Vec<String>> = terms.iter().map(vars_of).collect();
    let mut present: std::collections::HashSet<Vec<String>> =
        queue.iter().map(|v| sorted(v)).collect();
    let mut margins = Vec::new();
    let mut i = 0;
    while i < queue.len() {
        let vars = queue[i].clone();
        i += 1;
        if vars.len() < 2 {
            continue;
        }
        for f in vars.iter().filter(|v| is_factor(v)) {
            let rest: Vec<String> = vars.iter().filter(|v| *v != f).cloned().collect();
            if present.insert(sorted(&rest)) {
                margins.push(match rest.as_slice() {
                    [name] => Term::Main { name: name.clone() },
                    _ => Term::Interaction { vars: rest.clone() },
                });
                queue.push(rest);
            }
        }
    }
    margins
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

    /// The added terms for formulas whose columns were read off R's
    /// `model.matrix` (f and g factors, x and z continuous): the formula's
    /// terms plus these span exactly R's columns.
    #[test]
    fn margin_terms_follow_r_factor_coding() {
        let margins = |formula: &str| -> Vec<String> {
            let parsed = parse_formula(formula).unwrap();
            margin_terms(&parsed.terms, |v| v == "f" || v == "g")
                .iter()
                .map(|t| match t {
                    Term::Main { name } => name.clone(),
                    Term::Interaction { vars } => vars.join(":"),
                })
                .collect()
        };
        assert_eq!(margins("y = f:x"), ["x"]);
        assert_eq!(margins("y = f + f:x"), ["x"]);
        assert!(margins("y = x + f:x").is_empty());
        assert_eq!(margins("y = f:g"), ["g", "f"]);
        assert!(margins("y = x:z").is_empty());
        assert_eq!(margins("y = f:x:z"), ["x:z"]);
        assert_eq!(margins("y = f:g:x"), ["g:x", "f:x", "x"]);
    }

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
    // off-by-one place.
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
        // Non-reference levels in declaration order (reference "B" dropped),
        // each a single 1-based column right after the (empty) non-factor block.
        assert_eq!(t.effect_names, vec!["group[A]", "group[C]"]);
        assert_eq!(t.effect_columns, vec![vec![1], vec![2]]);
        assert_eq!(t.factor_names, vec!["group"]);
        assert_eq!(
            t.factor_levels,
            vec![vec!["A".to_string(), "C".to_string()]]
        );
        assert_eq!(t.factor_references, vec!["B"]);
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
        // x1:group expands to the Cartesian product with group's non-reference levels.
        assert_eq!(
            t.effect_names,
            vec!["x1", "group[B]", "group[C]", "x1:group[B]", "x1:group[C]"]
        );
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
}
