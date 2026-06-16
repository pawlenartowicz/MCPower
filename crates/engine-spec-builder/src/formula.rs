//! Parses R-style formula strings into a `ParsedFormula` with fixed and random effects; `*` expands to main effects plus all-way interactions.
//!
//! Random-effect extraction order is load-bearing: nested `(1|A/B)` before slope `(1+…|g)`
//! before plain intercept `(1|g)` — the intercept regex would greedily match the inner var of a
//! nested pattern if it ran first. A suppression pre-check (step 0) rejects
//! intercept-suppressed terms (`(0+…|g)` / `(-1+…|g)`) up front, and an implicit-intercept slope
//! pass (step 2.5, between explicit slope and plain intercept) treats `(x|g)` as `(1+x|g)`; its
//! pattern is letter-led so it cannot collide with the digit-led explicit patterns regardless of
//! order.

use crate::error::SpecError;
use serde::Serialize;

/// Parsed view of the right-hand side of a formula. The builder consumes this
/// alongside the user's `predictors` list to produce a `PredictorTable` and an
/// effect ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParsedFormula {
    pub dependent: String,
    /// Predictor names in formula order. Duplicates are dropped.
    pub predictors: Vec<String>,
    /// Effect terms in formula order. Main effects appear as
    /// `Term::Main("x1")`; interactions as `Term::Interaction(vec!["x1", "x2"])`.
    pub terms: Vec<Term>,
    /// Random effects parsed from the formula (e.g. `(1|group)`); empty when the
    /// formula declares none.
    pub random_effects: Vec<RandomEffect>,
}

/// Internally tagged as `{"kind": "main", "name": …}` / `{"kind": "interaction",
/// "vars": […]}`. Struct variants (rather than `Main(String)`) so the derive can
/// reproduce the named-payload wire shape that an internally-tagged tuple variant
/// cannot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Term {
    Main { name: String },
    Interaction { vars: Vec<String> },
}

/// A single random-effect specification extracted from an LME formula.
///
/// Internally tagged as `{"kind": "intercept"|"slope", …}`. `parent` keeps its
/// default `null` serialization when `None` (no `skip_serializing_if`); the
/// formula fixtures rely on the explicit `"parent": null`. Struct variants let
/// the derive reproduce the wire shape exactly (cf. [`Term`], whose tuple
/// variants need a hand-written impl).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RandomEffect {
    /// Random intercept for `group`. `parent` is set when the grouping factor
    /// is nested (e.g. `A:B` → `parent = Some("A")`).
    Intercept {
        group: String,
        parent: Option<String>,
    },
    /// Random slope(s) `vars` for grouping factor `group`.
    Slope { group: String, vars: Vec<String> },
}

/// Parse an R-style formula string into a [`ParsedFormula`].
///
/// Random-effect terms (`(1|g)`, `(1|A/B)`, `(1+x|g)`) are extracted and stripped before
/// fixed-effect parsing; the resulting `random_effects` vec may be non-empty even when
/// fixed-effect vecs are empty (RE-only formula).
///
/// # Errors
/// - [`SpecError::EmptyFormula`]: input is blank or the RHS after the separator is empty.
/// - [`SpecError::FormulaSyntax`]: a term contains a non-identifier token.
/// - [`SpecError::TermRemovalUnsupported`]: a bare `-` (not a digit sign) appears outside parens.
/// - [`SpecError::EmptySlopeTerm`]: a slope clause `(1+…|g)` has no variables after stripping.
/// - [`SpecError::DuplicateGroupingVar`]: the same grouping variable appears more than once.
/// - [`SpecError::RandomInterceptSuppressionUnsupported`]: an RE term suppresses the random
///   intercept (`(0 + x | g)` / `(-1 + x | g)`); MCPower requires a random intercept with a slope.
pub fn parse_formula(input: &str) -> Result<ParsedFormula, SpecError> {
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return Err(SpecError::EmptyFormula);
    }

    let (mut dep, rhs) = split_at_separator(&cleaned);
    if dep.is_empty() {
        // Empty LHS (e.g. `~X` or `=X`) — fall back to the same default name
        // we use when there is no separator at all.
        dep = "explained_variable".to_string();
    }
    if rhs.is_empty() {
        return Err(SpecError::EmptyFormula);
    }

    // Extract random-effect terms first so the remaining RHS is a plain
    // fixed-effects string. Mirrors parsers.py:342-401.
    let (random_effects, rhs_stripped) = extract_random_effects(&rhs)?;

    // Reject term removal: a '-' that isn't inside parens and isn't a digit sign.
    // Mirrors parsers.py:430.
    if let Some(pos) = find_term_removal(&rhs_stripped) {
        let _ = pos; // position-aware error variant could be added later.
        return Err(SpecError::TermRemovalUnsupported);
    }

    let (predictors, terms) = if rhs_stripped.is_empty() {
        // RE-only RHS is valid (e.g. `y ~ (1|g)`).
        (Vec::new(), Vec::new())
    } else {
        parse_rhs(&rhs_stripped)?
    };

    Ok(ParsedFormula {
        dependent: dep,
        predictors,
        terms,
        random_effects,
    })
}

fn extract_random_effects(rhs: &str) -> Result<(Vec<RandomEffect>, String), SpecError> {
    use std::collections::BTreeSet;

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut effects: Vec<RandomEffect> = Vec::new();
    let mut work = rhs.to_string();

    // 0. Reject intercept suppression: (0+…|g), (-1+…|g), (0|g), (-1|g). MCPower
    //    has no slope-without-random-intercept model (tau_squared is the
    //    intercept variance a slope correlates against), so this is a clear error
    //    rather than a downstream FormulaSyntax surprise. Whitespace is already
    //    stripped by parse_formula, and only a '(' can start the match, so a digit
    //    like '1' or a name like '10' is never mis-flagged.
    let re_suppress = regex::Regex::new(r"\((?:0|-1)(?:\+[^|]*)?\|[^)]*\)").unwrap();
    if re_suppress.is_match(rhs) {
        return Err(SpecError::RandomInterceptSuppressionUnsupported);
    }

    // 1. Nested intercept: (1|A/B) → push pair, replace with empty.
    let re_nested =
        regex::Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)/([_A-Za-z][_A-Za-z0-9]*)\)").unwrap();
    loop {
        let snapshot = work.clone();
        let Some(m) = re_nested.captures(&snapshot) else {
            break;
        };
        let parent_name = m.get(1).unwrap().as_str().to_string();
        let child_name = m.get(2).unwrap().as_str().to_string();
        let joined = format!("{parent_name}:{child_name}");
        for name in [&parent_name, &joined] {
            if !seen.insert(name.clone()) {
                return Err(SpecError::DuplicateGroupingVar { name: name.clone() });
            }
        }
        effects.push(RandomEffect::Intercept {
            group: parent_name.clone(),
            parent: None,
        });
        effects.push(RandomEffect::Intercept {
            group: joined,
            parent: Some(parent_name),
        });
        work = re_nested.replacen(&snapshot, 1, "").into_owned();
    }

    // 2. Random slope: (1+x+y|g)
    let re_slope = regex::Regex::new(r"\(1\+([^|]+?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap();
    loop {
        let snapshot = work.clone();
        let Some(m) = re_slope.captures(&snapshot) else {
            break;
        };
        let var_list_raw = m.get(1).unwrap().as_str();
        let group = m.get(2).unwrap().as_str().to_string();
        let raw_tokens: Vec<&str> = var_list_raw
            .split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if raw_tokens.is_empty() {
            // Pure separators, e.g. (1++|g) — malformed, not an intercept.
            return Err(SpecError::EmptySlopeTerm { group });
        }
        // Drop redundant explicit intercept tokens (mirrors the implicit-slope
        // branch): (1+1|g) ≡ (1|g), and a bare "1" must never reach the host as
        // a slope predictor name.
        let vars: Vec<String> = raw_tokens
            .iter()
            .filter(|s| **s != "1")
            .map(|s| String::from(*s))
            .collect();
        if !seen.insert(group.clone()) {
            return Err(SpecError::DuplicateGroupingVar {
                name: group.clone(),
            });
        }
        if vars.is_empty() {
            // Only intercept tokens inside the parens — a plain random intercept.
            effects.push(RandomEffect::Intercept { group, parent: None });
        } else {
            effects.push(RandomEffect::Slope { group, vars });
        }
        work = re_slope.replacen(&snapshot, 1, "").into_owned();
    }

    // 2.5. Implicit-intercept slope: (x|g), (x+z|g) — equivalent to (1+x|g).
    //      The random intercept is implicit (lme4 convention; suppression is
    //      caught in step 0). The LHS starts with a letter, so this NEVER overlaps
    //      the digit-led explicit patterns ((1+…|g) / (1|g) / (1|A/B)). A redundant
    //      explicit '1' token (e.g. (x+1|g)) is dropped. Nested groups (A/B) are
    //      not matched here — same as the explicit slope pattern — so a slope on a
    //      nested factor falls through to the downstream error.
    let re_islope = regex::Regex::new(r"\(([_A-Za-z][^|]*?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap();
    loop {
        let snapshot = work.clone();
        let Some(m) = re_islope.captures(&snapshot) else {
            break;
        };
        let var_list_raw = m.get(1).unwrap().as_str();
        let group = m.get(2).unwrap().as_str().to_string();
        let vars: Vec<String> = var_list_raw
            .split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != "1") // drop a redundant explicit intercept
            .map(String::from)
            .collect();
        // No empty-vars guard here (unlike the explicit slope loop): the regex
        // forces a letter-led first token, so `vars` is always non-empty.
        if !seen.insert(group.clone()) {
            return Err(SpecError::DuplicateGroupingVar { name: group.clone() });
        }
        effects.push(RandomEffect::Slope { group, vars });
        work = re_islope.replacen(&snapshot, 1, "").into_owned();
    }

    // 3. Random intercept: (1|g)
    let re_int = regex::Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap();
    loop {
        let snapshot = work.clone();
        let Some(m) = re_int.captures(&snapshot) else {
            break;
        };
        let name = m.get(1).unwrap().as_str().to_string();
        if !seen.insert(name.clone()) {
            return Err(SpecError::DuplicateGroupingVar { name });
        }
        effects.push(RandomEffect::Intercept {
            group: name,
            parent: None,
        });
        work = re_int.replacen(&snapshot, 1, "").into_owned();
    }

    // Clean stray "+" — collapse "++", trim leading/trailing "+".
    let cleaned = clean_residual_plusses(&work);
    Ok((effects, cleaned))
}

fn clean_residual_plusses(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_plus = false;
    for ch in s.chars() {
        if ch == '+' {
            if !prev_plus && !out.is_empty() {
                out.push('+');
                prev_plus = true;
            }
        } else if !ch.is_whitespace() {
            out.push(ch);
            prev_plus = false;
        }
    }
    // Strip leading/trailing "+".
    while out.starts_with('+') {
        out.remove(0);
    }
    while out.ends_with('+') {
        out.pop();
    }
    out
}

fn split_at_separator(s: &str) -> (String, String) {
    if let Some((l, r)) = s.split_once('~') {
        (l.to_string(), r.to_string())
    } else if let Some((l, r)) = s.split_once('=') {
        (l.to_string(), r.to_string())
    } else {
        ("explained_variable".to_string(), s.to_string())
    }
}

fn find_term_removal(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'-' if depth == 0 => {
                let next = bytes.get(i + 1).copied().unwrap_or(b' ');
                if !next.is_ascii_digit() {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_rhs(rhs: &str) -> Result<(Vec<String>, Vec<Term>), SpecError> {
    use std::collections::BTreeSet;

    let mut predictors: Vec<String> = Vec::new();
    let mut seen_pred: BTreeSet<String> = BTreeSet::new();
    let mut terms: Vec<Term> = Vec::new();
    let mut seen_term: BTreeSet<String> = BTreeSet::new();

    for raw_term in rhs.split('+') {
        let term = raw_term.trim();
        if term.is_empty() {
            continue;
        }

        if term.contains('*') {
            // x1*x2*x3 → main effects + all-way interactions (2..=n).
            let vars = parse_identifier_list(term, &['*'])?;
            register_vars(&vars, &mut predictors, &mut seen_pred);
            for v in &vars {
                if seen_term.insert(v.clone()) {
                    terms.push(Term::Main { name: v.clone() });
                }
            }
            for r in 2..=vars.len() {
                for combo in combinations(&vars, r) {
                    let key = combo.join(":");
                    if seen_term.insert(key) {
                        terms.push(Term::Interaction { vars: combo });
                    }
                }
            }
        } else if term.contains(':') {
            let vars = parse_identifier_list(term, &[':'])?;
            register_vars(&vars, &mut predictors, &mut seen_pred);
            let key = vars.join(":");
            if seen_term.insert(key) {
                terms.push(Term::Interaction { vars });
            }
        } else {
            let name = parse_single_identifier(term)?;
            if seen_pred.insert(name.clone()) {
                predictors.push(name.clone());
            }
            if seen_term.insert(name.clone()) {
                terms.push(Term::Main { name });
            }
        }
    }

    Ok((predictors, terms))
}

fn parse_single_identifier(s: &str) -> Result<String, SpecError> {
    if is_identifier(s) {
        Ok(s.to_string())
    } else {
        Err(SpecError::FormulaSyntax {
            pos: 0,
            msg: format!("expected identifier, got '{s}'"),
        })
    }
}

fn parse_identifier_list(s: &str, seps: &[char]) -> Result<Vec<String>, SpecError> {
    let mut parts: Vec<&str> = vec![s];
    for sep in seps {
        parts = parts.into_iter().flat_map(|p| p.split(*sep)).collect();
    }
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(parse_single_identifier)
        .collect()
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn register_vars(
    vars: &[String],
    predictors: &mut Vec<String>,
    seen: &mut std::collections::BTreeSet<String>,
) {
    for v in vars {
        if seen.insert(v.clone()) {
            predictors.push(v.clone());
        }
    }
}

fn combinations<T: Clone>(items: &[T], r: usize) -> Vec<Vec<T>> {
    let n = items.len();
    if r == 0 || r > n {
        return vec![];
    }
    let mut idx: Vec<usize> = (0..r).collect();
    let mut out: Vec<Vec<T>> = Vec::new();
    loop {
        out.push(idx.iter().map(|&i| items[i].clone()).collect());
        // Generate next combination in lexicographic order.
        let mut i = r;
        while i > 0 && idx[i - 1] == n - r + (i - 1) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        idx[i - 1] += 1;
        for j in i..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_additive_formula() {
        let f = parse_formula("y = x1 + x2").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.predictors, vec!["x1", "x2"]);
        assert_eq!(
            f.terms,
            vec![
                Term::Main { name: "x1".into() },
                Term::Main { name: "x2".into() }
            ]
        );
    }

    #[test]
    fn parses_tilde_separator() {
        let f = parse_formula("y ~ x1 + x2").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.predictors, vec!["x1", "x2"]);
    }

    #[test]
    fn parses_specific_interaction() {
        let f = parse_formula("y = x1 + x2 + x1:x2").unwrap();
        assert_eq!(f.predictors, vec!["x1", "x2"]);
        assert_eq!(
            f.terms,
            vec![
                Term::Main { name: "x1".into() },
                Term::Main { name: "x2".into() },
                Term::Interaction {
                    vars: vec!["x1".into(), "x2".into()]
                },
            ]
        );
    }

    #[test]
    fn parses_star_expands_main_plus_all_interactions() {
        let f = parse_formula("y = x1*x2*x3").unwrap();
        assert_eq!(f.predictors, vec!["x1", "x2", "x3"]);
        assert_eq!(
            f.terms,
            vec![
                Term::Main { name: "x1".into() },
                Term::Main { name: "x2".into() },
                Term::Main { name: "x3".into() },
                Term::Interaction {
                    vars: vec!["x1".into(), "x2".into()]
                },
                Term::Interaction {
                    vars: vec!["x1".into(), "x3".into()]
                },
                Term::Interaction {
                    vars: vec!["x2".into(), "x3".into()]
                },
                Term::Interaction {
                    vars: vec!["x1".into(), "x2".into(), "x3".into()]
                },
            ]
        );
    }

    #[test]
    fn no_lhs_defaults_to_explained_variable() {
        let f = parse_formula("x1 + x2").unwrap();
        assert_eq!(f.dependent, "explained_variable");
    }

    #[test]
    fn empty_lhs_with_tilde_defaults_to_explained_variable() {
        let f = parse_formula("~ x1 + x2").unwrap();
        assert_eq!(f.dependent, "explained_variable");
        assert_eq!(f.predictors, vec!["x1", "x2"]);
    }

    #[test]
    fn empty_lhs_with_equals_defaults_to_explained_variable() {
        let f = parse_formula("= x1 + x2").unwrap();
        assert_eq!(f.dependent, "explained_variable");
        assert_eq!(f.predictors, vec!["x1", "x2"]);
    }

    #[test]
    fn empty_formula_errors() {
        assert!(matches!(parse_formula(""), Err(SpecError::EmptyFormula)));
        assert!(matches!(parse_formula("y ="), Err(SpecError::EmptyFormula)));
    }

    #[test]
    fn term_removal_rejected() {
        assert!(matches!(
            parse_formula("y = x1 - x2"),
            Err(SpecError::TermRemovalUnsupported)
        ));
    }

    // Duplicate predictor names in the RHS collapse — each name appears once
    // in `predictors` (first occurrence wins) and is not re-emitted as a
    // duplicate `Term::Main`.
    #[test]
    fn duplicate_predictor_names_collapse() {
        let f = parse_formula("y = x1 + x2 + x1").unwrap();
        assert_eq!(f.predictors, vec!["x1", "x2"]);
        assert_eq!(
            f.terms,
            vec![
                Term::Main { name: "x1".into() },
                Term::Main { name: "x2".into() }
            ]
        );
    }

    // A non-identifier token in a term position is a syntax error.
    #[test]
    fn non_identifier_token_is_syntax_error() {
        assert!(matches!(
            parse_formula("y = x1 + 2bad"),
            Err(SpecError::FormulaSyntax { .. })
        ));
    }

    // A RE-only RHS is accepted and yields empty fixed-effect vecs with a
    // non-empty random-effects vec.
    #[test]
    fn re_only_formula_has_empty_fixed_effects() {
        let f = parse_formula("y ~ (1|g)").unwrap();
        assert!(f.predictors.is_empty());
        assert!(f.terms.is_empty());
        // The RE-only path still extracts the actual random effect, not
        // merely "something non-empty" — assert the parsed content (group/parent).
        assert_eq!(
            f.random_effects,
            vec![RandomEffect::Intercept {
                group: "g".into(),
                parent: None,
            }]
        );
    }

    // A slope clause that matches the `(1+…|g)` shape but whose var list
    // collapses to nothing (e.g. `(1++|g)` — the captured segment is only
    // separators) is an EmptySlopeTerm error.
    #[test]
    fn empty_slope_var_list_errors() {
        assert!(matches!(
            parse_formula("y ~ x + (1++|g)"),
            Err(SpecError::EmptySlopeTerm { .. })
        ));
    }

    // (1+1|g) is just a doubly-written random intercept — it must not surface
    // "1" as a slope predictor name (hosts resolve slope names to columns).
    #[test]
    fn redundant_explicit_intercept_token_collapses_to_intercept() {
        let f = parse_formula("y ~ x + (1+1|g)").unwrap();
        assert_eq!(
            f.random_effects,
            vec![RandomEffect::Intercept {
                group: "g".into(),
                parent: None,
            }]
        );
        let f = parse_formula("y ~ x + (1+1+x|g)").unwrap();
        assert_eq!(
            f.random_effects,
            vec![RandomEffect::Slope {
                group: "g".into(),
                vars: vec!["x".into()],
            }]
        );
    }
}
