//! Parses R-style formula strings into a `ParsedFormula` with fixed and random effects; `*` expands to main effects plus all-way interactions.
//!
//! Random-effect extraction order is load-bearing: nested `(1|A/B)` before slope `(1+…|g)`
//! before plain intercept `(1|g)` — the intercept regex would greedily match the inner var of a
//! nested pattern if it ran first. A suppression pre-check (step 0) rejects
//! intercept-suppressed terms (`(0+…|g)` / `(-1+…|g)`) up front, and an implicit-intercept slope
//! pass (step 2.5, between explicit slope and plain intercept) treats `(x|g)` as `(1+x|g)`; its
//! pattern is letter-led so it cannot collide with the digit-led explicit patterns regardless of
//! order. Nested spellings the parser does not take (the colon form `(1|a:b)`, a slope in a
//! nested term) are refused in step 0.5, and a term with `|` left after every pattern is refused
//! in step 4, so neither reaches the fixed-effect parser as a bare syntax error.

use crate::error::SpecError;
use regex::Regex;
use serde::Serialize;
use std::sync::LazyLock;

// Random-effect extraction patterns. Compiled once on first use rather than per
// `extract_random_effects` call — the patterns are constant, and the function is
// hot under WASM-worker / benchmark load. Match order is documented in step 0–3
// below; these mirror those steps one-to-one.
static RE_DOUBLE_BAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\([^()]*\|\|[^()]*\)").unwrap());
static RE_SUPPRESS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\((?:0|-1)(?:\+[^|]*)?\|[^)]*\)").unwrap());
static RE_NESTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)/([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
static RE_SLOPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(1\+([^|]+?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_ISLOPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(([_A-Za-z][^|]*?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_INT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
// Refusal patterns (step 0.5): a nested grouping in its colon spelling
// `(…|a:b)`, and any left-hand side on the slash spelling `(…|a/b)` (only the
// plain intercept `(1|a/b)` is supported).
static RE_COLON_GROUP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(([^()|]*)\|([_A-Za-z][_A-Za-z0-9]*):([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
static RE_NESTED_ANY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(([^()|]*)\|([_A-Za-z][_A-Za-z0-9]*)/([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
// Any parenthesised term with a '|': what is left of one after every pattern
// has run is an unsupported random-effect term.
static RE_ANY_TERM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\([^()]*\|[^()]*\)").unwrap());

/// The first random-effect term of a formula, as written (any parenthesised
/// term containing '|'), or `None` when it has none.
pub(crate) fn first_random_effect_term(formula: &str) -> Option<&str> {
    RE_ANY_TERM.find(formula).map(|m| m.as_str())
}

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
/// - [`SpecError::UncorrelatedRandomEffectsUnsupported`]: an RE term uses the double bar
///   (`(1 + x || g)`), which MCPower does not parse.
/// - [`SpecError::NestedColonFormUnsupported`]: a nested intercept in its colon spelling
///   (`(1|a:b)`); the supported spelling is `(1|a/b)`.
/// - [`SpecError::NestedSlopeUnsupported`]: a random slope in a nested term (`(1 + x|a/b)`,
///   `(x|a/b)`, `(1 + x|a:b)`); such a slope is declared through `set_cluster`.
/// - [`SpecError::MultipleNestedTermsUnsupported`]: more than one nested term
///   (`(1|a/b) + (1|c/d)`).
/// - [`SpecError::GroupingNameUnsupported`]: a term whose grouping name is not letters, digits
///   and `_` (e.g. `(1|school.id)`).
/// - [`SpecError::RandomEffectTermUnsupported`]: any other parenthesised term with `|` that no
///   random-effect pattern matches.
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
    // fixed-effects string. Mirrors parsers.py.
    let (random_effects, rhs_stripped) = extract_random_effects(&rhs)?;

    // Reject term removal: a '-' that isn't inside parens and isn't a digit sign.
    // Mirrors parsers.py.
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
    if RE_SUPPRESS.is_match(rhs) {
        return Err(SpecError::RandomInterceptSuppressionUnsupported);
    }
    // lme4's double bar `(1 + x || g)` (uncorrelated random effects) would
    // otherwise fall through every pattern below to a bare syntax error. The
    // same model is a correlated term with the correlation set to 0.
    if RE_DOUBLE_BAR.is_match(rhs) {
        return Err(SpecError::UncorrelatedRandomEffectsUnsupported);
    }
    // 0.5. Nested spellings MCPower does not parse. The colon form `(1|a:b)`
    //      (alone or beside `(1|a)`) has its own message pointing at `(1|a/b)`;
    //      a slope on either nested spelling points at set_cluster, the only
    //      route to a slope on the nested grouping.
    for m in RE_COLON_GROUP.captures_iter(rhs) {
        let (parent, child) = (m[2].to_string(), m[3].to_string());
        match re_lhs_tokens(&m[1]) {
            Some(tokens) if tokens.iter().all(|t| *t == "1") => {
                return Err(SpecError::NestedColonFormUnsupported { parent, child });
            }
            Some(tokens) => {
                return Err(SpecError::NestedSlopeUnsupported {
                    term: format!("({}|{parent}:{child})", tokens.join(" + ")),
                    parent,
                    child,
                });
            }
            None => {}
        }
    }
    for m in RE_NESTED_ANY.captures_iter(rhs) {
        if let Some(tokens) = re_lhs_tokens(&m[1]) {
            if tokens.iter().any(|t| *t != "1") {
                let (parent, child) = (m[2].to_string(), m[3].to_string());
                return Err(SpecError::NestedSlopeUnsupported {
                    term: format!("({}|{parent}/{child})", tokens.join(" + ")),
                    parent,
                    child,
                });
            }
        }
    }

    // 1. Nested intercept: (1|A/B) → push pair, replace with empty. The engine
    //    carries one nested grouping, so a second nested term is refused here,
    //    naming both terms.
    let re_nested = &*RE_NESTED;
    let mut first_nested: Option<String> = None;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_nested.captures(&snapshot) else {
            break;
        };
        let parent_name = m.get(1).unwrap().as_str().to_string();
        let child_name = m.get(2).unwrap().as_str().to_string();
        let term = format!("(1|{parent_name}/{child_name})");
        if let Some(first) = first_nested {
            return Err(SpecError::MultipleNestedTermsUnsupported {
                first,
                second: term,
            });
        }
        first_nested = Some(term);
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
    let re_slope = &*RE_SLOPE;
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
            effects.push(RandomEffect::Intercept {
                group,
                parent: None,
            });
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
    //      not matched here (same as the explicit slope pattern); a slope on a
    //      nested factor is refused in step 0.5.
    let re_islope = &*RE_ISLOPE;
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
            return Err(SpecError::DuplicateGroupingVar {
                name: group.clone(),
            });
        }
        effects.push(RandomEffect::Slope { group, vars });
        work = re_islope.replacen(&snapshot, 1, "").into_owned();
    }

    // 3. Random intercept: (1|g)
    let re_int = &*RE_INT;
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

    // 4. A term with '|' that no pattern above took would otherwise reach the
    //    fixed-effect parser as a bare syntax error at position 0.
    //    A term whose only fault is its grouping name (`(1|school.id)`) says
    //    so, rather than suggesting the shape the user already wrote.
    if let Some(m) = RE_ANY_TERM.find(&work) {
        let term = m.as_str().to_string();
        if let Some(name) = bad_grouping_name(&term) {
            return Err(SpecError::GroupingNameUnsupported { term, name });
        }
        return Err(SpecError::RandomEffectTermUnsupported { term });
    }
    // Extra parentheses hide a term from every pattern: `(1|(g))` keeps its '|',
    // and `((1|g))` leaves `()` behind once the inner term is taken.
    if work.contains('(') && (work.contains('|') || (!effects.is_empty() && work.contains("()"))) {
        return Err(SpecError::RandomEffectTermUnsupported {
            term: rhs.trim().to_string(),
        });
    }

    // Clean stray "+" — collapse "++", trim leading/trailing "+".
    let cleaned = clean_residual_plusses(&work);
    Ok((effects, cleaned))
}

/// The grouping name of a one-bar term `(lhs|name)` whose name is not a valid
/// grouping (ASCII letters, digits and '_', not starting with a digit,
/// optionally `a/b`), or `None` when the term has another fault. Deeper
/// nesting (`a/b/c`) and colon groupings (`a:b:c`) are other faults: their
/// names may be valid, so they keep the generic unsupported-term message.
fn bad_grouping_name(term: &str) -> Option<String> {
    let inner = term.strip_prefix('(')?.strip_suffix(')')?;
    let (lhs, name) = inner.split_once('|')?;
    if lhs.is_empty()
        || name.is_empty()
        || name.contains('|')
        || name.contains(':')
        || name.matches('/').count() > 1
    {
        return None;
    }
    let valid = |s: &str| {
        let mut chars = s.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let ok = match name.split_once('/') {
        Some((a, b)) => valid(a) && valid(b),
        None => valid(name),
    };
    (!ok).then(|| name.to_string())
}

/// The '+'-separated tokens of a random-effect term's left-hand side, or
/// `None` when it has none (left to the unsupported-term refusal).
fn re_lhs_tokens(lhs: &str) -> Option<Vec<&str>> {
    let tokens: Vec<&str> = lhs.split('+').filter(|t| !t.is_empty()).collect();
    (!tokens.is_empty()).then_some(tokens)
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
            // parse_identifier_list drops empty pieces, so a stray colon
            // (`x2:`, `:x2`) leaves a single variable.
            if vars.len() < 2 {
                return Err(SpecError::FormulaSyntax {
                    pos: 0,
                    msg: format!("interaction '{term}' needs at least two variables"),
                });
            }
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
    //! Parser edge cases outside the shared corpus
    //! (`configs/formula-fixtures/cases.json`, driven by `tests/formula_fixtures.rs`).
    use super::*;

    #[test]
    fn empty_lhs_with_separator_defaults_to_explained_variable() {
        // Both separator tokens (`~` and `=`) must trigger the empty-LHS default.
        for formula in ["~ x1 + x2", "= x1 + x2"] {
            let f = parse_formula(formula).unwrap();
            assert_eq!(f.dependent, "explained_variable", "formula={formula:?}");
            assert_eq!(f.predictors, vec!["x1", "x2"], "formula={formula:?}");
        }
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

    // A non-identifier token in a later term position is a syntax error (the
    // corpus row covers a bad first term).
    #[test]
    fn non_identifier_token_is_syntax_error() {
        assert!(matches!(
            parse_formula("y = x1 + 2bad"),
            Err(SpecError::FormulaSyntax { .. })
        ));
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

    // lme4's double bar (uncorrelated random effects) gets its own refusal
    // instead of a syntax error, in both the explicit and implicit forms.
    #[test]
    fn double_bar_random_effect_is_refused() {
        for f in ["y ~ x + (1 + x || g)", "y ~ x + (x || g)"] {
            assert!(
                matches!(
                    parse_formula(f),
                    Err(SpecError::UncorrelatedRandomEffectsUnsupported)
                ),
                "{f}"
            );
        }
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
