//! User-facing assignment-string grammar for `name=value` pairs.
//!
//! `AssignmentKey::Pair` keys are stored with names sorted (a ≤ b) so lookups must present sorted pairs.
//!
//! Mirrors the Python `_AssignmentParser` (`ports/py/mcpower/parsers.py:17-296`):
//! a comma-separated list where each item is either
//! - `name=value`           (variable_type, effect)
//! - `corr(a,b)=value`      (correlation)
//!
//! Per-item failures (unknown names, malformed values) are collected into
//! `Assignments.errors`; the top level only returns `Err` when the input has no
//! `=` at all.

use crate::error::SpecError;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentKind {
    VariableType,
    Correlation,
    Effect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentKey {
    Name(String),
    Pair(String, String),
}

/// Structured parameters for a variable-type assignment value. Defaults for the
/// bare forms are filled here (binary → `proportion 0.5`; factor → 3 equal
/// levels) so every host reads identical numbers without re-deriving them.
#[derive(Debug, Clone, PartialEq)]
pub enum VarTypeParams {
    /// No params (normal / right_skewed / left_skewed / high_kurtosis / uniform).
    Bare,
    Binary {
        proportion: f64,
    },
    Factor {
        n_levels: i32,
        proportions: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignmentValue {
    /// Retained for back-compat; the parser now emits `VariableTypeWithParams`.
    VariableType(String),
    /// Variable type plus its structured params (tuple syntax `(binary,0.3)` /
    /// `(factor,0.2,0.3,0.5)` and bare-form defaults), single-sourced from Rust.
    VariableTypeWithParams {
        var_type: String,
        params: VarTypeParams,
    },
    Correlation(f64),
    Effect(f64),
}

#[derive(Debug, Default)]
pub struct Assignments {
    pub items: Vec<(AssignmentKey, AssignmentValue)>,
    pub errors: Vec<SpecError>,
}

#[derive(Debug, Clone, Copy)]
pub struct KnownNames<'a> {
    pub predictors: &'a [String],
    pub interaction_terms: &'a [String],
}

/// User-facing variable types accepted in `name=value` assignments — the single
/// source the Python/R ports bind to (previously `_SUPPORTED_VAR_TYPES` /
/// `.SUPPORTED_VAR_TYPES`). Bare `factor`/`binary` get default params.
const SUPPORTED_VAR_TYPES: &[&str] = &[
    "normal",
    "binary",
    "right_skewed",
    "left_skewed",
    "high_kurtosis",
    "uniform",
    "factor",
];

/// Parse a comma-separated `name=value` assignment string of the given `kind`.
///
/// Per-item failures (unknown names, malformed values, unrecognised variable types) are
/// collected into `Assignments.errors` rather than short-circuiting — the caller decides
/// whether to propagate them. The top level returns `Err(MalformedAssignment)` only
/// when the entire input contains no `=`, or has unbalanced parentheses.
///
/// # Errors
/// - [`SpecError::MalformedAssignment`]: input contains no `=` at all, or unbalanced parens.
pub fn parse_assignments(
    input: &str,
    kind: AssignmentKind,
    known: &KnownNames<'_>,
) -> Result<Assignments, SpecError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(Assignments::default());
    }

    // Top-level malformed: no '=' anywhere.
    if !trimmed.contains('=') {
        return Err(SpecError::MalformedAssignment {
            input: trimmed.to_string(),
        });
    }

    let parts = split_top_level_commas(trimmed).map_err(|_| SpecError::MalformedAssignment {
        input: trimmed.to_string(),
    })?;

    let mut items = Vec::new();
    let mut errors = Vec::new();

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match parse_single_item(part, kind, known) {
            Ok(Some(item)) => items.push(item),
            Ok(None) => {}
            Err(e) => errors.push(e),
        }
    }

    Ok(Assignments { items, errors })
}

/// Split on top-level commas. Errors on unbalanced parens (matching the Python
/// `_split_assignments` behaviour which raises `ValueError`).
fn split_top_level_commas(s: &str) -> Result<Vec<&str>, ()> {
    let mut depth: i32 = 0;
    let mut start = 0;
    let mut out = Vec::new();
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return Err(());
                }
            }
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(());
    }
    out.push(&s[start..]);
    Ok(out)
}

/// Split a comma-separated assignment string at top-level commas (parens
/// suppress splitting), trimming each segment and dropping empties — the shared
/// splitter the host correlation/effect normalisers bind to (previously the
/// per-port `_split_assignments` / `.split_assignments`). Errors on unbalanced
/// parens (matching the hosts' `ValueError`/`stop`).
pub fn split_assignments(input: &str) -> Result<Vec<String>, SpecError> {
    let parts = split_top_level_commas(input).map_err(|_| SpecError::MalformedAssignment {
        input: input.to_string(),
    })?;
    Ok(parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect())
}

fn parse_single_item(
    part: &str,
    kind: AssignmentKind,
    known: &KnownNames<'_>,
) -> Result<Option<(AssignmentKey, AssignmentValue)>, SpecError> {
    let Some(eq_pos) = part.rfind('=') else {
        return Err(SpecError::MalformedAssignment {
            input: part.to_string(),
        });
    };
    let lhs = part[..eq_pos].trim();
    let rhs = part[eq_pos + 1..].trim();

    match kind {
        AssignmentKind::VariableType => {
            if !known.predictors.iter().any(|n| n == lhs) {
                return Err(SpecError::UnknownAssignmentName {
                    name: lhs.to_string(),
                });
            }
            // Accept optional surrounding quotes for parity with Python `x1="binary"`.
            let rhs_unquoted = strip_quotes(rhs);
            match parse_var_type_value(rhs_unquoted) {
                Ok((var_type, params)) => Ok(Some((
                    AssignmentKey::Name(lhs.to_string()),
                    AssignmentValue::VariableTypeWithParams { var_type, params },
                ))),
                Err(message) => Err(SpecError::InvalidVariableTypeValue {
                    name: lhs.to_string(),
                    message,
                }),
            }
        }
        AssignmentKind::Correlation => {
            let inner = lhs
                .strip_prefix("corr(")
                .or_else(|| lhs.strip_prefix("corr ("))
                .and_then(|s| s.strip_suffix(')'));
            let Some(inner) = inner else {
                return Err(SpecError::MalformedAssignment {
                    input: part.to_string(),
                });
            };
            let names: Vec<&str> = inner.split(',').map(str::trim).collect();
            if names.len() != 2 {
                return Err(SpecError::MalformedAssignment {
                    input: part.to_string(),
                });
            }
            let (a, b) = (names[0], names[1]);
            if a == b {
                return Err(SpecError::MalformedAssignment {
                    input: format!("corr({a},{a}): cannot correlate a variable with itself"),
                });
            }
            let pool: BTreeSet<&String> = known
                .predictors
                .iter()
                .chain(known.interaction_terms.iter())
                .collect();
            if !pool.iter().any(|n| n.as_str() == a) {
                return Err(SpecError::UnknownAssignmentName {
                    name: a.to_string(),
                });
            }
            if !pool.iter().any(|n| n.as_str() == b) {
                return Err(SpecError::UnknownAssignmentName {
                    name: b.to_string(),
                });
            }
            let value: f64 = rhs.parse().map_err(|_| SpecError::MalformedAssignment {
                input: part.to_string(),
            })?;
            let (a_sorted, b_sorted) = if a <= b { (a, b) } else { (b, a) };
            Ok(Some((
                AssignmentKey::Pair(a_sorted.to_string(), b_sorted.to_string()),
                AssignmentValue::Correlation(value),
            )))
        }
        AssignmentKind::Effect => {
            let pool: BTreeSet<&String> = known
                .predictors
                .iter()
                .chain(known.interaction_terms.iter())
                .collect();
            if !pool.iter().any(|n| n.as_str() == lhs) {
                return Err(SpecError::UnknownAssignmentName {
                    name: lhs.to_string(),
                });
            }
            let value: f64 = rhs.parse().map_err(|_| SpecError::MalformedAssignment {
                input: part.to_string(),
            })?;
            Ok(Some((
                AssignmentKey::Name(lhs.to_string()),
                AssignmentValue::Effect(value),
            )))
        }
    }
}

/// Parse a single variable-type RHS (`"binary"`, `"(factor,0.2,0.3,0.5)"`, …)
/// into `(var_type, VarTypeParams)`. Mirrors the host `_parse_variable_type_value`
/// verbatim — including error wording and the bare-form defaults — so routing
/// `set_variable_type` through Rust does not move behaviour. The factor level
/// ceiling is single-sourced from the embedded config. Returns the per-value
/// diagnostic string on failure (the caller prefixes the predictor name).
fn parse_var_type_value(value: &str) -> Result<(String, VarTypeParams), String> {
    if value.starts_with('(') && value.ends_with(')') {
        let content = &value[1..value.len() - 1];
        if !content.contains(',') {
            return Err(
                "Invalid tuple format. Expected '(type,value)' or '(type,val1,val2,...)'"
                    .to_string(),
            );
        }
        let parts: Vec<&str> = content.split(',').map(str::trim).collect();
        if parts.len() < 2 {
            return Err("Expected at least 2 values in tuple".to_string());
        }
        let var_type = parts[0];
        if !SUPPORTED_VAR_TYPES.contains(&var_type) {
            return Err(format!("Unsupported type '{var_type}'"));
        }

        if var_type == "binary" {
            if parts.len() != 2 {
                return Err(
                    "Binary type expects exactly 2 values: (binary, proportion)".to_string()
                );
            }
            let proportion: f64 = parts[1]
                .parse()
                .map_err(|_| format!("Invalid proportion value '{}'", parts[1]))?;
            if !(0.0..=1.0).contains(&proportion) {
                return Err("Proportion must be between 0 and 1".to_string());
            }
            return Ok(("binary".to_string(), VarTypeParams::Binary { proportion }));
        }

        if var_type == "factor" {
            let max_levels = engine_contract::config().limits.factor_levels[1] as usize;
            if parts.len() == 2 {
                // n_levels form. Like Python `int()`, reject non-integer text ("3.5").
                let n_levels: i32 = parts[1].parse().map_err(|_| {
                    format!("Invalid number of levels '{}'. Must be integer", parts[1])
                })?;
                if n_levels < 2 {
                    return Err("Factor must have at least 2 levels".to_string());
                }
                if n_levels as usize > max_levels {
                    return Err(format!("Factor cannot have more than {max_levels} levels"));
                }
                let proportions = vec![1.0 / n_levels as f64; n_levels as usize];
                return Ok((
                    "factor".to_string(),
                    VarTypeParams::Factor {
                        n_levels,
                        proportions,
                    },
                ));
            }
            // proportions form.
            let mut proportions: Vec<f64> = Vec::with_capacity(parts.len() - 1);
            for p in &parts[1..] {
                match p.parse::<f64>() {
                    Ok(v) => proportions.push(v),
                    Err(_) => {
                        return Err("Invalid proportions. All values must be numeric".to_string())
                    }
                }
            }
            let n_levels = proportions.len() as i32;
            if (n_levels as usize) < 2 {
                return Err("Factor must have at least 2 levels".to_string());
            }
            if n_levels as usize > max_levels {
                return Err(format!("Factor cannot have more than {max_levels} levels"));
            }
            if proportions.iter().any(|&p| p <= 0.0) {
                return Err("All proportions must be positive (greater than 0)".to_string());
            }
            let total: f64 = proportions.iter().sum();
            let proportions: Vec<f64> = proportions.iter().map(|&p| p / total).collect();
            return Ok((
                "factor".to_string(),
                VarTypeParams::Factor {
                    n_levels,
                    proportions,
                },
            ));
        }

        return Err("Tuple format only supported for binary and factor variables".to_string());
    }

    // Bare form.
    if !SUPPORTED_VAR_TYPES.contains(&value) {
        return Err(format!(
            "Unsupported type '{value}'. Valid: {}",
            SUPPORTED_VAR_TYPES.join(", ")
        ));
    }
    let params = match value {
        "binary" => VarTypeParams::Binary { proportion: 0.5 },
        "factor" => VarTypeParams::Factor {
            n_levels: 3,
            proportions: vec![1.0 / 3.0; 3],
        },
        _ => VarTypeParams::Bare,
    };
    Ok((value.to_string(), params))
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        let first = bytes[0];
        let last = bytes[s.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return s[1..s.len() - 1].trim();
        }
    }
    s
}
