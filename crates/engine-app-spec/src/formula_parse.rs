//! Host-agnostic formula parsing consumed by both GUI shells (Tauri and WASM).
//!
//! Thin wrapper over `engine_spec_builder::parse_formula` that serializes to the
//! exact JSON shape the Python PyO3 bridge emits (`parsed_formula_to_pydict`):
//! `{dependent, predictors, terms:[{kind,..}], random_effects:[{kind,..}]}`.
//! This keeps the cross-port formula-effect suite byte-comparable across hosts.

use engine_spec_builder::{parse_formula as sb_parse, RandomEffect, Term};
use serde::Serialize;

use crate::error::AdapterError;

/// JSON-facing mirror of `engine_spec_builder::Term`, serialised with a `kind` tag.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TermJson {
    Main { name: String },
    Interaction { vars: Vec<String> },
}

/// JSON-facing mirror of `engine_spec_builder::RandomEffect`, serialised with a `kind` tag.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RandomEffectJson {
    Intercept {
        group: String,
        parent: Option<String>,
    },
    Slope {
        group: String,
        vars: Vec<String>,
    },
}

/// Full parsed-formula view, mirroring the Python bridge dict shape.
#[derive(Debug, Clone, Serialize)]
pub struct FormulaParse {
    pub dependent: String,
    pub predictors: Vec<String>,
    pub terms: Vec<TermJson>,
    pub random_effects: Vec<RandomEffectJson>,
}

/// Parse a model formula into the host-agnostic [`FormulaParse`] shape.
///
/// Errors map transparently from `engine_spec_builder::SpecError` via
/// [`AdapterError`]; callers (Tauri/WASM) surface `.to_string()`.
pub fn parse_formula(input: &str) -> Result<FormulaParse, AdapterError> {
    let p = sb_parse(input)?;
    Ok(FormulaParse {
        dependent: p.dependent,
        predictors: p.predictors,
        terms: p
            .terms
            .into_iter()
            .map(|t| match t {
                Term::Main { name } => TermJson::Main { name },
                Term::Interaction { vars } => TermJson::Interaction { vars },
            })
            .collect(),
        random_effects: p
            .random_effects
            .into_iter()
            .map(|r| match r {
                RandomEffect::Intercept { group, parent } => {
                    RandomEffectJson::Intercept { group, parent }
                }
                RandomEffect::Slope { group, vars } => RandomEffectJson::Slope { group, vars },
            })
            .collect(),
    })
}
