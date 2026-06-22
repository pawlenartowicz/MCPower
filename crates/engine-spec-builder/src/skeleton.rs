//! The `EffectSkeleton` — a structured, index-only description of every
//! design-matrix effect (β-column) for one analysis.
//!
//! The engine speaks only indices; display strings like `origin[Japan]` are
//! owned by each port. This skeleton is the single, Rust-sourced layout the
//! ports render names from: each `FactorLevel` carries a `level` *index* the
//! port maps to its own `labels[factor][level]` store, so factor expansion and
//! the flat β-column layout are never re-implemented per port.
//!
//! Returned once per analysis from the build call (it is identical across
//! scenarios — perturbations don't change the design). It is **not** a wire or
//! result field; the `SimulationContract` / `PowerResult` shapes are untouched.

use serde::{Deserialize, Serialize};

/// One design-matrix effect, described structurally with indices only.
///
/// `predictor` / `factor` are the **formula identifiers** the user wrote (e.g.
/// `x1`, `group`), not factor-level labels — returning them does not put labels
/// in the engine. The only level information is the integer `level`, which the
/// port resolves against its own label store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectDescriptor {
    /// The model intercept (design column 0). Never a power target — present
    /// only so positions line up with `target_indices`' 1-based space.
    Intercept,
    /// A continuous or binary main effect. `predictor` is the formula identifier.
    Continuous { predictor: String },
    /// One factor dummy. `factor` is the formula identifier; `level` is the
    /// **0-based index of this dummy's level within the factor's FULL ordered
    /// label list (reference level included)** — a name lookup, not an
    /// arithmetic offset from the contract's 1-based `DummyOf.level_index`.
    /// The reference may sit anywhere in declaration order, so the port simply
    /// renders `labels[factor][level]` with no reference-offset logic.
    FactorLevel { factor: String, level: u32 },
    /// An interaction term. Each component is a `Continuous` or `FactorLevel`.
    Interaction { components: Vec<EffectDescriptor> },
}

/// β-column-aligned effect layout for one analysis: index 0 is the intercept,
/// and the remaining entries follow `design_test` term order — the same 1-based
/// space as `PowerResult.target_indices`. Built once per analysis.
pub type EffectSkeleton = Vec<EffectDescriptor>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_tags_are_snake_case() {
        let s = serde_json::to_string(&EffectDescriptor::Intercept).unwrap();
        assert_eq!(s, r#"{"kind":"intercept"}"#);
        let s = serde_json::to_string(&EffectDescriptor::Continuous {
            predictor: "x1".into(),
        })
        .unwrap();
        assert_eq!(s, r#"{"kind":"continuous","predictor":"x1"}"#);
        let s = serde_json::to_string(&EffectDescriptor::FactorLevel {
            factor: "group".into(),
            level: 2,
        })
        .unwrap();
        assert_eq!(s, r#"{"kind":"factor_level","factor":"group","level":2}"#);
    }

    #[test]
    fn skeleton_serde_round_trips() {
        let skeleton: EffectSkeleton = vec![
            EffectDescriptor::Intercept,
            EffectDescriptor::Continuous {
                predictor: "x1".into(),
            },
            EffectDescriptor::FactorLevel {
                factor: "group".into(),
                level: 1,
            },
            EffectDescriptor::Interaction {
                components: vec![
                    EffectDescriptor::Continuous {
                        predictor: "x1".into(),
                    },
                    EffectDescriptor::FactorLevel {
                        factor: "group".into(),
                        level: 2,
                    },
                ],
            },
        ];
        let json = serde_json::to_string(&skeleton).unwrap();
        let back: EffectSkeleton = serde_json::from_str(&json).unwrap();
        assert_eq!(skeleton, back);
    }
}
