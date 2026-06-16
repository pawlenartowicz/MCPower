//! `DesignSpec` and `DesignTerm` — the post-parse model formula structure passed across the boundary.

use serde::{Deserialize, Serialize};

use crate::ids::ColumnId;

/// Ordered term list defining a design matrix. Term order is part of the
/// wire contract: the index into `terms` is the `term` coordinate that
/// `TestTarget`s and `coefficients` address.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DesignSpec {
    pub terms: Vec<DesignTerm>,
}

/// One design-matrix column, expressed against `GenerationSpec.columns`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DesignTerm {
    /// Intercept (constant-1) column.
    Const,
    /// The generated column at `column`, used as-is (continuous predictor).
    Direct {
        column: ColumnId,
    },
    /// 0/1 indicator for one level of the factor at `column`.
    DummyOf {
        column: ColumnId,
        level_index: u32,
    },
    /// A materialised interaction cell: the raw elementwise product of its
    /// component columns. Each component is a `Direct` (continuous) or
    /// `DummyOf` (factor cell) — `Const` and nested `Interaction` are rejected
    /// by `validate()` (invariant 18). At least 2 components.
    Interaction {
        components: Vec<DesignTerm>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_spec_roundtrip_keeps_term_order() {
        let spec = DesignSpec {
            terms: vec![
                DesignTerm::Const,
                DesignTerm::Direct {
                    column: ColumnId(0),
                },
                DesignTerm::DummyOf {
                    column: ColumnId(1),
                    level_index: 1,
                },
                DesignTerm::DummyOf {
                    column: ColumnId(1),
                    level_index: 2,
                },
            ],
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: DesignSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn design_spec_roundtrip_keeps_interaction_term() {
        let spec = DesignSpec {
            terms: vec![
                DesignTerm::Const,
                DesignTerm::Direct {
                    column: ColumnId(0),
                },
                DesignTerm::DummyOf {
                    column: ColumnId(1),
                    level_index: 2,
                },
                DesignTerm::Interaction {
                    components: vec![
                        DesignTerm::Direct {
                            column: ColumnId(0),
                        },
                        DesignTerm::DummyOf {
                            column: ColumnId(1),
                            level_index: 2,
                        },
                    ],
                },
            ],
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: DesignSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }
}
