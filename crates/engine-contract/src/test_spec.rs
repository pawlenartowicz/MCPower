//! Hypothesis-test configuration: `TestSpec`, `TestTarget`, `CorrectionMethod`, and `PosthocSpec`.

use serde::{Deserialize, Serialize};

use crate::ids::ColumnId;

/// What is tested and how: targets addressing `design_test.terms` by
/// position, the family-wise correction across them, and the test α.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestSpec {
    pub targets: Vec<TestTarget>,
    pub correction: CorrectionMethod,
    pub alpha: f64,
}

/// One tested hypothesis; all indices are term positions into
/// `design_test.terms`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TestTarget {
    Marginal {
        term: u32,
    },
    Joint {
        terms: Vec<u32>,
    },
    /// Pairwise contrast `β_positive − β_negative`. Both indices are term
    /// positions into `design_test.terms` (same convention as `Marginal`).
    /// When `negative` maps to `DesignTerm::Const` (the dummy-coding reference),
    /// the contract adapter collapses this to a `Marginal` on `positive`.
    Contrast {
        positive: u32,
        negative: u32,
    },
}

/// Family-wise correction applied across the test targets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMethod {
    None,
    Bonferroni,
    Holm,
    BenjaminiHochberg,
    TukeyHsd,
}

impl CorrectionMethod {
    /// Integer code consumed by `engine_core::SimulationSpec.correction_method`:
    /// none=0, bonferroni=1, holm=2, bh=3, tukey_hsd=4. Byte-for-byte alignment
    /// with the hosts' code maps (Python `_CORRECTION_CODE`) is load-bearing —
    /// the `correction_codes_match_spec_builder_mapping` test pins it.
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Bonferroni => 1,
            Self::Holm => 2,
            Self::BenjaminiHochberg => 3,
            Self::TukeyHsd => 4,
        }
    }
}

/// Pairwise post-hoc block for one factor: its generation column, the term
/// positions of its dummies, and an optional separate α (`None` → test α).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PosthocSpec {
    pub factor_column: ColumnId,
    pub target_term_indices: Vec<u32>,
    pub posthoc_alpha: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_roundtrip_with_marginal_and_joint() {
        let spec = TestSpec {
            targets: vec![
                TestTarget::Marginal { term: 1 },
                TestTarget::Joint { terms: vec![2, 3] },
            ],
            correction: CorrectionMethod::Bonferroni,
            alpha: 0.05,
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: TestSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn correction_codes_match_spec_builder_mapping() {
        assert_eq!(CorrectionMethod::None.code(), 0);
        assert_eq!(CorrectionMethod::Bonferroni.code(), 1);
        assert_eq!(CorrectionMethod::Holm.code(), 2);
        assert_eq!(CorrectionMethod::BenjaminiHochberg.code(), 3);
        assert_eq!(CorrectionMethod::TukeyHsd.code(), 4);
    }

    #[test]
    fn contrast_roundtrip() {
        let spec = TestSpec {
            targets: vec![TestTarget::Contrast {
                positive: 1,
                negative: 2,
            }],
            correction: CorrectionMethod::None,
            alpha: 0.05,
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: TestSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn posthoc_spec_roundtrip() {
        let spec = PosthocSpec {
            factor_column: ColumnId(1),
            target_term_indices: vec![2, 3],
            posthoc_alpha: Some(0.025),
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: PosthocSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }
}
