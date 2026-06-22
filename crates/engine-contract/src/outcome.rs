//! DGP outcome axis: `OutcomeSpec`, `OutcomeKind`, and residual distribution.

use serde::{Deserialize, Serialize};

use crate::ids::ColumnId;

/// The Y side of the DGP: link kind, true coefficients, and residual shape.
/// Perturbation magnitudes (λ, heterogeneity) are scenario-only — the model
/// states structure, scenarios state how wrong it might be.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutcomeSpec {
    pub kind: OutcomeKind,
    pub intercept: f64,
    /// One per `design_generation.terms` entry (invariant 1).
    pub coefficients: Vec<f64>,
    pub residual: ResidualSpec,
    /// Predictor driving heteroskedastic residual variance, or `None` for the
    /// linear predictor Xβ. The variance ratio λ comes from the active
    /// scenario (`ScenarioPerturbations.heteroskedasticity_ratio`).
    #[serde(default)]
    pub heteroskedasticity_driver: Option<ColumnId>,
}

/// How the linear predictor becomes Y. `Binary` implies a logit link in 1.0
/// (probit is a later additive parameter). Carries no solver role.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Continuous,
    Binary,
}

/// Residual marginal for continuous outcomes. No df field: `HighKurtosis` is
/// t with the active scenario's `residual_df` (shape is the user's assertion,
/// severity stays scenario-graded).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResidualSpec {
    pub distribution: ResidualDist,
    /// Explicitly chosen by the user (incl. explicit `normal`) — scenario
    /// residual swaps leave it alone. Unpinned `Normal` is the only
    /// swap-eligible state.
    #[serde(default)]
    pub pinned: bool,
}

/// Residual marginal families — the same five parameterless shapes as
/// `SyntheticKind`'s continuous arms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidualDist {
    Normal,
    RightSkewed,
    LeftSkewed,
    HighKurtosis,
    Uniform,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_spec_roundtrip_binary_with_het_driver() {
        let spec = OutcomeSpec {
            kind: OutcomeKind::Binary,
            intercept: -1.3863,
            coefficients: vec![-1.3863, 0.4, 0.2, 0.5],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: true,
            },
            heteroskedasticity_driver: Some(ColumnId(0)),
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: OutcomeSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn residual_spec_pinned_defaults_false_on_old_payload() {
        // Pre-pin payloads carry only `distribution`; serde default fills
        // pinned = false (= swap-eligible when Normal).
        #[derive(Serialize)]
        struct OldResidualSpec {
            distribution: ResidualDist,
        }
        let bytes = rmp_serde::to_vec_named(&OldResidualSpec {
            distribution: ResidualDist::Normal,
        })
        .unwrap();
        let back: ResidualSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert!(!back.pinned);
    }
}
