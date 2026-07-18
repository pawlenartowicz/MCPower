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
    /// Non-canonical link override. `None` = canonical for the kind
    /// (`Binary` → logit, `Count` → log); `Some(Probit)` overrides `Binary` to
    /// a probit link. Rejected by validate() for any `kind` other than `Binary`
    /// (invariant_24_link_matches_kind). Additive at 1.1.0 — old payloads
    /// default to `None`.
    #[serde(default)]
    pub link: Option<LinkKind>,
}

/// How the linear predictor becomes Y. Canonical link per kind (`Binary` →
/// logit, `Count` → Poisson log); a non-canonical override rides on
/// `OutcomeSpec.link`. Carries no solver role — the estimator dispatch reads
/// kind + link to pick the `glmm::Family`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Continuous,
    Binary,
    Count,
}

/// Non-canonical link override for an outcome kind. Names only the overrides
/// that exist — canonical links (Binary→logit, Count→log) are the `None` case
/// on `OutcomeSpec.link`, not variants here.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Probit,
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
            link: None,
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: OutcomeSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn outcome_spec_roundtrip_probit_and_count() {
        for (kind, link) in [
            (OutcomeKind::Binary, Some(LinkKind::Probit)),
            (OutcomeKind::Count, None),
        ] {
            let spec = OutcomeSpec {
                kind,
                intercept: 0.5,
                coefficients: vec![0.5, 0.3],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link,
            };
            let bytes = rmp_serde::to_vec_named(&spec).unwrap();
            let back: OutcomeSpec = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(spec, back);
        }
    }

    #[test]
    fn outcome_spec_link_defaults_none_on_old_payload() {
        // Pre-1.1.0 payloads carry no `link` field; serde default fills None
        // (= canonical link for the kind).
        #[derive(Serialize)]
        struct OldOutcomeSpec {
            kind: OutcomeKind,
            intercept: f64,
            coefficients: Vec<f64>,
            residual: ResidualSpec,
        }
        let bytes = rmp_serde::to_vec_named(&OldOutcomeSpec {
            kind: OutcomeKind::Binary,
            intercept: 0.0,
            coefficients: vec![0.0],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: false,
            },
        })
        .unwrap();
        let back: OutcomeSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.link, None);
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
