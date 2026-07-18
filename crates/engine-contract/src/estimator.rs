//! `EstimatorSpec` — append-only enum identifying the fitted-model solver applied to generated data.

use serde::{Deserialize, Serialize};

/// The solver applied to the generated data. Orthogonal to `OutcomeKind`:
/// the engine generates per `outcome.kind` and fits per `estimator`.
/// Algorithm-class names — `Glm` is the IRLS class (logit in 1.0), `Mle` is
/// the likelihood-optimisation class (intercept-only RE in 1.0). Parameters
/// (links, optimiser settings) are added when those features land.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorSpec {
    Ols,
    Glm,
    Mle,
}

impl EstimatorSpec {
    /// Whether this estimator's Wald-type decision statistic is compared against
    /// the Student-t distribution (OLS, df = N − P) rather than the standard
    /// normal (GLM/MLE Wald-z, df-independent). Single source for the t-vs-z
    /// decision shared by the critical-value kernel and the debug report.
    pub fn uses_student_t(self) -> bool {
        matches!(self, EstimatorSpec::Ols)
    }

    /// Degrees-of-freedom slots reported for this estimator: OLS reports its
    /// residual df; the z-based estimators (GLM/MLE) report none. Same OLS-vs-z
    /// partition as [`Self::uses_student_t`].
    pub fn df_slots(self, df_resid: f64) -> Vec<f64> {
        if self.uses_student_t() {
            vec![df_resid]
        } else {
            Vec::new()
        }
    }
}

/// Fixed-effect Wald-SE denominator selection for the clustered-binary/count
/// GLMM. Additive contract field; no-op for OLS/LMM. `Rx` is the 1.1.0 default
/// (fastmode: rx + Laplace is accurate enough for power analysis, where
/// Monte-Carlo noise dominates); `Hessian` (lme4 `use.hessian = TRUE`) is the
/// rigor opt-in. This default mirrors `configs/config.json` `estimation.wald_se`
/// — the single home for the cross-port default; change both together.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WaldSe {
    /// Per-fit FD-Hessian SE (glmer `use.hessian = TRUE`) — the lme4 "correct"
    /// denominator; the accurate/AGQ opt-in.
    Hessian,
    /// RX/PLS Schur (glmer `use.hessian = FALSE`) — the 1.1.0 default; faster
    /// but anticonservative for GLMM (assumes β–θ orthogonality, false under
    /// IRLS weight coupling), acceptable when MC noise dominates.
    #[default]
    Rx,
}

impl WaldSe {
    /// `wald_se` only affects the clustered-binary GLMM estimator. OLS/LMM and
    /// unclustered GLM SEs are already exact, so the orchestrator ignores the
    /// switch there rather than computing a Hessian. (design §7.)
    pub fn affects(self, est: EstimatorSpec, clustered: bool) -> bool {
        matches!(est, EstimatorSpec::Glm) && clustered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wald_se_serde_snake_case_and_default() {
        // wire tags are snake_case
        assert_eq!(
            serde_json::to_string(&WaldSe::Hessian).unwrap(),
            "\"hessian\""
        );
        assert_eq!(serde_json::to_string(&WaldSe::Rx).unwrap(), "\"rx\"");
        // default flipped to rx at 1.1.0 (fastmode); mirrors config.json estimation.wald_se
        assert_eq!(WaldSe::default(), WaldSe::Rx);
        // affects() is true only for clustered GLM
        assert!(WaldSe::Hessian.affects(EstimatorSpec::Glm, true));
        assert!(!WaldSe::Hessian.affects(EstimatorSpec::Glm, false)); // unclustered GLM
        assert!(!WaldSe::Hessian.affects(EstimatorSpec::Ols, true));
        assert!(!WaldSe::Hessian.affects(EstimatorSpec::Mle, true)); // LMM
    }
}
