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
