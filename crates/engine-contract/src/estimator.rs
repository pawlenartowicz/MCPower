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

    /// Per-target degrees-of-freedom slots, mirroring the shape of
    /// `CritValueTable::df_per_target`'s row for one sample size: reads
    /// `df_per_target` — already resolved by the crit-table builder, including
    /// the between-within denominator df for a mixed spec — rather than
    /// recomputing it here. An empty `df_per_target` (every standard-normal /
    /// Wald-z build, which is OLS's `uses_student_t() == false` twin `Glm`/`Mle`
    /// under `DfMethod::Asymptotic`) reports `n_targets` empty slots; a
    /// non-empty row (OLS always; a mixed target under the between-within
    /// reference) reports one `[df]` slot per target.
    pub fn df_slots(self, df_per_target: &[f64], n_targets: usize) -> Vec<Vec<f64>> {
        if df_per_target.is_empty() {
            vec![Vec::new(); n_targets]
        } else {
            debug_assert_eq!(
                df_per_target.len(),
                n_targets,
                "a non-empty df_per_target row must carry exactly one df per target"
            );
            df_per_target.iter().map(|&d| vec![d]).collect()
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

/// Reference distribution for mixed-model fixed-effect Wald tests. Additive
/// contract field; no-op for `Ols` and unclustered `Glm`. This default mirrors
/// `configs/config.json` `estimation.df_method` — the single home for the
/// cross-port default; change both together.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DfMethod {
    /// Wald z against the standard normal, applied to every clustered fit
    /// regardless of estimator — the opt-in.
    Asymptotic,
    /// Between-within denominator degrees of freedom (SAS `DDFM=BETWITHIN`)
    /// applied to mixed fits — `Mle`, or `Glm` with a cluster spec — the
    /// default. Unclustered `Glm` and `Ols` ignore the knob.
    #[default]
    BetweenWithin,
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
    }

    #[test]
    fn df_method_serde_snake_case_and_default() {
        // wire tags are snake_case
        assert_eq!(
            serde_json::to_string(&DfMethod::Asymptotic).unwrap(),
            "\"asymptotic\""
        );
        assert_eq!(
            serde_json::to_string(&DfMethod::BetweenWithin).unwrap(),
            "\"between_within\""
        );
        // default is between-within; mirrors config.json estimation.df_method
        assert_eq!(DfMethod::default(), DfMethod::BetweenWithin);
    }
}
