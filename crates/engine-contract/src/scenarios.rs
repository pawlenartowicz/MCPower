//! Per-block perturbation payload: `ScenarioPerturbations` and `LmeScenarioPerturbations`.

use serde::{Deserialize, Serialize};

use crate::generation::SyntheticKind;
use crate::outcome::ResidualDist;

/// One scenario's perturbation block — each field nudges one DGP axis away
/// from the optimistic baseline; the `Default` values mean "off".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioPerturbations {
    pub name: String,
    /// Between-study heterogeneity: one realized β-shift is drawn per simulation
    /// (the same jittered effects apply to every row of that sim), used directly
    /// (clamped at 0.0) with no model-level baseline combine. SD = value·|βⱼ| for
    /// each effect coefficient; binary outcomes additionally jitter the intercept
    /// on the log-odds scale with SD = the value. 0.0 = off.
    pub heterogeneity: f64,
    /// Variance ratio λ for residual heteroskedasticity. 1.0 = homoskedastic.
    pub heteroskedasticity_ratio: f64,
    pub correlation_noise_sd: f64,
    pub distribution_change_prob: f64,
    pub new_distributions: Vec<SyntheticKind>,
    pub residual_change_prob: f64,
    pub residual_dists: Vec<ResidualDist>,
    pub residual_df: f64,
    /// Factor-proportion sampling. `false` (default, optimistic) → each synthetic
    /// factor's per-level counts are the closest integer split of N to the
    /// requested proportions (deterministic largest-remainder walk, identical on
    /// every sim, no RNG consumed). `true` → independent per-row categorical draw
    /// (simple randomization; Multinomial count jitter). Additive — absent in
    /// older payloads, defaulting to `false` (exact), preserving v1 behaviour.
    #[serde(default)]
    pub sampled_factor_proportions: bool,
    pub lme: Option<LmeScenarioPerturbations>,
}

impl Default for ScenarioPerturbations {
    fn default() -> Self {
        // `heteroskedasticity_ratio` defaults to 1.0 (homoskedastic / off), NOT
        // the f64 zero a derived `Default` would give — a ratio of 0 is
        // nonsensical and would make `is_optimistic()` (which tests `== 1.0`)
        // reject every default-constructed baseline scenario.
        Self {
            name: String::new(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: Vec::new(),
            residual_change_prob: 0.0,
            residual_dists: Vec::new(),
            residual_df: 0.0,
            sampled_factor_proportions: false,
            lme: None,
        }
    }
}

/// LME-specific perturbations: random-effect distribution swap + ICC noise.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LmeScenarioPerturbations {
    pub random_effect_dist: ResidualDist,
    pub random_effect_df: f64,
    pub icc_noise_sd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_perturbations_roundtrip_with_pools() {
        let spec = ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.1,
            heteroskedasticity_ratio: 2.0,
            correlation_noise_sd: 0.15,
            distribution_change_prob: 0.2,
            new_distributions: vec![SyntheticKind::RightSkewed, SyntheticKind::HighKurtosis],
            residual_change_prob: 0.5,
            residual_dists: vec![ResidualDist::HighKurtosis, ResidualDist::RightSkewed],
            residual_df: 8.0,
            sampled_factor_proportions: true,
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 5.0,
                icc_noise_sd: 0.05,
            }),
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: ScenarioPerturbations = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn sampled_factor_proportions_defaults_false() {
        assert!(!ScenarioPerturbations::default().sampled_factor_proportions);
    }
}
