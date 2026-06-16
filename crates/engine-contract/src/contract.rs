//! Top-level boundary envelope type (`SimulationContract`); the root of what every host passes in.

use serde::{Deserialize, Serialize};

use crate::design::DesignSpec;
use crate::estimator::EstimatorSpec;
use crate::generation::GenerationSpec;
use crate::outcome::OutcomeSpec;
use crate::scenarios::ScenarioPerturbations;
use crate::test_spec::{PosthocSpec, TestSpec};

/// Root envelope every host passes in: the DGP truth (`generation` →
/// `outcome`) on one side, the analysis run against it (`design_test`,
/// `estimator`, `test`) on the other. msgpack-encoded at the FFI boundary;
/// evolution is additive-only at minor versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationContract {
    // ---- DGP side (the truth) ----
    pub generation: GenerationSpec,
    /// Design matrix the *generator* materialises (the truth).
    pub design_generation: DesignSpec,
    pub outcome: OutcomeSpec,
    // ---- Estimation side (the analysis) ----
    /// Design matrix the *fitted model* uses; `None` = same as
    /// `design_generation` (the adapter resolves the default).
    #[serde(default)]
    pub design_test: Option<DesignSpec>,
    pub estimator: EstimatorSpec,
    pub test: TestSpec,
    #[serde(default)]
    pub posthoc: Vec<PosthocSpec>,
    /// Perturbation block for this contract's scenario — hosts fan out one
    /// contract per scenario.
    pub scenario: ScenarioPerturbations,
    /// Post-batch convergence threshold (fraction of failed sims); evaluated
    /// host-side, not by the kernel.
    pub max_failed_fraction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::example1_simple_ols;

    #[test]
    fn simulation_contract_msgpack_roundtrip_example1() {
        let c = example1_simple_ols();
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let back: SimulationContract = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(c, back);
    }

}
