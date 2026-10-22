//! Top-level boundary envelope type (`SimulationContract`); the root of what every host passes in.

use serde::{Deserialize, Serialize};

use crate::design::DesignSpec;
use crate::estimator::{DfMethod, EstimatorSpec, WaldSe};
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
    /// Fixed-effect Wald-SE mode for the clustered-binary/count GLMM (no-op
    /// elsewhere). Additive; serde default flipped to `Rx` at 1.1.0 (fastmode),
    /// mirroring `config.json` estimation.wald_se. (design §7.)
    #[serde(default)]
    pub wald_se: WaldSe,
    /// Reference distribution for mixed-model fixed-effect Wald tests
    /// (no-op for `Ols` and unclustered `Glm`). Additive; serde default keeps
    /// older payloads valid, mirroring `config.json` estimation.df_method.
    #[serde(default)]
    pub df_method: DfMethod,
    /// Adaptive Gauss–Hermite quadrature node count for the GLMM likelihood.
    /// `1` = Laplace (the default). `> 1` requires an eligible shape
    /// (Binary/Count GLMM, single grouping factor, ≤ 3 REs per group, odd
    /// `k ≤ 25`) — validate() invariant_25_nagq_backstop is the backstop;
    /// hosts strip ineligible values first. Additive; serde default `1` keeps
    /// older payloads valid. Mirrors `config.json` estimation.nagq.
    #[serde(default = "default_nagq")]
    pub nagq: u8,
    pub test: TestSpec,
    #[serde(default)]
    pub posthoc: Vec<PosthocSpec>,
    /// Perturbation block for this contract's scenario — hosts fan out one
    /// contract per scenario.
    pub scenario: ScenarioPerturbations,
    /// Post-batch convergence threshold (fraction of failed sims); evaluated
    /// host-side, not by the kernel.
    pub max_failed_fraction: f64,
    /// Curvature hints for the mixed-model gradient search, one entry per
    /// sample-size grid point, set only by the WASM host. `None` (the
    /// default): the engine runs its own one-draw pilot to compute them.
    /// `Some(v)`: no pilot runs and grid point `k` fits with `v[k]`;
    /// `Some([])` means no hints, and a non-empty `v` whose length is not the
    /// grid length is an error. Lets every single-core worker of one run fit
    /// with the same hints, computed once from the master seed.
    #[serde(default)]
    pub curvature_hints: Option<Vec<Option<CurvatureHintParts>>>,
}

/// One glmm `CurvatureHint` as plain numbers, so it can cross a WASM worker
/// boundary: the row-major `dim × dim` inverse Hessian and the units the hint
/// was taken in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CurvatureHintParts {
    pub dim: usize,
    pub inv_hessian: Vec<f64>,
    pub units: f64,
}

/// Serde default for `SimulationContract.nagq` — Laplace (mirrors
/// `config.json` estimation.nagq).
fn default_nagq() -> u8 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::{DfMethod, WaldSe};
    use crate::fixtures::example1_simple_ols;

    #[test]
    fn contract_wald_se_defaults_when_absent() {
        // Serialize a named-map payload, strip the wald_se keys to mimic an older
        // producer, then confirm the new struct deserializes it to the default.
        let c = example1_simple_ols();
        let v = rmp_serde::to_vec_named(&c).unwrap();
        let mut val: rmpv::Value = rmp_serde::from_slice(&v).unwrap();
        if let rmpv::Value::Map(entries) = &mut val {
            entries.retain(|(k, _)| {
                k.as_str() != Some("wald_se")
                    && k.as_str() != Some("nagq")
                    && k.as_str() != Some("df_method")
            });
        }
        let mut stripped = Vec::new();
        rmpv::encode::write_value(&mut stripped, &val).unwrap();
        let back: SimulationContract = rmp_serde::from_slice(&stripped).unwrap();
        // Absent wald_se/nagq/df_method deserialize to their 1.1.0 defaults.
        assert_eq!(back.wald_se, WaldSe::Rx);
        assert_eq!(back.nagq, 1);
        assert_eq!(back.df_method, DfMethod::BetweenWithin);
    }

    #[test]
    fn contract_curvature_hints_default_to_none_when_absent() {
        let c = example1_simple_ols();
        let v = rmp_serde::to_vec_named(&c).unwrap();
        let mut val: rmpv::Value = rmp_serde::from_slice(&v).unwrap();
        if let rmpv::Value::Map(entries) = &mut val {
            entries.retain(|(k, _)| k.as_str() != Some("curvature_hints"));
        }
        let mut stripped = Vec::new();
        rmpv::encode::write_value(&mut stripped, &val).unwrap();
        let back: SimulationContract = rmp_serde::from_slice(&stripped).unwrap();
        assert_eq!(back.curvature_hints, None);
    }

    #[test]
    fn contract_curvature_hints_msgpack_roundtrip() {
        let mut c = example1_simple_ols();
        c.curvature_hints = Some(vec![
            Some(CurvatureHintParts {
                dim: 2,
                inv_hessian: vec![2.0, 0.5, 0.5, 1.0],
                units: 4.0,
            }),
            None,
        ]);
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let back: SimulationContract = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(c, back);
    }
}
