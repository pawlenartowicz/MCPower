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
/// GLMM. Additive contract field; no-op for OLS/LMM. `Hessian` (lme4
/// `use.hessian = TRUE`) is the default; `Rx` is
/// the faster opt-in. This default mirrors `configs/config.json` `estimation.wald_se`
/// — the single home for the cross-port default; change both together.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WaldSe {
    /// Per-fit Hessian SE (glmer `use.hessian = TRUE`). It is computed at a
    /// singular fit too, where glmer has no Hessian and its summary reports
    /// the RX SE.
    #[default]
    Hessian,
    /// RX/PLS Schur (glmer `use.hessian = FALSE`) — faster, but anticonservative
    /// for GLMM (assumes β–θ orthogonality, false under IRLS weight coupling).
    Rx,
}

/// Reference distribution for mixed-model fixed-effect Wald tests. Additive
/// contract field; no-op for `Ols` and unclustered `Glm`. Each value other
/// than `Auto` names one test, and a design that test does not cover is
/// refused by `validate()` rather than tested another way. This default
/// mirrors `configs/config.json` `estimation.df_method` — the single home for
/// the cross-port default; change both together.
///
/// `Auto` resolves once per batch:
///
/// | Fit  | Grouping          | Primary random slope | Test           |
/// |------|-------------------|----------------------|----------------|
/// | LMM  | single            | none                 | between-within |
/// | LMM  | single            | any                  | Satterthwaite  |
/// | LMM  | crossed or nested | any                  | Satterthwaite  |
/// | GLMM | single            | any                  | between-within |
/// | GLMM | crossed or nested | any                  | Wald z         |
///
/// A slope counts when it is in the fitted random structure, whatever its
/// variance. Under any value, a draw whose Satterthwaite pass fails is tested
/// against Wald z.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DfMethod {
    /// Wald z against the standard normal, applied to every clustered fit
    /// regardless of estimator.
    Asymptotic,
    /// Between-within denominator degrees of freedom (SAS `DDFM=BETWITHIN`)
    /// applied to mixed fits — `Mle`, or `Glm` with a cluster spec. Invalid on
    /// a crossed or nested design, where the between/within split of the
    /// residual df is not defined. Unclustered `Glm` and `Ols` ignore the knob.
    BetweenWithin,
    /// Per-draw Satterthwaite df, as lmerTest computes them, on any Gaussian
    /// LMM (`Mle`). Invalid on a GLMM, which has no such df.
    Satterthwaite,
    /// The per-design choice in the table above; the default.
    #[default]
    Auto,
}

/// The fixed-effect test a mixed fit runs, as [`DfMethod::resolve`] picks it
/// for one design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedTest {
    /// Student t at the between-within denominator df (one table per batch).
    BetweenWithin,
    /// Student t at each draw's Satterthwaite df; Wald z on a draw whose df
    /// pass fails.
    Satterthwaite,
    /// Wald z against the standard normal.
    WaldZ,
}

impl DfMethod {
    /// The test this value selects for a fit by `estimator` on a design with
    /// cluster spec `cluster`. `primary_slope` says whether the primary
    /// grouping carries a random slope in the fitted structure. `Ok(None)` for
    /// a fit that is not mixed (`Ols`, or `Glm` without a cluster spec), where
    /// the value is ignored. A named test the design does not admit is an
    /// error, never another test: `BetweenWithin` on crossed or nested
    /// groupings, `Satterthwaite` on a GLMM.
    pub fn resolve(
        self,
        estimator: EstimatorSpec,
        cluster: Option<&crate::generation::ClusterSpec>,
        primary_slope: bool,
    ) -> Result<Option<MixedTest>, crate::error::ContractError> {
        use crate::error::ContractError;
        let Some(cluster) = cluster else {
            return Ok(None);
        };
        let lmm = match estimator {
            EstimatorSpec::Ols => return Ok(None),
            EstimatorSpec::Mle => true,
            EstimatorSpec::Glm => false,
        };
        let single = cluster.extra_groupings.is_empty();
        let test = match self {
            DfMethod::Asymptotic => MixedTest::WaldZ,
            DfMethod::Auto => match (lmm, single) {
                (true, true) if !primary_slope => MixedTest::BetweenWithin,
                (true, _) => MixedTest::Satterthwaite,
                (false, true) => MixedTest::BetweenWithin,
                (false, false) => MixedTest::WaldZ,
            },
            DfMethod::BetweenWithin if single => MixedTest::BetweenWithin,
            DfMethod::BetweenWithin => {
                return Err(ContractError::BetweenWithinNeedsSingleGrouping { lmm });
            }
            DfMethod::Satterthwaite if lmm => MixedTest::Satterthwaite,
            DfMethod::Satterthwaite => return Err(ContractError::SatterthwaiteNeedsLmm),
        };
        Ok(Some(test))
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
        // mirrors config.json estimation.wald_se
        assert_eq!(WaldSe::default(), WaldSe::Hessian);
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
        assert_eq!(
            serde_json::to_string(&DfMethod::Satterthwaite).unwrap(),
            "\"satterthwaite\""
        );
        assert_eq!(serde_json::to_string(&DfMethod::Auto).unwrap(), "\"auto\"");
        // default is auto; mirrors config.json estimation.df_method
        assert_eq!(DfMethod::default(), DfMethod::Auto);
    }

    fn cluster(extra: bool) -> crate::generation::ClusterSpec {
        use crate::generation::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let mut c =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 10 }, 0.25);
        if extra {
            c.extra_groupings.push(GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                tau_squared: 0.2,
                slopes: vec![],
            });
        }
        c
    }

    /// The `auto` table, one row per (fit, grouping, primary slope) cell.
    #[test]
    fn auto_resolves_per_design() {
        use EstimatorSpec::{Glm, Mle};
        use MixedTest::{BetweenWithin, Satterthwaite, WaldZ};
        let single = cluster(false);
        let crossed = cluster(true);
        for (estimator, c, slope, want) in [
            (Mle, &single, false, BetweenWithin),
            (Mle, &single, true, Satterthwaite),
            (Mle, &crossed, false, Satterthwaite),
            (Mle, &crossed, true, Satterthwaite),
            (Glm, &single, false, BetweenWithin),
            (Glm, &single, true, BetweenWithin),
            (Glm, &crossed, false, WaldZ),
            (Glm, &crossed, true, WaldZ),
        ] {
            let got = DfMethod::Auto.resolve(estimator, Some(c), slope).unwrap();
            assert_eq!(
                got,
                Some(want),
                "{estimator:?}, crossed {}, slope {slope}",
                !c.extra_groupings.is_empty()
            );
        }
    }

    /// Each named value is one test wherever it is defined, and an error
    /// where it is not; a fit that is not mixed ignores the value.
    #[test]
    fn named_values_run_their_test_or_refuse() {
        use EstimatorSpec::{Glm, Mle, Ols};
        let single = cluster(false);
        let crossed = cluster(true);
        for slope in [false, true] {
            for (estimator, c) in [
                (Mle, &single),
                (Mle, &crossed),
                (Glm, &single),
                (Glm, &crossed),
            ] {
                let r = |d: DfMethod| d.resolve(estimator, Some(c), slope);
                assert_eq!(r(DfMethod::Asymptotic).unwrap(), Some(MixedTest::WaldZ));
                let single_grouping = c.extra_groupings.is_empty();
                match r(DfMethod::BetweenWithin) {
                    Ok(t) => {
                        assert!(single_grouping);
                        assert_eq!(t, Some(MixedTest::BetweenWithin));
                    }
                    Err(e) => {
                        assert!(!single_grouping);
                        assert_eq!(
                            e,
                            crate::error::ContractError::BetweenWithinNeedsSingleGrouping {
                                lmm: estimator == Mle
                            }
                        );
                    }
                }
                match r(DfMethod::Satterthwaite) {
                    Ok(t) => {
                        assert_eq!(estimator, Mle);
                        assert_eq!(t, Some(MixedTest::Satterthwaite));
                    }
                    Err(e) => {
                        assert_eq!(estimator, Glm);
                        assert_eq!(e, crate::error::ContractError::SatterthwaiteNeedsLmm);
                    }
                }
            }
        }
        for d in [
            DfMethod::Auto,
            DfMethod::BetweenWithin,
            DfMethod::Satterthwaite,
            DfMethod::Asymptotic,
        ] {
            assert_eq!(d.resolve(Ols, Some(&crossed), true).unwrap(), None);
            assert_eq!(d.resolve(Glm, None, false).unwrap(), None);
        }
    }
}
