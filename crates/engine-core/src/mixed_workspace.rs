//! Spec-reading glue between MCPower's `SimulationSpec`/`ClusterSpec` and the
//! `glmm` crate's fit kernels. These dispatch helpers read sim-layer types that
//! stay in engine-core (so they cannot live in `glmm`), build a `glmm::ModelSpec`
//! via the free conversion fns below, and call the relocated workspace
//! constructors `glmm::mcpower::{LmmWorkspace,GlmmWorkspace}::for_cluster_spec`.

use crate::spec::SimulationSpec;

/// `ClusterSpec` → `glmm::ModelSpec`. A **free fn, not a `From` impl**: both
/// types are foreign to engine-core, so the orphan rule forbids the impl here,
/// and routing it through `engine-contract` would drag `glmm`'s faer/bobyqa/pulp
/// into that lightweight crate. Populates `estimator`/`wald_se` from the
/// `SimulationSpec` in one place so a half-built `ModelSpec` (e.g. an OLS spec
/// mislabelled `Mle`) can never reach the friendly `glmm::fit` dispatch.
pub(crate) fn cluster_to_model_spec(
    c: &engine_contract::ClusterSpec,
    estimator: engine_contract::EstimatorSpec,
    wald_se: engine_contract::WaldSe,
) -> glmm::ModelSpec {
    glmm::ModelSpec {
        sizing: match c.sizing {
            engine_contract::ClusterSizing::FixedClusters { n_clusters } => {
                glmm::Sizing::FixedClusters { n_clusters }
            }
            engine_contract::ClusterSizing::FixedSize { cluster_size } => {
                glmm::Sizing::FixedSize { cluster_size }
            }
        },
        tau_squared: c.tau_squared,
        slopes: c.slopes.iter().map(slope_to_glmm).collect(),
        extra_groupings: c
            .extra_groupings
            .iter()
            .map(|g| glmm::Grouping {
                relation: match g.relation {
                    engine_contract::GroupingRelation::Crossed { n_clusters } => {
                        glmm::GroupingRelation::Crossed { n_clusters }
                    }
                    engine_contract::GroupingRelation::NestedWithin { n_per_parent } => {
                        glmm::GroupingRelation::NestedWithin { n_per_parent }
                    }
                },
                tau_squared: g.tau_squared,
                slopes: g.slopes.iter().map(slope_to_glmm).collect(),
            })
            .collect(),
        // Set from the SimulationSpec's estimator/wald_se HERE — the one place that
        // knows them. The kernel constructors ignore both fields; only the friendly
        // `fit` reads them, so populating them correctly here keeps a half-built
        // ModelSpec from ever escaping.
        estimator: match estimator {
            engine_contract::EstimatorSpec::Ols => glmm::Estimator::Ols,
            engine_contract::EstimatorSpec::Glm => glmm::Estimator::Glm,
            engine_contract::EstimatorSpec::Mle => glmm::Estimator::Mle,
        },
        wald_se: wald_se_to_glmm(wald_se),
    }
}

fn slope_to_glmm(s: &engine_contract::SlopeTerm) -> glmm::SlopeTerm {
    glmm::SlopeTerm {
        // ColumnId is a newtype `ColumnId(pub u32)`; glmm::ColumnId = u32 —
        // read `.0`, do NOT `as`-cast.
        column: s.column.0,
        variance: s.variance,
        corr_with_intercept: s.corr_with_intercept,
        corr_with: s.corr_with.clone(),
    }
}

/// `engine_contract::WaldSe` → `glmm::WaldSe`. Shared by `cluster_to_model_spec`
/// and the `fit_glmm` call sites (batch.rs / introspect.rs), which pass
/// `spec.wald_se` to the now-`glmm::WaldSe`-typed kernel parameter.
pub(crate) fn wald_se_to_glmm(wald_se: engine_contract::WaldSe) -> glmm::WaldSe {
    match wald_se {
        engine_contract::WaldSe::Hessian => glmm::WaldSe::Hessian,
        engine_contract::WaldSe::Rx => glmm::WaldSe::Rx,
    }
}

/// Dispatch helper: Some iff this spec routes to the general path
/// (estimator Mle + non-degenerate ClusterSpec). Degenerate specs keep the
/// scalar Brent path — shipped workloads never enter the new code.
pub fn build_lmm_workspace(
    spec: &SimulationSpec,
    max_n: usize,
    n_predictors: usize,
) -> Option<Box<glmm::mcpower::LmmWorkspace>> {
    let cluster = spec.cluster.as_ref()?;
    // General path = Mle + non-degenerate ClusterSpec, where non-degenerate means
    // extra groupings OR primary slopes. A degenerate single-intercept spec keeps
    // the scalar Brent path in `lme.rs`.
    if spec.estimator != engine_contract::EstimatorSpec::Mle
        || (cluster.extra_groupings.is_empty() && cluster.slopes.is_empty())
    {
        return None;
    }
    let slope_cols: Vec<usize> = spec
        .cluster_slope_design_cols
        .iter()
        .map(|&c| c as usize)
        .collect();
    let model = cluster_to_model_spec(cluster, spec.estimator, spec.wald_se);
    Some(Box::new(glmm::mcpower::LmmWorkspace::for_cluster_spec(
        n_predictors,
        &model,
        max_n,
        &slope_cols,
    )))
}

/// Dispatch filter: Some iff Glm + cluster present. Mirrors `build_lmm_workspace`.
pub fn build_glmm_workspace(
    spec: &SimulationSpec,
    max_n: usize,
    n_predictors: usize,
) -> Option<Box<glmm::mcpower::GlmmWorkspace>> {
    let cluster = spec.cluster.as_ref()?;
    if spec.estimator != engine_contract::EstimatorSpec::Glm {
        return None;
    }
    let slope_cols: Vec<usize> = spec
        .cluster_slope_design_cols
        .iter()
        .map(|&c| c as usize)
        .collect();
    // `test_formula` reduced fit: the FIXED design fits only `spec.fit_columns`
    // (ascending kernel cols, intercept always present). Size the β-dimension `p`
    // to the reduced count so the joint [θ|β] BOBYQA and every p-sized inference
    // scratch match. The RE design Z and θ-truth stay FULL — `slope_cols` index
    // the full generation design, and a random slope may reference a fixed term
    // the test formula drops. Empty / covers-all ⇒ full (current behaviour). The
    // batch GLMM branch gathers the matching reduced X/β/targets per (sim, N).
    let p_fit = if !spec.fit_columns.is_empty() && spec.fit_columns.len() < n_predictors {
        spec.fit_columns.len()
    } else {
        n_predictors
    };
    let model = cluster_to_model_spec(cluster, spec.estimator, spec.wald_se);
    Some(Box::new(glmm::mcpower::GlmmWorkspace::for_cluster_spec(
        p_fit,
        &model,
        max_n,
        &slope_cols,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cluster_to_model_spec` must reproduce every field the `glmm` kernels read
    /// — sizing, tau_squared, the full slope vector (column/variance/corrs), and
    /// the extra groupings (relation/tau/slopes) — plus the estimator/wald_se the
    /// friendly `fit` dispatch reads. Pins the cross-crate conversion the carve
    /// relies on (nothing else enforces it).
    #[test]
    fn cluster_to_model_spec_round_trips_every_field() {
        let c = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 30 },
            tau_squared: 0.5,
            slopes: vec![
                engine_contract::SlopeTerm {
                    column: engine_contract::ColumnId(1),
                    variance: 0.2,
                    corr_with_intercept: 0.1,
                    corr_with: vec![],
                },
                engine_contract::SlopeTerm {
                    column: engine_contract::ColumnId(2),
                    variance: 0.3,
                    corr_with_intercept: -0.2,
                    corr_with: vec![0.05],
                },
            ],
            extra_groupings: vec![
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::Crossed { n_clusters: 12 },
                    tau_squared: 0.4,
                    slopes: vec![],
                },
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::NestedWithin { n_per_parent: 4 },
                    tau_squared: 0.6,
                    slopes: vec![engine_contract::SlopeTerm {
                        column: engine_contract::ColumnId(3),
                        variance: 0.7,
                        corr_with_intercept: 0.0,
                        corr_with: vec![],
                    }],
                },
            ],
        };
        let m = cluster_to_model_spec(
            &c,
            engine_contract::EstimatorSpec::Mle,
            engine_contract::WaldSe::Rx,
        );

        let expected = glmm::ModelSpec {
            sizing: glmm::Sizing::FixedClusters { n_clusters: 30 },
            tau_squared: 0.5,
            slopes: vec![
                glmm::SlopeTerm {
                    column: 1,
                    variance: 0.2,
                    corr_with_intercept: 0.1,
                    corr_with: vec![],
                },
                glmm::SlopeTerm {
                    column: 2,
                    variance: 0.3,
                    corr_with_intercept: -0.2,
                    corr_with: vec![0.05],
                },
            ],
            extra_groupings: vec![
                glmm::Grouping {
                    relation: glmm::GroupingRelation::Crossed { n_clusters: 12 },
                    tau_squared: 0.4,
                    slopes: vec![],
                },
                glmm::Grouping {
                    relation: glmm::GroupingRelation::NestedWithin { n_per_parent: 4 },
                    tau_squared: 0.6,
                    slopes: vec![glmm::SlopeTerm {
                        column: 3,
                        variance: 0.7,
                        corr_with_intercept: 0.0,
                        corr_with: vec![],
                    }],
                },
            ],
            estimator: glmm::Estimator::Mle,
            wald_se: glmm::WaldSe::Rx,
        };
        assert_eq!(m, expected);
    }
}
