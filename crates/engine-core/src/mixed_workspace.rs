//! Spec-reading glue between MCPower's `SimulationSpec`/`ClusterSpec` and the
//! `glmm` crate's fit kernels. These dispatch helpers read sim-layer types that
//! stay in engine-core (so they cannot live in `glmm`), build a `glmm::ModelSpec`
//! via the free conversion fn below, and call the relocated workspace
//! constructors `glmm::loop_advanced::{LmmWorkspace,GlmmWorkspace}::for_cluster_spec`.

use crate::spec::{LinkKind, OutcomeKind, SimulationSpec};

/// Map the DGP outcome (kind + optional link override) to the `glmm::Family`
/// the estimator fits: Binary → logit / probit, Count → Poisson log. The
/// GLM/GLMM dispatch only reaches this for Binary/Count (contract invariant 12);
/// Continuous falls through to `Gaussian` defensively. This mirrors the D4
/// contract→Family table used at all three fit sites (GLMM workspace, and the
/// unclustered GLM in `batch.rs`/`introspect.rs`) — change together.
pub(crate) fn glmm_family(outcome_kind: OutcomeKind, link: Option<LinkKind>) -> glmm::Family {
    match (outcome_kind, link) {
        (OutcomeKind::Binary, Some(LinkKind::Probit)) => glmm::Family::Binomial {
            link: glmm::BinomialLink::Probit,
        },
        (OutcomeKind::Binary, _) => glmm::Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        (OutcomeKind::Count, _) => glmm::Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        (OutcomeKind::Continuous, _) => glmm::Family::Gaussian,
    }
}

/// `ClusterSpec` → `glmm::ModelSpec`. A **free fn, not a `From` impl**: both
/// types are foreign to engine-core, so the orphan rule forbids the impl here,
/// and routing it through `engine-contract` would drag `glmm`'s faer/bobyqa/pulp
/// into that lightweight crate.
///
/// `ModelSpec` is structure-only (topology + column indices, never fitted
/// magnitudes): `tau_squared` and the per-slope variance/correlation fields on
/// `engine_contract::SlopeTerm`/`GroupingSpec` stay engine-core-side (they drive
/// data generation, not the fit kernels) and are dropped here. `family` is not
/// derivable from `ClusterSpec` — every call site is always mixed-model, so
/// `re` is always `Some`; the caller passes the `Family` it already knows
/// (`Family::Gaussian` for the LMM builder, `Family::Binomial{Logit}` for GLMM).
pub(crate) fn cluster_to_model_spec(
    c: &engine_contract::ClusterSpec,
    family: glmm::Family,
) -> glmm::ModelSpec {
    glmm::ModelSpec {
        family,
        re: Some(glmm::ReStructure {
            sizing: match c.sizing {
                engine_contract::ClusterSizing::FixedClusters { n_clusters } => {
                    glmm::Sizing::FixedClusters { n_clusters }
                }
                engine_contract::ClusterSizing::FixedSize { cluster_size } => {
                    glmm::Sizing::FixedSize { cluster_size }
                }
            },
            slopes: c.slopes.iter().map(column_id).collect(),
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
                    slopes: g.slopes.iter().map(column_id).collect(),
                })
                .collect(),
        }),
    }
}

/// The DGP-truth warm-start θ (RE Cholesky parameters) in the kernel's
/// column-major vech layout — what `glmm::loop_advanced::fit_lmm`/`fit_glmm`
/// accept as `theta_start`. One block per grouping (primary, then extras in
/// declaration order); each is the column-major lower-triangular vech of
/// `Λ = chol(D)`, where `D = diag(τ)·R·diag(τ)` is that grouping's RE covariance
/// from the contract (`τ₀ = √tau_squared`, `τ_{k+1} = √slopes[k].variance`,
/// `R = re_correlation_matrix`). Data-gen draws unit-variance residuals, so
/// `σ² = 1` and θ is `chol(D)` directly (no σ rescale); GLMM families fix
/// dispersion at 1, so the identical layout seeds both LMM and GLMM. Mirrors
/// data_gen's per-grouping `D → chol_lower` construction — change together.
///
/// This is a scenario ASSUMPTION, not a speed knob: seeding the optimizer at the
/// truth asserts well-behaved estimation. The MLE is start-sensitive (BOBYQA can
/// settle in different local optima), so a truth start and a blind start are not
/// interchangeable — the caller gates it on the scenario's `truth_start`.
///
/// `include_slopes = false` collapses every grouping to intercept-only (`q = 1`,
/// θ block = `τ`): the reduced-p exclusion fit in batch.rs drops all random
/// slopes, so its warm start must match that reduced RE structure.
pub(crate) fn truth_theta(c: &engine_contract::ClusterSpec, include_slopes: bool) -> Vec<f64> {
    let mut theta = Vec::new();
    let (q, r) = c.re_correlation_matrix();
    push_grouping_vech(&mut theta, c.tau_squared, &c.slopes, q, &r, include_slopes);
    for g in &c.extra_groupings {
        let (qg, rg) = g.re_correlation_matrix();
        push_grouping_vech(
            &mut theta,
            g.tau_squared,
            &g.slopes,
            qg,
            &rg,
            include_slopes,
        );
    }
    theta
}

/// Append one grouping's θ block (column-major vech of `chol(D)`) to `theta`.
/// Intercept-only (`include_slopes = false` or no slopes) is the `q = 1` scalar
/// `τ = √tau_squared`. `r` is the `q×q` row-major RE correlation matrix.
fn push_grouping_vech(
    theta: &mut Vec<f64>,
    tau_squared: f64,
    slopes: &[engine_contract::SlopeTerm],
    q: usize,
    r: &[f64],
    include_slopes: bool,
) {
    if !include_slopes || slopes.is_empty() {
        theta.push(tau_squared.max(0.0).sqrt());
        return;
    }
    let mut tau_vec = Vec::with_capacity(q);
    tau_vec.push(tau_squared.max(0.0).sqrt());
    for s in slopes {
        tau_vec.push(s.variance.max(0.0).sqrt());
    }
    let mut dmat = vec![0.0f64; q * q];
    for i in 0..q {
        for j in 0..q {
            dmat[i * q + j] = tau_vec[i] * r[i * q + j] * tau_vec[j];
        }
    }
    let l = crate::data_gen::chol_lower(&dmat, q);
    // Column-major lower-triangular vech: matches primary_lambda's unpack.
    for col in 0..q {
        for row in col..q {
            theta.push(l[row * q + col]);
        }
    }
}

/// `SlopeTerm` → `glmm::ColumnId`. `ColumnId` is a newtype `ColumnId(pub u32)`;
/// `glmm::ColumnId = u32` — read `.0`, do NOT `as`-cast.
fn column_id(s: &engine_contract::SlopeTerm) -> glmm::ColumnId {
    s.column.0
}

/// `engine_contract::WaldSe` → `glmm::WaldSe`. Shared by the `fit_glmm` call
/// sites (batch.rs / introspect.rs), which pass `spec.wald_se` to the
/// `glmm::WaldSe`-typed kernel parameter.
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
) -> Option<Box<glmm::loop_advanced::LmmWorkspace>> {
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
    let extra_slope_cols: Vec<Vec<usize>> = spec
        .extra_slope_cols
        .iter()
        .map(|v| v.iter().map(|&c| c as usize).collect())
        .collect();
    let model = cluster_to_model_spec(cluster, glmm::Family::Gaussian);
    Some(Box::new(
        glmm::loop_advanced::LmmWorkspace::for_cluster_spec_ext(
            n_predictors,
            &model,
            max_n,
            &slope_cols,
            &extra_slope_cols,
        ),
    ))
}

/// Dispatch filter: Some iff Glm + cluster present. Mirrors `build_lmm_workspace`.
pub fn build_glmm_workspace(
    spec: &SimulationSpec,
    max_n: usize,
    n_predictors: usize,
) -> Option<Box<glmm::loop_advanced::GlmmWorkspace>> {
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
    // scratch match. The RE design Z stays FULL — `slope_cols` index the full
    // generation design, and a random slope may reference a fixed term the test
    // formula drops. Empty / covers-all ⇒ full (current behaviour). The batch
    // GLMM branch gathers the matching reduced X/β/targets per (sim, N).
    let p_fit = if !spec.fit_columns.is_empty() && spec.fit_columns.len() < n_predictors {
        spec.fit_columns.len()
    } else {
        n_predictors
    };
    let model = cluster_to_model_spec(cluster, glmm_family(spec.outcome_kind, spec.link));
    // nagq threaded from the contract (1 = Laplace default). The eligibility
    // backstop (Binary/Count GLMM, single grouping, ≤ 3 REs, odd k ≤ 25) ran in
    // validate() — mirrors glmm's `fit.rs::assert_model_shape`, change together.
    Some(Box::new(
        glmm::loop_advanced::GlmmWorkspace::for_cluster_spec(
            p_fit,
            &model,
            max_n,
            &slope_cols,
            spec.nagq,
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cluster_to_model_spec` must reproduce every structural field the `glmm`
    /// kernels read — sizing, slope column ids, and the extra groupings
    /// (relation/slope column ids) — plus thread the caller-supplied `family`
    /// straight through. Pins the cross-crate conversion the carve relies on
    /// (nothing else enforces it).
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
        let m = cluster_to_model_spec(&c, glmm::Family::Gaussian);

        let expected = glmm::ModelSpec {
            family: glmm::Family::Gaussian,
            re: Some(glmm::ReStructure {
                sizing: glmm::Sizing::FixedClusters { n_clusters: 30 },
                slopes: vec![1, 2],
                extra_groupings: vec![
                    glmm::Grouping {
                        relation: glmm::GroupingRelation::Crossed { n_clusters: 12 },
                        slopes: vec![],
                    },
                    glmm::Grouping {
                        relation: glmm::GroupingRelation::NestedWithin { n_per_parent: 4 },
                        slopes: vec![3],
                    },
                ],
            }),
        };
        assert_eq!(m, expected);
    }

    fn approx(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "θ length: {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "θ entry: {a:?} vs {b:?}");
        }
    }

    /// Intercept-only primary + intercept-only extra: one scalar τ = √tau_squared
    /// per grouping, primary first. No slopes ⇒ `include_slopes` is inert.
    #[test]
    fn truth_theta_intercept_only_is_sqrt_tau_per_grouping() {
        let c = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 10 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![engine_contract::GroupingSpec {
                relation: engine_contract::GroupingRelation::Crossed { n_clusters: 8 },
                tau_squared: 0.09,
                slopes: vec![],
            }],
        };
        approx(&truth_theta(&c, true), &[0.5, 0.3]);
        // No slopes anywhere, so the include_slopes flag changes nothing.
        approx(&truth_theta(&c, false), &[0.5, 0.3]);
    }

    /// One random slope: θ is the column-major lower-tri vech of Λ = chol(D).
    /// Chosen so D = ΛΛ′ has a known factor: Λ = [[2,0],[0.5,1]] (vech [2,0.5,1])
    /// ⇒ D = [[4,1],[1,1.25]], i.e. τ₀² = 4, τ₁² = 1.25, ρ = 1/(2·√1.25).
    #[test]
    fn truth_theta_with_slope_is_column_major_vech_of_chol() {
        let rho = 1.0 / (2.0 * 1.25_f64.sqrt());
        let c = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 10 },
            tau_squared: 4.0,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(1),
                variance: 1.25,
                corr_with_intercept: rho,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        approx(&truth_theta(&c, true), &[2.0, 0.5, 1.0]);
        // The exclusion fit drops the slope (q = 1): θ collapses to just √τ₀².
        approx(&truth_theta(&c, false), &[2.0]);
    }
}
