//! Spec-reading glue between MCPower's `SimulationSpec`/`ClusterSpec` and the
//! `glmm` crate's unified fit core. These dispatch helpers read sim-layer types
//! that stay in engine-core (so they cannot live in `glmm`), build a
//! `glmm::ModelSpec` via the free conversion fn below, and hand it to
//! `glmm::loop_advanced::build_workspace` — which classifies the design once and
//! pins the solver, replacing the per-kernel `LmmWorkspace`/`GlmmWorkspace`
//! constructors this module used to call directly.

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
///
/// Slope columns come from the caller, NOT from `SlopeTerm.column`. glmm reads
/// `ModelSpec.slopes` as **0-based indices into the fit x matrix** (intercept =
/// column 0), so a random slope must name its `x_full` design column. But
/// `SlopeTerm.column` is a GENERATION column index (0 = first continuous
/// predictor), which the contract adapter resolves to the design column via
/// `column_position_for_continuous` and stores in `spec.cluster_slope_design_cols`
/// / `spec.extra_slope_cols`. Passing the raw generation index would make glmm
/// read the wrong x column (off by the intercept, and by any factor
/// interleaving) — turning a random slope into a second random intercept. So the
/// caller threads those precomputed design columns here; `primary_slope_cols`
/// and `extra_slope_cols` must line up 1:1 with `c.slopes` and each
/// `c.extra_groupings[i].slopes`.
pub(crate) fn cluster_to_model_spec(
    c: &engine_contract::ClusterSpec,
    family: glmm::Family,
    primary_slope_cols: &[u32],
    extra_slope_cols: &[Vec<u32>],
) -> glmm::ModelSpec {
    // Empty is the deliberate "slopes are stripped right after" call (batch.rs's
    // reduced/exclusion fit); anything else must match 1:1 or a grouping would
    // silently lose its slopes and fit as a second random intercept.
    debug_assert!(
        primary_slope_cols.is_empty() || primary_slope_cols.len() == c.slopes.len(),
        "primary_slope_cols must line up 1:1 with ClusterSpec.slopes"
    );
    debug_assert!(
        extra_slope_cols.is_empty() || extra_slope_cols.len() == c.extra_groupings.len(),
        "extra_slope_cols must line up 1:1 with ClusterSpec.extra_groupings"
    );
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
            slopes: primary_slope_cols.to_vec(),
            extra_groupings: c
                .extra_groupings
                .iter()
                .enumerate()
                .map(|(i, g)| glmm::Grouping {
                    relation: match g.relation {
                        engine_contract::GroupingRelation::Crossed { n_clusters } => {
                            glmm::GroupingRelation::Crossed { n_clusters }
                        }
                        engine_contract::GroupingRelation::NestedWithin { n_per_parent } => {
                            glmm::GroupingRelation::NestedWithin { n_per_parent }
                        }
                    },
                    slopes: extra_slope_cols.get(i).cloned().unwrap_or_default(),
                })
                .collect(),
        }),
    }
}

/// The DGP-truth warm-start θ (RE Cholesky parameters) in the kernel's
/// column-major vech layout — what `glmm::loop_advanced::fit_on`
/// accepts as `theta_start`. One block per grouping (primary, then extras in
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

/// `engine_contract::WaldSe` → `glmm::WaldSe`. Shared by the `fit_on` call
/// sites (batch.rs / introspect.rs), which pass `spec.wald_se` to the
/// `glmm::WaldSe`-typed kernel parameter.
pub(crate) fn wald_se_to_glmm(wald_se: engine_contract::WaldSe) -> glmm::WaldSe {
    match wald_se {
        engine_contract::WaldSe::Hessian => glmm::WaldSe::Hessian,
        engine_contract::WaldSe::Rx => glmm::WaldSe::Rx,
    }
}

/// The per-grid-point `fit_on` inputs for one mixed-model batch: a workspace and
/// the level ids for every sample-size grid point, plus the frozen options each
/// call must present unchanged. `None` for OLS / unclustered GLM, which never
/// reach `fit_on`.
///
/// **One workspace per grid point, not one per batch.** `fit_on` pins the primary
/// cluster count at build and hard-panics when a draw's count differs. A
/// `FixedClusters` design holds that count constant across the grid, but a
/// `FixedSize` design grows it with N, so a single `n_max` workspace would panic
/// at every smaller-N point. Building per point covers both uniformly — K
/// allocations at batch entry, dwarfed by the `n_sims` inner loop.
///
/// Unlike the retired `build_lmm_workspace`, this does NOT exempt the degenerate
/// single-intercept LMM: the scalar Brent path is gone, so every clustered `Mle`
/// spec now routes through `fit_on` (BOBYQA) here.
pub fn build_mixed_workspaces(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_predictors: usize,
    cluster_ids: &[u32],
    extra_grouping_ids: &[Vec<u32>],
) -> Option<(
    Vec<glmm::loop_advanced::FitWorkspace>,
    Vec<glmm::GroupIds>,
    glmm::FitOptions,
)> {
    let cluster = spec.cluster.as_ref()?;
    let family = match spec.estimator {
        engine_contract::EstimatorSpec::Mle => glmm::Family::Gaussian,
        engine_contract::EstimatorSpec::Glm => glmm_family(spec.outcome_kind, spec.link),
        engine_contract::EstimatorSpec::Ols => return None,
    };
    // `test_formula` reduced fit (GLM only): the FIXED design fits just
    // `spec.fit_columns` (ascending kernel cols, intercept always present), so the
    // β-dimension `p` is the reduced count and the joint [θ|β] BOBYQA plus every
    // p-sized inference scratch match it. The RE design Z stays FULL — a random
    // slope may reference a fixed term the test formula drops. Empty / covers-all
    // ⇒ full. The LMM arm keeps the full width; its reduced/exclusion fits build
    // their own one-shot workspace on the cold path.
    let p = match spec.estimator {
        engine_contract::EstimatorSpec::Glm
            if !spec.fit_columns.is_empty() && spec.fit_columns.len() < n_predictors =>
        {
            spec.fit_columns.len()
        }
        _ => n_predictors,
    };
    // Frozen across the batch: `fit_on` asserts nagq / weights-presence /
    // offset-presence / parallel_inner match the build. `target_indices` is NOT
    // frozen (it only sizes the OLS/GLM scratch, unused by the mixed arms), so the
    // cold reduced fits may vary it on their own workspace. nagq's eligibility
    // backstop ran in validate() — mirrors glmm's `assert_model_shape`, change
    // together.
    let opts = glmm::FitOptions {
        target_indices: spec.target_indices.clone(),
        wald_se: wald_se_to_glmm(spec.wald_se),
        nagq: spec.nagq,
        ..Default::default()
    };
    let model = cluster_to_model_spec(
        cluster,
        family,
        &spec.cluster_slope_design_cols,
        &spec.extra_slope_cols,
    );
    let mut workspaces = Vec::with_capacity(sample_sizes.len());
    let mut ids_per_point = Vec::with_capacity(sample_sizes.len());
    for &n in sample_sizes {
        let n = n as usize;
        // Structural and invariant across sims (`SimWorkspace::new` fills them
        // once), so the ids are materialized here per grid point rather than
        // rebuilt on every draw.
        let ids = glmm::GroupIds {
            primary: cluster_ids[..n].to_vec(),
            extra: extra_grouping_ids.iter().map(|v| v[..n].to_vec()).collect(),
        };
        // `build_workspace` requires a spec whose RE level counts already match
        // the data; reuse glmm's own normalizer instead of reimplementing the
        // crossed/nested count derivation.
        let sized = glmm::loop_advanced::spec_sized_from_ids_pub(&model, &ids);
        workspaces.push(glmm::loop_advanced::build_workspace(&sized, n, p, &opts));
        ids_per_point.push(ids);
    }
    Some((workspaces, ids_per_point, opts))
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
        // Design columns come from the caller (the contract adapter's
        // `column_position_for_continuous` resolution), not from `SlopeTerm.column`.
        // Pick values distinct from the raw `.column.0` to prove they are threaded
        // through verbatim rather than re-read off the ClusterSpec.
        let m = cluster_to_model_spec(&c, glmm::Family::Gaussian, &[4, 5], &[vec![], vec![6]]);

        let expected = glmm::ModelSpec {
            family: glmm::Family::Gaussian,
            re: Some(glmm::ReStructure {
                sizing: glmm::Sizing::FixedClusters { n_clusters: 30 },
                slopes: vec![4, 5],
                extra_groupings: vec![
                    glmm::Grouping {
                        relation: glmm::GroupingRelation::Crossed { n_clusters: 12 },
                        slopes: vec![],
                    },
                    glmm::Grouping {
                        relation: glmm::GroupingRelation::NestedWithin { n_per_parent: 4 },
                        slopes: vec![6],
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
