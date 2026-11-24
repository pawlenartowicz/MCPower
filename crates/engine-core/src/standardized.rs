//! Population standardized effects of a contract's generating model: the
//! standardized β of every effect, Cohen's d of every binary predictor and
//! factor level, and the model R². Display values only — nothing in a run
//! reads them. Deterministic (no RNG, no simulated data, no fit); computed
//! from the column moments of [`SimulationSpec::column_covariance`], so the
//! accuracy notes of `compute_het_coeffs` apply here too, except for
//! strict-mode (bootstrap) uploads: their columns take the exact empirical
//! moments of the uploaded rows, and an uploaded binary or factor's d comes
//! from the uploaded rows in each group.
//!
//! With β the coefficients (intercept excluded) and Σ the population
//! covariance of the design columns:
//!
//! - `V_η = β′Σβ` (`lp_pop_std²`);
//! - `σ²_ε` = 1 for a continuous outcome, π²/3 for logit, 1 for probit (the
//!   latent residual of the two binary links);
//! - `T` = Σ τ² over every grouping (absolute scale, latent for binary links);
//! - `V_y = V_η + T + σ²_ε`;
//! - standardized `β_j = β_j · SD(x_j) / √V_y`;
//! - `R² = V_η / V_y`: Nakagawa & Schielzeth (2013) marginal R² with clusters,
//!   McKelvey & Zavoina (1975) for the binary links;
//! - Cohen's d of a binary predictor g (a factor level k against the
//!   reference): `d = (m₁ − m₀) / √(w·v₁ + (1 − w)·v₀ + T + σ²_ε)`, with
//!   `m_g`, `v_g` the mean and variance of η within group g and w the share
//!   of group 1 (`p_k / (p_ref + p_k)` for a factor level). Other predictors
//!   are not held fixed, so this is the d a large two-group comparison shows.
//!
//! None of Σ, the SDs, the noise term `T + σ²_ε`, or the group means and
//! covariances depend on β, so [`effect_moments`] returns them alone and a
//! host can evaluate the formulas above for any β without another engine
//! call ([`standardized_effects`] does both steps).
//!
//! Undefined, so `None`: a count outcome (the observation-level variance of a
//! log-link Poisson depends on the mean rate; Nakagawa, Johnson & Schielzeth
//! 2017), and any random slope (the variance of y then depends on x).
//!
//! Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for
//! obtaining R² from generalized linear mixed-effects models. Methods in
//! Ecology and Evolution, 4(2), 133–142.
//! Nakagawa, S., Johnson, P. C. D., & Schielzeth, H. (2017). The coefficient
//! of determination R² and intra-class correlation coefficient from
//! generalized linear mixed-effects models revisited and expanded. Journal of
//! the Royal Society Interface, 14(134), 20170213.
//! McKelvey, R. D., & Zavoina, W. (1975). A statistical model for the
//! analysis of ordinal level dependent variables. Journal of Mathematical
//! Sociology, 4(1), 103–120.

use engine_contract::{ColumnSpec, DesignTerm, LinkKind, OutcomeKind, SimulationContract};
use serde::{Deserialize, Serialize};

use crate::contract_adapter::{
    AdapterError, contract_to_simulation_spec, factor_position_for, term_to_kernel_col,
};
use crate::spec::{ColumnCovariance, FrameMoments, SimulationSpec};

/// Standardized effects of one contract's generating model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandardizedEffects {
    /// One per `design_generation.terms` entry, intercept excluded, in that order.
    pub effects: Vec<StandardizedEffect>,
    /// `V_η / V_y`; `None` where `V_y` is undefined.
    pub r2: Option<f64>,
}

/// Standardized values of one design term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandardizedEffect {
    /// `β · SD(x) / √V_y`; `None` where `V_y` is undefined.
    pub std_beta: Option<f64>,
    /// Some only for a binary predictor or a factor level (against the
    /// reference), and only where `V_y` is defined.
    pub cohen_d: Option<f64>,
}

/// The β-free parts of the standardized values, over the k design terms
/// (`design_generation.terms`, intercept excluded, in that order). With β the
/// k effect sizes in that order (module docs for the definitions):
/// `V_η = β′·cov·β`, `std_beta_j = β_j·sd_j / √(V_η + noise)`,
/// `R² = V_η / (V_η + noise)`, and per group entry
/// `d = β′·mean_diff / √(w·β′·cov_1·β + (1 − w)·β′·cov_0·β + noise)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectMoments {
    /// `T + σ²_ε`.
    pub noise: f64,
    /// SD of each term's design column (k).
    pub sd: Vec<f64>,
    /// Covariance of the terms' design columns, `k × k` row-major.
    pub cov: Vec<f64>,
    /// Per term: the two groups of a binary predictor or factor level; None
    /// for every other term, and where a group is empty.
    pub groups: Vec<Option<GroupMoments>>,
}

/// The two groups behind one Cohen's d: group 1 (the binary at 1, or factor
/// level k) against group 0 (the binary at 0, or the reference level).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupMoments {
    /// Group 1's share of the two groups.
    pub w: f64,
    /// Column means of the k terms in group 1 minus those in group 0.
    pub mean_diff: Vec<f64>,
    /// Covariance of the k terms' columns within group 1, `k × k` row-major.
    pub cov_1: Vec<f64>,
    /// Covariance of the k terms' columns within group 0, `k × k` row-major.
    pub cov_0: Vec<f64>,
}

/// Standardized β, Cohen's d and R² of `c`'s generating model (see the module
/// docs for the definitions). Scenario perturbations play no part.
///
/// # Errors
/// Any error of [`contract_to_simulation_spec`] (an invalid contract).
pub fn standardized_effects(c: &SimulationContract) -> Result<StandardizedEffects, AdapterError> {
    let mut spec = contract_to_simulation_spec(c)?;
    let cols = term_columns(c, &spec);
    Ok(match moments_of(c, &mut spec, &cols) {
        Some(m) => {
            let beta: Vec<f64> = cols.iter().map(|&kc| spec.effect_sizes[kc]).collect();
            evaluate(&m, &beta)
        }
        None => StandardizedEffects {
            effects: vec![
                StandardizedEffect {
                    std_beta: None,
                    cohen_d: None,
                };
                cols.len()
            ],
            r2: None,
        },
    })
}

/// [`EffectMoments`] of `c`'s generating model; `None` where the values are
/// undefined (a count outcome, any random slope).
///
/// # Errors
/// Any error of [`contract_to_simulation_spec`] (an invalid contract).
pub fn effect_moments(c: &SimulationContract) -> Result<Option<EffectMoments>, AdapterError> {
    let mut spec = contract_to_simulation_spec(c)?;
    let cols = term_columns(c, &spec);
    Ok(moments_of(c, &mut spec, &cols))
}

/// Kernel (full-design) column of every design term, intercept excluded.
fn term_columns(c: &SimulationContract, spec: &SimulationSpec) -> Vec<usize> {
    let (n_nf, n_fd) = (spec.n_non_factor, spec.n_factor_dummies);
    c.design_generation
        .terms
        .iter()
        .enumerate()
        .filter(|(_, t)| !matches!(t, DesignTerm::Const))
        .map(|(i, _)| term_to_kernel_col(c, i as u32, n_nf, n_fd, &c.design_generation) as usize)
        .collect()
}

/// Mirrors `evaluateStandardized` in the app's `standardized.svelte.ts`;
/// change together. `beta` has one entry per term of `m`.
fn evaluate(m: &EffectMoments, beta: &[f64]) -> StandardizedEffects {
    let quad = |cov: &[f64]| -> f64 {
        let k = beta.len();
        (0..k)
            .map(|i| beta[i] * (0..k).map(|j| cov[i * k + j] * beta[j]).sum::<f64>())
            .sum()
    };
    let v_eta = quad(&m.cov);
    let sd_y = (v_eta + m.noise).sqrt();
    let effects = beta
        .iter()
        .zip(&m.sd)
        .zip(&m.groups)
        .map(|((b, sd), g)| StandardizedEffect {
            std_beta: Some(b * sd / sd_y),
            cohen_d: g.as_ref().map(|g| {
                let diff: f64 = beta.iter().zip(&g.mean_diff).map(|(b, d)| b * d).sum();
                diff / (g.w * quad(&g.cov_1) + (1.0 - g.w) * quad(&g.cov_0) + m.noise).sqrt()
            }),
        })
        .collect();
    StandardizedEffects {
        effects,
        r2: Some(v_eta / (v_eta + m.noise)),
    }
}

/// [`EffectMoments`] over the design terms at kernel columns `cols`; `spec`
/// is restored before returning.
fn moments_of(
    c: &SimulationContract,
    spec: &mut SimulationSpec,
    cols: &[usize],
) -> Option<EffectMoments> {
    let has_slopes = c.generation.cluster.as_ref().is_some_and(|cl| {
        !cl.slopes.is_empty() || cl.extra_groupings.iter().any(|g| !g.slopes.is_empty())
    });
    if c.outcome.kind == OutcomeKind::Count || has_slopes {
        return None;
    }

    let sigma2_eps = match (c.outcome.kind, c.outcome.link) {
        (OutcomeKind::Binary, None) => std::f64::consts::PI.powi(2) / 3.0,
        (OutcomeKind::Binary, Some(LinkKind::Probit)) | (OutcomeKind::Continuous, _) => 1.0,
        (OutcomeKind::Count, _) => unreachable!("count outcomes returned above"),
    };
    let tau2_total = c.generation.cluster.as_ref().map_or(0.0, |cl| {
        cl.tau_squared
            + cl.extra_groupings
                .iter()
                .map(|g| g.tau_squared)
                .sum::<f64>()
    });

    // Strict mode only; elsewhere no column is frame-copied. The frame
    // moments do not depend on any synthetic column, so the synthetic binary
    // and factor groups below reuse them.
    let frame_all = (!spec.bootstrap_frame_map.is_empty()).then(|| spec.frame_moments());
    let overall = spec.column_covariance(frame_all.as_ref());
    let group = |g1: &ColumnCovariance, g0: &ColumnCovariance, w: f64| GroupMoments {
        w,
        mean_diff: cols
            .iter()
            .map(|&a| g1.col_mean[a] - g0.col_mean[a])
            .collect(),
        cov_1: sub_cov(g1, cols),
        cov_0: sub_cov(g0, cols),
    };
    // Groups of strict-mode frame rows, `(rows, frame moments)` each; None
    // when either group has no rows.
    let frame_group =
        |spec: &SimulationSpec, g1: &(usize, FrameMoments), g0: &(usize, FrameMoments)| {
            let ((n1, f1), (n0, f0)) = (g1, g0);
            (*n1 > 0 && *n0 > 0).then(|| {
                group(
                    &spec.column_covariance(Some(f1)),
                    &spec.column_covariance(Some(f0)),
                    *n1 as f64 / (n1 + n0) as f64,
                )
            })
        };

    let n_nf = spec.n_non_factor as usize;
    // Per uploaded factor: the frame moments within each level, shared by
    // all of its levels.
    let mut level_groups: Vec<Option<Vec<(usize, FrameMoments)>>> =
        vec![None; spec.factor_n_levels.len()];
    let terms = c
        .design_generation
        .terms
        .iter()
        .filter(|t| !matches!(t, DesignTerm::Const));
    let mut groups = Vec::with_capacity(cols.len());
    for (term, &kc) in terms.zip(cols) {
        let g = match term {
            DesignTerm::Direct { column } => match &c.generation.columns[column.0 as usize] {
                ColumnSpec::Synthetic {
                    kind: engine_contract::SyntheticKind::Binary { .. },
                    ..
                }
                | ColumnSpec::ResampledBinary { .. } => {
                    // Non-factor column j of the kernel sits at x_full column j + 1.
                    let j = kc - 1;
                    let p = spec.var_params[j];
                    match spec.bootstrap_frame_map.get(j).copied().flatten() {
                        Some(fc) => {
                            let g = spec.grouped_frame_moments(2, |row| {
                                let v = crate::spec::recover_uploaded_binary(row[fc as usize], p);
                                Some(usize::from(v == 1.0))
                            });
                            frame_group(spec, &g[1], &g[0])
                        }
                        None => {
                            // Fix the binary at 1, then at 0; it is independent
                            // of every other column (the spec builder refuses a
                            // correlation that names a binary), so the others
                            // keep their distribution.
                            spec.var_params[j] = 1.0;
                            let g1 = spec.column_covariance(frame_all.as_ref());
                            spec.var_params[j] = 0.0;
                            let g0 = spec.column_covariance(frame_all.as_ref());
                            spec.var_params[j] = p;
                            Some(group(&g1, &g0, p))
                        }
                    }
                }
                _ => None,
            },
            DesignTerm::DummyOf {
                column,
                level_index,
            } => {
                let f = factor_position_for(c, column.0) as usize;
                match spec.bootstrap_frame_map.get(n_nf + f).copied().flatten() {
                    Some(fc) => {
                        let n_levels = spec.factor_n_levels[f] as f64;
                        if level_groups[f].is_none() {
                            level_groups[f] =
                                Some(spec.grouped_frame_moments(n_levels as usize, |row| {
                                    let code = row[fc as usize].round();
                                    (code >= 0.0 && code < n_levels).then_some(code as usize)
                                }));
                        }
                        let g = level_groups[f].as_ref().expect("filled above");
                        frame_group(spec, &g[*level_index as usize], &g[0])
                    }
                    None => {
                        // Put all of the factor on level k, then on the
                        // reference; factors are drawn independently of every
                        // other column.
                        let off: usize =
                            spec.factor_n_levels[..f].iter().map(|&n| n as usize).sum();
                        let n_levels = spec.factor_n_levels[f] as usize;
                        let k = off + *level_index as usize;
                        let saved: Vec<f64> = spec.factor_proportions[off..off + n_levels].to_vec();
                        let (p_ref, p_k) = (saved[0], saved[k - off]);
                        // A pair with no mass on either level has no groups to compare.
                        (p_ref + p_k > 0.0).then(|| {
                            let mut all_on = |level: usize| {
                                spec.factor_proportions[off..off + n_levels].fill(0.0);
                                spec.factor_proportions[level] = 1.0;
                                spec.column_covariance(frame_all.as_ref())
                            };
                            let g1 = all_on(k);
                            let g0 = all_on(off);
                            spec.factor_proportions[off..off + n_levels].copy_from_slice(&saved);
                            group(&g1, &g0, p_k / (p_ref + p_k))
                        })
                    }
                }
            }
            _ => None,
        };
        groups.push(g);
    }

    Some(EffectMoments {
        noise: tau2_total + sigma2_eps,
        sd: cols.iter().map(|&a| overall.col_std[a]).collect(),
        cov: sub_cov(&overall, cols),
        groups,
    })
}

/// The covariance of columns `cols` from `h`'s full-design matrix, row-major.
fn sub_cov(h: &ColumnCovariance, cols: &[usize]) -> Vec<f64> {
    let p = h.col_mean.len();
    cols.iter()
        .flat_map(|&a| cols.iter().map(move |&b| h.cov[a * p + b]))
        .collect()
}
