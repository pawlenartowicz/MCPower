//! Engine throughput bench — frozen fits/sec baseline over a 26-case grid.
//!
//! Run from the workspace root (`mcpower/`):
//!
//! ```text
//! cargo run --release --bin throughput -p engine-core              # run grid, compare vs baseline
//! cargo run --release --bin throughput -p engine-core -- --save    # overwrite the baseline
//! cargo run --release --bin throughput -p engine-core -- --case glm_rare   # iterate on one kernel
//! cargo run --release --bin throughput -p engine-core -- --smoke   # n_sims=4, no timing/compare
//! cargo run --release --bin throughput -p engine-core -- --case glm_rare --mode on  # one mode only (profiling)
//! cargo run --release --bin throughput -p engine-core -- --dump-cases  # case specs as JSON (wasm bench input)
//! ```
//!
//! Each case runs in two modes, mirroring the two real host call shapes:
//!
//! - `off` — the single optimistic (zero-perturbation) spec: what a default
//!   `scenario_analysis=False` call runs, and the pure-kernel regression signal.
//! - `on` — the three presets (optimistic, realistic, doomer) run back-to-back
//!   and timed as one row, at **half** the row's n_sims: the triple is 3× the
//!   fits, so halving keeps the row affordable. fits/sec stays comparable to
//!   `off` (it is per-fit), so the on−off delta is still the scenario-path cost
//!   as users pay it — only the raw elapsed/n_sims columns differ by mode.
//!
//! Every case carries the `on` preset triple — OLS, plain GLM, clustered GLMM,
//! and the Mle (LME/LMM) rows alike — so the scenario-path cost is benched on
//! every estimator. The presets leave the LME-specific RE knobs (`lme`) None;
//! only the general perturbations (heterogeneity, heteroskedasticity, distribution
//! and residual swaps) vary, which all estimators accept (`invariant_13` permits
//! `scenario.lme = None` for any estimator) — see `has_on_mode`.
//!
//! Specs are FROZEN Rust constructors: a baseline comparison is only valid if
//! the spec is bit-identical across runs, so nothing here reads
//! `benchmark_cases.json` or `configs/scenarios.json` at runtime. The scenario
//! presets below are a hand-frozen copy of the config values as of 2026-06-04;
//! a deliberate preset retune must be re-frozen here as a visible diff.
//!
//! Timing: single-threaded (`run_batch_st`) — the stable regression signal;
//! one discarded warm-up call, then median of 3 reps per row. `--save` writes
//! the baseline JSON to `benchmarks/results/engine.json` (gitignored scratch);
//! later runs print the current/baseline fits/sec ratio per row, with a marker
//! on drops beyond 10%. No thresholds, no exit codes on regressions, no CI —
//! a human reads the table. Run baseline + comparison on a stabilized machine.

use std::collections::BTreeMap;
use std::time::Instant;

use engine_contract::{ColumnId, GroupingRelation, GroupingSpec, SlopeTerm};
use engine_core::batch::run_batch_st;
use engine_core::rng::splitmix64_finalize;
use engine_core::spec::{
    BatchResult, ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution,
    EstimatorSpec, HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations,
    SimulationSpec, WaldSe,
};
use serde::{Deserialize, Serialize};

/// Base seed for every row; per-scenario seeds are derived the way the
/// orchestrator derives them (golden-ratio Weyl pre-mix + SplitMix64
/// finalizer), so power/convergence numbers match what a host call sees.
const BASE_SEED: u64 = 2137;

/// Timed repetitions per row (after one discarded warm-up); the median is kept.
const N_REPS: usize = 3;

/// Baseline path relative to this crate's manifest dir, so the bin works
/// regardless of cwd. Gitignored scratch, same stance as all benchmark results.
const BASELINE_REL: &str = "../../benchmarks/results/engine.json";

// ---------------------------------------------------------------------------
// Frozen scenario presets — hand-frozen copy of the workspace scenario config
// as of 2026-06-04. `heavy_tailed` → T, `skewed` → RightSkewed (the legacy
// alias mapping every host applies). LME knobs stay None: only the general
// perturbations vary, which every estimator (incl. Mle) accepts.
// ---------------------------------------------------------------------------

fn preset(name: &str) -> ScenarioPerturbations {
    let menus = (
        vec![
            Distribution::RightSkewed,
            Distribution::LeftSkewed,
            Distribution::Uniform,
        ],
        vec![ResidualDist::HighKurtosis, ResidualDist::RightSkewed],
    );
    match name {
        // Zero-perturbation baseline; menus populated (hosts always send them)
        // but never consulted at zero probability — still the fast path.
        "optimistic" => ScenarioPerturbations {
            name: "optimistic".into(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: menus.0,
            residual_change_prob: 0.0,
            residual_dists: menus.1,
            residual_df: 10.0,
            sampled_factor_proportions: false,
            lme: None,
        },
        "realistic" => ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.2,
            heteroskedasticity_ratio: 2.0,
            correlation_noise_sd: 0.15,
            distribution_change_prob: 0.5,
            new_distributions: menus.0,
            residual_change_prob: 0.5,
            residual_dists: menus.1,
            residual_df: 8.0,
            sampled_factor_proportions: true,
            lme: None,
        },
        "doomer" => ScenarioPerturbations {
            name: "doomer".into(),
            heterogeneity: 0.4,
            heteroskedasticity_ratio: 4.0,
            correlation_noise_sd: 0.3,
            distribution_change_prob: 0.8,
            new_distributions: menus.0,
            residual_change_prob: 0.8,
            residual_dists: menus.1,
            residual_df: 5.0,
            sampled_factor_proportions: true,
            lme: None,
        },
        other => unreachable!("unknown preset {other}"),
    }
}

/// Call-level seed shared by every scenario, mirroring the orchestrator's
/// `lower_contracts` (scenarios in one call are paired runs on one stream).
fn scenario_seed() -> u64 {
    splitmix64_finalize(BASE_SEED)
}

// ---------------------------------------------------------------------------
// Frozen case grid — 26 cases. The OLS/GLM/LME rows mirror the cross-port
// bench's ids; the M2 (crossed/nested), M3 (random-slope), and M4 (GLMM) rows
// exercise the general-path kernels. Each of the 5 M4 GLMM rows also carries a
// `_hessian` copy (FD-Hessian Wald SE). Specs are the record (effect sizes
// follow the cross-port sketches, frozen here).
// ---------------------------------------------------------------------------

struct Case {
    id: &'static str,
    n: u32,
    n_sims: u32,
    spec: SimulationSpec,
}

impl Case {
    /// Every row carries the `on` preset triple. The presets leave the LME RE
    /// knobs (`lme`) None, so only the general perturbations vary — accepted by
    /// every estimator including Mle (`invariant_13` permits `scenario.lme = None`
    /// for any estimator), verified by the on-mode smoke pass.
    fn has_on_mode(&self) -> bool {
        true
    }
}

/// All-defaults OLS spec: `k` independent standard-normal predictors, zero
/// effects, no factors/interactions, optimistic scenario. Cases override from here.
fn base_spec(k: u32) -> SimulationSpec {
    let k_us = k as usize;
    let mut correlation = vec![0.0; k_us * k_us];
    for i in 0..k_us {
        correlation[i * k_us + i] = 1.0;
    }
    SimulationSpec {
        n_non_factor: k,
        n_factor_dummies: 0,
        correlation,
        var_types: vec![Distribution::Normal; k_us],
        var_pinned: vec![],
        var_params: vec![0.0; k_us],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes: vec![0.0; 1 + k_us],
        target_indices: vec![1, 2],
        contrast_pairs: vec![],
        interactions: vec![],
        correction_method: CorrectionMethod::None,
        crit_values: CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        },
        heteroskedasticity_driver: None,
        residual_dist: ResidualDist::Normal,
        residual_pinned: false,
        outcome_kind: OutcomeKind::Continuous,
        estimator: EstimatorSpec::Ols,
        wald_se: WaldSe::default(),
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.1,
        cluster: None,
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    }
}

/// Logit variant: binary outcome, GLM estimator, `effect_sizes[0]` =
/// logit(baseline_p) (the constant term the kernel consumes), host-default
/// `max_failed_fraction` 0.25.
fn glm_spec(k: u32, baseline_p: f64) -> SimulationSpec {
    let mut s = base_spec(k);
    s.outcome_kind = OutcomeKind::Binary;
    s.estimator = EstimatorSpec::Glm;
    s.intercept = (baseline_p / (1.0 - baseline_p)).ln();
    s.effect_sizes[0] = s.intercept;
    s.max_failed_fraction = 0.25;
    s
}

/// LME variant: continuous outcome, MLE estimator, fixed cluster count with
/// τ² = ICC/(1−ICC) at ICC 0.2 (the host translation), `max_failed_fraction` 0.25.
fn lme_spec(k: u32, n_clusters: u32) -> SimulationSpec {
    let mut s = base_spec(k);
    s.estimator = EstimatorSpec::Mle;
    s.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters },
        tau_squared: 0.25,
        slopes: vec![],
        extra_groupings: vec![],
    });
    s.max_failed_fraction = 0.25;
    s
}

/// GLMM variant: clustered binary outcome — `glm_spec` plus a random intercept,
/// the `Glm + cluster.is_some()` dispatch. `tau_squared` is the **latent-scale**
/// random-intercept variance (logistic residual var = π²/3, so ICC 0.2 →
/// τ² = 0.2/0.8·π²/3 ≈ 0.822). Estimator stays `Glm`, so the engine's scenario
/// gate does not reject these — but the bench still runs them off-only (see
/// `has_on_mode`): the dense-RE GLMM fit is too heavy to also pay the preset triple.
fn glmm_spec(k: u32, baseline_p: f64, n_clusters: u32, tau_squared: f64) -> SimulationSpec {
    let mut s = glm_spec(k, baseline_p);
    s.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters },
        tau_squared,
        slopes: vec![],
        extra_groupings: vec![],
    });
    // Base GLMM rows are pinned to the Rx Schur SE (the fast legacy baseline) so the
    // `<id>` vs `<id>_hessian` pair times the per-fit Hessian tax side-by-side. Without
    // this the base would inherit `WaldSe::default()` = Hessian and the pair collapses.
    s.wald_se = WaldSe::Rx;
    s
}

fn cases() -> Vec<Case> {
    // OLS baseline, p=5 (the historical throughput case): x1=x2=0.18 (~75% power).
    let mut ols_multi = base_spec(5);
    ols_multi.effect_sizes[1] = 0.18;
    ols_multi.effect_sizes[2] = 0.18;

    // p=15 — design-matrix width scaling: x1=x2=0.2.
    let mut ols_wide = base_spec(15);
    ols_wide.effect_sizes[1] = 0.2;
    ols_wide.effect_sizes[2] = 0.2;

    // n=5000 — per-row-dominated regime: x1=x2=0.037 (~75% power).
    let mut ols_large_n = base_spec(3);
    ols_large_n.effect_sizes[1] = 0.037;
    ols_large_n.effect_sizes[2] = 0.037;

    // Factor dummy draw + cont×factor product column:
    // y = x1 + f + x1:f, f 2-level (0.5, 0.5); x1=0.27 (~75% power), f[2]=0.5, x1:f[2]=0.3.
    // Kernel columns: 0 intercept, 1 x1, 2 f-dummy, 3 interaction.
    let mut ols_factor_inter = base_spec(1);
    ols_factor_inter.n_factor_dummies = 1;
    ols_factor_inter.factor_n_levels = vec![2];
    ols_factor_inter.factor_proportions = vec![0.5, 0.5];
    ols_factor_inter.interactions = vec![vec![1, 2]];
    ols_factor_inter.effect_sizes = vec![0.0, 0.27, 0.5, 0.3];
    ols_factor_inter.target_indices = vec![1];

    // IRLS baseline: x1=x2=0.4 at baseline_p 0.3.
    let mut glm_multi = glm_spec(5, 0.3);
    glm_multi.effect_sizes[1] = 0.4;
    glm_multi.effect_sizes[2] = 0.4;

    // IRLS wide design.
    let mut glm_wide = glm_spec(15, 0.3);
    glm_wide.effect_sizes[1] = 0.4;
    glm_wide.effect_sizes[2] = 0.4;

    // IRLS large-n: x1=x2=0.09 (~75% power).
    let mut glm_large_n = glm_spec(3, 0.3);
    glm_large_n.effect_sizes[1] = 0.09;
    glm_large_n.effect_sizes[2] = 0.09;

    // baseline_p=0.05 — IRLS iteration/convergence stress: x1=x2=0.45 (~75% power, first-cut).
    let mut glm_rare = glm_spec(2, 0.05);
    glm_rare.effect_sizes[1] = 0.45;
    glm_rare.effect_sizes[2] = 0.45;

    // Brent + profiled deviance, 20 clusters: x1=x2=0.15.
    let mut lme_multi = lme_spec(5, 20);
    lme_multi.effect_sizes[1] = 0.15;
    lme_multi.effect_sizes[2] = 0.15;

    // 100 clusters — random-effect dimension scaling: x1=x2=0.07.
    let mut lme_many_clusters = lme_spec(3, 100);
    lme_many_clusters.effect_sizes[1] = 0.07;
    lme_many_clusters.effect_sizes[2] = 0.07;

    // --- M2 general-path rows (lmm dispatch) + the matched shipped baseline.
    // All Ns are atom multiples. The shipped comparator matches p and the
    // primary count so the general/shipped ratio isolates the lmm machinery.
    let lme_matched_q1 = {
        let mut s = lme_spec(4, 20);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        s
    };
    let lmm_crossed = {
        let mut s = lme_spec(4, 20);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        s.cluster
            .as_mut()
            .unwrap()
            .extra_groupings
            .push(GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 12 },
                tau_squared: 0.15,
                slopes: vec![],
            });
        s
    };
    let lmm_nested = {
        let mut s = lme_spec(4, 20);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        s.cluster
            .as_mut()
            .unwrap()
            .extra_groupings
            .push(GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 4 },
                tau_squared: 0.10,
                slopes: vec![],
            });
        s
    };
    let lmm_crossed_nested = {
        let mut s = lme_spec(4, 10);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        let c = s.cluster.as_mut().unwrap();
        c.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 6 },
            tau_squared: 0.15,
            slopes: vec![],
        });
        c.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.08,
            slopes: vec![],
        });
        s
    };

    // --- M3 random-slope rows (lmm slope path). Same lme_spec(4,20) shape as
    // the M2 matched row, plus a random slope on x1 (and x2 for multi-slope);
    // cluster_slope_design_cols maps slope predictor col k → x_full index k+1.
    let lmm_slope = {
        let mut s = lme_spec(4, 20);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        s.cluster.as_mut().unwrap().slopes.push(SlopeTerm {
            column: ColumnId(0),
            variance: 0.10,
            corr_with_intercept: 0.3,
            corr_with: vec![],
        });
        s.cluster_slope_design_cols = vec![1];
        s
    };
    let lmm_multislope = {
        let mut s = lme_spec(4, 20);
        s.effect_sizes[1] = 0.15;
        s.effect_sizes[2] = 0.15;
        let slopes = &mut s.cluster.as_mut().unwrap().slopes;
        slopes.push(SlopeTerm {
            column: ColumnId(0),
            variance: 0.10,
            corr_with_intercept: 0.3,
            corr_with: vec![],
        });
        slopes.push(SlopeTerm {
            column: ColumnId(1),
            variance: 0.08,
            corr_with_intercept: 0.1,
            corr_with: vec![0.2],
        });
        s.cluster_slope_design_cols = vec![1, 2];
        s
    };

    // --- M4 GLMM rows (Glm + cluster). Latent τ² ≈ 0.822 (ICC 0.2), x1=0.28
    // (intercept, ~75% power) / 0.5 (slope), baseline_p 0.3 — the losf-23/24
    // validation shapes. k=1 ⇒ target is [1].
    // n_clusters is kept SMALL (8): the dense-RE Laplace fit is ~O(clusters²·n)
    // per PIRLS step (glmm.rs), so a realistic count makes these rows dominate
    // the whole bench. Throughput-probe sizing — not a power scenario.
    let mut glmm_intercept = glmm_spec(1, 0.3, 8, 0.822);
    glmm_intercept.effect_sizes[1] = 0.28;
    glmm_intercept.target_indices = vec![1];
    let glmm_slope = {
        let mut s = glmm_spec(1, 0.3, 8, 0.822);
        s.effect_sizes[1] = 0.5;
        s.target_indices = vec![1];
        s.cluster.as_mut().unwrap().slopes.push(SlopeTerm {
            column: ColumnId(0),
            variance: 0.10,
            corr_with_intercept: 0.3,
            corr_with: vec![],
        });
        s.cluster_slope_design_cols = vec![1];
        s
    };

    // --- M4 GLMM crossed/nested rows — the GLMM mirror of the M2 lmm rows.
    // Non-empty `extra_groupings` routes these through the DENSE PIRLS path
    // (`pirls_solve`, k = total RE dims), which the blocked intercept/slope rows
    // never touch — these are the only grid rows profiling that kernel. Extra
    // τ² mirror the lmm rows on the latent scale (×π²/3): 0.15→0.49 crossed,
    // 0.10→0.33 nested, 0.08→0.26 second extra. Primary stays at 8 clusters so
    // dense-vs-blocked is comparable across the M4 rows; n_sims is small —
    // the dense fit is O(n·k²) per PIRLS pass (k=14/24/30 here). No multislope
    // row: too slow for the grid at current dense-path cost.
    let glmm_crossed = {
        let mut s = glmm_spec(1, 0.3, 8, 0.822);
        s.effect_sizes[1] = 0.30;
        s.target_indices = vec![1];
        s.cluster
            .as_mut()
            .unwrap()
            .extra_groupings
            .push(GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 6 },
                tau_squared: 0.49,
                slopes: vec![],
            });
        s
    };
    let glmm_nested = {
        let mut s = glmm_spec(1, 0.3, 8, 0.822);
        s.effect_sizes[1] = 0.30;
        s.target_indices = vec![1];
        s.cluster
            .as_mut()
            .unwrap()
            .extra_groupings
            .push(GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                tau_squared: 0.33,
                slopes: vec![],
            });
        s
    };
    let glmm_crossed_nested = {
        let mut s = glmm_spec(1, 0.3, 8, 0.822);
        s.effect_sizes[1] = 0.30;
        s.target_indices = vec![1];
        let c = s.cluster.as_mut().unwrap();
        c.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 6 },
            tau_squared: 0.49,
            slopes: vec![],
        });
        c.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.26,
            slopes: vec![],
        });
        s
    };

    // --- M4 GLMM hessian copies — every GLMM shape above, re-run with the
    // FD-Hessian Wald SE (the engine default) instead of the base rows' pinned Rx
    // Schur SE, so the grid times the per-fit Hessian tax side-by-side (`<id>` vs
    // `<id>_hessian`). Same n/n_sims as their originals (true copies); fits/sec is
    // per-fit so the ratio reads straight off the table.
    let with_hessian = |s: &SimulationSpec| {
        let mut s = s.clone();
        s.wald_se = WaldSe::Hessian;
        s
    };
    let glmm_intercept_hessian = with_hessian(&glmm_intercept);
    let glmm_slope_hessian = with_hessian(&glmm_slope);
    let glmm_crossed_hessian = with_hessian(&glmm_crossed);
    let glmm_nested_hessian = with_hessian(&glmm_nested);
    let glmm_crossed_nested_hessian = with_hessian(&glmm_crossed_nested);

    vec![
        Case {
            id: "ols_multi",
            n: 200,
            n_sims: 10_000,
            spec: ols_multi,
        },
        Case {
            id: "ols_wide",
            n: 200,
            n_sims: 10_000,
            spec: ols_wide,
        },
        Case {
            id: "ols_large_n",
            n: 5000,
            n_sims: 2_000,
            spec: ols_large_n,
        },
        Case {
            id: "ols_factor_inter",
            n: 200,
            n_sims: 10_000,
            spec: ols_factor_inter,
        },
        Case {
            id: "glm_multi",
            n: 200,
            n_sims: 10_000,
            spec: glm_multi,
        },
        Case {
            id: "glm_wide",
            n: 200,
            n_sims: 5_000,
            spec: glm_wide,
        },
        Case {
            id: "glm_large_n",
            n: 5000,
            n_sims: 1_000,
            spec: glm_large_n,
        },
        Case {
            id: "glm_rare",
            n: 500,
            n_sims: 10_000,
            spec: glm_rare,
        },
        Case {
            id: "lme_multi",
            n: 500,
            n_sims: 2_000,
            spec: lme_multi,
        },
        Case {
            id: "lme_many_clusters",
            n: 1000,
            n_sims: 1_000,
            spec: lme_many_clusters,
        },
        Case {
            id: "lme_matched_q1",
            n: 480,
            n_sims: 1_000,
            spec: lme_matched_q1,
        },
        Case {
            id: "lmm_crossed",
            n: 480,
            n_sims: 1_000,
            spec: lmm_crossed,
        }, // atom 240
        Case {
            id: "lmm_nested",
            n: 480,
            n_sims: 1_000,
            spec: lmm_nested,
        }, // atom 80
        Case {
            id: "lmm_crossed_nested",
            n: 480,
            n_sims: 1_000,
            spec: lmm_crossed_nested,
        }, // atom 120
        Case {
            id: "lmm_slope",
            n: 480,
            n_sims: 1_000,
            spec: lmm_slope,
        }, // M3 single slope
        Case {
            id: "lmm_multislope",
            n: 480,
            n_sims: 1_000,
            spec: lmm_multislope,
        }, // M3 two slopes
        Case {
            id: "glmm_intercept",
            n: 480,
            n_sims: 1_000,
            spec: glmm_intercept,
        }, // M4 GLMM (Glm+cluster), random intercept
        Case {
            id: "glmm_intercept_hessian",
            n: 480,
            n_sims: 1_000,
            spec: glmm_intercept_hessian,
        }, // hessian copy of glmm_intercept
        Case {
            id: "glmm_slope",
            n: 480,
            n_sims: 1_000,
            spec: glmm_slope,
        }, // M4 GLMM + slope, heaviest fit in grid
        Case {
            id: "glmm_slope_hessian",
            n: 480,
            n_sims: 1_000,
            spec: glmm_slope_hessian,
        }, // hessian copy of glmm_slope
        Case {
            id: "glmm_crossed",
            n: 480,
            n_sims: 500,
            spec: glmm_crossed,
        }, // M4 dense path, k=14 (~62 fits/s)
        Case {
            id: "glmm_crossed_hessian",
            n: 480,
            n_sims: 500,
            spec: glmm_crossed_hessian,
        }, // hessian copy of glmm_crossed
        Case {
            id: "glmm_nested",
            n: 480,
            n_sims: 500,
            spec: glmm_nested,
        }, // M4 dense path, k=24 (~27 fits/s)
        Case {
            id: "glmm_nested_hessian",
            n: 480,
            n_sims: 500,
            spec: glmm_nested_hessian,
        }, // hessian copy of glmm_nested
        Case {
            id: "glmm_crossed_nested",
            n: 480,
            n_sims: 500,
            spec: glmm_crossed_nested,
        }, // M4 dense path, k=30 (~13 fits/s)
        Case {
            id: "glmm_crossed_nested_hessian",
            n: 480,
            n_sims: 500,
            spec: glmm_crossed_nested_hessian,
        }, // hessian copy of glmm_crossed_nested
    ]
}

// ---------------------------------------------------------------------------
// Row execution
// ---------------------------------------------------------------------------

/// The pass specs a row runs: one optimistic spec for `off`, the three presets
/// in order for `on`. Shared by `run_row_once` and `--dump-cases`, so the dump
/// can never drift from what the bench actually runs.
fn pass_specs(case: &Case, mode: &str) -> Vec<SimulationSpec> {
    let preset_names: &[&str] = match mode {
        "off" => &["optimistic"],
        "on" => &["optimistic", "realistic", "doomer"],
        other => unreachable!("unknown mode {other}"),
    };
    preset_names
        .iter()
        .map(|name| {
            let mut spec = case.spec.clone();
            spec.scenario = preset(name);
            spec
        })
        .collect()
}

/// One run of a row's full batch: one spec for `off`, the three presets
/// back-to-back for `on`. Returns the per-scenario results (order = preset order).
fn run_row_once(
    case: &Case,
    mode: &str,
    n_sims: u32,
) -> Result<Vec<BatchResult>, engine_core::spec::EngineError> {
    let specs = pass_specs(case, mode);
    let mut results = Vec::with_capacity(specs.len());
    for spec in &specs {
        results.push(run_batch_st(
            spec,
            &[case.n],
            n_sims,
            scenario_seed(),
            None,
        )?);
    }
    Ok(results)
}

/// FNV-1a 64 over `bytes`, continuing from `h`. Mirrors `fnv1a` in
/// `crates/engine-wasm/src/lib.rs` (the dev-only bench entry) — change together.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// FNV-1a 64 offset basis — the initial `hash_state` of every row digest.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// First-target success count of one pass — the integer behind
/// `power = k / n_sims` (single sample size, first target only).
fn first_target_count(batch: &BatchResult) -> u64 {
    let n_sims = batch.shape.n_sims as usize;
    let n_targets = batch.shape.n_targets as usize;
    if n_targets == 0 {
        return 0;
    }
    (0..n_sims)
        .map(|sim| batch.uncorrected[sim * n_targets] as u64)
        .sum()
}

/// Converged-sim count of one pass (single sample size).
fn converged_count(batch: &BatchResult) -> u64 {
    let n_sims = batch.shape.n_sims as usize;
    (0..n_sims).map(|sim| batch.converged[sim] as u64).sum()
}

/// First-target significance rate over all sims (the orchestrator's power
/// definition: successes / n_sims, single sample size).
fn first_target_power(batch: &BatchResult) -> f64 {
    let n_sims = batch.shape.n_sims as usize;
    if n_sims == 0 {
        return 0.0;
    }
    first_target_count(batch) as f64 / n_sims as f64
}

fn convergence_rate(batch: &BatchResult) -> f64 {
    let n_sims = batch.shape.n_sims as usize;
    if n_sims == 0 {
        return 1.0;
    }
    converged_count(batch) as f64 / n_sims as f64
}

// ---------------------------------------------------------------------------
// Baseline JSON — `{meta, records}`, the cross-port results shape.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Baseline {
    meta: Meta,
    records: Vec<Record>,
}

/// `--dump-cases` row: everything a non-Rust runner needs to replay a grid row
/// without re-declaring specs. `seed` is the call-level seed (`scenario_seed()`)
/// stringified — u64 does not survive `JSON.parse` in JS (f64 mantissa), so the
/// runner forwards it as a BigInt. `modes` maps mode → pass specs in preset
/// order ("on" present only when the case has it).
#[derive(Serialize)]
struct CaseDump<'a> {
    id: &'a str,
    n: u32,
    n_sims: u32,
    has_on_mode: bool,
    seed: String,
    modes: BTreeMap<&'static str, Vec<SimulationSpec>>,
}

#[derive(Serialize, Deserialize)]
struct Meta {
    timestamp_utc: String,
    os: String,
    cpu_model: String,
    cores_logical: u32,
    threads_mode: String,
    engine_core_version: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    case_id: String,
    mode: String,
    n: u32,
    n_sims: u32,
    elapsed_s: f64,
    fits_per_sec: f64,
    convergence_rate: f64,
    power_first_target: f64,
    /// Hidden control columns for cross-build sanity gates (not printed in the
    /// table): first-target success count and converged count, each summed
    /// over the row's passes, plus the FNV-1a 64 hex digest folded over every
    /// pass's `uncorrected` then `converged` bitstream in preset order.
    /// serde-defaulted so baselines saved before these fields still load.
    #[serde(default)]
    k_unc: u64,
    #[serde(default)]
    k_conv: u64,
    #[serde(default)]
    sig_hash: String,
    /// BOBYQA objective evals per fit (mean over the row's warm-up sims),
    /// overall and converged-only — `None` for rows with no BOBYQA-backed
    /// fits (OLS/GLM/Brent-LME). Read from `optim_diag` snapshot deltas.
    #[serde(default)]
    evals_per_fit: Option<f64>,
    #[serde(default)]
    evals_per_fit_conv: Option<f64>,
    /// Fraction of BOBYQA-backed fits that pinned a variance component at the
    /// boundary (`boundary_hit == 1`) — the E2 npt-sweep regression signal.
    #[serde(default)]
    pinned_rate: Option<f64>,
}

fn baseline_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(BASELINE_REL)
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn meta() -> Meta {
    Meta {
        timestamp_utc: iso8601_utc_now(),
        os: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        cpu_model: cpu_model(),
        cores_logical: std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(0),
        threads_mode: "1".into(),
        engine_core_version: env!("CARGO_PKG_VERSION").into(),
    }
}

/// `YYYY-MM-DDTHH:MM:SS+00:00` from the system clock, no chrono dependency.
fn iso8601_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's civil_from_days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    save: bool,
    smoke: bool,
    list: bool,
    dump_cases: bool,
    case_filter: Option<String>,
    mode_filter: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        save: false,
        smoke: false,
        list: false,
        dump_cases: false,
        case_filter: None,
        mode_filter: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--save" => args.save = true,
            "--smoke" => args.smoke = true,
            "--list" => args.list = true,
            "--dump-cases" => args.dump_cases = true,
            "--case" => match it.next() {
                Some(id) => args.case_filter = Some(id),
                None => die("--case requires a case id"),
            },
            "--scenarios" => match it.next().as_deref() {
                Some("off") => args.mode_filter = Some("off".into()),
                Some("on") => args.mode_filter = Some("on".into()),
                _ => die("--scenarios requires off or on"),
            },
            other => die(&format!(
                "unknown flag {other}; usage: throughput [--save] [--smoke] [--list] [--dump-cases] [--case <id>] [--scenarios off|on]"
            )),
        }
    }
    if args.dump_cases
        && (args.save
            || args.smoke
            || args.list
            || args.case_filter.is_some()
            || args.mode_filter.is_some())
    {
        die("--dump-cases is exclusive; drop the other flags");
    }
    if args.save && args.case_filter.is_some() {
        die("--save with --case would overwrite the full baseline with a partial run; drop one");
    }
    if args.save && args.mode_filter.is_some() {
        die("--save with --scenarios would overwrite the full baseline with a partial run; drop one");
    }
    if args.save && args.smoke {
        die("--smoke runs are untimed; nothing to save");
    }
    args
}

fn die(msg: &str) -> ! {
    eprintln!("throughput: {msg}");
    std::process::exit(1);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    if cfg!(debug_assertions) {
        die("refusing to run a debug build — use `cargo run --release --bin throughput -p engine-core`");
    }
    let args = parse_args();

    let all = cases();
    // `--list`: case ids one per line — the machine-readable source for tools
    // (scripts/profile.sh) that take the same ids.
    if args.list {
        for c in &all {
            println!("{}", c.id);
        }
        return;
    }
    // `--dump-cases`: the full grid as JSON on stdout — the single-sourced
    // case input for the wasm bench runner (ports/wasm/bench/throughput.mjs).
    // Regenerated at bench time, never committed, so it cannot go stale
    // against the frozen Rust constructors.
    if args.dump_cases {
        let dumps: Vec<CaseDump> = all
            .iter()
            .map(|c| {
                let mut modes = BTreeMap::new();
                modes.insert("off", pass_specs(c, "off"));
                if c.has_on_mode() {
                    modes.insert("on", pass_specs(c, "on"));
                }
                CaseDump {
                    id: c.id,
                    n: c.n,
                    n_sims: c.n_sims,
                    has_on_mode: c.has_on_mode(),
                    seed: scenario_seed().to_string(),
                    modes,
                }
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&dumps).expect("cases serialize")
        );
        return;
    }
    if let Some(id) = &args.case_filter {
        if !all.iter().any(|c| c.id == *id) {
            let ids: Vec<&str> = all.iter().map(|c| c.id).collect();
            die(&format!("unknown case {id:?}; valid: {}", ids.join(", ")));
        }
    }
    let selected: Vec<&Case> = all
        .iter()
        .filter(|c| match &args.case_filter {
            Some(id) => c.id == *id,
            None => true,
        })
        .collect();

    if args.smoke {
        run_smoke(&selected, args.mode_filter.as_deref());
        return;
    }
    run_timed(&selected, args.save, args.mode_filter.as_deref());
}

/// Modes a row runs under the optional `--scenarios` filter: the case's natural
/// modes (`off`+`on` for every row) intersected with the filter.
fn row_modes(case: &Case, mode_filter: Option<&str>) -> Vec<&'static str> {
    let natural: &[&str] = if case.has_on_mode() {
        &["off", "on"]
    } else {
        &["off"]
    };
    natural
        .iter()
        .copied()
        .filter(|m| mode_filter.is_none_or(|f| *m == f))
        .collect()
}

/// `--smoke`: every row at n_sims=4, no timing, no compare — the
/// did-anything-break check after spec-struct changes.
fn run_smoke(selected: &[&Case], mode_filter: Option<&str>) {
    let mut failed = false;
    for case in selected {
        for mode in &row_modes(case, mode_filter) {
            match run_row_once(case, mode, 4) {
                Ok(results) => {
                    let conv = results.iter().map(convergence_rate).fold(1.0, f64::min);
                    println!(
                        "{:<28} {:<4} n={:<5} n_sims=4 conv={conv:.2} ok",
                        case.id, mode, case.n
                    );
                }
                Err(e) => {
                    failed = true;
                    println!("{:<28} {:<4} n={:<5} ERROR: {e}", case.id, mode, case.n);
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

fn run_timed(selected: &[&Case], save: bool, mode_filter: Option<&str>) {
    // Load baseline for comparison (skipped silently if absent/unreadable —
    // the first run has nothing to compare against).
    let baseline: Option<Baseline> = std::fs::read_to_string(baseline_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let baseline_record = |case_id: &str, mode: &str| -> Option<&Record> {
        baseline.as_ref().and_then(|b| {
            b.records
                .iter()
                .find(|r| r.case_id == case_id && r.mode == mode)
        })
    };

    println!(
        "{:<28} {:<9} {:>5} {:>7} {:>10} {:>12} {:>6} {:>7} {:>7} {:>7} {:>6}  vs_baseline",
        "case",
        "scenarios",
        "n",
        "n_sims",
        "elapsed_s",
        "fits_per_sec",
        "conv",
        "power",
        "ev/fit",
        "ev/f_cv",
        "pin"
    );

    let mut records: Vec<Record> = Vec::new();
    for case in selected {
        for mode in &row_modes(case, mode_filter) {
            // The `on` triple runs 3 presets back-to-back, so it runs at half
            // the row's n_sims — fits/sec stays comparable, wall-clock affordable.
            let n_sims = if *mode == "on" {
                case.n_sims / 2
            } else {
                case.n_sims
            };
            // One discarded warm-up call, then median of 3 reps. The eval
            // counters are diffed around the warm-up only — the timed reps
            // re-run the same deterministic fits, so the delta is per-row.
            let ev0 = engine_core::batch::optim_diag::snapshot();
            let results = match run_row_once(case, mode, n_sims) {
                Ok(r) => r,
                Err(e) => die(&format!("{} {mode}: {e}", case.id)),
            };
            let ev1 = engine_core::batch::optim_diag::snapshot();
            let per_fit = |evals: u64, fits: u64| -> Option<f64> {
                (fits > 0).then(|| evals as f64 / fits as f64)
            };
            let evals_per_fit = per_fit(ev1[0] - ev0[0], ev1[1] - ev0[1]);
            let evals_per_fit_conv = per_fit(ev1[2] - ev0[2], ev1[3] - ev0[3]);
            let pinned_rate = per_fit(ev1[4] - ev0[4], ev1[1] - ev0[1]);
            // Control columns from the (deterministic) warm-up results.
            let k_unc: u64 = results.iter().map(first_target_count).sum();
            let k_conv: u64 = results.iter().map(converged_count).sum();
            let mut h = FNV_OFFSET;
            for r in &results {
                h = fnv1a(h, &r.uncorrected);
                h = fnv1a(h, &r.converged);
            }
            let sig_hash = format!("{h:016x}");
            let mut elapsed: Vec<f64> = (0..N_REPS)
                .map(|_| {
                    let start = Instant::now();
                    if let Err(e) = run_row_once(case, mode, n_sims) {
                        die(&format!("{} {mode}: {e}", case.id));
                    }
                    start.elapsed().as_secs_f64()
                })
                .collect();
            elapsed.sort_by(f64::total_cmp);
            let median = elapsed[N_REPS / 2];

            let n_passes = results.len() as f64; // 1 for off, 3 for on
            let fits = n_passes * n_sims as f64;
            let fits_per_sec = fits / median;
            // Convergence: min across presets (the interesting one for glm_rare);
            // power: first target of the optimistic pass, eyeball sanity only.
            let conv = results.iter().map(convergence_rate).fold(1.0, f64::min);
            let power = first_target_power(&results[0]);

            let vs = match baseline_record(case.id, mode) {
                Some(base) if base.fits_per_sec > 0.0 => {
                    let ratio = fits_per_sec / base.fits_per_sec;
                    // Drops beyond 10% get a marker for scanability.
                    let mut marker = if ratio < 0.90 { "  <<< REGRESSION" } else { "" }.to_string();
                    // Result-identity check vs baselines that carry the control
                    // columns (k_unc==k_conv==0 + empty hash = pre-control save).
                    let base_has_sig =
                        base.k_unc != 0 || base.k_conv != 0 || !base.sig_hash.is_empty();
                    if base_has_sig
                        && (base.k_unc != k_unc
                            || base.k_conv != k_conv
                            || base.sig_hash != sig_hash)
                    {
                        marker.push_str("  <<< SIG-DIFF");
                    }
                    format!("{ratio:.2}x{marker}")
                }
                _ => "-".into(),
            };
            let fmt_ev = |v: Option<f64>| match v {
                Some(e) => format!("{e:>7.1}"),
                None => format!("{:>7}", "-"),
            };
            let fmt_pin = match pinned_rate {
                Some(p) => format!("{p:>6.3}"),
                None => format!("{:>6}", "-"),
            };
            println!(
                "{:<28} {:<9} {:>5} {:>7} {:>10.3} {:>12.0} {:>6.3} {:>7.3} {} {} {}  {}",
                case.id,
                mode,
                case.n,
                n_sims,
                median,
                fits_per_sec,
                conv,
                power,
                fmt_ev(evals_per_fit),
                fmt_ev(evals_per_fit_conv),
                fmt_pin,
                vs
            );

            records.push(Record {
                case_id: case.id.into(),
                mode: (*mode).into(),
                n: case.n,
                n_sims,
                elapsed_s: median,
                fits_per_sec,
                convergence_rate: conv,
                power_first_target: power,
                k_unc,
                k_conv,
                sig_hash,
                evals_per_fit,
                evals_per_fit_conv,
                pinned_rate,
            });
        }
    }

    if save {
        let path = baseline_path();
        let out = Baseline {
            meta: meta(),
            records,
        };
        let json = serde_json::to_string_pretty(&out).expect("baseline serializes");
        if let Err(e) = std::fs::write(&path, json) {
            die(&format!("cannot write baseline {}: {e}", path.display()));
        }
        println!("\nbaseline saved to {}", path.display());
    }
}
