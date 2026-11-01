//! `df_method` end-to-end gate: `BetweenWithin`, `Satterthwaite` and `Auto`
//! against `Asymptotic`.
//!
//! Two levels:
//! - the debug report surfaces the per-target denominator df and statistic
//!   kind (`T` vs `Z`) the crit table actually resolved; a design tested at
//!   per-draw Satterthwaite df has no df before the fit, so its design-time
//!   report is `Z` with empty df;
//! - the entry points (`find_power` / `single_core_find_power`) route the
//!   wider Student-t/F threshold into the decision, which can only lower
//!   power relative to the asymptotic Wald-z reference, and `Auto` runs the
//!   between-within test without a random slope and the Satterthwaite test
//!   with one.

mod common;
use common::{lme_spec, logit_spec, minimal_ols_contract};

use engine_contract::{ClusterSizing, ClusterSpec, DfMethod, OutcomeKind, SimulationContract};
use engine_orchestrator::debug::{StageMask, debug_report};
use engine_orchestrator::{CancellationToken, find_power, single_core_find_power};
use engine_spec_builder::build_contract;

const N: usize = 120;
const N_CLUSTERS: usize = 6;

fn cluster(tau_squared: f64) -> ClusterSpec {
    ClusterSpec {
        sizing: ClusterSizing::FixedClusters {
            n_clusters: N_CLUSTERS as u32,
        },
        tau_squared,
        slopes: vec![],
        extra_groupings: vec![],
    }
}

/// Continuous outcome + one random intercept ⇒ `Mle`. `lme_spec` has two
/// continuous predictors (`x1`, `x2`), neither cluster-level, so `p_between`
/// is the intercept alone.
fn bw_mle_contract() -> SimulationContract {
    let mut contracts = build_contract(
        &lme_spec(),
        OutcomeKind::Continuous,
        None,
        None,
        0.0,
        vec![cluster(0.3)],
    )
    .expect("build_contract mle");
    let mut c = contracts.remove(0);
    c.df_method = DfMethod::BetweenWithin;
    c
}

/// `bw_mle_contract`'s design with a random slope on `x1` (column 0), under
/// `df_method` = `d`.
fn slope_mle_contract(d: DfMethod) -> SimulationContract {
    let mut c = bw_mle_contract();
    c.generation.cluster.as_mut().unwrap().slopes = vec![engine_contract::SlopeTerm {
        column: engine_contract::ColumnId(0),
        variance: 0.2,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    c.df_method = d;
    c
}

/// Binary outcome + one random intercept ⇒ `Glm` + cluster (GLMM path).
/// `logit_spec` has three continuous predictors + one binary, none
/// cluster-level.
fn bw_glm_cluster_contract() -> SimulationContract {
    let mut contracts = build_contract(
        &logit_spec(),
        OutcomeKind::Binary,
        None,
        None,
        -0.5,
        vec![cluster(0.3)],
    )
    .expect("build_contract glm+cluster");
    let mut c = contracts.remove(0);
    c.df_method = DfMethod::BetweenWithin;
    c
}

// ---------------------------------------------------------------------------
// Deliverable 2 — debug report df / statistic-kind
// ---------------------------------------------------------------------------

/// N=120, p=3 (intercept + x1 + x2), g=6, p_between=1 (intercept only: neither
/// predictor is cluster-level) ⇒ df_resid = 120-3 = 117, df_between = 6-1 = 5,
/// df_within = 117-5 = 112. Target `x1` (term 1, `col_1`) is within-level.
#[test]
fn debug_report_mle_between_within_reports_t_and_hand_computed_df() {
    let c = bw_mle_contract();
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: false,
        stats: true,
        crit: true,
    };
    let report = debug_report(&c, 2137, N, 40, mask).unwrap();
    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");

    assert_eq!(crit.targets.len(), 1);
    let tc = &crit.targets[0];
    assert_eq!(
        tc.df,
        vec![112.0],
        "hand computation: df_resid 117 - df_between 5 = df_within 112"
    );
    assert!(matches!(
        tc.distribution,
        engine_orchestrator::debug::StatisticKind::T
    ));
    assert!(matches!(
        stats.targets[0].statistic_kind,
        engine_orchestrator::debug::StatisticKind::T
    ));
}

/// N=120, p=5 (intercept + x1,x2,x3 + one binary dummy), g=6, p_between=1 ⇒
/// df_resid = 120-5 = 115, df_between = 6-1 = 5, df_within = 115-5 = 110.
#[test]
fn debug_report_glm_cluster_between_within_reports_t_and_hand_computed_df() {
    let c = bw_glm_cluster_contract();
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: false,
        stats: true,
        crit: true,
    };
    let report = debug_report(&c, 2137, N, 40, mask).unwrap();
    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");

    assert_eq!(crit.targets.len(), 1);
    let tc = &crit.targets[0];
    assert_eq!(
        tc.df,
        vec![110.0],
        "hand computation: df_resid 115 - df_between 5 = df_within 110"
    );
    assert!(matches!(
        tc.distribution,
        engine_orchestrator::debug::StatisticKind::T
    ));
    assert!(matches!(
        stats.targets[0].statistic_kind,
        engine_orchestrator::debug::StatisticKind::T
    ));
}

#[test]
fn debug_report_asymptotic_mixed_reports_z_and_empty_df() {
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: false,
        stats: true,
        crit: true,
    };

    for c in [
        {
            let mut c = bw_mle_contract();
            c.df_method = DfMethod::Asymptotic;
            c
        },
        {
            let mut c = bw_glm_cluster_contract();
            c.df_method = DfMethod::Asymptotic;
            c
        },
    ] {
        let report = debug_report(&c, 2137, N, 40, mask).unwrap();
        let stats = report.stats.expect("stats");
        let crit = report.crit.expect("crit");
        let tc = &crit.targets[0];
        assert!(
            tc.df.is_empty(),
            "asymptotic mixed must report no df, got {:?}",
            tc.df
        );
        assert!(matches!(
            tc.distribution,
            engine_orchestrator::debug::StatisticKind::Z
        ));
        assert!(matches!(
            stats.targets[0].statistic_kind,
            engine_orchestrator::debug::StatisticKind::Z
        ));
    }
}

/// A slope LMM under `auto` (and named `satterthwaite`) is tested at each
/// draw's Satterthwaite df, which the design-time crit stage cannot know: it
/// reports the asymptotic stand-in, `Z` with no df, as a crossed design does.
/// Named `between_within` on the same design keeps its table df.
#[test]
fn debug_report_slope_lmm_auto_has_no_design_time_df() {
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: false,
        stats: false,
        crit: true,
    };
    for d in [DfMethod::Auto, DfMethod::Satterthwaite] {
        let report = debug_report(&slope_mle_contract(d), 2137, N, 4, mask).unwrap();
        let tc = &report.crit.expect("crit").targets[0];
        assert!(tc.df.is_empty(), "{d:?}: got {:?}", tc.df);
        assert!(matches!(
            tc.distribution,
            engine_orchestrator::debug::StatisticKind::Z
        ));
    }
    let report = debug_report(
        &slope_mle_contract(DfMethod::BetweenWithin),
        2137,
        N,
        4,
        mask,
    )
    .unwrap();
    assert_eq!(report.crit.expect("crit").targets[0].df, vec![112.0]);
}

// ---------------------------------------------------------------------------
// Deliverable 3 — find_power / single_core_find_power entry points
// ---------------------------------------------------------------------------

const SEED: u64 = 2137;
const N_SIMS: usize = 300;

/// Target power from both dispatch twins; `single_core_find_power` walks the
/// same per-sim seed chain as `find_power` (`run_batch_st` vs `run_batch`), so
/// the two must agree bit-for-bit.
fn twin_power(c: &SimulationContract) -> f64 {
    let cancel = CancellationToken::new();
    let mc = find_power(std::slice::from_ref(c), N, N_SIMS, SEED, None, &cancel)
        .unwrap()
        .scenarios[0]
        .1
        .power_uncorrected[0];
    let sc = single_core_find_power(std::slice::from_ref(c), N, N_SIMS, SEED, None, &cancel)
        .unwrap()
        .scenarios[0]
        .1
        .power_uncorrected[0];
    assert_eq!(mc, sc, "dispatch twins must agree on {:?}", c.df_method);
    mc
}

/// Recorded literals at (seed=2137, N=120, n_sims=300). The between-within
/// literal sits strictly below the asymptotic one: the wider t(112) threshold
/// reached the decision. An unwired `df_method` would make the two equal.
#[test]
fn find_power_between_within_never_exceeds_asymptotic_mle() {
    let bw = bw_mle_contract();
    let mut asymp = bw.clone();
    asymp.df_method = DfMethod::Asymptotic;

    let p_bw = twin_power(&bw);
    let p_asymp = twin_power(&asymp);

    assert_eq!(p_asymp, 0.9166666666666666);
    assert_eq!(p_bw, 0.9133333333333333);
    assert!(
        p_bw < p_asymp,
        "between-within power {p_bw} must sit below asymptotic power {p_asymp}"
    );
}

/// `Auto` without a random slope is the between-within test, bit for bit; the
/// GLMM's `Auto` is too.
#[test]
fn find_power_auto_is_between_within_without_a_slope() {
    for bw in [bw_mle_contract(), bw_glm_cluster_contract()] {
        let mut auto = bw.clone();
        auto.df_method = DfMethod::Auto;
        assert_eq!(twin_power(&auto), twin_power(&bw), "{:?}", bw.estimator);
    }
}

/// With a random slope `Auto` is the Satterthwaite test: equal to named
/// `satterthwaite`, below `between_within` (6 clusters: the slope variable's
/// Satterthwaite df follow the clusters, its between-within df is the
/// within df 112), and no higher than the Wald z. Both dispatch twins agree.
#[test]
fn find_power_auto_is_satterthwaite_with_a_slope() {
    let p_auto = twin_power(&slope_mle_contract(DfMethod::Auto));
    let p_satt = twin_power(&slope_mle_contract(DfMethod::Satterthwaite));
    let p_bw = twin_power(&slope_mle_contract(DfMethod::BetweenWithin));
    let p_z = twin_power(&slope_mle_contract(DfMethod::Asymptotic));
    assert_eq!(p_auto, p_satt);
    assert!(
        p_auto < p_bw && p_bw <= p_z,
        "auto {p_auto}, between-within {p_bw}, asymptotic {p_z}"
    );
}

/// A named test the design does not admit stops the run instead of running
/// another test.
#[test]
fn find_power_refuses_named_test_the_design_does_not_admit() {
    let cancel = CancellationToken::new();
    let mut glmm = bw_glm_cluster_contract();
    glmm.df_method = DfMethod::Satterthwaite;
    let err = find_power(&[glmm], N, 10, SEED, None, &cancel).unwrap_err();
    assert!(err.to_string().contains("'satterthwaite'"), "{err}");
}

/// Clustered-GLM counterpart: t(110) between-within threshold vs Wald z.
#[test]
fn find_power_between_within_never_exceeds_asymptotic_glm_cluster() {
    let bw = bw_glm_cluster_contract();
    let mut asymp = bw.clone();
    asymp.df_method = DfMethod::Asymptotic;

    let p_bw = twin_power(&bw);
    let p_asymp = twin_power(&asymp);

    assert_eq!(p_asymp, 0.31);
    assert_eq!(p_bw, 0.30333333333333334);
    assert!(
        p_bw < p_asymp,
        "between-within power {p_bw} must sit below asymptotic power {p_asymp}"
    );
}

#[test]
fn ols_between_within_is_bit_identical_to_asymptotic() {
    let cancel = CancellationToken::new();
    let mut bw = minimal_ols_contract();
    bw.df_method = DfMethod::BetweenWithin;
    let mut asymp = bw.clone();
    asymp.df_method = DfMethod::Asymptotic;

    let r_bw = find_power(&[bw.clone()], N, N_SIMS, SEED, None, &cancel).unwrap();
    let r_asymp = find_power(&[asymp.clone()], N, N_SIMS, SEED, None, &cancel).unwrap();
    assert_eq!(r_bw.scenarios[0].1, r_asymp.scenarios[0].1);

    let s_bw = single_core_find_power(&[bw], N, N_SIMS, SEED, None, &cancel).unwrap();
    let s_asymp = single_core_find_power(&[asymp], N, N_SIMS, SEED, None, &cancel).unwrap();
    assert_eq!(s_bw.scenarios[0].1, s_asymp.scenarios[0].1);
}
