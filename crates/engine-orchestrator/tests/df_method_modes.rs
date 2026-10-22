//! `df_method` end-to-end gate: `DfMethod::BetweenWithin` vs `DfMethod::Asymptotic`.
//!
//! Two levels:
//! - the debug report surfaces the per-target denominator df and statistic
//!   kind (`T` vs `Z`) the crit table actually resolved;
//! - the entry points (`find_power` / `single_core_find_power`) route the
//!   wider Student-t/F threshold into the decision, which can only lower
//!   power relative to the asymptotic Wald-z reference.

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
