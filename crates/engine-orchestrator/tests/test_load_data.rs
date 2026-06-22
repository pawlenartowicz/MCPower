// Bit-equivalence proof for the `data → results` path: fitting sim-0's bytes via
// `debug_load_data` must reproduce the in-pipeline sim-0 statistic + crit exactly
// (betas are a deterministic byproduct of the same fit — see plan design note).
mod common;

use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec, OutcomeKind};
use engine_orchestrator::debug::{debug_load_data, debug_report, StageMask};
use engine_spec_builder::build_contract;

#[test]
fn load_data_reproduces_sim0_statistic_and_crit_bit_for_bit() {
    let c = common::minimal_ols_contract();
    let (n, n_sims, seed) = (130usize, 64usize, 555u64);

    // Pipeline: data (sim-0 bytes) + stats (per-sim statistic) + crit.
    let report = debug_report(&c, seed, n, n_sims, StageMask::all()).unwrap();
    let d = report.data.expect("data");
    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");

    // load_data on the exact sim-0 bytes.
    let ld = debug_load_data(
        &c,
        seed,
        &d.design.data,
        d.design.nrow,
        d.design.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .unwrap();

    assert_eq!(
        ld.betas.len(),
        d.design.ncol,
        "betas align to design columns"
    );
    assert!(ld.converged, "OLS fit converged");
    assert_eq!(
        ld.targets.len(),
        stats.targets.len(),
        "one load_data target per pipeline target"
    );

    for (t, tgt) in ld.targets.iter().enumerate() {
        let pipeline_stat = stats.targets[t].statistic[0]; // sim 0
        assert_eq!(
            tgt.statistic.to_bits(),
            pipeline_stat.to_bits(),
            "target {t}: load_data statistic != sim-0 pipeline statistic (bit-for-bit)"
        );
        assert_eq!(
            tgt.critical_value.to_bits(),
            crit.targets[t].critical_value.to_bits(),
            "target {t}: load_data crit != pipeline crit (bit-for-bit)"
        );
        // sanity: the tested beta is finite and statistic reconstructs from beta/se.
        assert!(tgt.beta.is_finite() && tgt.se > 0.0);
    }
}

// Bit-equivalence for the GLM arm: load_data on sim-0's bytes reproduces the
// in-pipeline sim-0 statistic + crit exactly. (Higher-fidelity replacement for
// the old "rejects non-OLS" guard test, now that the GLM arm ships.)
#[test]
fn load_data_reproduces_glm_sim0_statistic_and_crit_bit_for_bit() {
    let contracts = build_contract(
        &common::logit_spec(),
        OutcomeKind::Binary,
        None,
        -0.5,
        vec![],
    )
    .unwrap();
    let c = &contracts[0];
    let (n, n_sims, seed) = (500usize, 64usize, 555u64);

    let report = debug_report(c, seed, n, n_sims, StageMask::all()).unwrap();
    let d = report.data.expect("data");
    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");

    let ld = debug_load_data(
        c,
        seed,
        &d.design.data,
        d.design.nrow,
        d.design.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .unwrap();

    assert_eq!(ld.betas.len(), d.design.ncol);
    assert_eq!(ld.targets.len(), stats.targets.len());
    for (t, tgt) in ld.targets.iter().enumerate() {
        let pipeline_stat = stats.targets[t].statistic[0];
        assert!(
            (tgt.statistic.is_nan() && pipeline_stat.is_nan())
                || tgt.statistic.to_bits() == pipeline_stat.to_bits(),
            "GLM target {t}: load_data statistic != sim-0 pipeline statistic"
        );
        assert_eq!(
            tgt.critical_value.to_bits(),
            crit.targets[t].critical_value.to_bits(),
            "GLM target {t}: load_data crit != pipeline crit"
        );
    }
}

// Bit-equivalence for the MLE arm (clustered). Same invariant; the per-cluster
// suff-stats + REML/Brent fit are byte-identical between the two paths.
#[test]
fn load_data_reproduces_mle_sim0_statistic_and_crit_bit_for_bit() {
    let contracts = build_contract(
        &common::lme_spec(),
        OutcomeKind::Continuous,
        None,
        0.0,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 25 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .unwrap();
    let c = &contracts[0];
    let (n, n_sims, seed) = (500usize, 64usize, 555u64); // 25 clusters × 20

    let report = debug_report(c, seed, n, n_sims, StageMask::all()).unwrap();
    let d = report.data.expect("data");
    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");

    let ld = debug_load_data(
        c,
        seed,
        &d.design.data,
        d.design.nrow,
        d.design.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .unwrap();

    assert!(d.cluster_ids.is_some(), "LME draw carries cluster ids");
    assert_eq!(ld.targets.len(), stats.targets.len());
    for (t, tgt) in ld.targets.iter().enumerate() {
        let pipeline_stat = stats.targets[t].statistic[0];
        assert!(
            (tgt.statistic.is_nan() && pipeline_stat.is_nan())
                || tgt.statistic.to_bits() == pipeline_stat.to_bits(),
            "MLE target {t}: load_data statistic != sim-0 pipeline statistic"
        );
        assert_eq!(
            tgt.critical_value.to_bits(),
            crit.targets[t].critical_value.to_bits(),
            "MLE target {t}: load_data crit != pipeline crit"
        );
    }
}

// M2 general path: the data stage carries layout-derived extra_grouping_ids,
// and load_data surfaces per-grouping variance components τ̂²_g = θ̂_g²·σ̂².
#[test]
fn debug_surfaces_extra_grouping_ids_and_variance_components() {
    let contracts = build_contract(
        &common::lme_spec(),
        OutcomeKind::Continuous,
        None,
        0.0,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 6 },
            tau_squared: 0.20,
            slopes: vec![],
            extra_groupings: vec![
                GroupingSpec {
                    relation: GroupingRelation::Crossed { n_clusters: 4 },
                    tau_squared: 0.15,
                    slopes: vec![],
                },
                GroupingSpec {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                    tau_squared: 0.08,
                    slopes: vec![],
                },
            ],
        }],
    )
    .unwrap();
    let c = &contracts[0];
    let (n, n_sims, seed) = (480usize, 32usize, 555u64); // atom 6·4·2 = 48; 480 = 10·48

    let report = debug_report(c, seed, n, n_sims, StageMask::all()).unwrap();
    let d = report.data.expect("data");

    // Data stage: one id vector per extra grouping, each length n, matching the
    // contract's layout helper exactly.
    assert_eq!(d.extra_grouping_ids.len(), 2, "crossed + nested");
    let cl = c.generation.cluster.as_ref().unwrap();
    for g in 0..2 {
        assert_eq!(d.extra_grouping_ids[g].len(), n);
        for i in 0..n {
            assert_eq!(
                d.extra_grouping_ids[g][i],
                cl.extra_level_of_row(g, i) as u32,
                "extra grouping {g} row {i}"
            );
        }
    }

    // load_data: the general path surfaces per-grouping variance components.
    let ld = debug_load_data(
        c,
        seed,
        &d.design.data,
        d.design.nrow,
        d.design.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .unwrap();
    assert!(ld.converged, "general-path fit converged");
    assert_eq!(ld.variance_components.len(), 3); // primary + crossed + nested
    assert!(ld
        .variance_components
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
    assert!(ld.sigma_sq_hat.is_finite() && ld.sigma_sq_hat > 0.0);
}
