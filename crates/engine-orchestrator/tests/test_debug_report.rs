mod common;

use engine_orchestrator::debug::{debug_report, StageMask};
use engine_orchestrator::{find_power, CancellationToken};

#[test]
fn invariant_3_stats_reproduce_power() {
    let c = common::minimal_ols_contract();
    let report = debug_report(&c, 777, 115, 100, StageMask::all()).unwrap();
    let stats = report.stats.unwrap();
    let crit = report.crit.unwrap();
    let power = report.power.unwrap();

    for (t, ts) in stats.targets.iter().enumerate() {
        let tc = &crit.targets[t];
        let thresh = tc.critical_value;
        // two_sided + natural statistic: reject iff |stat| > crit. Engine stats
        // are >= 0 (sqrt of squared), so |stat| == stat.
        let recomputed: u64 = ts
            .statistic
            .iter()
            .filter(|&&s| !s.is_nan() && s > thresh)
            .count() as u64;
        assert_eq!(
            recomputed, power.success_counts_uncorrected[t],
            "target {t}: stats→power mismatch"
        );
    }
}

#[test]
fn debug_report_data_stage_matches_design_shape() {
    let c = common::minimal_ols_contract();
    let mask = StageMask {
        input: false,
        data: true,
        dispatch: false,
        stats: false,
        crit: false,
    };
    let report = debug_report(&c, 42, 90, 50, mask).unwrap();
    let d = report.data.expect("data present");
    assert_eq!(d.design.nrow, 90);
    assert_eq!(d.outcome.len(), 90);
    assert_eq!(d.design_columns.len(), d.design.ncol);
    assert_eq!(d.design_columns[0], "intercept");
    // Intercept column (col-major first nrow entries) is all ones.
    assert!(d.design.data[..90].iter().all(|&v| v == 1.0));
}

#[test]
fn debug_report_input_stage_resolves() {
    let c = common::minimal_ols_contract();
    let mask = StageMask {
        input: true,
        data: false,
        dispatch: false,
        stats: false,
        crit: false,
    };
    let report = debug_report(&c, 42, 100, 50, mask).unwrap();
    let inp = report.input.expect("input present");
    assert_eq!(inp.resolved_alpha, c.test.alpha);
    // Effect names are synthesized positional labels.
    assert!(inp
        .effective_effects
        .iter()
        .all(|(name, _)| name.starts_with("col_") || name == "intercept"));
}

#[test]
fn debug_report_dispatch_only_runs_no_sims() {
    let c = common::minimal_ols_contract();
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: true,
        stats: false,
        crit: false,
    };
    let report = debug_report(&c, 42, 100, 50, mask).unwrap();
    assert!(report.dispatch.is_some());
    assert!(report.input.is_none());
    assert!(report.data.is_none());
    assert!(report.stats.is_none());
    assert!(report.crit.is_none());
    assert!(report.power.is_none());
    let d = report.dispatch.unwrap();
    assert!(matches!(
        d.estimator,
        engine_orchestrator::debug::Estimator::Ols
    ));
    assert!(matches!(
        d.route,
        engine_orchestrator::debug::DispatchRoute::Simulated
    ));
}

#[test]
fn debug_report_stats_crit_power_assemble() {
    let c = common::minimal_ols_contract();
    let mask = StageMask {
        input: false,
        data: false,
        dispatch: false,
        stats: true,
        crit: true,
    };
    let report = debug_report(&c, 42, 120, 64, mask).unwrap();

    let stats = report.stats.expect("stats");
    let crit = report.crit.expect("crit");
    let power = report.power.expect("power present iff stats && crit");

    assert_eq!(stats.converged.len(), 64);
    assert_eq!(stats.targets.len(), crit.targets.len());
    for ts in &stats.targets {
        assert_eq!(ts.statistic.len(), 64);
        assert!(matches!(
            ts.statistic_kind,
            engine_orchestrator::debug::StatisticKind::T
        ));
    }
    for tc in &crit.targets {
        assert!(tc.two_sided);
        assert_eq!(tc.df.len(), 1); // OLS: [N - P]
        assert!(tc.critical_value.is_finite());
    }
    assert_eq!(power.n, 120);
    assert_eq!(power.n_sims, 64);
}

#[test]
fn invariant_1_determinism() {
    let c = common::minimal_ols_contract();
    let mask = StageMask::all();
    let a = debug_report(&c, 2024, 110, 80, mask).unwrap();
    let b = debug_report(&c, 2024, 110, 80, mask).unwrap();
    assert_eq!(a, b); // full DebugReport PartialEq, byte-for-byte.
}

#[test]
fn invariant_2_bit_equivalence_to_find_power() {
    let c = common::minimal_ols_contract();
    let (n, n_sims, seed) = (130usize, 96usize, 555u64);

    let report = debug_report(&c, seed, n, n_sims, StageMask::all()).unwrap();
    let dbg_power = report.power.expect("power");

    let cancel = CancellationToken::new();
    let fp = find_power(std::slice::from_ref(&c), n, n_sims, seed, None, &cancel).unwrap();
    let ref_power = &fp.scenarios[0].1;

    // Exact equality on RAW INTEGER COUNTERS — not float rates.
    assert_eq!(
        dbg_power.success_counts_uncorrected,
        ref_power.success_counts_uncorrected
    );
    assert_eq!(
        dbg_power.success_counts_corrected,
        ref_power.success_counts_corrected
    );
    assert_eq!(dbg_power.convergence_count, ref_power.convergence_count);
    assert_eq!(
        dbg_power.overall_significant_count,
        ref_power.overall_significant_count
    );
}

#[test]
fn invariant_4_data_refit_reproduces_stat_under_hsk() {
    use engine_orchestrator::debug::debug_load_data;
    // Scenario λ=4 so the test fails whenever the data stage and the stats
    // stage generate from differently-prepared specs (the pre-fix divergence).
    let mut c = common::minimal_ols_contract();
    c.scenario.heteroskedasticity_ratio = 4.0;
    let report = debug_report(&c, 777, 120, 1, StageMask::all()).unwrap();
    let stats = report.stats.unwrap();
    let d = report.data.unwrap();

    let fit = debug_load_data(
        &c,
        777,
        &d.design.data,
        d.design.nrow,
        d.design.ncol,
        &d.outcome,
        None,
    )
    .unwrap();
    for tg in &fit.targets {
        let ts = stats
            .targets
            .iter()
            .find(|t| t.target_label == tg.target_label)
            .expect("matching stats target");
        assert_eq!(
            tg.statistic.to_bits(),
            ts.statistic[0].to_bits(),
            "{}: refit stat {} != stats.statistic[0] {}",
            tg.target_label,
            tg.statistic,
            ts.statistic[0]
        );
    }
}
