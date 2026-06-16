//! Standing L1 parity gate: the general machine (`lmm`) reproduces the
//! shipped `lme.rs` Brent fit on the same bytes — q=1-through-general-machine
//! parity at the Gate-0 amended tolerances (rel 1e-4 with abs floors stat
//! 1e-4 / β̂ 1e-5, the measured Brent θ̂-placement-noise floor; any
//! optimizer-vs-optimizer gate needs an abs floor ≥ the reference's own
//! placement noise). Corpus mirrors the Gate-0 spike grid — FROZEN: parity
//! numbers are only comparable across runs if cases stay bit-identical.
//!
//! Two runs over the same corpus: blind (`theta_start = None` — the M1 gate
//! proper, apples-to-apples vs the shipped Brent) and truth-started
//! (`Some(√τ²)` — the P1 flip proof: same bytes, same gates, θ-search
//! started at the spec's known θ).

use engine_core::introspect::{fit_provided_data, run_introspect, IntrospectMask};
use engine_core::lmm::{fit_lmm, LmmWorkspace};
use engine_core::spec::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    HeteroskedasticityCoeffs, OutcomeKind, ResidualDist,
    ScenarioPerturbations, SimulationSpec,
};
use faer::Mat;

const BASE_SEED: u64 = 2137;
/// Datasets per case — 6 cases × 25 seeds = 150 same-bytes comparisons.
const N_SEEDS: u64 = 25;

struct CaseDef {
    id: &'static str,
    /// Non-factor predictors; design p = 1 + k, targets = cols 1 and 2.
    k: u32,
    n: u32,
    n_clusters: u32,
    tau_sq: f64,
    effect: f64,
}

/// The Gate-0 spike corpus (n_sims column dropped — each seed is one draw).
const CASES: &[CaseDef] = &[
    CaseDef { id: "mid",    k: 5, n: 500,  n_clusters: 20,  tau_sq: 0.25, effect: 0.15 },
    CaseDef { id: "many",   k: 3, n: 1000, n_clusters: 100, tau_sq: 0.25, effect: 0.07 },
    CaseDef { id: "small",  k: 3, n: 60,   n_clusters: 6,   tau_sq: 0.25, effect: 0.35 },
    CaseDef { id: "hi_tau", k: 3, n: 200,  n_clusters: 20,  tau_sq: 4.0,  effect: 0.25 },
    CaseDef { id: "zero",   k: 3, n: 200,  n_clusters: 20,  tau_sq: 0.0,  effect: 0.14 },
    CaseDef { id: "tiny",   k: 3, n: 200,  n_clusters: 20,  tau_sq: 0.01, effect: 0.14 },
];

/// Frozen spec constructor — hand-copied from bin/throughput.rs's
/// base_spec()+lme_spec() shape (the same provenance as the Gate-0 spike's
/// corpus); the test must not read configs at runtime.
fn spec_for(c: &CaseDef) -> SimulationSpec {
    let k = c.k as usize;
    let mut correlation = vec![0.0; k * k];
    for i in 0..k {
        correlation[i * k + i] = 1.0;
    }
    let mut effect_sizes = vec![0.0; 1 + k];
    effect_sizes[1] = c.effect;
    effect_sizes[2] = c.effect;
    SimulationSpec {
        n_non_factor: c.k,
        n_factor_dummies: 0,
        correlation,
        var_types: vec![Distribution::Normal; k],
        var_pinned: vec![],
        var_params: vec![0.0; k],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes,
        target_indices: vec![1, 2],
        contrast_pairs: vec![],
        interactions: vec![],
        correction_method: CorrectionMethod::None,
        crit_values: CritValues { alpha: 0.05, posthoc_alpha: None },
        heteroskedasticity_driver: None,
        residual_dist: ResidualDist::Normal,
        residual_pinned: false,
        outcome_kind: OutcomeKind::Continuous,
        estimator: EstimatorSpec::Mle,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.25,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: c.n_clusters },
            tau_squared: c.tau_sq,
            slopes: vec![],
            extra_groupings: vec![],
        }),
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    }
}

/// |a−b| within rel of the larger magnitude, with an absolute floor.
fn close(a: f64, b: f64, rel: f64, abs_floor: f64) -> bool {
    let d = (a - b).abs();
    d <= abs_floor || d <= rel * a.abs().max(b.abs())
}

/// Shared corpus walk — both #[test] entry points below run the identical
/// cases/seeds/gates and differ only in the θ start handed to `fit_lmm`.
fn corpus_parity(theta_start_for: fn(&CaseDef) -> Option<f64>) {
    for c in CASES {
        let spec = spec_for(c);
        let p = (1 + c.k) as usize;
        let mut ws = LmmWorkspace::new(p, c.n_clusters as usize);

        for s in 0..N_SEEDS {
            // Distinct seed per (case, draw); offsets keep streams disjoint
            // across cases.
            let seed = BASE_SEED
                + 10_000 * (CASES.iter().position(|cc| cc.id == c.id).unwrap() as u64)
                + s;
            let out = run_introspect(
                &spec,
                c.n,
                1,
                seed,
                IntrospectMask { stats: false, data: true, crit: true, power: false },
            )
            .expect("introspect");
            let d = out.data.expect("data capture");
            let crit_sq = out.crit.expect("crit").crit_sq_uncorrected;

            // Shipped path — the B↔C debug vehicle fits the captured bytes
            // through the identical lme.rs kernel the hot loop uses.
            let ship = fit_provided_data(
                &spec,
                &d.design,
                d.nrow,
                d.ncol,
                &d.outcome,
                d.cluster_ids.as_deref(),
            )
            .expect("shipped fit");

            // General path — the SAME bytes through the lmm machine.
            // Design/outcome captured as f64; narrow to f32 for the data plane.
            let ids = d.cluster_ids.as_ref().expect("cluster ids");
            let mut x = Mat::<f32>::zeros(d.nrow, d.ncol);
            for j in 0..d.ncol {
                for i in 0..d.nrow {
                    x[(i, j)] = d.design[j * d.nrow + i] as f32;
                }
            }
            let outcome_f32: Vec<f32> = d.outcome.iter().map(|&v| v as f32).collect();
            ws.suff.reset();
            ws.suff.add_rows(x.as_ref(), &outcome_f32, ids);
            let start_buf = theta_start_for(c).map(|t| [t]);
            let fit = fit_lmm(
                &mut ws,
                &spec.target_indices,
                start_buf.as_ref().map(|a| a.as_slice()),
            );

            // Gate 1: convergence classification agrees (the spike held this
            // 5400/5400 — zero ship_only/gen_only anywhere, hi_tau included).
            assert_eq!(
                ship.converged, fit.converged,
                "{}/{s}: convergence classification diverged",
                c.id
            );
            if !fit.converged {
                continue;
            }

            // Gate 2: β̂ per design column (rel 1e-4, abs floor 1e-5).
            for j in 0..p {
                assert!(
                    close(ship.betas[j], ws.fit.betas[j], 1e-4, 1e-5),
                    "{}/{s} β[{j}]: shipped {} vs general {}",
                    c.id, ship.betas[j], ws.fit.betas[j]
                );
            }

            // Gate 3 + 4: per-target statistic (rel 1e-4, abs floor 1e-4) and
            // the significance CALL. Both paths share the one CritValueTable
            // authority, so crit parity is exact by construction.
            for (t, &tj) in spec.target_indices.iter().enumerate() {
                let gen_t_sq = ws.fit.t_sq[tj as usize];
                let gen_stat = gen_t_sq.sqrt();
                assert!(
                    close(ship.statistic[t], gen_stat, 1e-4, 1e-4),
                    "{}/{s} stat[{t}]: shipped {} vs general {}",
                    c.id, ship.statistic[t], gen_stat
                );
                let ship_sig = ship.statistic[t] * ship.statistic[t] > crit_sq;
                let gen_sig = gen_t_sq > crit_sq;
                assert_eq!(
                    ship_sig, gen_sig,
                    "{}/{s} target {t}: significance call flipped",
                    c.id
                );
            }
        }
    }
}

/// The M1 gate proper — blind start, apples-to-apples vs the shipped Brent.
#[test]
fn general_machine_reproduces_shipped_q1_fit_on_same_bytes() {
    corpus_parity(|_| None);
}

/// P1 truth-start flip proof (the last M1 parity step): same corpus, same
/// gates, θ-search started at the spec's known θ_true = τ/σ (unit residual
/// σ by corpus construction ⇒ √τ²). The zero case passes Some(0.0) and
/// exercises the THETA_TRUTH_FLOOR clamp on real fits.
#[test]
fn truth_started_fit_holds_the_same_parity_gates() {
    corpus_parity(|c| Some(c.tau_sq.sqrt()));
}
