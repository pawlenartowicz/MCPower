//! `standardized::standardized_effects`: exact closed-form checks, the
//! undefined (`None`) cases, and checks against the sample moments of a large
//! seeded draw from the generator.

use engine_contract::fixtures::example1_simple_ols;
use engine_contract::{
    ClusterSizing, ClusterSpec, ColumnId, ColumnSpec, Correlations, DesignTerm, EstimatorSpec,
    LinkKind, OutcomeKind, SimulationContract, SlopeTerm, SyntheticKind, TestTarget, UploadedFrame,
};
use engine_core::contract_adapter::contract_to_simulation_spec;
use engine_core::data_gen::generate_sim_data;
use engine_core::standardized::{StandardizedEffects, standardized_effects};
use engine_core::workspace::SimWorkspace;

const PI2_3: f64 = std::f64::consts::PI * std::f64::consts::PI / 3.0;

fn normal() -> ColumnSpec {
    ColumnSpec::Synthetic {
        kind: SyntheticKind::Normal,
        pinned: false,
    }
}

fn binary(p: f64) -> ColumnSpec {
    ColumnSpec::Synthetic {
        kind: SyntheticKind::Binary { p },
        pinned: false,
    }
}

fn factor(proportions: &[f64]) -> ColumnSpec {
    ColumnSpec::FactorSynthetic {
        n_levels: proportions.len() as u32,
        proportions: proportions.to_vec(),
        sampled_proportions: None,
    }
}

fn direct(c: u32) -> DesignTerm {
    DesignTerm::Direct {
        column: ColumnId(c),
    }
}

fn dummy(c: u32, level_index: u32) -> DesignTerm {
    DesignTerm::DummyOf {
        column: ColumnId(c),
        level_index,
    }
}

/// OLS contract over `columns`, generating `terms` (Const first) with `coefs`
/// (0 for the Const slot); every non-Const term is a target.
fn contract(
    columns: Vec<ColumnSpec>,
    terms: Vec<DesignTerm>,
    coefs: Vec<f64>,
) -> SimulationContract {
    let mut c = example1_simple_ols();
    c.test.targets = (1..terms.len() as u32)
        .map(|term| TestTarget::Marginal { term })
        .collect();
    c.generation.columns = columns;
    c.design_generation.terms = terms;
    c.outcome.coefficients = coefs;
    c
}

fn random_intercept(mut c: SimulationContract, tau_squared: f64) -> SimulationContract {
    c.generation.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
        tau_squared,
        slopes: vec![],
        extra_groupings: vec![],
    });
    if c.outcome.kind == OutcomeKind::Continuous {
        c.estimator = EstimatorSpec::Mle;
    }
    c
}

fn binary_outcome(mut c: SimulationContract, link: Option<LinkKind>) -> SimulationContract {
    c.outcome.kind = OutcomeKind::Binary;
    c.outcome.link = link;
    c.outcome.intercept = -0.5;
    c.estimator = EstimatorSpec::Glm;
    c
}

fn close(got: Option<f64>, want: f64, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: got None, want {want}"));
    assert!((got - want).abs() < 1e-12, "{what}: got {got}, want {want}");
}

fn run(c: &SimulationContract) -> StandardizedEffects {
    standardized_effects(c).expect("valid contract")
}

#[test]
fn one_normal_predictor() {
    let b = 0.5_f64;
    let r = run(&contract(
        vec![normal()],
        vec![DesignTerm::Const, direct(0)],
        vec![0.0, b],
    ));
    assert_eq!(r.effects.len(), 1);
    close(r.effects[0].std_beta, b / (b * b + 1.0).sqrt(), "std beta");
    assert_eq!(
        r.effects[0].cohen_d, None,
        "a continuous predictor has no d"
    );
    close(r.r2, b * b / (b * b + 1.0), "R2");
}

#[test]
fn one_balanced_binary_predictor() {
    let b = 0.8_f64;
    let r = run(&contract(
        vec![binary(0.5)],
        vec![DesignTerm::Const, direct(0)],
        vec![0.0, b],
    ));
    close(r.effects[0].cohen_d, b, "d");
    close(
        r.effects[0].std_beta,
        0.5 * b / (0.25 * b * b + 1.0).sqrt(),
        "std beta",
    );
}

#[test]
fn one_factor_gives_d_equal_to_each_level_effect() {
    let r = run(&contract(
        vec![factor(&[0.5, 0.3, 0.2])],
        vec![DesignTerm::Const, dummy(0, 1), dummy(0, 2)],
        vec![0.0, 0.4, -0.7],
    ));
    close(r.effects[0].cohen_d, 0.4, "d level 1");
    close(r.effects[1].cohen_d, -0.7, "d level 2");
}

#[test]
fn random_intercept_enters_the_total_variance() {
    let (b, tau2) = (0.5_f64, 0.4_f64);
    let c = random_intercept(
        contract(
            vec![normal()],
            vec![DesignTerm::Const, direct(0)],
            vec![0.0, b],
        ),
        tau2,
    );
    let r = run(&c);
    close(
        r.effects[0].std_beta,
        b / (b * b + 1.0 + tau2).sqrt(),
        "std beta",
    );
    close(r.r2, b * b / (b * b + 1.0 + tau2), "marginal R2");
}

#[test]
fn binary_links_use_their_latent_residual_variance() {
    let b = 0.5_f64;
    let base = contract(
        vec![normal()],
        vec![DesignTerm::Const, direct(0)],
        vec![0.0, b],
    );
    let logit = run(&binary_outcome(base.clone(), None));
    close(
        logit.effects[0].std_beta,
        b / (b * b + PI2_3).sqrt(),
        "logit std beta",
    );
    close(logit.r2, b * b / (b * b + PI2_3), "logit R2");
    let probit = run(&binary_outcome(base, Some(LinkKind::Probit)));
    close(
        probit.effects[0].std_beta,
        b / (b * b + 1.0).sqrt(),
        "probit std beta",
    );
}

#[test]
fn binary_d_beside_an_uncorrelated_predictor_is_marginal() {
    // Randomized binary, no interactions: d = β / √(V_y − β²p(1−p)).
    let (p, bg, bx) = (0.3_f64, 0.6_f64, 0.5_f64);
    let r = run(&contract(
        vec![binary(p), normal()],
        vec![DesignTerm::Const, direct(0), direct(1)],
        vec![0.0, bg, bx],
    ));
    let v_y = bg * bg * p * (1.0 - p) + bx * bx + 1.0;
    close(
        r.effects[0].cohen_d,
        bg / (v_y - bg * bg * p * (1.0 - p)).sqrt(),
        "d",
    );
    close(r.effects[1].std_beta, bx / v_y.sqrt(), "std beta of x");
}

#[test]
fn count_outcome_and_random_slopes_are_undefined() {
    let base = contract(
        vec![binary(0.5), normal()],
        vec![DesignTerm::Const, direct(0), direct(1)],
        vec![0.0, 0.3, 0.2],
    );
    let mut count = base.clone();
    count.outcome.kind = OutcomeKind::Count;
    count.estimator = EstimatorSpec::Glm;
    let mut slope = random_intercept(base, 0.3);
    slope.generation.cluster.as_mut().unwrap().slopes = vec![SlopeTerm {
        column: ColumnId(1),
        variance: 0.1,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    for (label, c) in [("count", count), ("random slope", slope)] {
        let r = run(&c);
        assert_eq!(r.effects.len(), 2, "{label}");
        assert_eq!(r.r2, None, "{label}");
        for e in &r.effects {
            assert_eq!((e.std_beta, e.cohen_d), (None, None), "{label}");
        }
    }
}

// ---------------------------------------------------------------------------
// Against generated data
// ---------------------------------------------------------------------------

const N_DRAW: usize = 400_000;

/// One generated dataset of `N_DRAW` rows: the design columns (`x_full`) and y.
fn draw(c: &SimulationContract) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut spec = contract_to_simulation_spec(c).unwrap();
    spec.het_coeffs = spec.compute_het_coeffs();
    let n_nf = spec.n_non_factor as usize;
    let p = 1 + n_nf + spec.n_factor_dummies as usize + spec.interactions.len();
    let mut ws = SimWorkspace::new(N_DRAW, p, n_nf, spec.factor_n_levels.len(), None);
    generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
    let cols = (0..p)
        .map(|j| (0..N_DRAW).map(|i| ws.x_full[(i, j)] as f64).collect())
        .collect();
    let y = (0..N_DRAW).map(|i| ws.y_full[i] as f64).collect();
    (cols, y)
}

fn mean_var(v: impl Iterator<Item = f64> + Clone) -> (f64, f64, f64) {
    let n = v.clone().count() as f64;
    let m = v.clone().sum::<f64>() / n;
    let var = v.map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, var, n)
}

/// Sample d between the rows where `in_1` holds and those where `in_0` does,
/// over the share-weighted pooled within-group variance.
fn sample_d(y: &[f64], in_1: impl Fn(usize) -> bool, in_0: impl Fn(usize) -> bool) -> f64 {
    let (m1, v1, n1) = mean_var((0..y.len()).filter(|&i| in_1(i)).map(|i| y[i]));
    let (m0, v0, n0) = mean_var((0..y.len()).filter(|&i| in_0(i)).map(|i| y[i]));
    let w = n1 / (n1 + n0);
    (m1 - m0) / (w * v1 + (1.0 - w) * v0).sqrt()
}

/// Monte-Carlo SE of a d estimate on groups of share w out of N_DRAW rows.
fn se_d(d: f64, w: f64) -> f64 {
    let n = N_DRAW as f64;
    (1.0 / (n * w * (1.0 - w)) + d * d / (2.0 * n)).sqrt()
}

fn sd(v: &[f64]) -> f64 {
    mean_var(v.iter().copied()).1.sqrt()
}

#[test]
fn binary_and_factor_d_inside_interactions_match_generated_data() {
    // y ~ g + x + f + g:x + f[2]:x, g binary (p = 0.3), f three levels. Exact
    // case: binaries and factors are independent of everything else.
    let c = contract(
        vec![binary(0.3), normal(), factor(&[0.5, 0.3, 0.2])],
        vec![
            DesignTerm::Const,
            direct(0),
            direct(1),
            dummy(2, 1),
            dummy(2, 2),
            DesignTerm::Interaction {
                components: vec![direct(0), direct(1)],
            },
            DesignTerm::Interaction {
                components: vec![dummy(2, 2), direct(1)],
            },
        ],
        vec![0.0, 0.5, 0.3, 0.4, -0.6, 0.35, 0.25],
    );
    let r = run(&c);
    let (x, y) = draw(&c);
    // x_full: [1, g, x, f1, f2, g:x, f2:x].
    let d_g = sample_d(&y, |i| x[1][i] == 1.0, |i| x[1][i] == 0.0);
    let ref_level = |i: usize| x[3][i] == 0.0 && x[4][i] == 0.0;
    let d_f2 = sample_d(&y, |i| x[4][i] == 1.0, ref_level);
    let got_g = r.effects[0].cohen_d.unwrap();
    let got_f2 = r.effects[3].cohen_d.unwrap();
    assert!(
        (got_g - d_g).abs() < 5.0 * se_d(d_g, 0.3),
        "binary d: analytic {got_g}, sample {d_g}"
    );
    assert!(
        (got_f2 - d_f2).abs() < 5.0 * se_d(d_f2, 0.2 / 0.7),
        "factor d: analytic {got_f2}, sample {d_f2}"
    );
}

/// Largest absolute error of the standardized β and R² against the sample,
/// where sample std β_j = β_j · SD(x_j) / SD(y) and sample R² = Var(η)/Var(y).
fn std_beta_and_r2_error(c: &SimulationContract, r: &StandardizedEffects) -> f64 {
    let (x, y) = draw(c);
    let spec = contract_to_simulation_spec(c).unwrap();
    let sd_y = sd(&y);
    let mut worst = 0.0_f64;
    for (k, e) in r.effects.iter().enumerate() {
        // Without factors or reordering, term k + 1 is x_full column k + 1.
        let j = k + 1;
        let sample = spec.effect_sizes[j] * sd(&x[j]) / sd_y;
        worst = worst.max((e.std_beta.unwrap() - sample).abs());
    }
    let eta: Vec<f64> = (0..y.len())
        .map(|i| (1..x.len()).map(|j| spec.effect_sizes[j] * x[j][i]).sum())
        .collect();
    let r2 = sd(&eta).powi(2) / sd_y.powi(2);
    worst.max((r.r2.unwrap() - r2).abs())
}

/// Accuracy probe for the approximate case (a non-normal predictor correlated
/// with another): prints the largest error against generated data. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "accuracy probe: prints errors, no target asserted"]
fn approximate_cases_error_probe() {
    // A right-skewed predictor correlated (r = 0.5) with a normal one, plus
    // their interaction.
    let mut skewed = contract(
        vec![
            ColumnSpec::Synthetic {
                kind: SyntheticKind::RightSkewed,
                pinned: true,
            },
            normal(),
        ],
        vec![
            DesignTerm::Const,
            direct(0),
            direct(1),
            DesignTerm::Interaction {
                components: vec![direct(0), direct(1)],
            },
        ],
        vec![0.0, 0.5, 0.3, 0.3],
    );
    skewed.generation.correlations = Correlations::Matrix {
        continuous_columns: vec![ColumnId(0), ColumnId(1)],
        values: vec![1.0, 0.5, 0.5, 1.0],
    };
    let r = run(&skewed);
    println!(
        "correlated right-skewed with interaction: max |error| of std beta and R2 = {:.5}",
        std_beta_and_r2_error(&skewed, &r)
    );
    skewed.design_generation.terms.pop();
    skewed.outcome.coefficients.pop();
    skewed.test.targets.pop();
    let r = run(&skewed);
    println!(
        "correlated right-skewed, main effects only: max |error| of std beta and R2 = {:.5}",
        std_beta_and_r2_error(&skewed, &r)
    );
}

#[test]
fn strict_upload_matches_generated_data() {
    // Strict (bootstrap) upload of three columns that go together: a
    // standardized continuous u, a binary b (1 when u + noise > 0.3) and a
    // three-level factor g cut from u + noise; plus a synthetic normal x.
    // Generated rows copy whole frame rows, so b and g carry u's effect into
    // their group differences; the moments must use the rows themselves.
    let rows = 2_000;
    let mut state = 12345_u64;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut u: Vec<f64> = Vec::with_capacity(rows);
    let mut b: Vec<f64> = Vec::with_capacity(rows);
    let mut g: Vec<f64> = Vec::with_capacity(rows);
    for _ in 0..rows {
        // Box–Muller pair: one draw for u, one for the noise.
        let (r1, r2) = (unif(), unif());
        let rad = (-2.0 * r1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * r2;
        let (z1, z2) = (rad * angle.cos(), rad * angle.sin());
        u.push(z1);
        b.push(if z1 + z2 > 0.3 { 1.0 } else { 0.0 });
        let t = z1 - 0.5 * z2;
        g.push(if t < -0.5 {
            0.0
        } else if t < 0.6 {
            1.0
        } else {
            2.0
        });
    }
    let (mu, vu, _) = mean_var(u.iter().copied());
    let p = b.iter().sum::<f64>() / rows as f64;
    let shares: Vec<f64> = (0..3)
        .map(|k| g.iter().filter(|&&v| v == k as f64).count() as f64 / rows as f64)
        .collect();
    let mut data = Vec::with_capacity(rows * 3);
    for i in 0..rows {
        data.push((u[i] - mu) / vu.sqrt());
        // Uploaded binaries travel centered (x − p), as the hosts send them.
        data.push(b[i] - p);
        data.push(g[i]);
    }
    // Columns u, b, g, x; terms ordered so term k is x_full column k:
    // [1, u, b, x, g1, g2, u:b, b:x].
    let mut strict = contract(
        vec![
            ColumnSpec::Resampled { frame_column: 0 },
            ColumnSpec::ResampledBinary {
                frame_column: 1,
                proportion: p,
            },
            ColumnSpec::FactorFromFrame {
                frame_column: 2,
                n_levels: 3,
                proportions: shares,
                sampled_proportions: None,
            },
            normal(),
        ],
        vec![
            DesignTerm::Const,
            direct(0),
            direct(1),
            direct(3),
            dummy(2, 1),
            dummy(2, 2),
            DesignTerm::Interaction {
                components: vec![direct(0), direct(1)],
            },
            DesignTerm::Interaction {
                components: vec![direct(1), direct(3)],
            },
        ],
        vec![0.0, 0.4, 0.5, 0.3, 0.2, -0.3, 0.25, 0.2],
    );
    strict.generation.uploaded_frame = Some(UploadedFrame {
        data,
        n_rows: rows as u32,
        n_cols: 3,
        bootstrap: true,
    });
    let r = run(&strict);
    let (x, y) = draw(&strict);
    let d_b = sample_d(&y, |i| x[2][i] == 1.0, |i| x[2][i] == 0.0);
    let ref_level = |i: usize| x[4][i] == 0.0 && x[5][i] == 0.0;
    let d_g2 = sample_d(&y, |i| x[5][i] == 1.0, ref_level);
    let got_b = r.effects[1].cohen_d.unwrap();
    let got_g2 = r.effects[4].cohen_d.unwrap();
    assert!(
        (got_b - d_b).abs() < 5.0 * se_d(d_b, p),
        "uploaded binary d: analytic {got_b}, sample {d_b}"
    );
    let n_g2 = x[5].iter().sum::<f64>();
    let w_g2 = n_g2 / (n_g2 + (0..N_DRAW).filter(|&i| ref_level(i)).count() as f64);
    assert!(
        (got_g2 - d_g2).abs() < 5.0 * se_d(d_g2, w_g2),
        "uploaded factor d: analytic {got_g2}, sample {d_g2}"
    );
    // Monte-Carlo SE of a ratio of SDs and of R² is about 1/√N_DRAW.
    let err = std_beta_and_r2_error(&strict, &r);
    assert!(
        err < 5.0 / (N_DRAW as f64).sqrt(),
        "strict upload: max |error| of std beta and R2 = {err}"
    );
}
