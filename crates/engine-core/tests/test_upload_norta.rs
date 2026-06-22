//! NORTA marginal path for uploaded data — B2 (UploadedData) and B3 (UploadedBinary) tests.
//!
//! All tests build `SimulationSpec` directly (the same pattern used in
//! `test_interactions_power.rs`) and call `generate_sim_data` to obtain
//! a single generated row-block, then compute moments over many rows.

use engine_core::data_gen::generate_sim_data;
use engine_core::spec::HeteroskedasticityCoeffs;
use engine_core::workspace::SimWorkspace;
use engine_core::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, OutcomeKind, ResidualDist,
    ScenarioPerturbations, SimulationSpec,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rational approximation of Φ⁻¹(p) — Peter Acklam's algorithm, max error ~3.65e-9.
/// Used only in test setup to build standard-normal quantile tables.
fn phi_inv(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0);
    // Coefficients for the rational approximation.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Build a standard-normal quantile table of `u_rows` entries (type-7 mid-point).
/// Values are in ascending order: `phi_inv((r + 0.5) / u_rows)` for r in 0..u_rows.
fn std_normal_quantile_table(u_rows: usize) -> Vec<f64> {
    (0..u_rows)
        .map(|r| phi_inv((r as f64 + 0.5) / u_rows as f64))
        .collect()
}

/// Build a minimal OLS spec with `n_non_factor` columns and supplied `upload_normal`.
/// Scenario is optimistic (no perturbation). Target: column 1 (first non-factor).
fn uploaded_data_spec(
    n_non_factor: u32,
    upload_normal: Vec<f64>,
    upload_normal_shape: (u32, u32),
    var_types: Vec<Distribution>,
    var_params: Vec<f64>,
    correlation: Vec<f64>,
) -> SimulationSpec {
    let n_nf = n_non_factor as usize;
    let n_predictors = 1 + n_nf; // intercept + non-factor
    let mut effect_sizes = vec![0.0; n_predictors];
    effect_sizes[1] = 0.5; // some effect on col 1 — just for valid spec
    SimulationSpec {
        n_non_factor,
        n_factor_dummies: 0,
        correlation,
        var_types,
        var_pinned: vec![],
        var_params,
        upload_normal,
        upload_normal_shape,
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes,
        target_indices: vec![1],
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
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 1.0,
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

/// Run `generate_sim_data` for `n_rows` rows and collect column `col_idx` (0-based,
/// into `x_full`; col 0 is intercept, col 1 is first non-factor, etc.).
fn collect_col(spec: &SimulationSpec, n_rows: usize, base_seed: u64, col_idx: usize) -> Vec<f64> {
    let n_nf = spec.n_non_factor as usize;
    let n_predictors = 1 + n_nf;
    let mut ws = SimWorkspace::new(n_rows, n_predictors, n_nf, 0, None);
    generate_sim_data(spec, 0, base_seed, &mut ws).expect("generate_sim_data failed");
    (0..n_rows)
        .map(|r| ws.x_full[(r, col_idx)] as f64)
        .collect()
}

/// Collect two columns (col_a, col_b) from many independent sims concatenated.
/// Returns paired (col_a_vals, col_b_vals) from the same x_full matrices.
fn collect_two_cols_many_sims(
    spec: &SimulationSpec,
    rows_per_sim: usize,
    n_sims: usize,
    base_seed: u64,
    col_a: usize,
    col_b: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_nf = spec.n_non_factor as usize;
    let n_predictors = 1 + n_nf;
    let mut ws = SimWorkspace::new(rows_per_sim, n_predictors, n_nf, 0, None);
    let mut all_a = Vec::with_capacity(rows_per_sim * n_sims);
    let mut all_b = Vec::with_capacity(rows_per_sim * n_sims);
    for sim in 0..n_sims as u64 {
        generate_sim_data(spec, sim, base_seed, &mut ws).expect("generate_sim_data failed");
        for r in 0..rows_per_sim {
            all_a.push(ws.x_full[(r, col_a)] as f64);
            all_b.push(ws.x_full[(r, col_b)] as f64);
        }
    }
    (all_a, all_b)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

// ---------------------------------------------------------------------------
// B2 tests — UploadedData: inverse empirical CDF
// ---------------------------------------------------------------------------

/// B2 (failing before implementation):
/// With `upload_normal` set to a standard-normal quantile table, the generated
/// column should have:
///   - mean ≈ 0   (|mean| < 0.05)
///   - var  ≈ 1   (|var - 1| < 0.05)
///   - every value ∈ [min_uploaded − 1e-9, max_uploaded + 1e-9]
///     (the identity stub returns z ~ N(0,1) which violates the bounds assertion
///     because the table's range is bounded while z is unbounded)
#[test]
fn b2_uploaded_data_moments_and_bounds() {
    const U: usize = 200;
    let table = std_normal_quantile_table(U);
    let min_val = table[0];
    let max_val = table[U - 1];

    let upload_normal = table.clone(); // (U, 1) row-major: col 0 = table
    let spec = uploaded_data_spec(
        1,
        upload_normal,
        (U as u32, 1),
        vec![Distribution::UploadedData],
        vec![0.0],
        vec![1.0], // 1×1 identity correlation
    );

    // Generate a single sim's column. n=40000 (was 2000): the ±0.05 moment
    // tolerance needs a well-powered estimator — at n=2000 the sample variance's
    // SD (~0.03) lets a valid realization sit ~2σ off (the Philox/f32 stream lands
    // at 0.93); n=40000 (SD ~0.007) converges to ≈1, confirming no f32 tail bias.
    let col = collect_col(&spec, 40000, 2137, 1);

    let m = mean(&col);
    let v = variance(&col);

    assert!(
        m.abs() < 0.05,
        "UploadedData mean should be ≈0 for std-normal table, got {m:.4}"
    );
    assert!(
        (v - 1.0).abs() < 0.05,
        "UploadedData variance should be ≈1 for std-normal table, got {v:.4}"
    );

    // Bounds: every generated value must lie within [min_val, max_val].
    // This assertion FAILS for the identity stub (z can exceed ±3 while the table
    // is bounded at ≈ ±2.97 for U=200).
    for (i, &val) in col.iter().enumerate() {
        assert!(
            val >= min_val - 1e-9 && val <= max_val + 1e-9,
            "row {i}: value {val} is outside uploaded range [{min_val}, {max_val}]"
        );
    }
}

/// B2 extra: with a non-standard table (bounded ramp [1..5]), all generated values
/// stay within [1, 5].
#[test]
fn b2_uploaded_data_bounded_by_table() {
    // Table: 50 evenly-spaced values in [1.0, 5.0], already sorted ascending.
    let table: Vec<f64> = (0..50).map(|i| 1.0 + 4.0 * i as f64 / 49.0).collect();
    let min_val = table[0];
    let max_val = table[49];

    let spec = uploaded_data_spec(
        1,
        table,
        (50, 1),
        vec![Distribution::UploadedData],
        vec![0.0],
        vec![1.0],
    );

    let col = collect_col(&spec, 1000, 2137, 1);
    for (i, &val) in col.iter().enumerate() {
        assert!(
            val >= min_val - 1e-9 && val <= max_val + 1e-9,
            "row {i}: {val} not in [{min_val}, {max_val}]"
        );
    }
}

// ---------------------------------------------------------------------------
// B3 tests — UploadedBinary + UploadedFactor
// ---------------------------------------------------------------------------

/// B3 (failing before B3 implementation):
/// UploadedBinary with proportion=0.3 should produce a binary column with
/// mean ≈ 0.3 and all values ∈ {0.0, 1.0}.
#[test]
fn b3_uploaded_binary_proportion_and_values() {
    // Single UploadedBinary column, proportion = 0.3.
    // upload_normal is unused for binary (the threshold test uses phi(z) < param),
    // but must be shape-consistent: set it to zeros.
    let n_nf = 1u32;
    let upload_normal = vec![0.0f64; 1]; // (1, 1) — dummy
    let spec = uploaded_data_spec(
        n_nf,
        upload_normal,
        (1, n_nf),
        vec![Distribution::UploadedBinary],
        vec![0.3],
        vec![1.0],
    );

    // Generate n=4000 rows.
    let col = collect_col(&spec, 4000, 2137, 1);

    // All values must be 0 or 1.
    for (i, &v) in col.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "row {i}: UploadedBinary value {v} is not 0 or 1"
        );
    }

    // Mean ≈ 0.3 (within ±0.03).
    let m = mean(&col);
    assert!(
        (m - 0.3).abs() < 0.03,
        "UploadedBinary mean should be ≈0.3, got {m:.4}"
    );
}

/// B3 attenuation: UploadedBinary + UploadedData with positive correlation.
/// The realized Pearson r between the two generated columns must be positive
/// but strictly less than the input correlation (no polychoric correction).
#[test]
fn b3_uploaded_binary_correlation_attenuated() {
    const U: usize = 200;
    let table = std_normal_quantile_table(U);
    let input_corr = 0.6f64;

    // 2-column spec: col 0 = UploadedBinary(p=0.3), col 1 = UploadedData.
    // Correlation matrix (2×2 column-major): [[1, r], [r, 1]].
    let corr = vec![1.0, input_corr, input_corr, 1.0];

    // upload_normal layout: (U, 2) row-major.
    // Col 0 (binary) — zeros (unused by UploadedBinary arm).
    // Col 1 (continuous) — standard normal quantile table.
    let mut upload_normal = vec![0.0f64; U * 2];
    for r in 0..U {
        upload_normal[r * 2 + 1] = table[r];
    }

    let n_nf = 2u32;
    let mut effect_sizes = vec![0.0f64; 3]; // intercept + 2 cols
    effect_sizes[1] = 0.3;
    effect_sizes[2] = 0.3;
    let spec = SimulationSpec {
        n_non_factor: n_nf,
        n_factor_dummies: 0,
        correlation: corr,
        var_types: vec![Distribution::UploadedBinary, Distribution::UploadedData],
        var_pinned: vec![],
        var_params: vec![0.3, 0.0],
        upload_normal,
        upload_normal_shape: (U as u32, n_nf),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes,
        target_indices: vec![1],
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
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 1.0,
        cluster: None,
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    };

    // Collect 4000 paired rows (same x_full per sim) for stable sample correlation.
    let (bin_col, cont_col) = collect_two_cols_many_sims(&spec, 200, 20, 2137, 1, 2);
    let n = bin_col.len();

    // Pearson correlation between binary and continuous columns.
    let mb = mean(&bin_col);
    let mc = mean(&cont_col);
    let cov: f64 = bin_col
        .iter()
        .zip(cont_col.iter())
        .map(|(b, c)| (b - mb) * (c - mc))
        .sum::<f64>()
        / n as f64;
    let sb = variance(&bin_col).sqrt();
    let sc = variance(&cont_col).sqrt();
    let realized_r = cov / (sb * sc);

    // The binary marginal assigns 1 to the HIGH-z tail (u ≥ 1 − param), so it is
    // monotone-INCREASING in z — like the continuous marginal. Consequence: the
    // Pearson correlation between binary and continuous columns is POSITIVE (same
    // sign as input_corr), with magnitude strictly less than |input_corr| because
    // binarisation attenuates the linear relationship (no polychoric correction).
    // NOTE: the spec-builder never produces a correlated binary column (predictor
    // correlation is continuous-only by design); this exercises the engine-core
    // marginal directly to guard the threshold direction (the sign-preservation fix).
    assert!(
        realized_r > 0.0,
        "realized correlation should be positive (binary assigns 1 to high-z tail, continuous increases with z), got {realized_r:.4}"
    );
    assert!(
        realized_r.abs() < input_corr,
        "realized |r|={:.4} should be strictly less than input r={input_corr} (attenuation, no polychoric correction)",
        realized_r.abs()
    );
}
