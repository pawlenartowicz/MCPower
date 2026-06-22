//! Strict-mode (bootstrap) row-sampling arm for uploaded data — tests for strict-mode behavior.
//!
//! In strict mode the engine draws ONE source row index per generated design row
//! and copies every uploaded column's values from that source row, preserving the
//! exact empirical joint across columns. Synthetic columns keep their drawn values.
//!
//! All tests build `SimulationSpec` directly (mirroring the helpers in
//! `data_gen.rs`'s test module and `test_upload_norta.rs`) and call
//! `generate_sim_data` to obtain a row-block, then inspect `ws.x_full`.

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

/// Build a strict-mode 2-continuous-column spec.
///
/// `frame` is the standardized uploaded frame as a list of rows `[x1, x2]`; it is
/// flattened row-major into `upload_data` with shape `(U, 2)`. Both columns are
/// `UploadedData`; `bootstrap_frame_map = [Some(0), Some(1)]`. Identity correlation
/// (continuous correlation is unused on the bootstrap path but must be valid).
fn strict_2cont_spec(frame: &[[f64; 2]]) -> SimulationSpec {
    let u = frame.len();
    let mut upload_data = Vec::with_capacity(u * 2);
    for row in frame {
        upload_data.push(row[0]);
        upload_data.push(row[1]);
    }
    SimulationSpec {
        n_non_factor: 2,
        n_factor_dummies: 0,
        correlation: vec![1.0, 0.0, 0.0, 1.0],
        var_types: vec![Distribution::UploadedData, Distribution::UploadedData],
        var_pinned: vec![],
        var_params: vec![0.0, 0.0],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data,
        upload_data_shape: (u as u32, 2),
        bootstrap_frame_map: vec![Some(0), Some(1)],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        // β layout: intercept, x1, x2.
        effect_sizes: vec![0.0, 0.5, 0.3],
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

/// A standardized 2-column frame with a strong NONLINEAR dependence: `x2 = x1²`.
/// A NORTA Gaussian-copula draw could NOT reproduce this joint, so it cleanly
/// distinguishes bootstrap row-copying from synthetic generation.
fn nonlinear_frame() -> Vec<[f64; 2]> {
    (0..40)
        .map(|i| {
            let x1 = -2.0 + 4.0 * (i as f64) / 39.0; // ramp across [-2, 2]
            [x1, x1 * x1]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Whole-row copy
// ---------------------------------------------------------------------------

/// Strict bootstrap copies WHOLE source rows: every generated `(x1, x2)` equals
/// some actual uploaded frame row. With `x2 = x1²` in the frame, a NORTA draw
/// could never satisfy this for all rows — that's the discriminating signal.
#[test]
fn strict_bootstrap_copies_whole_rows() {
    let frame = nonlinear_frame();
    let spec = strict_2cont_spec(&frame);

    let n = 500;
    let mut ws = SimWorkspace::new(n, 3, 2, 0, None);
    generate_sim_data(&spec, 0, 2137, &mut ws).expect("generate_sim_data failed");

    for i in 0..n {
        let x1 = ws.x_full[(i, 1)];
        let x2 = ws.x_full[(i, 2)];
        let matched = frame
            .iter()
            // f[] is f64 (uploaded frame); x1/x2 are f32 (data plane) — widen for comparison.
            .any(|f| (f[0] - x1 as f64).abs() < 1e-6 && (f[1] - x2 as f64).abs() < 1e-6);
        assert!(
            matched,
            "row {i}: generated ({x1}, {x2}) is not any uploaded frame row \
             (bootstrap must copy whole rows, not synthesize)"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Row stability across max_n
// ---------------------------------------------------------------------------

/// Strict bootstrap is row-stable: the first 100 generated rows at max_n=100 are
/// bit-identical to the first 100 at max_n=1000 (same seed). The bootstrap arm
/// adds exactly one `next_uniform()` per row, so the per-row RNG call count is
/// constant ⇒ prefix-stable.
#[test]
fn strict_bootstrap_is_row_stable() {
    let frame = nonlinear_frame();
    let spec = strict_2cont_spec(&frame);

    let mut ws_small = SimWorkspace::new(100, 3, 2, 0, None);
    let mut ws_big = SimWorkspace::new(1000, 3, 2, 0, None);
    generate_sim_data(&spec, 5, 2137, &mut ws_small).unwrap();
    generate_sim_data(&spec, 5, 2137, &mut ws_big).unwrap();

    for i in 0..100 {
        for j in 0..3 {
            let small = ws_small.x_full[(i, j)];
            let big = ws_big.x_full[(i, j)];
            assert_eq!(
                small, big,
                "strict bootstrap x_full[{i},{j}] differs across max_n: small={small}, big={big}"
            );
        }
        assert_eq!(
            ws_small.y_full[i], ws_big.y_full[i],
            "strict bootstrap y differs at i={i}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Factor re-expansion
// ---------------------------------------------------------------------------

/// Build a strict spec: one continuous uploaded column + one uploaded factor
/// (3 levels → 2 reference-coded dummies). `bootstrap_frame_map = [Some(0), Some(1)]`:
/// frame col 0 holds the continuous value, frame col 1 holds the level code (0/1/2).
fn strict_cont_plus_factor_spec(frame: &[[f64; 2]]) -> SimulationSpec {
    let u = frame.len();
    let mut upload_data = Vec::with_capacity(u * 2);
    for row in frame {
        upload_data.push(row[0]); // continuous
        upload_data.push(row[1]); // level code
    }
    // n_pred = intercept(1) + non_factor(1) + factor_dummies(2) = 4.
    SimulationSpec {
        n_non_factor: 1,
        n_factor_dummies: 2,
        correlation: vec![1.0], // 1×1 identity (one continuous column)
        var_types: vec![Distribution::UploadedData],
        var_pinned: vec![],
        var_params: vec![0.0],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data,
        upload_data_shape: (u as u32, 2),
        bootstrap_frame_map: vec![Some(0), Some(1)],
        between_var_indices: vec![],
        factor_n_levels: vec![3],
        factor_proportions: vec![0.5, 0.3, 0.2],
        factor_sampled: Vec::new(),
        // β layout: intercept, x_cont, dummy0, dummy1.
        effect_sizes: vec![0.0, 0.5, 0.0, 0.0],
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

/// Strict bootstrap re-expands a copied factor level code into reference-coded
/// dummies. Each generated dummy pair must be valid reference-coding (each ∈ {0,1},
/// at most one-hot) AND correspond to exactly one source row's level.
#[test]
fn strict_bootstrap_factor_reexpands_level() {
    // Frame: continuous values arbitrary, level codes cycling 0,1,2.
    let frame: Vec<[f64; 2]> = (0..30)
        .map(|i| {
            let lvl = (i % 3) as f64;
            [(i as f64) * 0.1 - 1.5, lvl]
        })
        .collect();
    let spec = strict_cont_plus_factor_spec(&frame);

    let n = 500;
    let mut ws = SimWorkspace::new(n, 4, 1, 1, None);
    generate_sim_data(&spec, 0, 2137, &mut ws).expect("generate_sim_data failed");

    // The set of valid (dummy0, dummy1) re-expansions, by source level:
    //   level 0 (reference) → (0, 0)
    //   level 1             → (1, 0)
    //   level 2             → (0, 1)
    let valid_pairs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];

    for i in 0..n {
        let d0 = ws.x_full[(i, 2)];
        let d1 = ws.x_full[(i, 3)];
        // Each dummy ∈ {0, 1}.
        assert!(
            d0 == 0.0 || d0 == 1.0,
            "row {i}: dummy0={d0} not in {{0,1}}"
        );
        assert!(
            d1 == 0.0 || d1 == 1.0,
            "row {i}: dummy1={d1} not in {{0,1}}"
        );
        // At most one-hot.
        assert!(
            d0 + d1 <= 1.0,
            "row {i}: dummies ({d0},{d1}) not at-most-one-hot"
        );
        // Corresponds to exactly one valid source-level re-expansion.
        assert!(
            valid_pairs.iter().any(|&(p0, p1)| p0 == d0 && p1 == d1),
            "row {i}: dummy pair ({d0},{d1}) is not a valid factor re-expansion"
        );

        // Cross-check: the continuous column was copied from the SAME source row
        // whose level produced these dummies. Recover the source level from the
        // dummies, then confirm a frame row exists with that (continuous, level).
        let recovered_level = if d0 == 1.0 {
            1.0
        } else if d1 == 1.0 {
            2.0
        } else {
            0.0
        };
        let x_cont = ws.x_full[(i, 1)];
        let matched = frame
            .iter()
            // x_cont is f32 (data plane); frame entries are f64 (uploaded) — widen.
            .any(|f| (f[0] - x_cont as f64).abs() < 1e-6 && (f[1] - recovered_level).abs() < 1e-12);
        assert!(
            matched,
            "row {i}: copied continuous {x_cont} + recovered level {recovered_level} \
             do not match any single source row (factor + non-factor must copy from the \
             SAME source row)"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Binary recovery — joint with continuous
// ---------------------------------------------------------------------------

/// Build a strict spec with TWO uploaded columns from the SAME frame:
/// - col 0: continuous (`UploadedData`), distinct standardized values.
/// - col 1: binary (`UploadedBinary`), stored CENTERED as `x - p`.
///
/// `bootstrap_frame_map = [Some(0), Some(1)]` so both are drawn from the SAME
/// source row index `r` on every bootstrap copy.  Identity correlation (2×2);
/// `n_pred = 3` (intercept + x_cont + x_bin).
fn strict_cont_plus_binary_spec(upload_data: Vec<f64>, u: usize, p: f64) -> SimulationSpec {
    SimulationSpec {
        n_non_factor: 2,
        n_factor_dummies: 0,
        // 2×2 identity — continuous correlation, unused on bootstrap path.
        correlation: vec![1.0, 0.0, 0.0, 1.0],
        var_types: vec![Distribution::UploadedData, Distribution::UploadedBinary],
        var_pinned: vec![],
        var_params: vec![0.0, p],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data,
        upload_data_shape: (u as u32, 2),
        bootstrap_frame_map: vec![Some(0), Some(1)],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        // β layout: intercept, x_cont, x_bin.
        effect_sizes: vec![0.0, 0.5, 0.3],
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

/// Strict bootstrap copies the binary column from the SAME source row as the
/// continuous column, preserving the empirical joint.
///
/// Construction: 20 source rows; continuous col spans -1.9 … 1.9 (distinct);
/// raw binary = 1 for the upper-half rows (i >= 10), 0 for the lower-half.
/// This strong association means an independent binary draw would produce
/// (high-continuous, 0) or (low-continuous, 1) pairs absent from the frame —
/// the joint assertion catches that.
///
/// Concretely, for each generated row `i`:
///   x1 = ws.x_full[(i, 1)]   (continuous col, copied from frame col 0)
///   b  = ws.x_full[(i, 2)]   (binary col, recovered to 0/1 from frame col 1)
/// The pair `(x1, b)` must equal some source-row pair
///   (upload_data[r*2+0],  if upload_data[r*2+1] + p >= 0.5 { 1.0 } else { 0.0 })
/// for at least one r in 0..U.  A pure domain check (`b ∈ {0,1}`) alone would
/// pass even if binary was drawn independently — this joint check does not.
#[test]
fn strict_bootstrap_binary_copies_joint_with_continuous() {
    let u: usize = 20;
    // Continuous col: distinct standardized values spanning [-1.9, 1.9].
    // Raw binary: 1 for upper-half rows (i >= 10), 0 for lower-half.
    // Empirical proportion p = 10/20 = 0.5.
    let p = 0.5_f64;
    let mut upload_data = Vec::with_capacity(u * 2);
    for i in 0..u {
        let x_cont = -1.9 + (i as f64) * (3.8 / (u - 1) as f64); // distinct across rows
        let raw_binary = if i >= u / 2 { 1.0_f64 } else { 0.0_f64 };
        let centered_binary = raw_binary - p;
        upload_data.push(x_cont);
        upload_data.push(centered_binary);
    }

    // Pre-build the set of valid (x_cont, recovered_binary) source-row pairs
    // so the per-generated-row assertion can look them up.
    let source_pairs: Vec<(f64, f64)> = (0..u)
        .map(|r| {
            let x_cont = upload_data[r * 2];
            let centered = upload_data[r * 2 + 1];
            let recovered = if centered + p >= 0.5 { 1.0 } else { 0.0 };
            (x_cont, recovered)
        })
        .collect();

    // Sanity: binary must not be constant and must track the continuous col,
    // so that an independent binary draw would break some pairs.
    let has_zero = source_pairs.iter().any(|&(_, b)| b == 0.0);
    let has_one = source_pairs.iter().any(|&(_, b)| b == 1.0);
    assert!(has_zero && has_one, "source binary must not be constant");
    // Lower-half rows must have b=0, upper-half b=1 — confirms the correlation.
    assert!(
        source_pairs[0].1 == 0.0 && source_pairs[u - 1].1 == 1.0,
        "binary must track continuous col (lower rows → 0, upper rows → 1)"
    );

    let spec = strict_cont_plus_binary_spec(upload_data, u, p);
    let n = 500;
    // n_pred = 3 (intercept + x_cont + x_bin), n_non_factor = 2, n_factor_dummies = 0.
    let mut ws = SimWorkspace::new(n, 3, 2, 0, None);
    generate_sim_data(&spec, 0, 2137, &mut ws).expect("generate_sim_data failed");

    for i in 0..n {
        let x1 = ws.x_full[(i, 1)]; // continuous
        let b = ws.x_full[(i, 2)]; // recovered binary

        // 1. Domain: recovered value must be 0 or 1.
        assert!(
            b == 0.0 || b == 1.0,
            "row {i}: binary value {b} is not 0.0 or 1.0"
        );

        // 2. Joint: the pair (x1, b) must equal some source-row pair within
        //    1e-12 on x1 (exact f64 copy) and exact on b.
        //    An independent draw of b would produce (high-x1, 0) or (low-x1, 1)
        //    pairs absent from the frame, so this assertion discriminates the copy.
        let matched = source_pairs
            .iter()
            // x1/b are f32 (data plane); source_pairs are f64 (uploaded) — widen.
            .any(|&(sx, sb)| (sx - x1 as f64).abs() < 1e-6 && (sb - b as f64).abs() < 1e-12);
        assert!(
            matched,
            "row {i}: generated (x1={x1}, b={b}) does not match any source-row pair \
             (binary must be copied from the SAME source row as the continuous column, \
             not drawn independently)"
        );
    }
}
