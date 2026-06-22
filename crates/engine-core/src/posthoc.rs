//! Posthoc contrast evaluation in t² space.
//!
//! Each contrast `c` produces an uncorrected/corrected pass/fail. Variance is
//! computed via `seᶜ² = σ̂² · ‖L⁻¹ c‖²` — forward solve on the lower-triangular
//! Cholesky factor `L` of `X'X` produced by `fit_suff_stats_t_sq`. Since
//! `(X'X)⁻¹ = L⁻ᵀ L⁻¹`, `cᵀ(X'X)⁻¹c = (L⁻¹c)'(L⁻¹c) = ‖L⁻¹c‖²` — same identity
//! the old QR path used with `R' = L`. No `(X'X)⁻¹` materialization. Same
//! method as the original OLS contrast path, modulo the t² rewrite.

use crate::correction::apply_correction;
use crate::ols::OlsFitView;
use crate::spec::CorrectionMethod;
use faer::MatRef;

use crate::FLOAT_NEAR_ZERO;

/// A single contrast `cᵀβ`. The vector is dense of length `P` so the engine
/// stays family-agnostic; sparse forms can be packed at the frontend.
pub type Contrast = Vec<f64>;

/// Evaluate a set of posthoc contrasts against an OLS fit.
///
/// - `fit`: borrowed view from `fit_suff_stats_t_sq`.
/// - `contrasts`: list of length-`P` contrast vectors.
/// - `posthoc_t_crit_sq`: scalar squared posthoc-α critical value (uncorrected).
/// - `correction_method`: typed enum applied over the contrast set
///   (independent of the main target correction).
/// - `correction_crit_sq`: per-rank thresholds for Bonferroni/Holm/BH; length `C`.
/// - `t_sq_scratch`: caller-owned per-contrast `t²` staging buffer. Grown to
///   `C` on first use, reused thereafter (the contrast count is not known at
///   workspace construction, so this is a `Vec` rather than a fixed slice).
/// - `u_scratch`: caller-owned forward-solve buffer, length `≥ P`. Shares the
///   role of the OLS-contrast forward-solve scratch (`lme_u_scratch`).
/// - `unc_out`: pass/fail vector of length `C` (uncorrected).
/// - `cor_out`: pass/fail vector of length `C` (corrected).
#[expect(
    clippy::too_many_arguments,
    reason = "hot-loop evaluator; each arg is a distinct caller-owned buffer or threshold"
)]
pub fn evaluate_posthoc(
    fit: &OlsFitView<'_>,
    contrasts: &[Contrast],
    posthoc_t_crit_sq: f64,
    correction_method: CorrectionMethod,
    correction_crit_sq: &[f64],
    t_sq_scratch: &mut Vec<f64>,
    u_scratch: &mut [f64],
    unc_out: &mut [u8],
    cor_out: &mut [u8],
) {
    let c_count = contrasts.len();
    debug_assert_eq!(unc_out.len(), c_count);
    debug_assert_eq!(cor_out.len(), c_count);

    for v in unc_out.iter_mut() {
        *v = 0;
    }
    for v in cor_out.iter_mut() {
        *v = 0;
    }

    if !fit.converged {
        return;
    }

    let p = fit.betas.len();
    let r = fit.factor;
    debug_assert!(
        p <= u_scratch.len(),
        "u_scratch must hold at least p entries"
    );
    // Per-contrast t² staging in reused scratch: grow to the contrast count on
    // first use (unknown at workspace construction), then reuse. NaN means "not
    // computed" — overwritten only for well-formed, finite contrasts.
    if t_sq_scratch.len() < c_count {
        t_sq_scratch.resize(c_count, f64::NAN);
    }
    let t_sq_all = &mut t_sq_scratch[..c_count];
    t_sq_all.fill(f64::NAN);

    for (ci, c) in contrasts.iter().enumerate() {
        if c.len() != p {
            // Skip malformed contrasts.
            continue;
        }
        // contrast_est = c · β̂
        let mut est = 0.0;
        for (j, &cj) in c.iter().enumerate().take(p) {
            est += cj * fit.betas[j];
        }
        // Forward solve L · u = c on the lower-triangular Cholesky factor.
        let nq_sq = forward_solve_l_norm_sq(r, c, &mut u_scratch[..p]);
        if !nq_sq.is_finite() {
            continue;
        }
        let se_sq = fit.sigma_sq * nq_sq;
        if se_sq > FLOAT_NEAR_ZERO && se_sq.is_finite() {
            t_sq_all[ci] = est * est / se_sq;
        }
    }

    // Uncorrected pass/fail.
    for ci in 0..c_count {
        if !t_sq_all[ci].is_nan() && t_sq_all[ci] > posthoc_t_crit_sq {
            unc_out[ci] = 1;
        }
    }

    // Corrected — delegate to `apply_correction` over the contrast set.
    apply_correction(
        correction_method,
        t_sq_all,
        posthoc_t_crit_sq,
        correction_crit_sq,
        cor_out,
    );
}

/// Solve `L u = b` by forward substitution on the lower-triangular Cholesky
/// factor `L`, then return `‖u‖²`. Writes the solve result into `u` (length `P`).
///
/// `L[i, j]` is non-zero only for `i ≥ j` (strict upper triangle is zeroed by
/// `Cholesky::compute_l` and the workspace copy in `fit_suff_stats_t_sq`).
/// Forward solve:
///   for i = 0..p: u[i] = (b[i] - Σ_{k<i} L[i, k] · u[k]) / L[i, i]
fn forward_solve_l_norm_sq(l: MatRef<'_, f64>, b: &[f64], u: &mut [f64]) -> f64 {
    let p = b.len();
    debug_assert_eq!(u.len(), p);
    for u_i in u.iter_mut() {
        *u_i = 0.0;
    }
    for i in 0..p {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[(i, k)] * u[k];
        }
        let l_ii = l[(i, i)];
        if l_ii.abs() < FLOAT_NEAR_ZERO {
            return f64::NAN;
        }
        u[i] = acc / l_ii;
    }
    let mut s = 0.0;
    for &v in u.iter() {
        s += v * v;
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ols::{fit_suff_stats_t_sq, OlsScratch, OlsSuffStats};
    use crate::rng::SimRng;
    use crate::workspace::SimWorkspace;
    use faer::Mat;

    /// EST-29: posthoc pass/fail is monotone in the uncorrected threshold —
    /// lowering the threshold can only turn failures into passes, never the
    /// reverse. So the pass set at a low threshold is a superset of the pass
    /// set at a high threshold (per-target, t² > crit_sq comparison). No
    /// contrast t² value is pinned — only the ordering invariant.
    #[test]
    fn posthoc_pass_monotone_in_threshold() {
        let n = 200;
        let p = 4; // intercept, x_cont, dummy_a, dummy_b
        let mut rng = SimRng::new(42, 1);
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = rng.next_normal();
            let lvl = i % 3;
            x[(i, 2)] = if lvl == 1 { 1.0 } else { 0.0 };
            x[(i, 3)] = if lvl == 2 { 1.0 } else { 0.0 };
        }
        let true_beta = [0.0f32, 0.5, 0.4, -0.3];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..p {
                s += x[(i, j)] * true_beta[j];
            }
            y[i] = s + rng.next_normal() * 0.5;
        }

        let mut ws = SimWorkspace::new(n, p, 1, 1, None);
        ws.reset_suff_stats();
        {
            let mut s = OlsSuffStats {
                xtx: ws.suff_xtx.as_mut(),
                xty: &mut ws.suff_xty,
                yty: &mut ws.suff_yty,
                sum_y: &mut ws.suff_sum_y,
                n_rows: &mut ws.suff_n_rows,
                panel_x: &mut ws.panel_x,
                panel_y: &mut ws.panel_y,
            };
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let ols = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[2, 3],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(ols.converged);

        // c1 = dummy_a (e_2); c2 = dummy_a − dummy_b (e_2 − e_3).
        let contrasts = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, -1.0]];

        let mut t_sq_scratch: Vec<f64> = Vec::new();
        let mut u_scratch = vec![0.0_f64; p];

        // Very high threshold → expect (almost) no passes; very low → all pass.
        let mut high_unc = vec![0u8; 2];
        let mut high_cor = vec![0u8; 2];
        evaluate_posthoc(
            &ols,
            &contrasts,
            1e12,
            CorrectionMethod::None,
            &[],
            &mut t_sq_scratch,
            &mut u_scratch,
            &mut high_unc,
            &mut high_cor,
        );

        let mut low_unc = vec![0u8; 2];
        let mut low_cor = vec![0u8; 2];
        evaluate_posthoc(
            &ols,
            &contrasts,
            0.0,
            CorrectionMethod::None,
            &[],
            &mut t_sq_scratch,
            &mut u_scratch,
            &mut low_unc,
            &mut low_cor,
        );

        // Monotonicity: a target passing at the high threshold must also pass
        // at the low threshold (superset relation).
        for t in 0..2 {
            assert!(
                low_unc[t] >= high_unc[t],
                "target {t}: lowering the threshold must not flip a pass to a fail"
            );
        }
        // At threshold 0 every (finite, positive) t² passes.
        assert_eq!(low_unc, vec![1, 1], "all contrasts pass at threshold 0");
        // At an astronomically high threshold none pass.
        assert_eq!(high_unc, vec![0, 0], "no contrast passes at threshold 1e12");
    }

    /// Posthoc contrast **golden** on a 3-level factor + continuous covariate,
    /// pinned against an external R `lm` oracle (scripts/posthoc_contrast_golden.R).
    /// The 12-row fixture has reference level A (rows 0–3), level B (4–7), level
    /// C (8–11). For each contrast `c` the engine's squared statistic is
    /// `t² = (c·β̂)² / (σ̂²·‖L⁻¹c‖²)` with σ̂² = RSS/(n−p) — identical to R's OLS
    /// contrast t². The crit-boundary sandwich pins each t² to ±1e-5 relative;
    /// a separate Bonferroni call pins the corrected pass/fail pattern and the
    /// malformed-contrast skip — the corrected path the monotone test leaves
    /// untraced.
    #[test]
    fn posthoc_contrast_golden_three_level_factor() {
        // Fixture + goldens generated by scripts/posthoc_contrast_golden.R.
        let x_cont = [
            -0.7750878248360481,
            -1.2252006806327926,
            -0.709_364_866_703_497_7,
            -0.018958944525548808,
            0.2101379587463299,
            2.2908625664918474,
            -0.35052443764750807,
            -0.863_563_430_859_728,
            0.892_631_781_054_136_3,
            1.4290259187331522,
            0.656_847_804_989_677_3,
            0.497_520_568_651_754_5,
        ];
        let y = [
            1.1802869053960685,
            -0.963_947_822_634_246,
            0.959_889_845_990_292_8,
            0.877_905_580_256_414_4,
            3.022_241_926_672_648,
            3.4920105259572303,
            2.3866765489473383,
            2.2500422884006044,
            2.1575189233134733,
            2.0193728119274468,
            1.9796161718742813,
            0.961_724_031_083_998_3,
        ];
        // Golden squared t-statistics from R `lm` (column order: 1, x, dB, dC).
        let tsq_b: f64 = 12.813078660741; // level B vs A
        let tsq_c: f64 = 0.667657954577; // level C vs A
        let tsq_b_minus_c: f64 = 8.296548322263; // level B vs C

        let n = 12;
        let p = 4; // intercept, x_cont, dummy_B, dummy_C
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x_cont[i] as f32;
            x[(i, 2)] = if (4..8).contains(&i) { 1.0 } else { 0.0 }; // level B
            x[(i, 3)] = if (8..12).contains(&i) { 1.0 } else { 0.0 }; // level C
        }
        let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();

        let mut ws = SimWorkspace::new(n, p, 1, 1, None);
        ws.reset_suff_stats();
        {
            let mut s = OlsSuffStats {
                xtx: ws.suff_xtx.as_mut(),
                xty: &mut ws.suff_xty,
                yty: &mut ws.suff_yty,
                sum_y: &mut ws.suff_sum_y,
                n_rows: &mut ws.suff_n_rows,
                panel_x: &mut ws.panel_x,
                panel_y: &mut ws.panel_y,
            };
            s.add_rows(x.as_ref(), &y_f32);
        }
        let ols = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[2, 3],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            OlsScratch {
                fit_betas: &mut ws.fit_betas,
                fit_var_diag: &mut ws.fit_var_diag,
                fit_t_sq: &mut ws.fit_t_sq,
                fit_u_scratch: &mut ws.fit_u_scratch,
                fit_factor: ws.fit_factor.as_mut(),
                fit_rhs: ws.fit_rhs.as_mut(),
            },
        );
        assert!(ols.converged);

        let c_b = vec![0.0, 0.0, 1.0, 0.0];
        let c_c = vec![0.0, 0.0, 0.0, 1.0];
        let c_b_minus_c = vec![0.0, 0.0, 1.0, -1.0];

        let mut t_sq_scratch: Vec<f64> = Vec::new();
        let mut u_scratch = vec![0.0_f64; p];

        // (1) Crit-boundary sandwich: pin each contrast's t² to ±1e-5 relative.
        //     `t² > crit` ⇒ pass, so a crit just below the golden must pass and
        //     a crit just above must fail. Engine↔R agreement is ~1e-11.
        for (contrast, golden) in [(&c_b, tsq_b), (&c_c, tsq_c), (&c_b_minus_c, tsq_b_minus_c)] {
            let list = vec![contrast.clone()];
            let lo = golden * (1.0 - 1e-5);
            let hi = golden * (1.0 + 1e-5);
            let mut unc = vec![0u8; 1];
            let mut cor = vec![0u8; 1];
            evaluate_posthoc(
                &ols,
                &list,
                lo,
                CorrectionMethod::None,
                &[],
                &mut t_sq_scratch,
                &mut u_scratch,
                &mut unc,
                &mut cor,
            );
            assert_eq!(unc, vec![1], "t² must exceed golden·(1−1e-5) = {lo}");
            evaluate_posthoc(
                &ols,
                &list,
                hi,
                CorrectionMethod::None,
                &[],
                &mut t_sq_scratch,
                &mut u_scratch,
                &mut unc,
                &mut cor,
            );
            assert_eq!(unc, vec![0], "t² must fall below golden·(1+1e-5) = {hi}");
        }

        // (2) Bonferroni corrected path + malformed-contrast skip in one call.
        //     crit 4.0 sits between tsq_c (0.67) and tsq_b_minus_c (8.30): B and
        //     B−C pass, C fails; the 4th contrast has length 3 ≠ p, so it is
        //     skipped and stays 0.
        let malformed = vec![0.0, 0.0, 1.0];
        let contrasts = vec![c_b.clone(), c_c.clone(), c_b_minus_c.clone(), malformed];
        let crit = 4.0;
        let crit_sq = vec![crit; contrasts.len()];
        let mut unc = vec![0u8; contrasts.len()];
        let mut cor = vec![0u8; contrasts.len()];
        evaluate_posthoc(
            &ols,
            &contrasts,
            crit,
            CorrectionMethod::Bonferroni,
            &crit_sq,
            &mut t_sq_scratch,
            &mut u_scratch,
            &mut unc,
            &mut cor,
        );
        assert_eq!(
            unc,
            vec![1, 0, 1, 0],
            "uncorrected pass pattern at crit=4.0 (malformed skipped)"
        );
        assert_eq!(
            cor,
            vec![1, 0, 1, 0],
            "Bonferroni corrected pass pattern at crit=4.0"
        );
    }

    #[test]
    fn d4_non_converged_ols_produces_all_zero() {
        let p = 3;
        // Drive a rank-deficient fit to produce a non-converged view from real
        // scratch, since the view borrows workspace storage and can't be hand-built.
        let n = 50;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) * 0.1;
            // Zero column → exactly-zero Cholesky pivot, robustly rank-deficient
            // in f32 (a duplicate column leaves a ~1e-7 roundoff pivot that f32
            // generation tips above eps_rank — the f32 LLT grey zone).
            x[(i, 2)] = 0.0;
        }
        let y: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut ws = SimWorkspace::new(n, p, 0, 0, None);
        ws.reset_suff_stats();
        {
            let mut s = OlsSuffStats {
                xtx: ws.suff_xtx.as_mut(),
                xty: &mut ws.suff_xty,
                yty: &mut ws.suff_yty,
                sum_y: &mut ws.suff_sum_y,
                n_rows: &mut ws.suff_n_rows,
                panel_x: &mut ws.panel_x,
                panel_y: &mut ws.panel_y,
            };
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let ols = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[1, 2],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(!ols.converged);

        let contrasts = vec![vec![1.0, 0.0, 0.0]];
        let mut unc_out = vec![0u8; 1];
        let mut cor_out = vec![0u8; 1];
        let mut t_sq_scratch: Vec<f64> = Vec::new();
        let mut u_scratch = vec![0.0_f64; p];
        evaluate_posthoc(
            &ols,
            &contrasts,
            1.0,
            CorrectionMethod::None,
            &[],
            &mut t_sq_scratch,
            &mut u_scratch,
            &mut unc_out,
            &mut cor_out,
        );
        assert_eq!(unc_out, vec![0]);
        assert_eq!(cor_out, vec![0]);
    }

    /// Warm-path allocation guard for `evaluate_posthoc` (Phase 2, item 6b).
    /// With the per-contrast `t²` staging buffer and the forward-solve `u`
    /// buffer both caller-owned (workspace scratch in production), a converged
    /// posthoc evaluation allocates **nothing** per call — the two per-call
    /// `Vec`s this hoist removed were the only heap work. Uses Bonferroni so
    /// `apply_correction` stays on its non-sorting arm (Holm / BH allocate a
    /// sort-index `Vec` — that is item 5, intentionally out of scope here). If
    /// a future change reintroduces a per-call allocation, raise the bound only
    /// after confirming the allocation is unavoidable — do not relax silently.
    ///
    /// `#[ignore]` because `dhat::Profiler` measures process-wide allocations
    /// and other concurrent tests in the same binary contaminate the count.
    /// Run explicitly with:
    ///   `cargo test -p engine-core posthoc_warm_path_bounded_alloc -- --ignored --test-threads=1`
    #[test]
    #[ignore]
    #[expect(
        clippy::absurd_extreme_comparisons,
        reason = "BOUND is currently 0; `<=` keeps raising the bound a one-line change"
    )]
    fn posthoc_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND: u64 = 0;

        // Converged OLS fit via the suff-stats path (factor holds L, which
        // posthoc consumes) on a small two-dummy factor design.
        let n = 200;
        let p = 4; // intercept, x_cont, dummy_a, dummy_b
        let mut rng = SimRng::new(7, 1);
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = rng.next_normal();
            let lvl = i % 3;
            x[(i, 2)] = if lvl == 1 { 1.0 } else { 0.0 };
            x[(i, 3)] = if lvl == 2 { 1.0 } else { 0.0 };
        }
        let true_beta = [0.0f32, 0.5, 0.4, -0.3];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..p {
                s += x[(i, j)] * true_beta[j];
            }
            y[i] = s + rng.next_normal() * 0.5;
        }

        let mut ws = SimWorkspace::new(n, p, 1, 1, None);
        ws.reset_suff_stats();
        {
            let mut s = OlsSuffStats {
                xtx: ws.suff_xtx.as_mut(),
                xty: &mut ws.suff_xty,
                yty: &mut ws.suff_yty,
                sum_y: &mut ws.suff_sum_y,
                n_rows: &mut ws.suff_n_rows,
                panel_x: &mut ws.panel_x,
                panel_y: &mut ws.panel_y,
            };
            s.add_rows(x.as_ref(), &y);
        }
        let ols = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[2, 3],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            OlsScratch {
                fit_betas: &mut ws.fit_betas,
                fit_var_diag: &mut ws.fit_var_diag,
                fit_t_sq: &mut ws.fit_t_sq,
                fit_u_scratch: &mut ws.fit_u_scratch,
                fit_factor: ws.fit_factor.as_mut(),
                fit_rhs: ws.fit_rhs.as_mut(),
            },
        );
        assert!(ols.converged);

        let contrasts = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, -1.0],
        ];
        let crit_sq = vec![ols.sigma_sq.max(1.0); contrasts.len()];
        let mut unc = vec![0u8; contrasts.len()];
        let mut cor = vec![0u8; contrasts.len()];
        let mut t_sq_scratch: Vec<f64> = Vec::new();
        let mut u_scratch = vec![0.0_f64; p];

        // Warmup drives the one-time `t_sq_scratch` grow.
        evaluate_posthoc(
            &ols,
            &contrasts,
            1.0,
            CorrectionMethod::Bonferroni,
            &crit_sq,
            &mut t_sq_scratch,
            &mut u_scratch,
            &mut unc,
            &mut cor,
        );

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            evaluate_posthoc(
                &ols,
                &contrasts,
                1.0,
                CorrectionMethod::Bonferroni,
                &crit_sq,
                &mut t_sq_scratch,
                &mut u_scratch,
                &mut unc,
                &mut cor,
            );
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "evaluate_posthoc allocated {} blocks across {} warm-path calls (expected ≤ {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }
}
