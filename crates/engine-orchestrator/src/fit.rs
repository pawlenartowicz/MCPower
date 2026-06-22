//! Model-based crossing fit for `find_sample_size`.
//!
//! Isotonic (PAVA) fit of one corrected power-vs-N series over the grid, read
//! off at the target power; 95% CI on the required N by Wilson band inversion;
//! probit-in-√N extrapolation hint when the target is never reached in-range.
//!
//! Pure and deterministic over `(grid, counts, n_sims, target, atom)` — no
//! I/O, no RNG — so the merge path recomputes identical values from pooled
//! counts (merge-safe by construction). Fit quality rests on the CRN
//! row-stable prefix-nesting invariant (adjacent grid points share rows, so
//! the series is smooth and near-monotone); see the curve-quality guard tests
//! `rng_rows_stable_across_max_n` and `cluster_draws_precede_row_loop` in
//! engine-core.
//!
//! **Barlow72** = Barlow, Bartholomew, Bremner & Brunk (1972), *Statistical
//! Inference Under Order Restrictions*, Wiley. **Cohen88** = Cohen, J. (1988),
//! *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., Erlbaum.

use crate::aggregation::wilson_ci;
use crate::grid::as_proportion;
use crate::result::CrossingFit;
use engine_core::critvals::norm_ppf;

/// Monotonicity-gate width in SE units. The threshold is the 2-SE band for a
/// difference of *independent* proportions; CRN makes grid points positively
/// correlated, so real MC noise is smaller than the threshold — the gate is
/// deliberately lenient to noise and fires only on real structure (model
/// misspecification, e.g. an interaction flipping sign with N).
const Z_GATE: f64 = 2.0;

/// The extrapolation hint requires the isotonic-fitted power at the last grid
/// point to reach this floor. Below the probit inflection the WLS sits in the
/// noisy flat tail and a numeric hint is false precision — the honest message
/// there is that the search range is the wrong order of magnitude.
const MIN_HINT_POWER: f64 = 0.5;

/// Hint suppressed when the probit crossing exceeds this multiple of the grid
/// ceiling. Catches the failure mode MIN_HINT_POWER doesn't: the curve rose
/// fine but the slope is too shallow for a credible extrapolation.
const EXTRAPOLATION_CAP: f64 = 2.0;

/// Fit one curve: corrected success counts per grid point (per-target counts,
/// or P(>=k) tail sums from the joint histograms), all at the same `n_sims`.
/// `target_power` accepts a percentage like the first-N lookups. `atom` is
/// the cluster atom `n_achievable` / `n_approx` are ceiled to.
pub(crate) fn fit_crossing(
    grid: &[usize],
    counts: &[u64],
    n_sims: u64,
    target_power: f64,
    atom: usize,
) -> CrossingFit {
    debug_assert_eq!(grid.len(), counts.len());
    if n_sims == 0 || grid.is_empty() {
        return CrossingFit::NotReached { n_approx: None };
    }
    let target = as_proportion(target_power);
    let n_f = n_sims as f64;
    let p_hat: Vec<f64> = counts.iter().map(|&k| k as f64 / n_f).collect();

    // 1. Monotonicity gate — fires ⇒ nothing else runs for this curve.
    if let Some(max_violation) = monotonicity_violation(&p_hat, n_f) {
        return CrossingFit::NonMonotone { max_violation };
    }

    // 2. PAVA (unweighted — n_sims is constant across grid points).
    let fitted = pava(&p_hat);

    // 3. Crossing by linear interpolation; leftmost on exact hits/flat blocks.
    if fitted[0] >= target {
        return CrossingFit::AtOrBelowMin { n_min: grid[0] };
    }
    if *fitted.last().unwrap() < target {
        let n_approx = probit_hint(grid, &p_hat, &fitted, n_f, target, atom);
        return CrossingFit::NotReached { n_approx };
    }
    let n_star =
        crossing(grid, &fitted, target).expect("fitted[0] < target <= fitted[last] => crossing");
    let n_achievable = ceil_to_atom(n_star, atom);

    // 4. CI by band inversion: PAVA the per-point Wilson lo/hi series
    // separately; CI on N* = [crossing of the hi band, crossing of the lo
    // band] — the optimistic (hi) band crosses earlier ⇒ lower bound, and
    // vice versa. PAVA is monotone in its input, so hi_band >= fitted >=
    // lo_band pointwise and the bounds bracket n_star.
    let lo_series: Vec<f64> = counts.iter().map(|&k| wilson_ci(k, n_sims).lo).collect();
    let hi_series: Vec<f64> = counts.iter().map(|&k| wilson_ci(k, n_sims).hi).collect();
    let hi_band = pava(&hi_series);
    let lo_band = pava(&lo_series);
    // hi band already >= target at the grid floor ⇒ the optimistic crossing
    // is somewhere below the searched range — unknown, hence None.
    let ci_lo = if hi_band[0] >= target {
        None
    } else {
        crossing(grid, &hi_band, target)
    };
    // lo band never reaching the target in-range ⇒ None (above the ceiling).
    let ci_hi = crossing(grid, &lo_band, target);

    CrossingFit::Fitted {
        n_star,
        n_achievable,
        ci_lo,
        ci_hi,
    }
}

/// Largest decrease `max_{i<j}(p̂ᵢ − p̂ⱼ)` compared against
/// `Z_GATE · √((p̂ᵢ(1−p̂ᵢ) + p̂ⱼ(1−p̂ⱼ))/n_sims)` for the argmax pair.
/// Returns `Some(max_decrease)` when the gate fires.
fn monotonicity_violation(p_hat: &[f64], n_sims: f64) -> Option<f64> {
    let mut max_drop = 0.0_f64;
    let mut drop_pair = None;
    for i in 0..p_hat.len() {
        for j in (i + 1)..p_hat.len() {
            let d = p_hat[i] - p_hat[j];
            if d > max_drop {
                max_drop = d;
                drop_pair = Some((p_hat[i], p_hat[j]));
            }
        }
    }
    let (pi, pj) = drop_pair?;
    let se = ((pi * (1.0 - pi) + pj * (1.0 - pj)) / n_sims).sqrt();
    (max_drop > Z_GATE * se).then_some(max_drop)
}

/// Pool-adjacent-violators (Barlow72): least-squares non-decreasing fit, unweighted.
/// Standard stack-of-blocks pass; ties (equal neighbours) are kept, only
/// strict decreases pool.
fn pava(y: &[f64]) -> Vec<f64> {
    let mut means: Vec<f64> = Vec::with_capacity(y.len());
    let mut lens: Vec<usize> = Vec::with_capacity(y.len());
    for &v in y {
        means.push(v);
        lens.push(1);
        while means.len() >= 2 && means[means.len() - 2] > means[means.len() - 1] {
            let m1 = means.pop().unwrap();
            let l1 = lens.pop().unwrap();
            let k = means.len() - 1;
            let l0 = lens[k];
            means[k] = (means[k] * l0 as f64 + m1 * l1 as f64) / (l0 + l1) as f64;
            lens[k] = l0 + l1;
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (m, l) in means.iter().zip(&lens) {
        out.extend(std::iter::repeat_n(*m, *l));
    }
    out
}

/// First crossing of a non-decreasing series with `target`, by linear
/// interpolation between adjacent grid points. Exact hits and flat blocks at
/// the target resolve leftmost (interpolation lands on the block's first grid
/// point). `None` when the series stays below target; a series already at or
/// above target at the first point returns `grid[0]`.
fn crossing(grid: &[usize], fitted: &[f64], target: f64) -> Option<f64> {
    if fitted[0] >= target {
        return Some(grid[0] as f64);
    }
    for i in 1..fitted.len() {
        if fitted[i] >= target {
            let (x0, x1) = (grid[i - 1] as f64, grid[i] as f64);
            let (y0, y1) = (fitted[i - 1], fitted[i]);
            // y0 < target <= y1 here, so the slope is strictly positive.
            return Some(x0 + (target - y0) * (x1 - x0) / (y1 - y0));
        }
    }
    None
}

/// Ceil a continuous N to the next integer multiple of the cluster atom.
/// The 1e-9 backoff absorbs interpolation float noise — a crossing that is
/// mathematically exactly 125 can arrive as 125.0000000000000x, and a bare
/// ceil would inflate the headline to 126.
fn ceil_to_atom(n: f64, atom: usize) -> usize {
    let a = atom.max(1);
    let n_int = (n - 1e-9).ceil().max(1.0) as usize;
    n_int.div_ceil(a) * a
}

/// Standard-normal density φ(z).
#[inline]
fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Out-of-range extrapolation hint for `NotReached`: the asymptotics-matched
/// probit fit (Cohen88). WLS of `Φ⁻¹(p̂ᵢ)` against `√Nᵢ` with delta-method
/// inverse-variance weights `n_sims·φ(Φ⁻¹(p̂ᵢ))²/(p̂ᵢ(1−p̂ᵢ))` — the
/// probit-scale information, not the logit's `n·p̂(1−p̂)` — inverted at the
/// target. `p̂` is clamped to `[0.5/n_sims, 1 − 0.5/n_sims]` before the
/// transform (and in the weights, which would otherwise divide by zero).
///
/// `None` when suppressed: fitted power at the last grid point below
/// `MIN_HINT_POWER`, non-positive or degenerate WLS slope, or crossing beyond
/// `EXTRAPOLATION_CAP ×` the grid ceiling.
fn probit_hint(
    grid: &[usize],
    p_hat: &[f64],
    fitted: &[f64],
    n_sims: f64,
    target: f64,
    atom: usize,
) -> Option<usize> {
    // Low-power gate on the FITTED endpoint: the last PAVA block pools
    // information, so a raw endpoint noise dip doesn't flunk the gate.
    if *fitted.last()? < MIN_HINT_POWER {
        return None;
    }
    let clamp_lo = 0.5 / n_sims;
    let clamp_hi = 1.0 - 0.5 / n_sims;
    let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, &n) in grid.iter().enumerate() {
        let p = p_hat[i].clamp(clamp_lo, clamp_hi);
        let z = norm_ppf(p);
        let pdf = norm_pdf(z);
        let w = n_sims * pdf * pdf / (p * (1.0 - p));
        let x = (n as f64).sqrt();
        sw += w;
        swx += w * x;
        swy += w * z;
        swxx += w * x * x;
        swxy += w * x * z;
    }
    let denom = sw * swxx - swx * swx;
    if !denom.is_finite() || denom <= 0.0 {
        return None; // degenerate WLS
    }
    let slope = (sw * swxy - swx * swy) / denom;
    if !slope.is_finite() || slope <= 0.0 {
        return None;
    }
    let intercept = (swy - slope * swx) / sw;
    let sqrt_n = (norm_ppf(target) - intercept) / slope;
    if !sqrt_n.is_finite() || sqrt_n <= 0.0 {
        return None;
    }
    let n_cross = sqrt_n * sqrt_n;
    let to = *grid.last()? as f64;
    if n_cross > EXTRAPOLATION_CAP * to {
        return None;
    }
    Some(ceil_to_atom(n_cross, atom))
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} !~ {b} (tol {tol})");
    }

    fn assert_vec_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_close(*x, *y, 1e-12);
        }
    }

    // ── PAVA goldens (hand-computed) ────────────────────────────────────────

    #[test]
    fn pava_identity_on_monotone() {
        assert_vec_close(&pava(&[0.1, 0.2, 0.3]), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn pava_pools_single_violation() {
        assert_vec_close(&pava(&[0.3, 0.1]), &[0.2, 0.2]);
    }

    #[test]
    fn pava_tie_block_pools_then_keeps_equal_neighbour() {
        // 0.5,0.3 pool to 0.4; the following raw 0.4 is NOT pooled (ties stay).
        assert_vec_close(
            &pava(&[0.1, 0.5, 0.3, 0.4, 0.9]),
            &[0.1, 0.4, 0.4, 0.4, 0.9],
        );
    }

    #[test]
    fn pava_cascading_merge() {
        // 0.6,0.5 pool to 0.55; appending 0.4 re-pools all three to 0.5.
        assert_vec_close(
            &pava(&[0.2, 0.6, 0.5, 0.4, 0.8]),
            &[0.2, 0.5, 0.5, 0.5, 0.8],
        );
    }

    #[test]
    fn pava_all_decreasing_pools_to_grand_mean() {
        assert_vec_close(&pava(&[0.9, 0.6, 0.3]), &[0.6, 0.6, 0.6]);
    }

    // ── Crossing + band inversion goldens ───────────────────────────────────
    // Golden values computed externally with the same Wilson (z = 1.96) and
    // Acklam Φ⁻¹ formulas, identical op order.

    const GRID: [usize; 4] = [50, 100, 150, 200];

    #[test]
    fn fitted_crossing_with_two_sided_ci() {
        // p̂ = [0.60, 0.75, 0.85, 0.95], monotone, crosses 0.8 between 100
        // and 150 at exactly 125. CI from PAVA'd Wilson bands.
        let fit = fit_crossing(&GRID, &[60, 75, 85, 95], 100, 0.8, 1);
        match fit {
            CrossingFit::Fitted {
                n_star,
                n_achievable,
                ci_lo,
                ci_hi,
            } => {
                assert_close(n_star, 125.0, 1e-9);
                assert_eq!(n_achievable, 125);
                assert_close(ci_lo.unwrap(), 90.83640209948472, 1e-9);
                assert_close(ci_hi.unwrap(), 163.5595794316606, 1e-9);
            }
            other => panic!("expected Fitted, got {other:?}"),
        }
    }

    #[test]
    fn exact_hit_resolves_leftmost() {
        // p̂ = [0.60, 0.70, 0.80, 0.84]: the fit hits 0.8 exactly at N=150 —
        // leftmost resolution puts n_star on that grid point, not beyond it.
        // The Wilson lo band tops out at lo(84/100) ≈ 0.756 < 0.8 ⇒ ci_hi None.
        let fit = fit_crossing(&GRID, &[60, 70, 80, 84], 100, 0.8, 1);
        match fit {
            CrossingFit::Fitted {
                n_star,
                n_achievable,
                ci_lo,
                ci_hi,
            } => {
                assert_close(n_star, 150.0, 1e-9);
                assert_eq!(n_achievable, 150);
                assert_close(ci_lo.unwrap(), 111.06986495758443, 1e-9);
                assert_eq!(ci_hi, None, "lo band never crosses in-grid");
            }
            other => panic!("expected Fitted, got {other:?}"),
        }
    }

    #[test]
    fn ci_lo_none_when_hi_band_at_or_above_target_at_floor() {
        // p̂ = [0.75, 0.85, 0.95, 0.98]: Wilson hi(75/100) ≈ 0.825 >= 0.8 at
        // the first grid point — the optimistic crossing is below the search
        // floor, so ci_lo is None while the fit itself crosses at 75.
        let fit = fit_crossing(&GRID, &[75, 85, 95, 98], 100, 0.8, 1);
        match fit {
            CrossingFit::Fitted {
                n_star,
                ci_lo,
                ci_hi,
                ..
            } => {
                assert_close(n_star, 75.0, 1e-9);
                assert_eq!(ci_lo, None, "hi band >= target at the floor");
                assert_close(ci_hi.unwrap(), 113.55957943166061, 1e-9);
            }
            other => panic!("expected Fitted, got {other:?}"),
        }
    }

    #[test]
    fn at_or_below_min_when_first_point_reaches_target() {
        let fit = fit_crossing(&GRID, &[85, 90, 95, 98], 100, 0.8, 1);
        assert_eq!(fit, CrossingFit::AtOrBelowMin { n_min: 50 });
    }

    #[test]
    fn n_achievable_ceils_to_cluster_atom() {
        // Same curve as fitted_crossing_with_two_sided_ci: n_star = 125,
        // atom 30 ⇒ 150.
        let fit = fit_crossing(&GRID, &[60, 75, 85, 95], 100, 0.8, 30);
        match fit {
            CrossingFit::Fitted { n_achievable, .. } => assert_eq!(n_achievable, 150),
            other => panic!("expected Fitted, got {other:?}"),
        }
    }

    #[test]
    fn percentage_target_equals_proportion_target() {
        let a = fit_crossing(&GRID, &[60, 75, 85, 95], 100, 80.0, 1);
        let b = fit_crossing(&GRID, &[60, 75, 85, 95], 100, 0.8, 1);
        assert_eq!(a, b);
    }

    // ── Monotonicity gate calibration ───────────────────────────────────────
    // n_sims = 100; threshold for the (0.5, p_j) pair is
    // 2·√((0.25 + p_j(1−p_j))/100): drop 0.10 vs thr 0.140 — under;
    // drop 0.15 vs thr 0.1382 — over.

    #[test]
    fn gate_does_not_fire_just_under_threshold() {
        let fit = fit_crossing(&[50, 100], &[50, 40], 100, 0.9, 1);
        assert!(
            !matches!(fit, CrossingFit::NonMonotone { .. }),
            "drop 0.10 < 2-SE 0.14 must pass the gate, got {fit:?}"
        );
        // PAVA pools the dip to a flat 0.45 < MIN_HINT_POWER ⇒ hint suppressed.
        assert_eq!(fit, CrossingFit::NotReached { n_approx: None });
    }

    #[test]
    fn gate_fires_just_over_threshold() {
        let fit = fit_crossing(&[50, 100], &[50, 35], 100, 0.9, 1);
        match fit {
            CrossingFit::NonMonotone { max_violation } => assert_close(max_violation, 0.15, 1e-12),
            other => panic!("expected NonMonotone, got {other:?}"),
        }
    }

    #[test]
    fn gate_precedes_crossing_and_hint() {
        // The curve recovers and ends above target, but the early collapse
        // (0.5 → 0.2) is structure, not noise — nothing else may run.
        let fit = fit_crossing(&GRID, &[50, 20, 30, 95], 100, 0.8, 1);
        match fit {
            CrossingFit::NonMonotone { max_violation } => assert_close(max_violation, 0.3, 1e-12),
            other => panic!("expected NonMonotone, got {other:?}"),
        }
    }

    // ── Probit extrapolation hint ───────────────────────────────────────────

    #[test]
    fn probit_hint_golden_value() {
        // p̂ = [0.30, 0.45, 0.60, 0.72] never reaches 0.9; WLS probit-in-√N
        // crossing = 351.48 (within the 2× cap), ceiled to 352.
        let fit = fit_crossing(&GRID, &[30, 45, 60, 72], 100, 0.9, 1);
        assert_eq!(
            fit,
            CrossingFit::NotReached {
                n_approx: Some(352)
            }
        );
    }

    #[test]
    fn probit_hint_ceils_to_cluster_atom() {
        let fit = fit_crossing(&GRID, &[30, 45, 60, 72], 100, 0.9, 30);
        assert_eq!(
            fit,
            CrossingFit::NotReached {
                n_approx: Some(360)
            }
        );
    }

    #[test]
    fn probit_hint_suppressed_beyond_extrapolation_cap() {
        // Shallower curve: crossing ≈ 446 > 2 × 200 ⇒ hint suppressed.
        let fit = fit_crossing(&GRID, &[30, 40, 55, 65], 100, 0.9, 1);
        assert_eq!(fit, CrossingFit::NotReached { n_approx: None });
    }

    #[test]
    fn probit_hint_low_power_gate_just_under_vs_just_over() {
        // Fitted endpoint 0.49 < 0.5 ⇒ suppressed; 0.51 ≥ 0.5 ⇒ hint runs
        // (crossing ≈ 262.6, within cap).
        let under = fit_crossing(&GRID, &[20, 30, 40, 49], 100, 0.6, 1);
        assert_eq!(under, CrossingFit::NotReached { n_approx: None });
        let over = fit_crossing(&GRID, &[20, 30, 40, 51], 100, 0.6, 1);
        assert_eq!(
            over,
            CrossingFit::NotReached {
                n_approx: Some(263)
            }
        );
    }

    #[test]
    fn probit_hint_suppressed_on_flat_slope() {
        // All points equal: fitted endpoint 0.6 passes the low-power gate but
        // the WLS slope is exactly 0 ⇒ suppressed.
        let fit = fit_crossing(&GRID, &[60, 60, 60, 60], 100, 0.7, 1);
        assert_eq!(fit, CrossingFit::NotReached { n_approx: None });
    }

    // ── Degenerate inputs ───────────────────────────────────────────────────

    #[test]
    fn zero_sims_yields_not_reached_without_hint() {
        let fit = fit_crossing(&GRID, &[0, 0, 0, 0], 0, 0.8, 1);
        assert_eq!(fit, CrossingFit::NotReached { n_approx: None });
    }
}
