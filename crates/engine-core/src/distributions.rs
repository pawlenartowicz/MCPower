//! Marginal transforms and residual samplers.
//!
//! Standard normals come out of `SimRng::next_normal`. Marginal transforms
//! map a `z ~ N(0, 1)` to the target distribution. Residual samplers (Student-t,
//! etc.) draw fresh from the RNG.
//!
//! # Reproducibility contract
//!
//! The implementations of `phi`, `marginal_uniform`, and `sample_t` — and the
//! inline transform surface in `data_gen.rs` (`apply_marginal`,
//! `draw_residual`, the bootstrap source-row draw) — are part of the
//! reproducibility contract, not internal numerics. Changing the polynomial
//! constants, the algorithm (e.g. switching `phi` to `libm::erfc` for better
//! accuracy), or the RNG-draw structure (e.g. swapping the chi² accumulator
//! in `sample_t`) moves results in every port. Such a change must land in all
//! ports simultaneously and carry a version bump — equal versions must mean
//! equal numbers — never as a silent "improvement". The golden tests in
//! `tests/golden_rng.rs` and the `data_gen.rs` golden unit tests fail loudly
//! on any such change.
//!
//! `phi`'s `erfc` composes the owned engine `exp` kernel
//! (`glmm::simd_transcendental::exp_nonpos`, ≤1 ULP of libm) rather than libm
//! `.exp()`; the A&S polynomial and constants are unchanged.

use crate::rng::SimRng;

/// Standard-normal CDF Φ(z). Uses `erfc` from Abramowitz–Stegun 7.1.26 / 26.2.17,
/// accurate to ~1.5e-7. Sufficient for marginal transforms; the resulting variable
/// is approximate by construction (Gaussian-copula CDF inversion is itself an
/// approximation for non-Gaussian marginals).
#[inline]
pub fn phi(z: f64) -> f64 {
    0.5 * erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Complementary error function. Abramowitz–Stegun 7.1.26 polynomial; max error
/// ~1.5e-7 over the real line.
fn erfc(x: f64) -> f64 {
    // erf(x) ≈ 1 - (a1 t + a2 t^2 + ... + a5 t^5) exp(-x²), t = 1 / (1 + p|x|)
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + P * abs_x);
    let y = 1.0
        - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1)
            * t
            * glmm::simd_transcendental::exp_nonpos(-abs_x * abs_x);
    let erf_x = sign * y;
    1.0 - erf_x
}

// ---------------------------------------------------------------------------
// Marginal transforms (post-Cholesky z → target distribution)
// ---------------------------------------------------------------------------

/// Uniform(a, b) via Gaussian-copula CDF inversion.
#[inline]
pub fn marginal_uniform(z: f64, a: f64, b: f64) -> f64 {
    a + (b - a) * phi(z)
}

// ---------------------------------------------------------------------------
// Residual samplers
// ---------------------------------------------------------------------------

/// Student-t sample with `df` degrees of freedom.
///
/// Builds t = Z / sqrt(χ²_df / df) where χ²_df is a sum of `df` independent
/// standard-normal squares (when df is integral). For non-integer `df`, falls
/// back to the same formula with `df.round()` squared normals — integer df is
/// the common case in practice, so the non-integer path is best-effort.
pub fn sample_t(rng: &mut SimRng, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 {
        // Degenerate — fall back to standard normal (f32 draw widened to f64).
        return rng.next_normal() as f64;
    }
    let z = rng.next_normal() as f64;
    let df_int = df.round() as i64;
    // Sum of df squared standard normals — chi-squared(df) when df is integer.
    let mut chi2 = 0.0;
    for _ in 0..df_int.max(1) {
        let g = rng.next_normal() as f64;
        chi2 += g * g;
    }
    let denom = (chi2 / df).sqrt();
    if denom <= 0.0 {
        z
    } else {
        z / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_basic_values() {
        // Φ(0) = 0.5 exactly.
        assert!((phi(0.0) - 0.5).abs() < 1e-7);
        // Φ(±∞) saturates.
        assert!(phi(8.0) > 1.0 - 1e-7);
        assert!(phi(-8.0) < 1e-7);
        // Symmetry: Φ(z) + Φ(-z) = 1.
        for z in [-2.5_f64, -1.0, 0.5, 1.7, 3.2] {
            let s = phi(z) + phi(-z);
            assert!((s - 1.0).abs() < 1e-6, "phi symmetry at z={z}: sum={s}");
        }
    }

    #[test]
    fn phi_monotone_non_decreasing() {
        // Φ is monotonically non-decreasing across the real line.
        let mut prev = phi(-10.0);
        let mut z = -10.0;
        while z <= 10.0 {
            let cur = phi(z);
            assert!(cur >= prev, "phi not monotone at z={z}: {cur} < {prev}");
            prev = cur;
            z += 0.1;
        }
    }

    #[test]
    fn marginal_uniform_in_range_and_monotone() {
        // marginal_uniform(z,a,b) lies in [a,b] for any finite z (Φ ∈ (0,1))
        // and is monotonically non-decreasing in z — deterministic properties
        // of the copula transform, not statistical content.
        let (a, b) = (-1.0, 3.0);
        let mut prev = marginal_uniform(-12.0, a, b);
        let mut z = -12.0;
        while z <= 12.0 {
            let u = marginal_uniform(z, a, b);
            assert!(u >= a && u <= b, "uniform {u} outside [{a},{b}] at z={z}");
            assert!(
                u >= prev,
                "marginal_uniform not monotone at z={z}: {u} < {prev}"
            );
            prev = u;
            z += 0.1;
        }
    }

    #[test]
    fn sample_t_degenerate_df_safe() {
        let mut rng = SimRng::new(1, 1);
        let x = sample_t(&mut rng, 0.0);
        assert!(x.is_finite());
        let y = sample_t(&mut rng, f64::NAN);
        assert!(y.is_finite());
    }

    // -----------------------------------------------------------------
    // C3 — sample_t moments match analytic Student-t theory
    // -----------------------------------------------------------------

    #[test]
    fn sample_t_moments_match_theory() {
        // Analytic: E[t(df)] = 0 (df > 1), Var[t(df)] = df/(df-2) (df > 2).
        // N = 50_000 draws per df; fixed seed → deterministic.
        // Mutation guard: chi2 / (df_int * df_int) instead of chi2 / df inflates
        // variance by factor df → measured Var ≈ df * expected_var → caught.
        let n = 50_000usize;
        for &df in &[5.0_f64, 10.0, 30.0] {
            let mut rng = SimRng::new(42, 0);
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for _ in 0..n {
                let x = sample_t(&mut rng, df);
                sum += x;
                sum_sq += x * x;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            let expected_var = df / (df - 2.0);
            assert!(
                mean.abs() < 0.02,
                "sample_t df={df}: mean={mean}, expected ~0"
            );
            assert!(
                (var - expected_var).abs() < 0.05,
                "sample_t df={df}: var={var}, expected {expected_var}"
            );
        }
    }

    #[test]
    fn phi_owned_exp_tracks_libm_formula() {
        // Reference: the pre-owned-exp erfc composition (libm .exp()), inlined.
        // The owned exp is ≤1 ULP of libm, so the composed phi must stay within
        // one rounding quantum of the old values — this is the regression net for
        // the swap. The right unit is ABSOLUTE difference, not relative ULP of
        // phi: `y = 1 − poly·t·e` quantizes the output at ULP(1.0) = 2⁻⁵², so a
        // sub-ULP shift in `e` can flip y by one quantum, which after the `1 − y`
        // re-cancellation is worth many relative ULP wherever erfc ≪ 1 (measured
        // 2026-06-11: exp ≤1 ULP, |Δphi| ≤ 2⁻⁵³ exactly, peaking at 16 relative
        // ULP near z = −2.12 — one y-stage quantum, the minimum nonzero movement
        // this composition can express).
        fn phi_libm(z: f64) -> f64 {
            const A1: f64 = 0.254829592;
            const A2: f64 = -0.284496736;
            const A3: f64 = 1.421413741;
            const A4: f64 = -1.453152027;
            const A5: f64 = 1.061405429;
            const P: f64 = 0.3275911;
            let x = -z * std::f64::consts::FRAC_1_SQRT_2;
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let abs_x = x.abs();
            let t = 1.0 / (1.0 + P * abs_x);
            let y =
                1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-abs_x * abs_x).exp();
            0.5 * (1.0 - sign * y)
        }
        let n = 20_003usize;
        for k in 0..n {
            let z = -9.0 + 18.0 * k as f64 / n as f64;
            let x = -z * std::f64::consts::FRAC_1_SQRT_2;
            let arg = -x.abs() * x.abs();
            let e_owned = glmm::simd_transcendental::exp_nonpos(arg);
            let e_libm = arg.exp();
            let e_ulp = ((e_owned.to_bits() as i64) - (e_libm.to_bits() as i64)).abs();
            assert!(e_ulp <= 2, "owned exp({arg}) drifted {e_ulp} ULP from libm");
            let (a, b) = (phi(z), phi_libm(z));
            let d = (a - b).abs();
            assert!(
                d <= 2.0f64.powi(-52),
                "phi({z}) moved {d:e} off the libm composition (> one y-stage quantum)"
            );
        }
    }

    // -----------------------------------------------------------------
    // C10 — phi golden values at critical thresholds (external oracle: A&S Φ)
    // -----------------------------------------------------------------

    #[test]
    fn phi_golden_values_at_critical_quantiles() {
        // Φ at the two critical z-thresholds used in all CI/power calculations.
        // Tolerance 1e-5 (A&S 7.1.26 max error ~1.5e-7; 1e-5 catches mis-coded polynomial).
        // A sign flip or coefficient error in A&S shifts phi(1.96) by ~0.001–0.01.
        assert!(
            (phi(1.959963985) - 0.975).abs() < 1e-5,
            "phi(1.96) = {}",
            phi(1.959963985)
        );
        assert!(
            (phi(1.644853627) - 0.95).abs() < 1e-5,
            "phi(1.645) = {}",
            phi(1.644853627)
        );
        assert!(
            (phi(-1.959963985) - 0.025).abs() < 1e-5,
            "phi(-1.96) = {}",
            phi(-1.959963985)
        );
    }
}
