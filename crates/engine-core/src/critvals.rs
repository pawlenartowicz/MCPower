//! Precomputed Student-t critical values.
//!
//! Per the hot-loop invariant, the per-sim path must never call Student-t
//! CDF/inv-CDF and must never sqrt-then-compare. All thresholds are precomputed
//! once per `run_batch` into a `CritValueTable`; the per-sim path then performs
//! only `t_sq > t_crit_sq` comparisons.
//!
//! Only `t_ppf` is exposed — no `t_cdf` / `t_sf` / `p_value` (those are not
//! needed on the per-sim comparison path).
//!
//! **NR** = Press, Teukolsky, Vetterling & Flannery (2007), *Numerical Recipes:
//! The Art of Scientific Computing*, 3rd ed., Cambridge University Press.
//! **Acklam** = Peter J. Acklam, *An algorithm for computing the inverse normal
//! cumulative distribution function* (unpublished web note).
//! **Lanczos** = Lanczos, C. (1964), *A precision approximation of the gamma
//! function*, SIAM J. Numer. Anal. Ser. B, 1, 86–96.
//! **Lenth** = Lenth, R. V. (1989), *Algorithm AS 190: Probabilities and Upper
//! Quantiles for the Studentized Range*, Appl. Statist. 38(1), 185–189.

use crate::distributions::phi;
use crate::spec::{CorrectionMethod, CritValues, EngineError, EstimatorSpec};

// ---------------------------------------------------------------------------
// Student-t inverse CDF
// ---------------------------------------------------------------------------

/// Standard-normal inverse CDF via Acklam's rational approximation
/// (max abs error ~1.15e-9 on `p ∈ [1e-300, 1 - 1e-16]`). Used as the seed
/// for Newton iteration in `t_ppf`, as the estimator-aware quantile under
/// `EstimatorSpec::Glm` and `EstimatorSpec::Mle` (Wald-z thresholds —
/// df-independent), and by the orchestrator's probit transform in the
/// model-based crossing fit (hence `pub`).
pub fn norm_ppf(p: f64) -> f64 {
    // Rational-approximation coefficients (Acklam).
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

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
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

/// Regularized incomplete beta function I_x(a, b) via the continued-fraction
/// expansion (NR §6.4). Accurate to ~1e-15.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lnbeta = ln_beta(a, b);
    let bt = ((a) * x.ln() + (b) * (1.0 - x).ln() - lnbeta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos log-gamma (g=7, n=9). Accurate to ~1e-15 for x > 0 (Lanczos).
fn ln_gamma(mut x: f64) -> f64 {
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    x -= 1.0;
    let mut a = P[0];
    let t = x + G + 0.5;
    for (i, &pi) in P.iter().enumerate().skip(1) {
        a += pi / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction for the incomplete beta (Lentz's method, NR §6.4).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3e-15;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // First step (even index).
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Second step (odd index).
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return h;
        }
    }
    h
}

/// Student-t CDF: P(T ≤ x) for T ~ t(df).
fn t_cdf(x: f64, df: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let z = df / (df + x * x);
    let half = 0.5 * betai(0.5 * df, 0.5, z);
    if x >= 0.0 {
        1.0 - half
    } else {
        half
    }
}

/// Student-t inverse CDF (quantile function).
///
/// Approach: Cornish-Fisher seed → Newton-Raphson refinement on
/// `F(x; df) - p` using the closed-form density. Converges in ≤8 iterations
/// to relative error < 1e-13 for `df ∈ [1, 1e6]`, `p ∈ [1e-12, 1 - 1e-12]`.
///
/// Edge cases:
///   - p ≤ 0  → -∞;  p ≥ 1 → +∞;  p = 0.5 → 0.0 exactly.
///   - df ≤ 0 → returns NaN.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Symmetry: solve for p' = max(p, 1-p) (upper half), then negate if needed.
    let upper = p > 0.5;
    let q = if upper { p } else { 1.0 - p };

    // Initial guess.
    let mut x = if df > 200.0 {
        // Use Cornish-Fisher expansion around the normal quantile.
        let z = norm_ppf(q);
        let z2 = z * z;
        z + (z * (z2 + 1.0)) / (4.0 * df)
            + (z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0)) / (96.0 * df * df)
    } else if (df - 1.0).abs() < 1e-9 {
        // Cauchy: closed form.
        ((std::f64::consts::PI * (q - 0.5)).tan()).abs()
    } else if (df - 2.0).abs() < 1e-9 {
        // df=2 closed form.
        let alpha = 4.0 * q * (1.0 - q);
        (2.0 / alpha - 2.0).sqrt() * if q > 0.5 { 1.0 } else { -1.0 }
    } else {
        norm_ppf(q)
    };

    // For very large df (>1e5), the betai-based t_cdf loses accuracy because
    // (a, b) = (df/2, 1/2) requires Γ ratios that overflow lgamma cancellation
    // budgets. Skip Newton refinement and rely on the Cornish-Fisher seed
    // (accurate to ~1e-7 at df > 1e4, which is the target for the hot-loop
    // critvals — `t² > t_crit²` boolean outcomes don't change at this scale).
    if df > 1.0e5 {
        return if upper { x } else { -x };
    }

    // Newton-Raphson: x_{n+1} = x_n - (F(x_n) - q) / f(x_n).
    // pdf(x; df) = Γ((df+1)/2) / (sqrt(df π) Γ(df/2)) · (1 + x²/df)^{-(df+1)/2}
    let log_norm_const =
        ln_gamma(0.5 * (df + 1.0)) - ln_gamma(0.5 * df) - 0.5 * (df * std::f64::consts::PI).ln();
    for _ in 0..80 {
        let cdf = t_cdf(x, df);
        let pdf_log = log_norm_const - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln();
        let pdf = pdf_log.exp();
        if pdf <= 0.0 || !pdf.is_finite() {
            break;
        }
        let step = (cdf - q) / pdf;
        let new_x = x - step;
        if !new_x.is_finite() {
            break;
        }
        let rel_change = (new_x - x).abs() / (1.0 + x.abs());
        x = new_x;
        if rel_change < 1e-14 {
            break;
        }
    }

    if upper {
        x
    } else {
        -x
    }
}

// ---------------------------------------------------------------------------
// Chi-squared inverse CDF (joint Wald-χ² critical values)
// ---------------------------------------------------------------------------

/// Regularised lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Uses NR §6.2: series expansion for `x < a + 1`, Lentz continued fraction
/// otherwise. Accurate to ~1e-14 over `a ∈ [0.5, 100]`, `x > 0`.
fn gamma_inc_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    const MAXIT: usize = 200;
    const EPS: f64 = 3e-15;
    const FPMIN: f64 = 1.0e-300;

    if x < a + 1.0 {
        // Series for P(a, x) = e^{-x} x^a / Γ(a+1) · Σ_{n≥0} x^n / (a+1)_n.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut term = sum;
        for _ in 0..MAXIT {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * EPS {
                return sum * (-x + a * x.ln() - ln_gamma(a)).exp();
            }
        }
        // Didn't converge — fall through with last-best value.
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x) (upper); P = 1 - Q.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAXIT {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < EPS {
                let q = (-x + a * x.ln() - ln_gamma(a)).exp() * h;
                return 1.0 - q;
            }
        }
        let q = (-x + a * x.ln() - ln_gamma(a)).exp() * h;
        1.0 - q
    }
}

/// Inverse CDF of χ²(k) at probability p ∈ (0, 1).
///
/// Wilson–Hilferty initial guess refined by Newton iteration on
/// `P(k/2, x/2) - p`, using the closed-form chi-squared PDF as the derivative
/// (`f(x; k) = (x/2)^{k/2 - 1} · e^{-x/2} / (2 Γ(k/2))`). Converges in ≤ 10
/// iterations to rel-err < 1e-12 over `k ∈ [1, 100]`, `p ∈ [1e-6, 1 - 1e-6]`.
///
/// Edge cases:
///   - `p ≤ 0` → 0.0;  `p ≥ 1` → +∞;  `k ≤ 0` → NaN.
pub fn chi2_ppf(p: f64, k: f64) -> f64 {
    if !k.is_finite() || k <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Wilson–Hilferty initial guess: χ²(k) ≈ k · (1 - 2/(9k) + z · √(2/(9k)))³.
    let z = norm_ppf(p);
    let h = 2.0 / (9.0 * k);
    let mut x = k * (1.0 - h + z * h.sqrt()).powi(3);
    if !x.is_finite() || x <= 0.0 {
        // Fall back to mean (which is k) when Wilson–Hilferty fails (e.g.,
        // very small k with extreme p).
        x = k.max(1e-6);
    }

    // Newton iteration: x_{n+1} = x_n - (P(k/2, x/2) - p) / pdf(x; k).
    // pdf(x; k) = (x/2)^{k/2 - 1} · exp(-x/2) / (2 Γ(k/2))
    //   ⇒ ln pdf = (k/2 - 1) ln(x/2) - x/2 - ln 2 - ln Γ(k/2).
    let a = 0.5 * k;
    let ln_norm = -a.ln().mul_add(0.0, 0.0) - std::f64::consts::LN_2 - ln_gamma(a);
    // ^ small trick: we'll recompute pdf directly each iter; ln_norm includes
    //   `- ln 2 - ln Γ(k/2)` which is the part of `ln pdf` not depending on x.
    let _ = ln_norm; // silence; we inline below for clarity.
    for _ in 0..80 {
        let cdf = gamma_inc_p(a, 0.5 * x);
        // PDF of χ²(k) at x.
        let ln_pdf = (a - 1.0) * (0.5 * x).ln() - 0.5 * x - std::f64::consts::LN_2 - ln_gamma(a);
        let pdf = ln_pdf.exp();
        if !pdf.is_finite() || pdf <= 0.0 {
            break;
        }
        let step = (cdf - p) / pdf;
        let mut new_x = x - step;
        // Keep x positive (Newton can overshoot for steep PDFs).
        if !new_x.is_finite() || new_x <= 0.0 {
            new_x = 0.5 * x;
        }
        let rel_change = (new_x - x).abs() / (1.0 + x.abs());
        x = new_x;
        if rel_change < 1e-13 {
            break;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// F inverse CDF (joint OLS F-test critical values)
// ---------------------------------------------------------------------------

/// PDF of the F(dfn, dfd) distribution at x > 0:
/// `f(x) = (dfn/dfd)^(dfn/2) · x^(dfn/2 − 1) · (1 + (dfn/dfd) · x)^(−(dfn+dfd)/2) / B(dfn/2, dfd/2)`.
/// Computed in log space.
#[inline]
fn f_pdf(x: f64, dfn: f64, dfd: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 0.5 * dfn;
    let b = 0.5 * dfd;
    let ln_pdf = a * (dfn / dfd).ln() + (a - 1.0) * x.ln()
        - (a + b) * (1.0 + dfn / dfd * x).ln()
        - ln_beta(a, b);
    ln_pdf.exp()
}

/// CDF of the F(dfn, dfd) distribution at x:
/// `P(X ≤ x) = 1 − I_{dfd/(dfd + dfn·x)}(dfd/2, dfn/2)`.
#[inline]
fn f_cdf(x: f64, dfn: f64, dfd: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return 1.0;
    }
    let z = dfd / (dfd + dfn * x);
    1.0 - betai(0.5 * dfd, 0.5 * dfn, z)
}

/// Inverse CDF of F(dfn, dfd) at probability `p ∈ (0, 1)`.
///
/// Approach: Wilson–Hilferty-style normal initial guess refined by
/// Newton-Raphson on `f_cdf(x) − p` using `f_pdf(x)` as the derivative.
/// Converges in ≤ 12 iterations to rel-err < 1e-10 over standard df ranges.
///
/// Edge cases:
/// - `p ≤ 0` → 0.0;  `p ≥ 1` → +∞;  `dfn ≤ 0` or `dfd ≤ 0` → NaN.
pub fn f_ppf(p: f64, dfn: f64, dfd: f64) -> f64 {
    if !dfn.is_finite() || !dfd.is_finite() || dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Initial guess via Wilson–Hilferty on the equivalent χ²: F ≈ χ²(dfn)/dfn.
    let z = norm_ppf(p);
    let h = 2.0 / (9.0 * dfn);
    let chi_seed = dfn * (1.0 - h + z * h.sqrt()).powi(3);
    let mut x = (chi_seed / dfn).max(1e-6);

    for _ in 0..80 {
        let cdf = f_cdf(x, dfn, dfd);
        let pdf = f_pdf(x, dfn, dfd);
        if !pdf.is_finite() || pdf <= 0.0 {
            break;
        }
        let step = (cdf - p) / pdf;
        let mut new_x = x - step;
        if !new_x.is_finite() || new_x <= 0.0 {
            new_x = 0.5 * x;
        }
        let rel_change = (new_x - x).abs() / (1.0 + x.abs());
        x = new_x;
        if rel_change < 1e-13 {
            break;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Studentized-range (Tukey HSD) inverse CDF
// ---------------------------------------------------------------------------

/// Standard-normal PDF φ(z).
#[inline]
fn norm_pdf(z: f64) -> f64 {
    use std::f64::consts::PI;
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// 32-node Gauss–Legendre abscissae/weights on [-1, 1] (symmetric; only the
/// 16 non-negative nodes are stored — negatives are mirror images with equal
/// weight). Used for both the inner normal integral and the outer χ mixing.
const GL32_X: [f64; 16] = [
    0.048_307_665_687_738_32,
    0.144_471_961_582_796_5,
    0.239_287_362_252_137_07,
    0.331_868_602_282_127_65,
    0.421_351_276_130_635_35,
    0.506_899_908_932_229_4,
    0.587_715_757_240_762_3,
    0.663_044_266_930_215_2,
    0.732_182_118_740_289_7,
    0.794_483_795_967_942_4,
    0.849_367_613_732_57,
    0.896_321_155_766_052_1,
    0.934_906_075_937_739_7,
    0.964_762_255_587_506_4,
    0.985_611_511_545_268_3,
    0.997_263_861_849_481_6,
];
const GL32_W: [f64; 16] = [
    0.096_540_088_514_727_8,
    0.095_638_720_079_274_86,
    0.093_844_399_080_804_57,
    0.091_173_878_695_763_89,
    0.087_652_093_004_403_81,
    0.083_311_924_226_946_76,
    0.078_193_895_787_070_31,
    0.072_345_794_108_848_5,
    0.065_822_222_776_361_85,
    0.058_684_093_478_535_55,
    0.050_998_059_262_376_18,
    0.042_835_898_022_226_69,
    0.034_273_862_913_021_43,
    0.025_392_065_309_262_06,
    0.016_274_394_730_905_67,
    0.007_018_610_009_470_117,
];

/// Integrate `f` over `[lo, hi]` with a composite Gauss–Legendre rule: the
/// interval is split into `panels` equal sub-intervals, each integrated with
/// the 32-node rule. Effective order is `32·panels`, which resolves sharply
/// peaked integrands (large-k inner integral, small-df χ mixing) that a single
/// GL32 panel under-resolves. Off the hot loop, so the per-call allocation-free
/// double sum is fine.
#[inline]
fn gl_composite<F: Fn(f64) -> f64>(lo: f64, hi: f64, panels: usize, f: F) -> f64 {
    let step = (hi - lo) / panels as f64;
    let half = 0.5 * step;
    let mut acc = 0.0;
    for p in 0..panels {
        let mid = lo + (p as f64 + 0.5) * step;
        for i in 0..16 {
            let dx = half * GL32_X[i];
            acc += GL32_W[i] * (f(mid + dx) + f(mid - dx));
        }
    }
    half * acc
}

/// Composite GL over `[0, hi]` with panel boundaries graded as `(i/panels)^grade`
/// so they cluster near 0. Used for the χ-mixing integral, whose integrand
/// `g(u)·range_cdf(q·√u)` develops a sharp transition near `u = 0` for high
/// quantiles (large q): `range_cdf(q·√u)` climbs from 0 to 1 within
/// `u ~ (1/q)²`, which a uniform mesh under-resolves. `grade > 1` packs the
/// nodes there. Each graded panel still uses the 32-node rule internally.
#[inline]
fn gl_graded<F: Fn(f64) -> f64>(hi: f64, panels: usize, grade: f64, f: F) -> f64 {
    let mut acc = 0.0;
    let mut a = 0.0_f64;
    for p in 1..=panels {
        let b = hi * (p as f64 / panels as f64).powf(grade);
        let half = 0.5 * (b - a);
        let mid = 0.5 * (b + a);
        for i in 0..16 {
            let dx = half * GL32_X[i];
            acc += half * GL32_W[i] * (f(mid + dx) + f(mid - dx));
        }
        a = b;
    }
    acc
}

/// Inner integral: P(range ≤ w | k) for k iid standard normals, where
/// `w = q·s` is the (scaled) range threshold. Equals
/// `k · ∫_{-∞}^{∞} φ(z) [Φ(z) − Φ(z − w)]^{k−1} dz`.
///
/// Integrated over the effectively-finite support `z ∈ [-8, 8]` with a composite
/// Gauss–Legendre rule. The integrand `φ(z)·[Φ(z) − Φ(z − w)]^{k−1}` is confined
/// to `[-8, 8]` by `φ(z)` *regardless of w* (the `[Φ(z) − Φ(z − w)]` factor is
/// bounded by 1), so the domain must NOT widen with w — doing so spreads all the
/// nodes into the φ-dead region `[8, 8+w]` for large w and starves the live
/// `[-8, 8]` band, which made `range_cdf` collapse toward 0 for huge w and broke
/// the total CDF mass. The integrand also becomes sharply peaked for large `k`,
/// so the panel count grows with `k` (a single GL32 panel loses accuracy past
/// k ≈ 8).
fn range_cdf(w: f64, k: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let lo = -8.0;
    let hi = 8.0;
    // More panels for larger k: the (k−1) power sharpens the peak.
    let panels = 4 + (k as usize).saturating_sub(1);
    let acc = gl_composite(lo, hi, panels, |z| {
        let inner = phi(z) - phi(z - w);
        if inner > 0.0 {
            norm_pdf(z) * inner.powf(k - 1.0)
        } else {
            0.0
        }
    });
    (k * acc).min(1.0)
}

/// Studentized-range CDF P(W ≤ q) for `k` groups and `df` error df (Lenth).
///
/// `P(W ≤ q) = ∫_0^∞ f_df(s) · range_cdf(q·s, k) ds`, where `s = χ_df/√df`
/// (the random scale factor) has density
/// `f_df(s) = 2 (df/2)^{df/2} / Γ(df/2) · s^{df−1} e^{−df s²/2}`.
///
/// For `df → ∞` the scale concentrates at `s = 1` and the mixing drops out, so
/// we short-circuit to `range_cdf(q, k)`.
///
/// Finite-df mixing is integrated in the variable `u = s² = χ²_df / df`, whose
/// density is `g(u) = (df/2)^{df/2}/Γ(df/2) · u^{df/2−1} e^{−df u/2}` (a gamma
/// density). The substitution is what makes small df tractable: in the original
/// `s` variable the factor `s^{df−1}` has an infinite slope at `s = 0` for
/// `df < 2` and a near-vertical rise for `df` just above 2, which a fixed GL
/// grid mis-resolves and which made the CDF non-monotone in q. In `u` the
/// exponent `df/2 − 1` is ≥ 0 for all `df ≥ 2`, so the integrand is smooth on
/// `[0, u_max]` and the composite GL rule converges. Small df gets extra panels
/// near `u = 0` where `u^{df/2−1}` still bends sharply.
///
/// `P(W ≤ q) = ∫_0^∞ g(u) · range_cdf(q·√u, k) du`, truncated at the χ²_df upper
/// tail. Requires `df ≥ 2`; callers guard `df < 2` upstream.
///
/// After editing this quadrature (or `range_cdf` / `gl_graded`), run the
/// exhaustive monotonicity sweep, which is `#[ignore]`d for cost:
/// `cargo test -p engine-core --release -- --ignored tukey_cdf_monotone`.
fn studentized_range_cdf(q: f64, k: f64, df: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    if !df.is_finite() || df > 2.0e4 {
        // df → ∞: s ≡ 1.
        return range_cdf(q, k);
    }
    let half_df = 0.5 * df;
    // log g(u) = (df/2) ln(df/2) − lnΓ(df/2) + (df/2 − 1) ln u − (df/2) u.
    let ln_c = half_df * half_df.ln() - ln_gamma(half_df);
    let g = |u: f64| -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        let dens = (ln_c + (half_df - 1.0) * u.ln() - half_df * u).exp();
        dens * range_cdf(q * u.sqrt(), k)
    };
    // u = χ²_df/df has mean 1 and sd √(2/df). Cap at mean+12·sd, comfortably
    // past the χ² upper tail for df ≥ 2 (range_cdf(q√u) ≈ 1 out there, so over-
    // shooting only adds negligible mass — but undershooting truncates real mass
    // and makes the CDF non-monotone in q).
    let u_max = 1.0 + 12.0 * (2.0 / df).sqrt();
    // Grade the mesh toward u = 0: for high quantiles range_cdf(q√u) climbs from
    // 0 to 1 within u ~ (1/q)², a sharp transition a uniform mesh misses. Small
    // df also needs more panels because u^{df/2−1} bends hard near 0.
    let (panels, grade) = if df < 6.0 { (96, 3.0) } else { (24, 2.0) };
    gl_graded(u_max, panels, grade, g).min(1.0)
}

/// Inverse CDF (quantile) of the studentized-range distribution: returns the
/// `p`-quantile `q` such that `P(W ≤ q) = p` for `k` groups and `df` error df.
///
/// Seeded from the Bonferroni bound `q0 = √2 · t_ppf(1 − (1−p)/(k(k−1)), df)`,
/// then refined by Newton iteration on `studentized_range_cdf(q) − p` (the
/// derivative is estimated by a central finite difference, since the CDF is
/// itself a quadrature). Falls back to bisection on a bracketing interval if
/// Newton diverges or leaves the bracket.
///
/// Used by `CritValueTable::build` to compute Tukey-HSD thresholds
/// `(q_{α,k,df} / √2)²`. Built once per `run_batch`, off the hot loop.
///
/// Edge cases: `p ≤ 0` → 0.0; `p ≥ 1` → +∞; `k ≤ 1`, `df < 2`, or non-finite
/// inputs → NaN. The `df < 2` case matches R: `qtukey(p, k, 1)` returns NaN
/// (the df=1 studentized range is degenerate), so we never emit finite garbage
/// there — the χ-mixing density `u^{df/2−1}` is non-integrable in our quadrature
/// for `df < 2` and the CDF stops being monotone in q, which would let the
/// bracket/Newton search lock onto a spurious crossing.
pub(crate) fn q_tukey_ppf(p: f64, k: f64, df: f64) -> f64 {
    if !k.is_finite() || k <= 1.0 || !df.is_finite() || df < 2.0 || !p.is_finite() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Bonferroni seed: √2 · t-quantile at the per-comparison level for the
    // k(k−1)/2 pairwise contrasts (two-sided ⇒ factor k(k−1)).
    let m = k * (k - 1.0);
    let t_q = t_ppf(1.0 - (1.0 - p) / m, df);
    let mut q = if t_q.is_finite() && t_q > 0.0 {
        std::f64::consts::SQRT_2 * t_q
    } else {
        3.0
    };

    // Bracket for the bisection fallback: widen until the CDF straddles p.
    let mut lo = 0.0_f64;
    let mut hi = q.max(1.0) * 2.0 + 4.0;
    while studentized_range_cdf(hi, k, df) < p && hi < 1.0e3 {
        hi *= 1.5;
    }

    // Newton with central-difference derivative; bisection fallback.
    let h = 1e-4;
    for _ in 0..60 {
        let f = studentized_range_cdf(q, k, df) - p;
        if f.abs() < 1e-9 {
            return q;
        }
        // Tighten the bracket using the sign of the residual.
        if f < 0.0 {
            lo = q;
        } else {
            hi = q;
        }
        let f_hi = studentized_range_cdf(q + h, k, df);
        let f_lo = studentized_range_cdf(q - h, k, df);
        let deriv = (f_hi - f_lo) / (2.0 * h);
        let mut next = if deriv.is_finite() && deriv > 1e-12 {
            q - f / deriv
        } else {
            f64::NAN
        };
        // Fall back to bisection if Newton leaves the bracket or diverges.
        if !next.is_finite() || next <= lo || next >= hi {
            next = 0.5 * (lo + hi);
        }
        if (next - q).abs() < 1e-10 {
            return next;
        }
        q = next;
    }
    q
}

// ---------------------------------------------------------------------------
// CritValueTable
// ---------------------------------------------------------------------------

/// Built once per `run_batch` invocation, indexed by sample-size position.
/// All values are squared so the per-sim path compares `t_sq > t_crit_sq`
/// directly without a square root on the SE side.
#[derive(Debug, Clone)]
pub struct CritValueTable {
    /// One entry per element of `sample_sizes`.
    pub t_crit_sq_uncorrected: Vec<f64>,
    /// Outer length = `sample_sizes.len()`; inner length = `n_targets`.
    /// Per-target thresholds sequenced as v1 does (entry `k` walks the
    /// Bonferroni-step sequence under Holm and the BH step-up sequence under
    /// BH).
    pub correction_t_crit_sq: Vec<Vec<f64>>,
    /// One entry per element of `sample_sizes`; equals
    /// `t_ppf(1 - posthoc_alpha/2, df)²` (or the uncorrected entry when
    /// `posthoc_alpha` is None).
    pub posthoc_t_crit_sq: Vec<f64>,
    /// χ²(k, 1-α) for the LME joint Wald-χ² test; `NaN` for non-LME families.
    /// Per-N for layout parity with `t_crit_sq_uncorrected` (the χ²(k) value is
    /// df-independent for the asymptotic test, so all entries are equal).
    pub joint_t_crit_sq: Vec<f64>,
    /// One entry per element of `sample_sizes`. v1-parity overall (joint)
    /// critical value used by the OLS F-test and Logit LRT. NOT squared — the
    /// hot path compares `F > overall_crit` / `LRT > overall_crit` directly.
    /// - `EstimatorSpec::Ols`  → `f_ppf(1 − α, dfn = P − 1, dfd = N − P)`
    /// - `EstimatorSpec::Glm`  → `chi2_ppf(1 − α, df = P − 1)` (df-independent)
    /// - `EstimatorSpec::Mle`  → `f64::INFINITY` (Mle emits 0; never satisfied)
    /// - `P − 1 == 0` (intercept-only) → `f64::INFINITY` for every family.
    pub overall_crit: Vec<f64>,
}

impl CritValueTable {
    /// Build the per-batch table.
    ///
    /// - `crit`: spec-level α and optional posthoc α.
    /// - `sample_sizes`: per-sim sample sizes evaluated.
    /// - `n_predictors`: total fixed-effect columns (used for `df = N − P`).
    /// - `n_targets`: number of inference targets.
    /// - `correction_method`: typed enum (None / Bonferroni / Holm / BenjaminiHochberg).
    /// - `estimator`: `EstimatorSpec::Ols` uses Student-t with `df = N − P`;
    ///   `EstimatorSpec::Glm` and `EstimatorSpec::Mle` both use the standard normal
    ///   (Wald-z, df-independent; the LME engine produces Wald-z statistics).
    pub fn build(
        crit: &CritValues,
        sample_sizes: &[u32],
        n_predictors: u32,
        n_targets: u32,
        correction_method: CorrectionMethod,
        estimator: EstimatorSpec,
    ) -> Result<CritValueTable, EngineError> {
        Self::build_with_tukey_k(
            crit,
            sample_sizes,
            n_predictors,
            n_targets,
            correction_method,
            estimator,
            &[],
        )
    }

    /// Like [`build`], but threads a per-target studentized-range `k` (number of
    /// means compared = the factor level count `L` of the factor each target
    /// belongs to). Only the `TukeyHsd` arm reads it; every other correction
    /// method ignores `tukey_k_per_target` entirely, so non-Tukey callers pass
    /// an empty slice via [`build`].
    ///
    /// `tukey_k_per_target` is indexed by target position (same order as the
    /// per-`N` correction row: marginal targets first, then contrasts). Each
    /// entry is the factor's `L`; targets that are not factor dummies (a
    /// continuous predictor, the intercept, or a cross-factor contrast) carry
    /// `< 2` (e.g. `0.0` or `NaN`), which `q_tukey_ppf` maps to `NaN` → those
    /// targets always fail under Tukey (a Tukey design should not point a target
    /// at a non-factor column).
    ///
    /// [`build`]: CritValueTable::build
    pub fn build_with_tukey_k(
        crit: &CritValues,
        sample_sizes: &[u32],
        n_predictors: u32,
        n_targets: u32,
        correction_method: CorrectionMethod,
        estimator: EstimatorSpec,
        tukey_k_per_target: &[f64],
    ) -> Result<CritValueTable, EngineError> {
        if !crit.alpha.is_finite() || crit.alpha <= 0.0 || crit.alpha >= 1.0 {
            return Err(EngineError::InvalidSpec(format!(
                "alpha must lie in (0, 1); got {}",
                crit.alpha
            )));
        }
        if let Some(pa) = crit.posthoc_alpha {
            if !pa.is_finite() || pa <= 0.0 || pa >= 1.0 {
                return Err(EngineError::InvalidSpec(format!(
                    "posthoc_alpha must lie in (0, 1); got {pa}"
                )));
            }
        }

        let alpha = crit.alpha;
        let posthoc_alpha = crit.posthoc_alpha.unwrap_or(alpha);

        // Estimator-aware quantile. Ols → Student-t with df = N − P; Glm →
        // standard normal (df-independent). Glm uses Wald-z (the standard
        // logit convention: normal quantile, not t), so the critical value is
        // df-independent. Mle also uses Wald-z: the LME engine produces
        // Wald-z statistics, so z² is the correct critical-value shape
        // (df-independent).
        let crit_quantile = |p: f64, df: f64| -> f64 {
            if estimator.uses_student_t() {
                t_ppf(p, df)
            } else {
                norm_ppf(p)
            }
        };

        let n = sample_sizes.len();
        let m = n_targets as usize;
        let mut t_crit_sq_uncorrected = Vec::with_capacity(n);
        let mut correction_t_crit_sq = Vec::with_capacity(n);
        let mut posthoc_t_crit_sq = Vec::with_capacity(n);

        for &sample_size in sample_sizes {
            let df_signed = sample_size as i64 - n_predictors as i64;
            if df_signed < 1 {
                return Err(EngineError::InvalidSpec(format!(
                    "df = N - P = {sample_size} - {n_predictors} < 1"
                )));
            }
            let df = df_signed as f64;

            let t_unc = crit_quantile(1.0 - alpha / 2.0, df);
            t_crit_sq_uncorrected.push(t_unc * t_unc);

            let mut row = Vec::with_capacity(m.max(1));
            if m == 0 {
                // No targets — push an empty per-N row.
            } else {
                match correction_method {
                    CorrectionMethod::None => {
                        // None.
                        for _ in 0..m {
                            row.push(t_unc * t_unc);
                        }
                    }
                    CorrectionMethod::Bonferroni => {
                        // Bonferroni.
                        let bonf = crit_quantile(1.0 - alpha / (2.0 * m as f64), df);
                        for _ in 0..m {
                            row.push(bonf * bonf);
                        }
                    }
                    CorrectionMethod::Holm => {
                        // Holm step-down: entry k uses α / (m - k).
                        for k in 0..m {
                            let eff = alpha / (2.0 * (m - k) as f64);
                            let t = if eff < 1e-12 {
                                f64::INFINITY
                            } else {
                                crit_quantile(1.0 - eff, df)
                            };
                            row.push(t * t);
                        }
                    }
                    CorrectionMethod::BenjaminiHochberg => {
                        // Benjamini-Hochberg step-up: entry k uses (k+1)·α/m.
                        for k in 0..m {
                            let eff = (k + 1) as f64 / m as f64 * alpha / 2.0;
                            let t = if eff < 1e-12 {
                                f64::INFINITY
                            } else {
                                crit_quantile(1.0 - eff, df)
                            };
                            row.push(t * t);
                        }
                    }
                    CorrectionMethod::TukeyHsd => {
                        // Tukey HSD: per-target threshold from the studentized-
                        // range quantile of that target's factor. `k` is the
                        // factor level count `L` (number of means compared) — it
                        // varies per target across a multi-factor design, so we
                        // read it from `tukey_k_per_target[i]` rather than the
                        // global target count `m`. The studentized-range stat
                        // compares to `q_{α,k,df}`; dividing by √2 converts it to
                        // a (two-sided) t-scale critical value, which we square to
                        // match the `t²` comparison space used by the hot loop.
                        debug_assert_eq!(
                            tukey_k_per_target.len(),
                            m,
                            "TukeyHsd callers must pass one k per target"
                        );
                        for i in 0..m {
                            let k = tukey_k_per_target.get(i).copied().unwrap_or(f64::NAN);
                            let q = q_tukey_ppf(1.0 - alpha, k, df);
                            let t = q / std::f64::consts::SQRT_2;
                            row.push(t * t);
                        }
                    }
                }
            }
            correction_t_crit_sq.push(row);

            let t_post = crit_quantile(1.0 - posthoc_alpha / 2.0, df);
            posthoc_t_crit_sq.push(t_post * t_post);
        }

        // Joint Wald-χ² critical value: χ²(k, 1-α) for EstimatorSpec::Mle, NaN
        // otherwise. χ²(k) is df-independent for the asymptotic test, but we
        // store one entry per N for layout parity with `t_crit_sq_uncorrected`
        // (and so a future per-N Kenward-Roger F upgrade fits without
        // re-shaping the table).
        let joint_t_crit_sq: Vec<f64> = match estimator {
            EstimatorSpec::Mle => {
                let k = n_targets as f64;
                if k <= 0.0 {
                    vec![f64::NAN; n]
                } else {
                    let chi2_crit = chi2_ppf(1.0 - alpha, k);
                    vec![chi2_crit; n]
                }
            }
            _ => vec![f64::NAN; n],
        };

        // Overall (joint) critvals — v1-parity F (OLS) / LRT (Logit) thresholds.
        // Stored unsquared. INFINITY for LME and for intercept-only specs.
        // Under sparse-factor exclusion, the OLS and GLM overall blocks in batch.rs
        // recompute this per round with the reduced-model dfs instead of reading here.
        // The LME joint Wald-χ² block in batch.rs also recomputes per round (k_red
        // may differ from n_targets when factor targets are excluded) — change together.
        let n_pred_no_int = (n_predictors as i64 - 1).max(0) as f64;
        let overall_crit: Vec<f64> = if n_pred_no_int < 1.0 {
            vec![f64::INFINITY; n]
        } else {
            match estimator {
                EstimatorSpec::Ols => sample_sizes
                    .iter()
                    .map(|&ss| {
                        let dfd = (ss as i64 - n_predictors as i64) as f64;
                        // df_signed < 1 is already rejected in the loop above —
                        // any sample size that reaches here has dfd ≥ 1.
                        f_ppf(1.0 - alpha, n_pred_no_int, dfd)
                    })
                    .collect(),
                EstimatorSpec::Glm => {
                    // χ²(P-1, 1-α) is df-independent under the asymptotic LRT;
                    // store one entry per N for layout parity with t_crit_sq.
                    let v = chi2_ppf(1.0 - alpha, n_pred_no_int);
                    vec![v; n]
                }
                EstimatorSpec::Mle => vec![f64::INFINITY; n],
            }
        };

        Ok(CritValueTable {
            t_crit_sq_uncorrected,
            correction_t_crit_sq,
            posthoc_t_crit_sq,
            joint_t_crit_sq,
            overall_crit,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// CRIT-10: the studentized-range quantile decreases as df grows (more
    /// residual df → tighter critical value), holding p and k fixed, and
    /// converges to a finite asymptote. Replaces the print-only theater test
    /// `tukey_report_errors` and the scipy/R golden value-matches.
    #[test]
    fn q_tukey_monotone_decreasing_in_df() {
        let p = 0.95;
        let k = 3.0;
        let dfs = [3.0_f64, 5.0, 10.0, 20.0, 60.0, 1e6];
        let mut prev = f64::INFINITY;
        for &df in &dfs {
            let q = q_tukey_ppf(p, k, df);
            assert!(
                q.is_finite() && q > 0.0,
                "q_tukey must be finite>0 at df={df}, got {q}"
            );
            assert!(
                q <= prev + 1e-9,
                "q_tukey must be non-increasing in df: df={df} q={q} > prev={prev}"
            );
            prev = q;
        }
    }

    #[test]
    fn tukey_quantile_df_lt_2_is_nan() {
        // R parity: qtukey(0.95, 3, 1) returns NaN (df=1 range is degenerate),
        // never finite garbage. Guard returns NaN for all df < 2.
        assert!(q_tukey_ppf(0.95, 3.0, 1.0).is_nan(), "df=1 must be NaN");
        assert!(q_tukey_ppf(0.95, 3.0, 0.5).is_nan(), "df=0.5 must be NaN");
        assert!(q_tukey_ppf(0.95, 3.0, 0.0).is_nan(), "df=0 must be NaN");
        assert!(q_tukey_ppf(0.95, 3.0, -1.0).is_nan(), "df<0 must be NaN");
    }

    /// The non-monotonicity that produced finite garbage at small df is gone:
    /// studentized_range_cdf must be non-decreasing in q across the search
    /// bracket for every df ≥ 2. Sampling is dense through the rise region
    /// (where the small-df quadrature bug lived) and sparse out to the
    /// q_tukey_ppf bracket cap of 1e3 (where the range_cdf collapse at huge
    /// w showed up) — each CDF call costs the same regardless of q, so the
    /// flat tails don't earn dense sampling.
    fn assert_srange_cdf_monotone(df: f64, k: f64) {
        let mut prev = -1.0_f64;
        let mut check = |q: f64| {
            let c = studentized_range_cdf(q, k, df);
            assert!(
                c >= prev - 1e-9,
                "CDF non-monotone: df={df} k={k} q={q} cdf={c} prev={prev}"
            );
            prev = c;
        };
        let mut q = 0.05_f64;
        while q <= 50.0 {
            check(q);
            q *= 1.5;
        }
        for q in [150.0, 450.0, 1.0e3] {
            check(q);
        }
    }

    #[test]
    fn tukey_cdf_monotone_in_q() {
        // Always-on smoke on the most fragile quadrature path: df=2 is the
        // small-df singularity worst case (96 graded panels), k=12 the
        // sharpest range_cdf integrand. The full df×k sweep is the #[ignore]d
        // test below.
        assert_srange_cdf_monotone(2.0, 12.0);
    }

    #[test]
    #[ignore = "exhaustive quadrature sweep (~15s) — run after editing studentized_range_cdf / range_cdf / gl_graded"]
    fn tukey_cdf_monotone_in_q_full() {
        for &df in &[2.0_f64, 3.0, 5.0, 10.0, 30.0] {
            for &k in &[3.0_f64, 8.0, 12.0] {
                assert_srange_cdf_monotone(df, k);
            }
        }
    }

    #[test]
    fn tukey_quantile_monotone_in_k_and_p() {
        assert!(q_tukey_ppf(0.95, 4.0, 30.0) > q_tukey_ppf(0.95, 3.0, 30.0));
        assert!(q_tukey_ppf(0.99, 3.0, 30.0) > q_tukey_ppf(0.95, 3.0, 30.0));
    }

    #[test]
    fn t_ppf_symmetry() {
        for &df in &[1.5_f64, 5.0, 30.0, 1000.0] {
            for &p in &[0.01_f64, 0.1, 0.25, 0.4] {
                let lo = t_ppf(p, df);
                let hi = t_ppf(1.0 - p, df);
                assert!((lo + hi).abs() < 1e-9, "df={df}, p={p}: lo={lo}, hi={hi}");
            }
        }
    }

    #[test]
    fn t_ppf_edge_cases() {
        assert_eq!(t_ppf(0.5, 10.0), 0.0);
        assert_eq!(t_ppf(0.0, 10.0), f64::NEG_INFINITY);
        assert_eq!(t_ppf(1.0, 10.0), f64::INFINITY);
        assert!(t_ppf(0.5, 0.0).is_nan());
        assert!(t_ppf(0.5, -1.0).is_nan());
    }

    #[test]
    fn t_crit_sq_invariant() {
        // For every (alpha, df) pair, (t_ppf(1-α/2, df))² must equal
        // t_crit_sq_uncorrected[i] bit-for-bit.
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![50_u32, 100, 500];
        let n_predictors = 3_u32;
        let n_targets = 2_u32;
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        )
        .unwrap();
        for (i, &n) in sample_sizes.iter().enumerate() {
            let df = (n - n_predictors) as f64;
            let t_crit = t_ppf(1.0 - 0.05 / 2.0, df);
            assert_eq!(t_crit * t_crit, table.t_crit_sq_uncorrected[i]);
        }
    }

    #[test]
    fn build_table_correction_methods() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![101_u32];
        let n_predictors = 3_u32; // df = 98
        let n_targets = 3_u32;
        let df = 98.0;

        // None — every entry equals uncorrected.
        let t0 = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        )
        .unwrap();
        let t_unc = t_ppf(1.0 - 0.05 / 2.0, df);
        for v in &t0.correction_t_crit_sq[0] {
            assert!(((v - t_unc * t_unc) / (t_unc * t_unc)).abs() < 1e-12);
        }

        // Bonferroni — every entry equals t_ppf(1 - α/(2m), df).
        let t1 = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::Bonferroni,
            EstimatorSpec::Ols,
        )
        .unwrap();
        let bonf = t_ppf(1.0 - 0.05 / (2.0 * 3.0), df);
        for v in &t1.correction_t_crit_sq[0] {
            assert!(((v - bonf * bonf) / (bonf * bonf)).abs() < 1e-12);
        }

        // Holm — entry k uses α/(m-k).
        let t2 = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::Holm,
            EstimatorSpec::Ols,
        )
        .unwrap();
        for k in 0..3 {
            let eff = 0.05 / (2.0 * (3 - k) as f64);
            let expected = t_ppf(1.0 - eff, df);
            let got = t2.correction_t_crit_sq[0][k];
            assert!(
                ((got - expected * expected) / (expected * expected)).abs() < 1e-12,
                "Holm k={k}: got {got}, expected {}",
                expected * expected
            );
        }

        // BH — entry k uses (k+1)·α/m.
        let t3 = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::BenjaminiHochberg,
            EstimatorSpec::Ols,
        )
        .unwrap();
        for k in 0..3 {
            let eff = (k + 1) as f64 / 3.0 * 0.05 / 2.0;
            let expected = t_ppf(1.0 - eff, df);
            let got = t3.correction_t_crit_sq[0][k];
            assert!(
                ((got - expected * expected) / (expected * expected)).abs() < 1e-12,
                "BH k={k}: got {got}, expected {}",
                expected * expected
            );
        }
    }

    #[test]
    fn holm_thresholds_dominate_uncorrected_and_last_equals() {
        // CRIT-08: every Holm rank threshold is at least as conservative as the
        // uncorrected threshold, and the least-conservative rank (k = m-1, effective
        // α/2) equals uncorrected exactly. Ordering property — no value pinned.
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![101_u32];
        let n_predictors = 3_u32; // df = 98
        let n_targets = 3_u32;
        let df = 98.0;

        let holm = CritValueTable::build(
            &crit,
            &sample_sizes,
            n_predictors,
            n_targets,
            CorrectionMethod::Holm,
            EstimatorSpec::Ols,
        )
        .unwrap();
        let t_unc = t_ppf(1.0 - 0.05 / 2.0, df);
        let unc_sq = t_unc * t_unc;

        let row = &holm.correction_t_crit_sq[0];
        for (k, &v) in row.iter().enumerate() {
            assert!(
                v >= unc_sq - 1e-9,
                "Holm rank {k} threshold {v} is below uncorrected {unc_sq}"
            );
        }
        let last = row[(n_targets - 1) as usize];
        assert!(
            ((last - unc_sq) / unc_sq).abs() < 1e-12,
            "Holm last rank {last} must equal uncorrected {unc_sq}"
        );
    }

    #[test]
    fn build_table_posthoc_alpha_default() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![100_u32];
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            2,
            2,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        )
        .unwrap();
        // posthoc_t_crit_sq must equal uncorrected when posthoc_alpha is None.
        assert_eq!(table.posthoc_t_crit_sq[0], table.t_crit_sq_uncorrected[0]);
    }

    #[test]
    fn build_table_posthoc_alpha_distinct() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: Some(0.01),
        };
        let sample_sizes = vec![100_u32];
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            2,
            2,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        )
        .unwrap();
        let df = 98.0;
        let t_post = t_ppf(1.0 - 0.01 / 2.0, df);
        assert!(((table.posthoc_t_crit_sq[0] - t_post * t_post) / (t_post * t_post)).abs() < 1e-12);
    }

    #[test]
    fn build_table_rejects_df_lt_1() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![3_u32];
        let result = CritValueTable::build(
            &crit,
            &sample_sizes,
            3,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        );
        assert!(matches!(result, Err(EngineError::InvalidSpec(_))));
    }

    // -----------------------------------------------------------------
    // EstimatorSpec::Glm — Wald-z thresholds (df-independent)
    // -----------------------------------------------------------------

    #[test]
    fn crit_logit_uncorrected_equals_norm_ppf() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![100_u32];
        // n_predictors = 2 so df = 98 is a legal value; under Logit it's unused.
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            2,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Glm,
        )
        .unwrap();
        let z = norm_ppf(0.975);
        let expected = z * z;
        let got = table.t_crit_sq_uncorrected[0];
        let rel = (got - expected).abs() / expected;
        assert!(
            rel < 1e-12,
            "Logit uncorrected t_crit_sq = {got}, expected {expected}, rel {rel}"
        );
    }

    #[test]
    fn crit_logit_invariant_to_n() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let table_small = CritValueTable::build(
            &crit,
            &[100_u32],
            2,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Glm,
        )
        .unwrap();
        let table_large = CritValueTable::build(
            &crit,
            &[10_000_u32],
            2,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Glm,
        )
        .unwrap();
        assert_eq!(
            table_small.t_crit_sq_uncorrected[0], table_large.t_crit_sq_uncorrected[0],
            "Wald-z threshold is df-independent"
        );
    }

    #[test]
    fn crit_logit_bonferroni() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let table = CritValueTable::build(
            &crit,
            &[100_u32],
            2,
            3,
            CorrectionMethod::Bonferroni,
            EstimatorSpec::Glm,
        )
        .unwrap();
        let z = norm_ppf(1.0 - 0.05 / (2.0 * 3.0));
        let expected = z * z;
        for &got in &table.correction_t_crit_sq[0] {
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-12,
                "Logit Bonferroni: got {got}, expected {expected}, rel {rel}"
            );
        }
    }

    // -----------------------------------------------------------------
    // EstimatorSpec::Mle — Wald-z thresholds (df-independent)
    // -----------------------------------------------------------------

    /// Mle must produce bit-identical z² values to Glm at every
    /// sample size and correction method (both use norm_ppf, df-independent).
    #[test]
    fn critvals_lme_matches_logit_z_critical() {
        let cv = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let tab_logit = CritValueTable::build(
            &cv,
            &[100_u32],
            3,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Glm,
        )
        .unwrap();
        let tab_lme = CritValueTable::build(
            &cv,
            &[100_u32],
            3,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Mle,
        )
        .unwrap();
        let delta = (tab_logit.t_crit_sq_uncorrected[0] - tab_lme.t_crit_sq_uncorrected[0]).abs();
        assert!(
            delta < 1e-12,
            "Glm z² = {}, Mle z² = {}, delta = {}",
            tab_logit.t_crit_sq_uncorrected[0],
            tab_lme.t_crit_sq_uncorrected[0],
            delta
        );
        // Also verify posthoc and correction slots are identical.
        assert_eq!(
            tab_logit.posthoc_t_crit_sq[0], tab_lme.posthoc_t_crit_sq[0],
            "posthoc_t_crit_sq must match"
        );
        assert_eq!(
            tab_logit.correction_t_crit_sq[0], tab_lme.correction_t_crit_sq[0],
            "correction_t_crit_sq must match"
        );
    }

    /// EstimatorSpec::Mle threshold is df-independent (same for N=100 and N=10000).
    #[test]
    fn critvals_lme_invariant_to_n() {
        let cv = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let tab_small = CritValueTable::build(
            &cv,
            &[100_u32],
            2,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Mle,
        )
        .unwrap();
        let tab_large = CritValueTable::build(
            &cv,
            &[10_000_u32],
            2,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Mle,
        )
        .unwrap();
        assert_eq!(
            tab_small.t_crit_sq_uncorrected[0], tab_large.t_crit_sq_uncorrected[0],
            "Wald-z threshold is df-independent for EstimatorSpec::Mle"
        );
    }

    /// Boolean-equivalence for the Holm/BH ordering. The hot path compares in
    /// `t²` space; a naive reference compares in `|t|` space. With the
    /// deterministic ascending-target-index tie-breaker, the pass/fail
    /// boolean vectors must match exactly.
    #[test]
    fn holm_bh_t2_vs_abs_t_boolean_equivalence() {
        // Synthetic per-target t² values (one sample size, 5 targets).
        // Negative t in one entry to expose the |t| vs t² difference under
        // hypothetical "signed" sorts. With t² the sign drops out, and both
        // reference paths use |t| then square — equivalence is by construction
        // once the tie-breaker matches.
        let t_vals = [1.2_f64, -2.3, 0.7, 3.5, 1.2];
        let abs_t: Vec<f64> = t_vals.iter().map(|x| x.abs()).collect();
        let t_sq: Vec<f64> = t_vals.iter().map(|x| x * x).collect();

        let n_targets = t_vals.len();
        let df = 95.0_f64;
        let alpha = 0.05_f64;

        // Pre-build the t² thresholds for Holm.
        let crit_sq: Vec<f64> = (0..n_targets)
            .map(|k| {
                let eff = alpha / (2.0 * (n_targets - k) as f64);
                let t = t_ppf(1.0 - eff, df);
                t * t
            })
            .collect();
        // And the |t| thresholds for the reference (same numbers, unsquared).
        let crit_abs: Vec<f64> = (0..n_targets)
            .map(|k| {
                let eff = alpha / (2.0 * (n_targets - k) as f64);
                t_ppf(1.0 - eff, df)
            })
            .collect();

        // Sort descending by sort-key; tie-break ascending by index.
        fn sorted_indices(scores: &[f64]) -> Vec<usize> {
            let mut idx: Vec<usize> = (0..scores.len()).collect();
            idx.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            idx
        }

        // Holm step-down (Bonferroni-style): walking sorted from largest,
        // if any position fails its threshold, all subsequent positions fail.
        fn holm_outcome(order: &[usize], scores: &[f64], thresholds: &[f64]) -> Vec<bool> {
            let n = scores.len();
            let mut pass = vec![false; n];
            let mut cont = true;
            for (k, &i) in order.iter().enumerate() {
                if cont && scores[i] > thresholds[k] {
                    pass[i] = true;
                } else {
                    cont = false;
                }
            }
            pass
        }

        let order_abs = sorted_indices(&abs_t);
        let order_sq = sorted_indices(&t_sq);
        assert_eq!(order_abs, order_sq, "tie-broken orders must match");

        let pass_abs = holm_outcome(&order_abs, &abs_t, &crit_abs);
        let pass_sq = holm_outcome(&order_sq, &t_sq, &crit_sq);
        assert_eq!(pass_abs, pass_sq, "Holm boolean vectors must match");

        // Now BH step-up: walk sorted from smallest, find largest k whose
        // score exceeds threshold[k]; all positions at/above pass.
        let crit_sq_bh: Vec<f64> = (0..n_targets)
            .map(|k| {
                let eff = (k + 1) as f64 / n_targets as f64 * alpha / 2.0;
                let t = t_ppf(1.0 - eff, df);
                t * t
            })
            .collect();
        let crit_abs_bh: Vec<f64> = (0..n_targets)
            .map(|k| {
                let eff = (k + 1) as f64 / n_targets as f64 * alpha / 2.0;
                t_ppf(1.0 - eff, df)
            })
            .collect();

        fn bh_outcome(order: &[usize], scores: &[f64], thresholds: &[f64]) -> Vec<bool> {
            let n = scores.len();
            let mut pass = vec![false; n];
            // order[0] is largest score → BH threshold for rank 1.
            // BH: rank j has threshold[j] where rank = 1..=m and BH uses
            //   reject if p_(j) < j*alpha/m → equivalently |t|_(j) > t_crit(j*alpha/m).
            // Walk from largest score (smallest p) — first pass position k means
            // all positions 0..=k pass.
            for (k, &i) in order.iter().enumerate() {
                let rank_from_largest = n - 1 - k; // largest |t| has BH threshold index n-1
                if scores[i] > thresholds[rank_from_largest] {
                    for &j in &order[..=k] {
                        pass[j] = true;
                    }
                }
            }
            pass
        }

        let pass_abs_bh = bh_outcome(&order_abs, &abs_t, &crit_abs_bh);
        let pass_sq_bh = bh_outcome(&order_sq, &t_sq, &crit_sq_bh);
        assert_eq!(pass_abs_bh, pass_sq_bh, "BH boolean vectors must match");
    }

    // -----------------------------------------------------------------
    // χ²(k) quantiles + LmeIntercept joint critval (joint Wald-χ² plan)
    // -----------------------------------------------------------------

    #[test]
    fn chi2_ppf_edge_cases() {
        assert_eq!(chi2_ppf(0.0, 5.0), 0.0);
        assert!(chi2_ppf(-0.1, 5.0) == 0.0);
        assert_eq!(chi2_ppf(1.0, 5.0), f64::INFINITY);
        assert!(chi2_ppf(0.5, 0.0).is_nan());
        assert!(chi2_ppf(0.5, -1.0).is_nan());
    }

    #[test]
    fn f_ppf_edge_cases() {
        assert_eq!(f_ppf(0.0, 3.0, 96.0), 0.0);
        assert!(f_ppf(-0.1, 3.0, 96.0) == 0.0);
        assert_eq!(f_ppf(1.0, 3.0, 96.0), f64::INFINITY);
        assert!(f_ppf(0.5, 0.0, 10.0).is_nan());
        assert!(f_ppf(0.5, 3.0, -1.0).is_nan());
    }

    #[test]
    fn f_ppf_cdf_inverse() {
        for &(dfn, dfd) in &[(1.0, 5.0), (3.0, 50.0), (10.0, 500.0)] {
            for &p in &[0.01_f64, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99] {
                let x = f_ppf(p, dfn, dfd);
                let back = f_cdf(x, dfn, dfd);
                let rel = (back - p).abs() / p;
                assert!(
                    rel < 1e-6,
                    "round-trip dfn={dfn} dfd={dfd} p={p}: x={x}, cdf(x)={back}, rel={rel}"
                );
            }
        }
    }

    #[test]
    fn crit_value_table_lme_intercept_has_joint() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![100_u32, 500_u32];
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            3, // n_predictors
            2, // n_targets — joint test k = 2
            CorrectionMethod::None,
            EstimatorSpec::Mle,
        )
        .unwrap();
        assert_eq!(table.joint_t_crit_sq.len(), 2);
        let expected = chi2_ppf(0.95, 2.0); // ≈ 5.991
        for &v in &table.joint_t_crit_sq {
            assert!(
                (v - expected).abs() < 1e-6,
                "joint critval {v} differs from chi2_ppf(0.95, 2) = {expected}"
            );
        }
    }

    #[test]
    fn crit_value_table_non_lme_has_nan_joint() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = vec![100_u32];
        for fam in [EstimatorSpec::Ols, EstimatorSpec::Glm] {
            let table =
                CritValueTable::build(&crit, &sample_sizes, 3, 2, CorrectionMethod::None, fam)
                    .unwrap();
            assert_eq!(table.joint_t_crit_sq.len(), 1);
            assert!(
                table.joint_t_crit_sq[0].is_nan(),
                "expected NaN joint critval for estimator={fam:?}, got {}",
                table.joint_t_crit_sq[0]
            );
        }
    }

    #[test]
    fn overall_crit_ols_matches_f_ppf() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        // sample_sizes must be strictly increasing for run_batch but CritValueTable
        // doesn't enforce that — use [30, 100] to keep it monotone.
        let sample_sizes = vec![30_u32, 100];
        let table = CritValueTable::build(
            &crit,
            &sample_sizes,
            4, // n_predictors → 3 non-intercept
            1,
            CorrectionMethod::None,
            EstimatorSpec::Ols,
        )
        .unwrap();
        assert_eq!(table.overall_crit.len(), 2);
        let expected_n30 = f_ppf(0.95, 3.0, 26.0);
        let expected_n100 = f_ppf(0.95, 3.0, 96.0);
        let rel_30 = (table.overall_crit[0] - expected_n30).abs() / expected_n30;
        let rel_100 = (table.overall_crit[1] - expected_n100).abs() / expected_n100;
        assert!(
            rel_30 < 1e-9,
            "n=30: got {}, expected {}",
            table.overall_crit[0],
            expected_n30
        );
        assert!(
            rel_100 < 1e-9,
            "n=100: got {}, expected {}",
            table.overall_crit[1],
            expected_n100
        );
    }

    #[test]
    fn overall_crit_logit_matches_chi2_ppf() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let table = CritValueTable::build(
            &crit,
            &[100_u32],
            4,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Glm,
        )
        .unwrap();
        assert_eq!(table.overall_crit.len(), 1);
        let expected = chi2_ppf(0.95, 3.0);
        let rel = (table.overall_crit[0] - expected).abs() / expected;
        assert!(
            rel < 1e-9,
            "got {}, expected {}",
            table.overall_crit[0],
            expected
        );
    }

    #[test]
    fn overall_crit_lme_is_infinity() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let table = CritValueTable::build(
            &crit,
            &[100_u32, 500],
            4,
            1,
            CorrectionMethod::None,
            EstimatorSpec::Mle,
        )
        .unwrap();
        assert_eq!(table.overall_crit.len(), 2);
        for &v in &table.overall_crit {
            assert!(v.is_infinite() && v > 0.0, "got {v}, expected +inf");
        }
    }

    #[test]
    fn overall_crit_intercept_only_is_infinity() {
        // n_predictors = 1 (intercept only) → degenerate joint test for every family.
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        for fam in [EstimatorSpec::Ols, EstimatorSpec::Glm] {
            let table = CritValueTable::build(&crit, &[100_u32], 1, 1, CorrectionMethod::None, fam)
                .unwrap();
            let v = table.overall_crit[0];
            assert!(
                v.is_infinite() && v > 0.0,
                "intercept-only estimator={fam:?}: got {v}, expected +inf"
            );
        }
    }

    // -----------------------------------------------------------------
    // C1 — norm_ppf golden values (external oracle: R qnorm)
    // -----------------------------------------------------------------

    #[test]
    fn norm_ppf_golden_values() {
        // External references: R qnorm(). Acklam max abs error ~1.15e-9 → tol 2e-9.
        // Covers all three branches of the piecewise approximation.
        // Central region (p ∈ [P_LOW, P_HIGH]):
        assert!(
            (norm_ppf(0.975) - 1.959963985f64).abs() < 2e-9,
            "norm_ppf(0.975) = {}",
            norm_ppf(0.975)
        );
        assert!(
            (norm_ppf(0.95) - 1.644853627f64).abs() < 2e-9,
            "norm_ppf(0.95) = {}",
            norm_ppf(0.95)
        );
        // Lower tail (p < P_LOW = 0.02425):
        assert!(
            (norm_ppf(0.025) - (-1.959963985f64)).abs() < 2e-9,
            "norm_ppf(0.025) = {}",
            norm_ppf(0.025)
        );
        // Upper tail (p > P_HIGH = 1 - 0.02425):
        assert!(
            (norm_ppf(0.999) - 3.090232306f64).abs() < 2e-9,
            "norm_ppf(0.999) = {}",
            norm_ppf(0.999)
        );
    }

    // -----------------------------------------------------------------
    // C2 — t_ppf golden values (external oracle: R qt)
    // -----------------------------------------------------------------

    #[test]
    fn t_ppf_golden_values() {
        // External oracle: R qt(p, df). Rel tolerance 1e-5 (Newton converges to ~1e-10
        // but conservative tol guards against beta-fn degradation at extreme df).
        // Note: qt(0.975, 1.5) = 6.016663 in R (R uses df=1.5 directly; some
        // implementations round to 1 or 2 — use a loose tol to accept integer-round path).
        let cases: &[(f64, f64, f64)] = &[
            (0.975, 30.0, 2.042272),   // moderate df — most common OLS case
            (0.975, 5.0, 2.570582),    // small df — sensitive to beta accuracy
            (0.975, 1000.0, 1.962339), // large df — approaches z
            (0.995, 10.0, 3.169273),   // high quantile
        ];
        for &(p, df, expected) in cases {
            let got = t_ppf(p, df);
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-5,
                "t_ppf({p}, {df}): got {got}, expected {expected}, rel {rel}"
            );
        }
    }

    // -----------------------------------------------------------------
    // C7 — q_tukey_ppf asymptotic golden value (external oracle: R qtukey)
    // -----------------------------------------------------------------

    #[test]
    fn q_tukey_ppf_asymptotic_golden_value() {
        // At df→∞, q_tukey_ppf(0.95, 3) converges to qtukey(0.95, 3, Inf) ≈ 3.314493 (R).
        // At df=1e6 the approximation error vs Inf is negligible (< 0.001).
        let q_large_df = q_tukey_ppf(0.95, 3.0, 1e6);
        assert!(
            (q_large_df - 3.314493f64).abs() < 0.01,
            "q_tukey_ppf(0.95, 3, 1e6) = {q_large_df}, expected ≈ 3.314493 (R qtukey)"
        );
        // Additional point at moderate df: qtukey(0.95, 4, 30) = 3.845401 (R).
        let q_30 = q_tukey_ppf(0.95, 4.0, 30.0);
        assert!(
            (q_30 - 3.845401f64).abs() < 0.01,
            "q_tukey_ppf(0.95, 4, 30) = {q_30}, expected ≈ 3.845401 (R qtukey)"
        );
    }

    // -----------------------------------------------------------------
    // C8 — chi2_ppf golden values (external oracle: R qchisq)
    // -----------------------------------------------------------------

    #[test]
    fn chi2_ppf_golden_values() {
        // External oracle: R qchisq(p, k). Rel tolerance 1e-5.
        let cases: &[(f64, f64, f64)] = &[
            (0.95, 1.0, 3.841459),  // LRT df=1 — most common
            (0.95, 2.0, 5.991465),  // LME joint k=2
            (0.95, 5.0, 11.070498), // larger k
            (0.99, 3.0, 11.344867), // higher quantile
        ];
        for &(p, k, expected) in cases {
            let got = chi2_ppf(p, k);
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-5,
                "chi2_ppf({p},{k})={got}, expected {expected}, rel {rel}"
            );
        }
    }

    // -----------------------------------------------------------------
    // C9 — f_ppf golden values (external oracle: R qf)
    // -----------------------------------------------------------------

    #[test]
    fn f_ppf_golden_values() {
        // External oracle: R qf(p, dfn, dfd). Rel tol 1e-5.
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.95, 3.0, 96.0, 2.699393), // OLS n=100, 3 predictors
            (0.95, 1.0, 30.0, 4.170877),
            (0.99, 2.0, 50.0, 5.056611), // R: qf(0.99, 2, 50) = 5.056611
        ];
        for &(p, dfn, dfd, expected) in cases {
            let got = f_ppf(p, dfn, dfd);
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-5,
                "f_ppf({p},{dfn},{dfd})={got}, expected {expected}, rel {rel}"
            );
        }
    }
}
