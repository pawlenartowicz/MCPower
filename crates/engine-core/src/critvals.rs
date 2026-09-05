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
use crate::spec::{
    ClusterSizing, CorrectionMethod, CritValues, DfMethod, EngineError, EstimatorSpec,
    SimulationSpec,
};

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
// Between-within denominator degrees of freedom
// ---------------------------------------------------------------------------

/// Inputs for the between-within denominator degrees of freedom (SAS
/// `DDFM=BETWITHIN`), resolved once per table build by [`BetweenWithinDf::resolve`].
///
/// Passing `None` to the builder selects the asymptotic (Wald-z) reference,
/// which is both today's default and the fallback for designs where the
/// between/within partition is not defined.
#[derive(Debug, Clone)]
pub struct BetweenWithinDf {
    /// Cluster count `g`, one entry per element of `sample_sizes`.
    pub clusters_per_n: Vec<usize>,
    /// Fitted columns that are constant within cluster: the intercept, every
    /// cluster-level predictor, and every interaction whose factors are all
    /// cluster-level.
    pub p_between: u32,
    /// One flag per target (marginals in `target_indices` order, then
    /// contrasts): `true` selects `df_between`, `false` `df_within`.
    pub between_per_target: Vec<bool>,
    /// The fit collapsed to a single-level model: every variance component
    /// pinned at zero (lme4's "singular fit"), so the fitted model is the
    /// plain OLS on the same columns and every target is referred to its
    /// residual df `N − P` instead of the between/within split. Set only by
    /// [`BetweenWithinDf::collapsed`]; `resolve` always yields `false`.
    pub collapsed: bool,
}

/// Whether kernel column `col` is constant within cluster. The intercept is;
/// `between_var_indices` names the cluster-level predictors (factor dummies
/// already expanded); an interaction inherits the property only if every one of
/// its factors has it — a product with any within-cluster factor varies within
/// cluster.
fn column_is_cluster_level(spec: &SimulationSpec, col: u32) -> bool {
    if col == 0 {
        return true;
    }
    if spec.between_var_indices.contains(&col) {
        return true;
    }
    let first_interaction = 1 + spec.n_non_factor + spec.n_factor_dummies;
    if col >= first_interaction {
        if let Some(factors) = spec.interactions.get((col - first_interaction) as usize) {
            return !factors.is_empty()
                && factors.iter().all(|c| spec.between_var_indices.contains(c));
        }
    }
    false
}

/// `p_between`: how many of the fitted columns are constant within cluster.
/// `kept` is the ascending kernel-column set of a reduced (sparse-exclusion /
/// test-formula) rebuild; `None` counts the full design.
pub fn between_column_count(spec: &SimulationSpec, kept: Option<&[u32]>) -> u32 {
    match kept {
        Some(cols) => cols
            .iter()
            .filter(|&&c| column_is_cluster_level(spec, c))
            .count() as u32,
        None => {
            let p = 1 + spec.n_non_factor + spec.n_factor_dummies + spec.interactions.len() as u32;
            (0..p).filter(|&c| column_is_cluster_level(spec, c)).count() as u32
        }
    }
}

impl BetweenWithinDf {
    /// Resolve the between-within inputs for `spec`, or `None` when the
    /// asymptotic reference applies: `df_method` is `Asymptotic`, the fit is
    /// not mixed (`Ols`, or `Glm` without a cluster spec), or the design
    /// carries crossed / nested extra groupings, for which the between/within
    /// partition of the residual df is not defined. `p_between` is supplied by
    /// the caller so a reduced-model rebuild can pass its own count.
    ///
    /// The clustered-`Glm` half of the predicate below is the same condition
    /// `WaldSe::affects` uses; `Mle` is in scope here and not there, so the two
    /// are not interchangeable.
    pub fn resolve(
        spec: &SimulationSpec,
        sample_sizes: &[u32],
        p_between: u32,
    ) -> Option<BetweenWithinDf> {
        if spec.df_method != DfMethod::BetweenWithin {
            return None;
        }
        // The `?` already established a cluster spec, so a surviving `Glm` is the
        // clustered GLMM; `Ols` over clustered data is not a mixed fit.
        let cluster = spec.cluster.as_ref()?;
        let mixed = matches!(spec.estimator, EstimatorSpec::Mle | EstimatorSpec::Glm);
        if !mixed || !cluster.extra_groupings.is_empty() {
            return None;
        }
        // `n_clusters_at` floors; the row→cluster map assigns the remainder rows
        // of a `FixedSize` design to one further partial cluster, so the count
        // that matches the data is the ceiling.
        let clusters_per_n = sample_sizes
            .iter()
            .map(|&n| match cluster.sizing {
                ClusterSizing::FixedClusters { n_clusters } => n_clusters.max(1) as usize,
                ClusterSizing::FixedSize { cluster_size } => {
                    let s = cluster_size.max(1) as usize;
                    (n as usize).div_ceil(s)
                }
            })
            .collect();
        let mut between_per_target: Vec<bool> = spec
            .target_indices
            .iter()
            .map(|&c| column_is_cluster_level(spec, c))
            .collect();
        // A contrast β_p − β_n is cluster-level only if both sides are; one
        // within-cluster side makes the difference vary within cluster.
        between_per_target.extend(
            spec.contrast_pairs.iter().map(|&(p, n)| {
                column_is_cluster_level(spec, p) && column_is_cluster_level(spec, n)
            }),
        );
        Some(BetweenWithinDf {
            clusters_per_n,
            p_between,
            between_per_target,
            collapsed: false,
        })
    }

    /// The reference for a fit whose variance components all pinned at zero.
    /// The between/within split assumes a real cluster variance; without one
    /// the fitted model is the single-level one on the same columns, and its
    /// exact test is the residual-df Student-t for a Gaussian outcome
    /// (`Some`, every target at `N − P`) and the Wald z for a GLM (`None`, the
    /// asymptotic table). A Satterthwaite reference does the same for a
    /// cluster-level effect on a balanced design: off the boundary its df
    /// equals `g − p_between`, on the boundary it jumps to the residual df.
    pub fn collapsed(&self, estimator: EstimatorSpec) -> Option<BetweenWithinDf> {
        match estimator {
            EstimatorSpec::Mle => Some(BetweenWithinDf {
                collapsed: true,
                ..self.clone()
            }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CritValueTable
// ---------------------------------------------------------------------------

/// Built once per `run_batch` invocation, indexed by sample-size position.
/// All values are squared so the per-sim path compares `t_sq > t_crit_sq`
/// directly without a square root on the SE side.
#[derive(Debug, Clone)]
pub struct CritValueTable {
    /// Outer length = `sample_sizes.len()`; inner length = `n_targets`.
    /// One threshold per target: they coincide except under the between-within
    /// denominator df, where a cluster-level target is referred to a different
    /// df than a within-cluster one.
    pub t_crit_sq_uncorrected: Vec<Vec<f64>>,
    /// Denominator df behind each entry of `t_crit_sq_uncorrected`, same shape.
    /// Inner rows are EMPTY whenever no df enters the threshold — every
    /// standard-normal (Wald-z) build, which is `Glm`/`Mle` under
    /// `DfMethod::Asymptotic` and every fallback in [`BetweenWithinDf::resolve`].
    /// OLS repeats its residual df `N − P` per target.
    pub df_per_target: Vec<Vec<f64>>,
    /// Outer length = `sample_sizes.len()`; inner length = `n_targets`.
    /// Per-target thresholds sequenced as v1 does (entry `k` walks the
    /// Bonferroni-step sequence under Holm and the BH step-up sequence under
    /// BH).
    pub correction_t_crit_sq: Vec<Vec<f64>>,
    /// One entry per element of `sample_sizes`; equals
    /// `t_ppf(1 - posthoc_alpha/2, df)²` (or the uncorrected entry when
    /// `posthoc_alpha` is None).
    pub posthoc_t_crit_sq: Vec<f64>,
    /// Joint Wald critical value over the whole target set, one entry per
    /// element of `sample_sizes`; `NaN` where no joint test is defined.
    /// - asymptotic mixed (`EstimatorSpec::Mle`) → χ²(k, 1−α), df-independent so
    ///   every entry is equal (the per-N layout matches `t_crit_sq_uncorrected`).
    /// - between-within mixed (`Mle`, or `Glm` with a cluster spec) →
    ///   `k · F(1−α; k, df_joint)`, with `df_joint` the smallest df in the
    ///   target set, so it does vary with N.
    /// - everything else → `NaN`.
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
    /// Whether a denominator df backs these thresholds — true for OLS (its
    /// residual df) and for a mixed build under the between-within reference,
    /// false for every standard-normal (Wald-z) build. Inside a mixed estimator
    /// arm it is exactly "the between-within split is active", which is what
    /// decides whether a reduced-model round has to rebuild the table.
    pub fn has_df(&self) -> bool {
        self.df_per_target.iter().any(|row| !row.is_empty())
    }

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
            None,
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
    /// `bw` is `Some` only for a mixed fit that asked for the between-within
    /// denominator degrees of freedom (SAS `DDFM=BETWITHIN`) and for which the
    /// partition is defined — see [`BetweenWithinDf::resolve`]. It replaces the
    /// standard-normal reference with a Student-t / F reference at a per-target
    /// df; `None` reproduces the asymptotic table exactly.
    ///
    /// [`build`]: CritValueTable::build
    #[expect(
        clippy::too_many_arguments,
        reason = "single df authority; each argument is one independent table input"
    )]
    pub fn build_with_tukey_k(
        crit: &CritValues,
        sample_sizes: &[u32],
        n_predictors: u32,
        n_targets: u32,
        correction_method: CorrectionMethod,
        estimator: EstimatorSpec,
        tukey_k_per_target: &[f64],
        bw: Option<&BetweenWithinDf>,
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
        let mut df_per_target = Vec::with_capacity(n);
        let mut correction_t_crit_sq = Vec::with_capacity(n);
        let mut posthoc_t_crit_sq = Vec::with_capacity(n);
        // Smallest reference df across the target set, one entry per N. Feeds
        // both the rank-indexed corrections and the joint Wald crit below.
        let mut df_family_per_n = Vec::with_capacity(n);
        let between_flags: &[bool] = bw.map(|b| b.between_per_target.as_slice()).unwrap_or(&[]);
        debug_assert!(
            bw.is_none() || between_flags.len() == m,
            "between-within callers must pass one between/within flag per target"
        );

        for (n_idx, &sample_size) in sample_sizes.iter().enumerate() {
            let df_signed = sample_size as i64 - n_predictors as i64;
            if df_signed < 1 {
                return Err(EngineError::InvalidSpec(format!(
                    "df = N - P = {sample_size} - {n_predictors} < 1"
                )));
            }
            let df = df_signed as f64;

            // Between-within denominator degrees of freedom (SAS
            // `DDFM=BETWITHIN`): the residual df splits into a between-cluster
            // part `g − p_between`, which carries the cluster-level effects, and
            // the within-cluster remainder, which carries the rest. `None` keeps
            // every target on the asymptotic reference.
            let bw_split: Option<(f64, f64)> = match bw {
                None => None,
                // Collapsed fit: the single-level model's residual df on every
                // target, still a Student-t reference.
                Some(b) if b.collapsed => Some((df, df)),
                Some(b) => {
                    let g = b.clusters_per_n[n_idx] as i64;
                    let df_between = g - b.p_between as i64;
                    let df_within = df_signed - df_between;
                    if df_between < 1 || df_within < 1 {
                        return Err(EngineError::InvalidSpec(format!(
                            "between-within df at N = {sample_size}: between = g - p_between = \
                             {g} - {} = {df_between}, within = {df_within}; both must be >= 1. \
                             Use df_method = \"asymptotic\" for the Wald z reference, or add clusters",
                            b.p_between
                        )));
                    }
                    Some((df_between as f64, df_within as f64))
                }
            };
            let df_at = |i: usize| -> f64 {
                match bw_split {
                    Some((between, within)) => {
                        if between_flags.get(i).copied().unwrap_or(false) {
                            between
                        } else {
                            within
                        }
                    }
                    None => df,
                }
            };
            // Under the between-within reference a mixed Wald statistic is
            // referred to Student-t at the target's own df instead of the
            // standard normal; every other build keeps `crit_quantile`.
            let quantile_at = |p: f64, i: usize| -> f64 {
                if bw_split.is_some() {
                    t_ppf(p, df_at(i))
                } else {
                    crit_quantile(p, df)
                }
            };
            // Holm's and BH's row entries are RANK-indexed — which target lands
            // at rank k is only known once a sim's t² values are sorted — so no
            // per-target df can be baked in. Both fall back to the smallest df in
            // the target set, the most conservative reference of the family, as
            // does the joint Wald crit. Equal to `df` outside the split.
            let df_family = if m == 0 {
                df
            } else {
                (0..m).map(&df_at).fold(f64::INFINITY, f64::min)
            };
            df_family_per_n.push(df_family);

            let unc_row: Vec<f64> = (0..m)
                .map(|i| {
                    let t = quantile_at(1.0 - alpha / 2.0, i);
                    t * t
                })
                .collect();
            df_per_target.push(if bw_split.is_some() {
                (0..m).map(&df_at).collect()
            } else if estimator.uses_student_t() {
                vec![df; m]
            } else {
                // Wald-z: no df enters the threshold, so there is none to report.
                Vec::new()
            });

            let mut row = Vec::with_capacity(m.max(1));
            if m == 0 {
                // No targets — push an empty per-N row.
            } else {
                match correction_method {
                    CorrectionMethod::None => {
                        // None.
                        row.extend_from_slice(&unc_row);
                    }
                    CorrectionMethod::Bonferroni => {
                        // Bonferroni.
                        for i in 0..m {
                            let bonf = quantile_at(1.0 - alpha / (2.0 * m as f64), i);
                            row.push(bonf * bonf);
                        }
                    }
                    CorrectionMethod::Holm => {
                        // Holm step-down: entry k uses α / (m - k).
                        for k in 0..m {
                            let eff = alpha / (2.0 * (m - k) as f64);
                            let t = if eff < 1e-12 {
                                f64::INFINITY
                            } else if bw_split.is_some() {
                                t_ppf(1.0 - eff, df_family)
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
                            } else if bw_split.is_some() {
                                t_ppf(1.0 - eff, df_family)
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
                            let q = q_tukey_ppf(1.0 - alpha, k, df_at(i));
                            let t = q / std::f64::consts::SQRT_2;
                            row.push(t * t);
                        }
                    }
                }
            }
            correction_t_crit_sq.push(row);
            t_crit_sq_uncorrected.push(unc_row);

            // Post-hoc contrasts are rejected outside OLS at batch entry, so the
            // between-within split never reaches this threshold.
            let t_post = crit_quantile(1.0 - posthoc_alpha / 2.0, df);
            posthoc_t_crit_sq.push(t_post * t_post);
        }

        // Joint Wald critical value over the whole target set. The asymptotic
        // mixed test is χ²(k, 1-α), df-independent, but stored one entry per N
        // for layout parity with `t_crit_sq_uncorrected`. Under the
        // between-within reference the joint statistic is referred to
        // `k · F(1−α; k, df_joint)` instead, so the entries do vary with N —
        // and the clustered GLMM gets a value too, where the asymptotic build
        // leaves NaN and batch.rs recomputes χ² inline.
        let joint_t_crit_sq: Vec<f64> = if n_targets == 0 {
            vec![f64::NAN; n]
        } else if bw.is_some() {
            let k = n_targets as f64;
            df_family_per_n
                .iter()
                .map(|&df_joint| k * f_ppf(1.0 - alpha, k, df_joint))
                .collect()
        } else {
            match estimator {
                EstimatorSpec::Mle => vec![chi2_ppf(1.0 - alpha, n_targets as f64); n],
                _ => vec![f64::NAN; n],
            }
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
            df_per_target,
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
            for &v in &table.t_crit_sq_uncorrected[i] {
                assert_eq!(t_crit * t_crit, v);
            }
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
        assert_eq!(
            table.posthoc_t_crit_sq[0],
            table.t_crit_sq_uncorrected[0][0]
        );
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
        let got = table.t_crit_sq_uncorrected[0][0];
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
        let logit_z_sq = tab_logit.t_crit_sq_uncorrected[0][0];
        let lme_z_sq = tab_lme.t_crit_sq_uncorrected[0][0];
        let delta = (logit_z_sq - lme_z_sq).abs();
        assert!(
            delta < 1e-12,
            "Glm z² = {}, Mle z² = {}, delta = {}",
            logit_z_sq,
            lme_z_sq,
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

    // -----------------------------------------------------------------
    // Between-within denominator degrees of freedom (SAS DDFM=BETWITHIN)
    // -----------------------------------------------------------------

    /// Balanced one-way mixed design: intercept + one cluster-level continuous
    /// predictor (kernel column 1) + one within-cluster one (column 2), so
    /// `p = 3` and `p_between = 2`. Targets are those two columns, in that
    /// order.
    fn bw_two_predictor_spec(estimator: EstimatorSpec, sizing: ClusterSizing) -> SimulationSpec {
        let mut spec = crate::spec::tests::minimal_ols_spec();
        spec.estimator = estimator;
        spec.df_method = DfMethod::BetweenWithin;
        spec.n_non_factor = 2;
        spec.between_var_indices = vec![1];
        spec.target_indices = vec![1, 2];
        spec.cluster = Some(crate::spec::ClusterSpec {
            sizing,
            tau_squared: 0.5,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec
    }

    /// Build the table for `spec` at one sample size, with `n_predictors` fitted
    /// columns of which `p_between` are cluster-level.
    fn bw_table(
        spec: &SimulationSpec,
        n: u32,
        n_predictors: u32,
        p_between: u32,
        correction_method: CorrectionMethod,
        tukey_k: &[f64],
    ) -> CritValueTable {
        let bw = BetweenWithinDf::resolve(spec, &[n], p_between);
        CritValueTable::build_with_tukey_k(
            &spec.crit_values,
            &[n],
            n_predictors,
            (spec.target_indices.len() + spec.contrast_pairs.len()) as u32,
            correction_method,
            spec.estimator,
            tukey_k,
            bw.as_ref(),
        )
        .unwrap()
    }

    /// CRIT-11: the between-within split of a balanced one-way design matches
    /// the hand arithmetic. N = 40 rows in clusters of 8 ⇒ g = 5; the fitted
    /// design is intercept + one cluster-level + one within-cluster predictor
    /// ⇒ p = 3, p_between = 2. So df_resid = 40 − 3 = 37,
    /// df_between = 5 − 2 = 3, df_within = 37 − 3 = 34.
    #[test]
    fn bw_df_of_balanced_one_way_matches_hand_computation() {
        let spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        let table = bw_table(&spec, 40, 3, 2, CorrectionMethod::None, &[]);
        assert_eq!(table.df_per_target[0], vec![3.0, 34.0]);
    }

    /// CRIT-12: the cluster-level target is referred to df_between and the
    /// within-cluster one to df_within — both as Student-t, not the standard
    /// normal the asymptotic build uses for `Mle`.
    #[test]
    fn bw_between_target_gets_between_df_within_target_gets_within_df() {
        let spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        let table = bw_table(&spec, 40, 3, 2, CorrectionMethod::None, &[]);
        let between = t_ppf(0.975, 3.0);
        let within = t_ppf(0.975, 34.0);
        assert_eq!(
            table.t_crit_sq_uncorrected[0],
            vec![between * between, within * within]
        );
        // The cluster-level target is the harder test: fewer df ⇒ larger crit.
        assert!(table.t_crit_sq_uncorrected[0][0] > table.t_crit_sq_uncorrected[0][1]);
        // ... and both exceed the asymptotic z² the same design would have used.
        let z = norm_ppf(0.975);
        assert!(table.t_crit_sq_uncorrected[0][1] > z * z);
    }

    /// CRIT-13: no-op tripwire. Under `DfMethod::Asymptotic` every value the
    /// table holds is bit-identical to what the pre-between-within code
    /// produced; the literals below were captured from a build of that
    /// code, not recomputed from the current one.
    #[test]
    fn asymptotic_tables_are_bit_identical_to_pre_change_values() {
        let a = CritValueTable::build(
            &CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            &[40, 80],
            4,
            3,
            CorrectionMethod::Holm,
            EstimatorSpec::Mle,
        )
        .unwrap();
        assert_eq!(
            a.t_crit_sq_uncorrected,
            vec![
                vec![3.841458826888164, 3.841458826888164, 3.841458826888164],
                vec![3.841458826888164, 3.841458826888164, 3.841458826888164],
            ]
        );
        assert_eq!(
            a.correction_t_crit_sq,
            vec![
                vec![5.731139288752486, 5.02388618220602, 3.841458826888164],
                vec![5.731139288752486, 5.02388618220602, 3.841458826888164],
            ]
        );
        assert_eq!(
            a.posthoc_t_crit_sq,
            vec![3.841458826888164, 3.841458826888164]
        );
        assert_eq!(
            a.joint_t_crit_sq,
            vec![7.814727903251182, 7.814727903251182]
        );
        assert_eq!(a.overall_crit, vec![f64::INFINITY, f64::INFINITY]);
        // Wald-z build ⇒ no df backs the threshold.
        assert!(a.df_per_target.iter().all(|r| r.is_empty()));

        let b = CritValueTable::build(
            &CritValues {
                alpha: 0.05,
                posthoc_alpha: Some(0.01),
            },
            &[20, 50],
            4,
            3,
            CorrectionMethod::BenjaminiHochberg,
            EstimatorSpec::Ols,
        )
        .unwrap();
        assert_eq!(
            b.t_crit_sq_uncorrected,
            vec![
                vec![4.493998477666361, 4.493998477666361, 4.493998477666361],
                vec![4.0517486921491965, 4.0517486921491965, 4.0517486921491965],
            ]
        );
        assert_eq!(
            b.correction_t_crit_sq,
            vec![
                vec![7.145101604364694, 5.421079040141435, 4.493998477666361],
                vec![6.173691995153475, 4.813165220184376, 4.0517486921491965],
            ]
        );
        assert_eq!(
            b.posthoc_t_crit_sq,
            vec![8.530965285896206, 7.220041507491684]
        );
        assert!(b.joint_t_crit_sq.iter().all(|v| v.is_nan()));
        assert_eq!(b.overall_crit, vec![3.238871517453583, 2.806844928806262]);
        // OLS reports its residual df N − P per target.
        assert_eq!(
            b.df_per_target,
            vec![vec![16.0, 16.0, 16.0], vec![46.0, 46.0, 46.0]]
        );

        let c = CritValueTable::build_with_tukey_k(
            &CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            &[30],
            3,
            2,
            CorrectionMethod::TukeyHsd,
            EstimatorSpec::Ols,
            &[3.0, 3.0],
            None,
        )
        .unwrap();
        assert_eq!(
            c.t_crit_sq_uncorrected,
            vec![vec![4.210008468359758, 4.210008468359758]]
        );
        assert_eq!(
            c.correction_t_crit_sq,
            vec![vec![6.147514869456612, 6.147514869456612]]
        );
        assert_eq!(c.posthoc_t_crit_sq, vec![4.210008468359758]);
        assert!(c.joint_t_crit_sq[0].is_nan());
        assert_eq!(c.overall_crit, vec![3.3541308285292044]);

        let d = CritValueTable::build(
            &CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            &[40, 80],
            4,
            3,
            CorrectionMethod::Bonferroni,
            EstimatorSpec::Glm,
        )
        .unwrap();
        assert_eq!(
            d.correction_t_crit_sq,
            vec![
                vec![5.731139288752486, 5.731139288752486, 5.731139288752486],
                vec![5.731139288752486, 5.731139288752486, 5.731139288752486],
            ]
        );
        assert!(d.joint_t_crit_sq.iter().all(|v| v.is_nan()));
        assert_eq!(d.overall_crit, vec![7.814727903251182, 7.814727903251182]);
    }

    /// CRIT-14: an interaction of two cluster-level columns is itself
    /// cluster-level and counts in `p_between`; an interaction that pulls in a
    /// within-cluster column does not. Design: three continuous predictors
    /// (columns 1-3) with 1 and 3 cluster-level, plus interactions x1:x3
    /// (column 4) and x1:x2 (column 5).
    #[test]
    fn bw_all_between_interaction_counts_in_p_between_mixed_one_does_not() {
        let mut spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 10 },
        );
        spec.n_non_factor = 3;
        spec.between_var_indices = vec![1, 3];
        spec.interactions = vec![vec![1, 3], vec![1, 2]];
        spec.target_indices = vec![4, 5];
        // intercept + columns 1 and 3 + the all-between interaction = 4 of 6.
        assert_eq!(between_column_count(&spec, None), 4);

        // N = 100 in clusters of 10 ⇒ g = 10; p = 6, p_between = 4 ⇒
        // df_between = 6, df_resid = 94, df_within = 88.
        let table = bw_table(&spec, 100, 6, 4, CorrectionMethod::None, &[]);
        assert_eq!(table.df_per_target[0], vec![6.0, 88.0]);
    }

    /// CRIT-15: a reduced-model rebuild that drops a cluster-level dummy carries
    /// the reduced `p_between`, which moves the split — dropping a within-cluster
    /// column would not. Design: intercept + one within-cluster continuous
    /// (column 1) + a cluster-level 3-level factor (dummies at columns 2 and 3).
    #[test]
    fn bw_reduced_rebuild_uses_reduced_p_between() {
        let mut spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 10 },
        );
        spec.n_non_factor = 1;
        spec.n_factor_dummies = 2;
        spec.factor_n_levels = vec![3];
        spec.between_var_indices = vec![2, 3];
        spec.target_indices = vec![2, 1];
        assert_eq!(between_column_count(&spec, None), 3);

        // Dropping the second dummy leaves columns [0, 1, 2]: p_red = 3 of which
        // p_between_red = 2 (intercept + the surviving dummy).
        let kept = [0u32, 1, 2];
        let p_between_red = between_column_count(&spec, Some(&kept));
        assert_eq!(p_between_red, 2);

        // N = 100 in clusters of 10 ⇒ g = 10. Full: p = 4, p_between = 3 ⇒
        // df_between = 7, df_within = 96 − 7 = 89. Reduced: p_red = 3,
        // p_between_red = 2 ⇒ df_between = 8, df_within = 97 − 8 = 89.
        let full = bw_table(&spec, 100, 4, 3, CorrectionMethod::None, &[]);
        assert_eq!(full.df_per_target[0], vec![7.0, 89.0]);
        let reduced = bw_table(&spec, 100, 3, p_between_red, CorrectionMethod::None, &[]);
        assert_eq!(reduced.df_per_target[0], vec![8.0, 89.0]);
    }

    /// CRIT-16: crossed or nested extra groupings leave the between-within
    /// partition undefined, so the build falls back to the asymptotic table —
    /// value for value, not merely without panicking.
    #[test]
    fn bw_falls_back_to_asymptotic_for_crossed_and_nested_groupings() {
        let base = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        let mut asymptotic = base.clone();
        asymptotic.df_method = DfMethod::Asymptotic;
        let reference = bw_table(&asymptotic, 40, 3, 2, CorrectionMethod::Bonferroni, &[]);

        for extra in [
            engine_contract::GroupingRelation::Crossed { n_clusters: 4 },
            engine_contract::GroupingRelation::NestedWithin { n_per_parent: 2 },
        ] {
            let mut spec = base.clone();
            spec.cluster.as_mut().unwrap().extra_groupings = vec![engine_contract::GroupingSpec {
                relation: extra.clone(),
                tau_squared: 0.3,
                slopes: vec![],
            }];
            assert!(
                BetweenWithinDf::resolve(&spec, &[40], 2).is_none(),
                "{extra:?} must refuse the between-within split"
            );
            let table = bw_table(&spec, 40, 3, 2, CorrectionMethod::Bonferroni, &[]);
            assert_eq!(
                table.t_crit_sq_uncorrected, reference.t_crit_sq_uncorrected,
                "{extra:?} must reproduce the asymptotic thresholds"
            );
            assert_eq!(table.correction_t_crit_sq, reference.correction_t_crit_sq);
            assert_eq!(table.joint_t_crit_sq, reference.joint_t_crit_sq);
            // The refusal is observable: no df backs an asymptotic threshold.
            assert!(table.df_per_target.iter().all(|r| r.is_empty()));
        }
    }

    /// CRIT-17: the split depends on the design, not the solver — a Gaussian LMM
    /// and a clustered GLMM over the same design get the same per-target df and
    /// the same thresholds.
    #[test]
    fn bw_df_identical_for_mle_and_clustered_glm() {
        let sizing = ClusterSizing::FixedSize { cluster_size: 8 };
        let mle = bw_two_predictor_spec(EstimatorSpec::Mle, sizing.clone());
        let glmm = bw_two_predictor_spec(EstimatorSpec::Glm, sizing);
        let a = bw_table(&mle, 40, 3, 2, CorrectionMethod::None, &[]);
        let b = bw_table(&glmm, 40, 3, 2, CorrectionMethod::None, &[]);
        assert_eq!(a.df_per_target, b.df_per_target);
        assert_eq!(a.df_per_target[0], vec![3.0, 34.0]);
        assert_eq!(a.t_crit_sq_uncorrected, b.t_crit_sq_uncorrected);
        // The clustered GLMM gets a joint crit under the split, where the
        // asymptotic build leaves NaN.
        assert_eq!(a.joint_t_crit_sq, b.joint_t_crit_sq);
        assert!(b.joint_t_crit_sq[0].is_finite());
    }

    /// CRIT-18: the joint Wald crit is `k · F(1−α; k, df_joint)` with `df_joint`
    /// the smallest df in the target set. The set here mixes a cluster-level
    /// target (df 3) with a within-cluster one (df 34), so the smaller wins.
    /// External oracle: R `qf(0.95, 2, 3)` = 9.552094.
    #[test]
    fn bw_joint_crit_uses_smallest_df_in_the_target_set() {
        let spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        let table = bw_table(&spec, 40, 3, 2, CorrectionMethod::None, &[]);
        let expected = 2.0 * 9.552094;
        let got = table.joint_t_crit_sq[0];
        assert!(
            (got - expected).abs() / expected < 1e-6,
            "joint crit {got}, expected 2 * qf(0.95, 2, 3) = {expected}"
        );
        // Both targets within-cluster ⇒ the larger df, hence a smaller crit.
        let mut all_within = spec.clone();
        all_within.between_var_indices = vec![];
        let within_only = bw_table(&all_within, 40, 3, 1, CorrectionMethod::None, &[]);
        assert!(within_only.joint_t_crit_sq[0] < got);
    }

    /// CRIT-19: `FixedSize` cluster counting rounds UP. `n_clusters_at` floors,
    /// but the row-to-cluster map puts the remainder rows in one further
    /// partial cluster, so the count that matches the data is the ceiling:
    /// N = 41 in clusters of 8 is 6 clusters, not 5.
    #[test]
    fn bw_cluster_count_is_ceiling_under_fixed_size() {
        let sizing = ClusterSizing::FixedSize { cluster_size: 8 };
        assert_eq!(sizing.n_clusters_at(41), 5);
        let spec = bw_two_predictor_spec(EstimatorSpec::Mle, sizing);
        let bw = BetweenWithinDf::resolve(&spec, &[41], 2).unwrap();
        assert_eq!(bw.clusters_per_n, vec![6]);
        // g = 6, p = 3, p_between = 2 ⇒ df_between = 4, df_within = 38 − 4 = 34.
        let table = bw_table(&spec, 41, 3, 2, CorrectionMethod::None, &[]);
        assert_eq!(table.df_per_target[0], vec![4.0, 34.0]);
    }

    /// CRIT-20: the studentized-range df follows the same between/within rule as
    /// the marginal test — a cluster-level factor's dummies go against
    /// df_between, a within-cluster factor's against df_within.
    #[test]
    fn bw_tukey_df_follows_the_target_between_within_level() {
        let mut spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        spec.n_non_factor = 0;
        spec.n_factor_dummies = 2;
        spec.factor_n_levels = vec![2, 2];
        // Column 1 = the cluster-level factor's dummy, column 2 = the
        // within-cluster factor's.
        spec.between_var_indices = vec![1];
        spec.target_indices = vec![1, 2];
        let table = bw_table(&spec, 40, 3, 2, CorrectionMethod::TukeyHsd, &[2.0, 2.0]);
        let row = &table.correction_t_crit_sq[0];
        let between = q_tukey_ppf(0.95, 2.0, 3.0) / std::f64::consts::SQRT_2;
        let within = q_tukey_ppf(0.95, 2.0, 34.0) / std::f64::consts::SQRT_2;
        assert_eq!(row[0], between * between);
        assert_eq!(row[1], within * within);
        assert!(row[0] > row[1]);
    }

    /// CRIT-21: the collapsed reference. An LMM whose variance components all
    /// pinned at zero is the OLS on the same columns, so every target — the
    /// cluster-level one included — goes against Student-t at the residual df
    /// `N − P` (here 40 − 3 = 37), and the joint test against `k · F(k, 37)`.
    /// A collapsed GLMM is a plain GLM, whose reference is the Wald z: `None`.
    #[test]
    fn bw_collapsed_fit_refers_every_target_to_the_residual_df() {
        let spec = bw_two_predictor_spec(
            EstimatorSpec::Mle,
            ClusterSizing::FixedSize { cluster_size: 8 },
        );
        let split = BetweenWithinDf::resolve(&spec, &[40], 2).unwrap();
        let collapsed = split.collapsed(EstimatorSpec::Mle).unwrap();
        assert!(collapsed.collapsed && !split.collapsed);
        let table = CritValueTable::build_with_tukey_k(
            &spec.crit_values,
            &[40],
            3,
            2,
            CorrectionMethod::None,
            EstimatorSpec::Mle,
            &[],
            Some(&collapsed),
        )
        .unwrap();
        assert_eq!(table.df_per_target[0], vec![37.0, 37.0]);
        let t = t_ppf(0.975, 37.0);
        assert_eq!(table.t_crit_sq_uncorrected[0], vec![t * t, t * t]);
        assert_eq!(table.joint_t_crit_sq[0], 2.0 * f_ppf(0.95, 2.0, 37.0));
        // Off the boundary the same design gives df 3 / 34 (CRIT-19).
        let live = bw_table(&spec, 40, 3, 2, CorrectionMethod::None, &[]);
        assert_eq!(live.df_per_target[0], vec![3.0, 34.0]);
        assert!(split.collapsed(EstimatorSpec::Glm).is_none());
    }
}
