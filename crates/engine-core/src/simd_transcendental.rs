//! Restricted-domain SIMD transcendentals for the GLM/GLMM fit path — a single
//! compiled-in, full-precision `exp`/`log1p` that vectorizes across the PIRLS
//! row loop (the scalar libm `exp`/`ln_1p` are extern calls the compiler cannot
//! vectorize; profiling put them at ~51% of the no-extras GLMM fit).
//!
//! Two primitives, on exactly the domains the fit path needs — the stable
//! sigmoid (`glm::sigmoid_stable`) and `log1pexp` — namely `exp` of a non-positive
//! argument and `log1p` of `z ∈ (0,1]`:
//!   - `exp` on `[−700, 700]` (clamped): Cody-Waite reduce `x = k·ln2 + r`,
//!     degree-11 minimax `exp(r)` on `[-ln2/2, ln2/2]`, scale by `2^k`. The
//!     reduction is sign-agnostic, so the same kernel serves the fit path's
//!     non-positive arguments and the generation-side full-domain entries
//!     (`exp_clamped`/`exp_fill`).
//!   - `log1p` on `(0,1]`: fdlibm form `f - (hfsq - s·(hfsq + R))`, `s = f/(2+f)`,
//!     `R = w·h(w)`, `w = s²`, degree-9 minimax `h`. Keeps the dominant `f - ½f²`
//!     exact → tail-safe (never forms `1+z`) and ≤1 ULP.
//!
//! Coefficients were derived offline by Remez minimax against an MPFR-class
//! oracle; the `≤1 ULP` accuracy is re-asserted in-repo by the L1 guard test
//! below (vs system libm, the same bar the scalar path meets). The composed
//! sigmoid `p = 1/(1+z)` inherits one extra division → 2 ULP, identical to the
//! scalar `sigmoid_stable` it replaces.
//!
//! **Not** bit-identical to the previous libm helper (different polynomial + a
//! lane-wise deviance reduction): the GLMM goldens move by ≤ a few ULP and are
//! re-frozen. Determinism is within-platform per engine version; SIMD lane width
//! is runtime-dispatched (AVX2→4, AVX-512→8, wasm128→2), so a horizontal
//! reduction sums a µarch-dependent number of partials — cross-µarch results
//! agree only to the last bits, by design (not gated on byte-identity).
//!
//! `2^k` is built with an FP magic-add + `u64` transmute/mask/mul rather than the
//! textbook `(k+1023)<<52`: pulp 0.22's `Simd` trait exposes no integer
//! shift/convert. Valid for `k ≥ -1023` (i.e. `x ⪆ -709`), always true on the
//! fit path where `η` is bounded by `BETA_CAP`.
//!
//! **fma policy.** wasm simd128 has no FMA instruction, so a guaranteed-fused
//! `mul_add` lowers to the soft-float compiler-builtins libcall there (measured
//! 9–41× on the GLM/GLMM wasm rows). Every fused site therefore goes through
//! `fmadd`/`fmadd_scalar`, keyed by a `const FUSED: bool` generic: native
//! instantiates `FUSED = true` (hardware fma, byte-identical to the pre-policy
//! code), wasm32 instantiates `FUSED = false` (plain mul/add, vectorizable
//! simd128). Native↔wasm byte-equality is deliberately dropped — the wasm bench
//! gate is tier-2 |Δk| instead. **All future fused ops in owned kernels go
//! through `fmadd`/`fmadd_scalar` — never raw `mul_add`.**
//! The unfused path is safe by construction: Cody-Waite's whole point is that
//! `kf·LN2HI` is exactly representable (≤11-bit integer × 33-bit mantissa), so
//! losing the fuse there costs nothing; the magic-add round-to-nearest tolerates
//! the separately-rounded `x·LOG2E + RND_MAGIC` (a ±1 shift in `k` at ties keeps
//! `r` inside the poly domain); only the Horner steps double-round, ~1–2 extra
//! ULP composed (pinned by `unfused_kernel_within_3ulp_of_libm`).

use pulp::Simd;

// exp(r) Horner coefficients (ascending), exp on (-∞,0], degree 11.
const EXP_C: [f64; 12] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fe0000000000010),
    f64::from_bits(0x3fc55555555554a2),
    f64::from_bits(0x3fa555555554f370),
    f64::from_bits(0x3f81111111130dd6),
    f64::from_bits(0x3f56c16c1878111c),
    f64::from_bits(0x3f2a01a0110572b2),
    f64::from_bits(0x3efa01992d0fe736),
    f64::from_bits(0x3ec71df4520aaeeb),
    f64::from_bits(0x3e928b311c7eb84f),
    f64::from_bits(0x3e5ad661c903688b),
];
// log1p h(w) Horner coefficients (ascending), fdlibm form, degree 9. h(0)=2/3.
const LOG1P_H: [f64; 10] = [
    f64::from_bits(0x3fe5555555555555),
    f64::from_bits(0x3fd999999999a455),
    f64::from_bits(0x3fd24924923cd3a0),
    f64::from_bits(0x3fcc71c727660721),
    f64::from_bits(0x3fc745cefc3caf8b),
    f64::from_bits(0x3fc3b18cab0fef6e),
    f64::from_bits(0x3fc10ab0536ce75b),
    f64::from_bits(0x3fbebaa07b021d58),
    f64::from_bits(0x3fb67ff2751e342c),
    f64::from_bits(0x3fc4b8585fced69a),
];
const EXP_DEG: usize = EXP_C.len() - 1; // 11
const LOG1P_DEG: usize = LOG1P_H.len() - 1; // 9

const LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000); // ln2, low mantissa zeroed (Cody-Waite)
const LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76); // ln2 - LN2HI
const LOG2E: f64 = f64::from_bits(0x3ff71547652b82fe); // 1/ln2
                                                       // round-to-nearest-int: (y + RND_MAGIC) - RND_MAGIC for |y| < 2^51.
const RND_MAGIC: f64 = 1.5 * (1u64 << 52) as f64;
// low 52 bits of (kf + BIAS_MAGIC) equal (k + 1023).
const BIAS_MAGIC: f64 = (1u64 << 52) as f64 + 1023.0;
const MANT_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
const SHIFT52: u64 = 1u64 << 52;
// exp(-|η|) underflows below ~e⁻⁷⁴⁵; clamp the argument so the FP-magic 2^k build
// stays in-domain (k ≥ -1022). For |η| > 700, z ≈ 0 and the saturated p/w/lp match
// the old libm path to rounding — this only guards the (off-fit-path) extreme tail,
// where the libm code saturated sigmoid → {0,1} anyway.
const EXP_ARG_FLOOR: f64 = -700.0;
// Upper clamp twin for the full-domain entries (`exp_clamped`/`exp_fill`): the
// Cody-Waite reduction and the EXP_C polynomial are sign-agnostic (r ∈
// [−ln2/2, ln2/2] either way) and the FP-magic 2^k build holds for k+1023 ≤ 2046
// (x ≤ ~709) — 700 keeps symmetric headroom. exp(700) ≈ 1e304, finite; the
// generation-side caller (hsk lognormal multiplier) saturates astronomically
// rarely there, where libm would be approaching f64::MAX/inf anyway.
const EXP_ARG_CEIL: f64 = 700.0;

// fma policy: guaranteed-fused mul_add on native; plain mul/add on wasm32.
// wasm simd128 has no FMA instruction, so `mul_add_f64s` / `f64::mul_add` lower
// to the exact soft-float compiler-builtins libcall there — measured 9–41× on
// the GLM/GLMM wasm rows (~89 pts of the glmm_intercept profile). Dropping the
// fuse on wasm breaks native↔wasm byte-equality by design; the wasm bench gate
// is tier-2 |Δk| from this change on. `FUSED` is a const generic (not a bare
// cfg!) so the native test suite can instantiate and ULP-guard the wasm
// arithmetic; production entries pick `FUSED_DEFAULT` for the compile target.
pub(crate) const FUSED_DEFAULT: bool = cfg!(not(target_arch = "wasm32"));

#[inline(always)]
fn fmadd<S: Simd, const FUSED: bool>(simd: S, a: S::f64s, b: S::f64s, c: S::f64s) -> S::f64s {
    if FUSED {
        simd.mul_add_f64s(a, b, c)
    } else {
        simd.add_f64s(simd.mul_f64s(a, b), c)
    }
}

#[inline(always)]
fn fmadd_scalar<const FUSED: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[inline(always)]
fn simd_exp_reduced<S: Simd, const FUSED: bool>(simd: S, x: S::f64s) -> S::f64s {
    let t = fmadd::<S, FUSED>(simd, x, simd.splat_f64s(LOG2E), simd.splat_f64s(RND_MAGIC));
    let kf = simd.sub_f64s(t, simd.splat_f64s(RND_MAGIC));
    let neg_kf = simd.neg_f64s(kf);
    let hi = fmadd::<S, FUSED>(simd, neg_kf, simd.splat_f64s(LN2HI), x); // x - kf·ln2hi
    let r = fmadd::<S, FUSED>(simd, neg_kf, simd.splat_f64s(LN2LO), hi); // - kf·ln2lo
    let mut acc = simd.splat_f64s(EXP_C[EXP_DEG]);
    let mut j = EXP_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd::<S, FUSED>(simd, acc, r, simd.splat_f64s(EXP_C[j]));
    }
    let e = simd.add_f64s(kf, simd.splat_f64s(BIAS_MAGIC));
    let m = simd.and_u64s(simd.transmute_u64s_f64s(e), simd.splat_u64s(MANT_MASK));
    let pow2 = simd.transmute_f64s_u64s(simd.mul_u64s(m, simd.splat_u64s(SHIFT52)));
    simd.mul_f64s(acc, pow2)
}

#[inline(always)]
fn simd_log1p_unit<S: Simd, const FUSED: bool>(simd: S, z: S::f64s) -> S::f64s {
    let f = z;
    let hfsq = simd.mul_f64s(simd.splat_f64s(0.5), simd.mul_f64s(f, f));
    let s = simd.div_f64s(f, simd.add_f64s(simd.splat_f64s(2.0), f));
    let w = simd.mul_f64s(s, s);
    let mut acc = simd.splat_f64s(LOG1P_H[LOG1P_DEG]);
    let mut j = LOG1P_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd::<S, FUSED>(simd, acc, w, simd.splat_f64s(LOG1P_H[j]));
    }
    let rr = simd.mul_f64s(w, acc);
    let inner = simd.mul_f64s(s, simd.add_f64s(hfsq, rr));
    simd.sub_f64s(f, simd.sub_f64s(hfsq, inner))
}

// All fit-path helpers share `z = exp(-|η|)` (clamped) and the `η ≥ 0` sign mask.
#[inline(always)]
fn simd_z_mask<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> (S::f64s, S::m64s) {
    let neg_abs = simd.max_f64s(
        simd.neg_f64s(simd.abs_f64s(eta)),
        simd.splat_f64s(EXP_ARG_FLOOR),
    );
    let z = simd_exp_reduced::<S, FUSED>(simd, neg_abs);
    let mask = simd.greater_than_or_equal_f64s(eta, simd.splat_f64s(0.0));
    (z, mask)
}

/// Fused `(p, w, log1pexp(η))` for the GLM IRLS / GLMM PIRLS row-pass — one
/// `exp` and one `log1p` of the same argument shared across all three.
/// `p = sigmoid(η)` (branchless `glm::sigmoid_stable`),
/// `w = max(p(1-p), WEIGHT_CLAMP)`, `lp = log(1 + exp(η))` (stable in both tails).
#[inline(always)]
fn simd_fused<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> (S::f64s, S::f64s, S::f64s) {
    let one = simd.splat_f64s(1.0);
    let (z, mask) = simd_z_mask::<S, FUSED>(simd, eta);
    let l = simd_log1p_unit::<S, FUSED>(simd, z);
    let opz = simd.add_f64s(one, z);
    let p = simd.select_f64s(mask, simd.div_f64s(one, opz), simd.div_f64s(z, opz));
    let lp = simd.select_f64s(mask, simd.add_f64s(eta, l), l);
    let w = simd.max_f64s(
        simd.mul_f64s(p, simd.sub_f64s(one, p)),
        simd.splat_f64s(crate::glm::WEIGHT_CLAMP),
    );
    (p, w, lp)
}

// Scalar mirror of `simd_fused`, bit-identical op-for-op (same coeffs, FMA via
// `f64::mul_add`), used for the sub-lane tail so the whole row range runs one
// kernel regardless of where the SIMD chunk boundary falls.
#[inline]
fn scalar_exp_reduced<const FUSED: bool>(x: f64) -> f64 {
    let kf = fmadd_scalar::<FUSED>(x, LOG2E, RND_MAGIC) - RND_MAGIC;
    let hi = fmadd_scalar::<FUSED>(-kf, LN2HI, x);
    let r = fmadd_scalar::<FUSED>(-kf, LN2LO, hi);
    let mut acc = EXP_C[EXP_DEG];
    let mut j = EXP_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd_scalar::<FUSED>(acc, r, EXP_C[j]);
    }
    let m = (kf + BIAS_MAGIC).to_bits() & MANT_MASK;
    acc * f64::from_bits(m.wrapping_mul(SHIFT52))
}
#[inline]
fn scalar_log1p_unit<const FUSED: bool>(z: f64) -> f64 {
    let f = z;
    let hfsq = 0.5 * (f * f);
    let s = f / (2.0 + f);
    let w = s * s;
    let mut acc = LOG1P_H[LOG1P_DEG];
    let mut j = LOG1P_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd_scalar::<FUSED>(acc, w, LOG1P_H[j]);
    }
    let rr = w * acc;
    f - (hfsq - s * (hfsq + rr))
}
#[inline]
fn scalar_z<const FUSED: bool>(eta: f64) -> f64 {
    scalar_exp_reduced::<FUSED>((-eta.abs()).max(EXP_ARG_FLOOR))
}
#[inline]
fn scalar_fused<const FUSED: bool>(eta: f64) -> (f64, f64, f64) {
    let z = scalar_z::<FUSED>(eta);
    let l = scalar_log1p_unit::<FUSED>(z);
    let (p, lp) = if eta >= 0.0 {
        (1.0 / (1.0 + z), eta + l)
    } else {
        (z / (1.0 + z), l)
    };
    let w = (p * (1.0 - p)).max(crate::glm::WEIGHT_CLAMP);
    (p, w, lp)
}

struct PwLog1pexpOp<'a, const FUSED: bool> {
    eta: &'a [f64],
    p: &'a mut [f64],
    w: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for PwLog1pexpOp<'_, FUSED> {
    type Output = f64;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (eh, et) = S::as_simd_f64s(self.eta);
        let (ph, pt) = S::as_mut_simd_f64s(self.p);
        let (wh, wt) = S::as_mut_simd_f64s(self.w);
        let mut dsum = simd.splat_f64s(0.0);
        for i in 0..eh.len() {
            let (p, w, lp) = simd_fused::<S, FUSED>(simd, eh[i]);
            ph[i] = p;
            wh[i] = w;
            dsum = simd.add_f64s(dsum, lp);
        }
        let mut acc = simd.reduce_sum_f64s(dsum);
        for i in 0..et.len() {
            let (p, w, lp) = scalar_fused::<FUSED>(et[i]);
            pt[i] = p;
            wt[i] = w;
            acc += lp;
        }
        acc
    }
}

/// Fill `p[i] = sigmoid(η[i])` and `w[i] = max(p(1-p), WEIGHT_CLAMP)` for the
/// whole slice via the SIMD kernel; return `Σ log1pexp(η[i])` (lane-wise SIMD
/// reduction + scalar tail). The PIRLS deviance is `2·(Σ log1pexp − Σ y·η)`,
/// with the `Σ y·η` half accumulated by the caller's scalar η-pass.
pub(crate) fn pw_and_log1pexp_sum(eta: &[f64], p: &mut [f64], w: &mut [f64]) -> f64 {
    debug_assert_eq!(eta.len(), p.len());
    debug_assert_eq!(eta.len(), w.len());
    pulp::Arch::new().dispatch(PwLog1pexpOp::<{ FUSED_DEFAULT }> { eta, p, w })
}

struct SigmoidInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for SigmoidInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for x in head.iter_mut() {
            let (z, mask) = simd_z_mask::<S, FUSED>(simd, *x);
            let opz = simd.add_f64s(one, z);
            *x = simd.select_f64s(mask, simd.div_f64s(one, opz), simd.div_f64s(z, opz));
        }
        for x in tail.iter_mut() {
            let z = scalar_z::<FUSED>(*x);
            *x = if *x >= 0.0 {
                1.0 / (1.0 + z)
            } else {
                z / (1.0 + z)
            };
        }
    }
}

/// In-place `buf[i] = sigmoid(buf[i])` for the generation-side binary-outcome
/// draw — SIMD head + bit-identical scalar tail, p only (the fit path's fused
/// (p, w, Σlog1pexp) variant is `pw_and_log1pexp_sum`).
pub(crate) fn sigmoid_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(SigmoidInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

/// Owned scalar `exp` on (−∞, 0], platform-default fma policy, argument clamped
/// to the kernel's 2^k domain. The generation-side replacement for libm `.exp()`
/// at non-positive arguments (erfc's `exp(−x²)`); ≤1 ULP of libm.
pub(crate) fn exp_nonpos(x: f64) -> f64 {
    scalar_exp_reduced::<{ FUSED_DEFAULT }>(x.max(EXP_ARG_FLOOR))
}

/// Owned scalar `exp` on the full certified domain `[−700, 700]` (two-sided
/// clamp); ≤1 ULP of libm. Scalar tail twin of `exp_fill`.
pub(crate) fn exp_clamped(x: f64) -> f64 {
    scalar_exp_reduced::<{ FUSED_DEFAULT }>(x.clamp(EXP_ARG_FLOOR, EXP_ARG_CEIL))
}

struct ExpInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for ExpInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let lo = simd.splat_f64s(EXP_ARG_FLOOR);
        let hi = simd.splat_f64s(EXP_ARG_CEIL);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for x in head.iter_mut() {
            *x = simd_exp_reduced::<S, FUSED>(simd, simd.min_f64s(simd.max_f64s(*x, lo), hi));
        }
        for x in tail.iter_mut() {
            *x = exp_clamped(*x);
        }
    }
}

/// In-place `buf[i] = exp(buf[i])`, argument clamped to `[−700, 700]` — the
/// generation-side column pass for the heteroskedasticity lognormal multiplier.
/// SIMD head + bit-identical scalar tail (`exp_clamped`).
pub(crate) fn exp_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(ExpInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

// ln(u) on the censored-Φ domain: bit-trick range reduction reusing LOG1P_H —
// ln(u) = k·ln2 + log1p(m−1) with the fdlibm √2-normalization m ∈ [√2/2, √2)
// (k = round(log2 u)): a plain m ∈ [1,2) reduction cancels catastrophically as
// u → 1 (k=−1 against log1p(2u−1) → ln2, measured ~5000 ULP); √2-normalizing
// gives k=0 there and feeds log1p the small argument directly. m comes from
// forcing the exponent field to 1023 (AND mantissa, OR bits(1.0)), then a
// halving select for the m ≥ √2 half. LOG1P_H stays valid: its w = s² domain
// only shrinks (|s| ≤ 0.172 → w ≤ 0.0295 ⊂ [0, 1/9]). pulp 0.22 exposes no
// integer shift or u64→f64 convert, so k is recovered with an 11-step
// compare-ladder kf = −Σ_{j=1..11} [u < √2·2^−j] (exactly consistent with the
// m ≥ √2 select — the thresholds are exact 2^−j scalings of the same √2). The
// two-sided clamp makes the ladder total on the skewed-marginal call site:
// u ≤ 2^−11 → ln(2^−11) = −7.625, past the −EXP_CAP censor either way (the
// censored-Exp(1) cap ≈ 6.96), so the capped output matches libm's exactly;
// u = 1.0 (Φ saturates only at |z| ≳ 8.3, outside the generated ±6-SD latent
// range) → ln(1 − 2^−53) ≈ −1.1e-16 instead of −0.0.
const LN_U_FLOOR: f64 = 4.8828125e-4; // 2^-11
const LN_U_CEIL: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF); // 1 − 2^-53
const ONE_BITS: u64 = 0x3FF0_0000_0000_0000;

#[inline]
fn scalar_ln_unit<const FUSED: bool>(u: f64) -> f64 {
    let u = u.clamp(LN_U_FLOOR, LN_U_CEIL);
    let mut kf = 0.0f64;
    let mut th = std::f64::consts::SQRT_2 * 0.5;
    for _ in 0..11 {
        if u < th {
            kf -= 1.0;
        }
        th *= 0.5;
    }
    let m = f64::from_bits((u.to_bits() & MANT_MASK) | ONE_BITS);
    let m = if m < std::f64::consts::SQRT_2 {
        m
    } else {
        0.5 * m
    };
    let l = scalar_log1p_unit::<FUSED>(m - 1.0);
    fmadd_scalar::<FUSED>(kf, LN2HI, fmadd_scalar::<FUSED>(kf, LN2LO, l))
}

struct LnInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for LnInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for v in head.iter_mut() {
            let u = simd.min_f64s(
                simd.max_f64s(*v, simd.splat_f64s(LN_U_FLOOR)),
                simd.splat_f64s(LN_U_CEIL),
            );
            let mut kf = simd.splat_f64s(0.0);
            let mut th = std::f64::consts::SQRT_2 * 0.5;
            for _ in 0..11 {
                let mask = simd.less_than_f64s(u, simd.splat_f64s(th));
                kf = simd.select_f64s(mask, simd.sub_f64s(kf, one), kf);
                th *= 0.5;
            }
            let m = simd.transmute_f64s_u64s(simd.or_u64s(
                simd.and_u64s(simd.transmute_u64s_f64s(u), simd.splat_u64s(MANT_MASK)),
                simd.splat_u64s(ONE_BITS),
            ));
            let lt = simd.less_than_f64s(m, simd.splat_f64s(std::f64::consts::SQRT_2));
            let m = simd.select_f64s(lt, m, simd.mul_f64s(simd.splat_f64s(0.5), m));
            let l = simd_log1p_unit::<S, FUSED>(simd, simd.sub_f64s(m, one));
            let inner = fmadd::<S, FUSED>(simd, kf, simd.splat_f64s(LN2LO), l);
            *v = fmadd::<S, FUSED>(simd, kf, simd.splat_f64s(LN2HI), inner);
        }
        for v in tail.iter_mut() {
            *v = scalar_ln_unit::<FUSED>(*v);
        }
    }
}

/// Owned scalar `ln` on the censored-Φ domain (clamped to `[2^−11, 1−2^−53]`) —
/// the skewed-marginal replacement for libm `.ln()`. Scalar twin of `ln_fill`,
/// bit-identical per element.
pub(crate) fn ln_owned(u: f64) -> f64 {
    scalar_ln_unit::<{ FUSED_DEFAULT }>(u)
}

/// In-place `buf[i] = ln(buf[i])` on the censored-Φ domain — SIMD head +
/// bit-identical scalar tail (`ln_owned`).
pub(crate) fn ln_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(LnInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

// A&S 7.1.26 constants — mirror crate::distributions::erfc, change together.
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;
const ERF_P: f64 = 0.3275911;

struct PhiInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for PhiInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let half = simd.splat_f64s(0.5);
        let c = simd.splat_f64s(std::f64::consts::FRAC_1_SQRT_2);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for v in head.iter_mut() {
            let x = simd.mul_f64s(simd.neg_f64s(*v), c); // x = (−z)·(1/√2)
            let neg = simd.less_than_f64s(x, simd.splat_f64s(0.0));
            let ax = simd.abs_f64s(x);
            let t = simd.div_f64s(
                one,
                simd.add_f64s(one, simd.mul_f64s(simd.splat_f64s(ERF_P), ax)),
            );
            // (((((A5·t + A4)·t) + A3)·t + A2)·t + A1)·t — plain mul/add, as scalar.
            let mut poly = simd.add_f64s(
                simd.mul_f64s(simd.splat_f64s(ERF_A5), t),
                simd.splat_f64s(ERF_A4),
            );
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A3));
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A2));
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A1));
            poly = simd.mul_f64s(poly, t);
            let e = simd_exp_reduced::<S, FUSED>(
                simd,
                simd.max_f64s(
                    simd.mul_f64s(simd.neg_f64s(ax), ax),
                    simd.splat_f64s(EXP_ARG_FLOOR),
                ),
            );
            let y = simd.sub_f64s(one, simd.mul_f64s(poly, e));
            let erf = simd.select_f64s(neg, simd.neg_f64s(y), y);
            *v = simd.mul_f64s(half, simd.sub_f64s(one, erf));
        }
        for v in tail.iter_mut() {
            *v = crate::distributions::phi(*v);
        }
    }
}

/// In-place `buf[i] = Φ(buf[i])` — SIMD mirror of `distributions::phi`
/// (bit-identical per element; the scalar tail calls it directly).
pub(crate) fn phi_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(PhiInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference: system libm (std f64). The spike proved the primitives are ≤1
    // ULP of an MPFR oracle; in-repo we re-assert SIMD == scalar-libm accuracy
    // (exp/log1p ≤1 ULP, composed p ≤2 ULP) — the regression net for the coeffs.
    fn ulp(a: f64, b: f64) -> i128 {
        let o = |x: f64| {
            let b = x.to_bits() as i64;
            (if b < 0 { i64::MIN.wrapping_sub(b) } else { b }) as i128
        };
        (o(a) - o(b)).abs()
    }
    fn libm_fused(eta: f64) -> (f64, f64) {
        if eta >= 0.0 {
            let z = (-eta).exp();
            (1.0 / (1.0 + z), eta + z.ln_1p())
        } else {
            let z = eta.exp();
            (z / (1.0 + z), z.ln_1p())
        }
    }

    #[test]
    fn simd_kernel_within_1ulp_of_libm() {
        // dense grid straddling the η=0 seam and into both tails
        let n = 20_003usize; // not a multiple of any lane count -> exercises the SIMD tail too
        let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();

        // Per-element accuracy of the kernel formula. `scalar_fused` is bit-identical
        // op-for-op to the SIMD path, so this guards the coefficients directly. The
        // primitives are ≤1 ULP of the true value (proved offline vs an MPFR oracle);
        // in-repo we bound the kernel-vs-libm difference (both ≤1 ULP of truth → ≤2 apart).
        let (mut pmax, mut lpmax) = (0i128, 0i128);
        for &e in &eta {
            let (p, w, lp) = scalar_fused::<{ FUSED_DEFAULT }>(e);
            let (libp, liblp) = libm_fused(e);
            pmax = pmax.max(ulp(p, libp));
            lpmax = lpmax.max(ulp(lp, liblp));
            assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        }
        assert!(pmax <= 2, "sigmoid p drifted {pmax} ULP from libm");
        assert!(lpmax <= 2, "log1pexp drifted {lpmax} ULP from libm");

        // End-to-end SIMD dispatch path: p filled within the same band, and the
        // lane-reduced Σlog1pexp tracks the scalar Σ (the reorder moves only last bits).
        let mut p = vec![0.0; n];
        let mut w = vec![0.0; n];
        let lp_sum = pw_and_log1pexp_sum(&eta, &mut p, &mut w);
        let mut p_simd_max = 0i128;
        let mut ref_sum = 0.0;
        for i in 0..n {
            let (libp, liblp) = libm_fused(eta[i]);
            p_simd_max = p_simd_max.max(ulp(p[i], libp));
            ref_sum += liblp;
        }
        assert!(
            p_simd_max <= 2,
            "SIMD-path p drifted {p_simd_max} ULP from libm"
        );
        assert!(
            (lp_sum - ref_sum).abs() <= 1e-9 * ref_sum.abs().max(1.0),
            "Σlog1pexp drift {lp_sum} vs {ref_sum}"
        );
    }

    #[test]
    fn unfused_kernel_within_3ulp_of_libm() {
        // The wasm32 arithmetic (plain mul/add, no fma), instantiated on native.
        // Cody-Waite reduction is fma-free-safe by design; only the Horner steps
        // double-round. Measured 2026-06-11 on the 20,003-pt grid: p 2 ULP,
        // log1pexp 2 ULP, SIMD p 2 ULP — bound pinned at measured+1 = 3. If it
        // ever exceeds that, a reduction step has become fma-dependent and the
        // policy is wrong — investigate before shipping.
        let n = 20_003usize;
        let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();
        let (mut pmax, mut lpmax) = (0i128, 0i128);
        for &e in &eta {
            let (p, w, lp) = scalar_fused::<false>(e);
            let (libp, liblp) = libm_fused(e);
            pmax = pmax.max(ulp(p, libp));
            lpmax = lpmax.max(ulp(lp, liblp));
            assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        }
        assert!(pmax <= 3, "unfused sigmoid p drifted {pmax} ULP from libm");
        assert!(lpmax <= 3, "unfused log1pexp drifted {lpmax} ULP from libm");

        // End-to-end dispatch of the unfused op on native SIMD lanes.
        let mut p = vec![0.0; n];
        let mut w = vec![0.0; n];
        let lp_sum = pulp::Arch::new().dispatch(PwLog1pexpOp::<false> {
            eta: &eta,
            p: &mut p,
            w: &mut w,
        });
        let mut ref_sum = 0.0;
        let mut p_simd_max = 0i128;
        for i in 0..n {
            let (libp, liblp) = libm_fused(eta[i]);
            p_simd_max = p_simd_max.max(ulp(p[i], libp));
            ref_sum += liblp;
        }
        assert!(
            p_simd_max <= 3,
            "unfused SIMD p drifted {p_simd_max} ULP from libm"
        );
        assert!((lp_sum - ref_sum).abs() <= 1e-9 * ref_sum.abs().max(1.0));
    }

    #[test]
    fn sigmoid_fill_within_2ulp_of_libm() {
        // Same grid discipline as the fused kernel test; in-place column op.
        let n = 20_003usize;
        let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();
        let mut buf = eta.clone();
        sigmoid_fill(&mut buf);
        let mut pmax = 0i128;
        for i in 0..n {
            let (libp, _) = libm_fused(eta[i]);
            pmax = pmax.max(ulp(buf[i], libp));
            assert!(buf[i].is_finite() && (0.0..=1.0).contains(&buf[i]));
        }
        assert!(pmax <= 2, "sigmoid_fill drifted {pmax} ULP from libm");
    }

    #[test]
    fn exp_fill_within_1ulp_of_libm_full_domain() {
        // Full certified domain [−700, 700] — positive arguments exercise the
        // same Cody-Waite reduction (sign-agnostic) and the 2^k build up to
        // k+1023 = 2033. Measured 2026-06-11 over this 20,003-pt grid: scalar
        // and SIMD both ≤1 ULP of libm — pinned at measured (1).
        let n = 20_003usize;
        let xs: Vec<f64> = (0..n)
            .map(|k| -700.0 + 1400.0 * k as f64 / n as f64)
            .collect();
        let mut emax = 0i128;
        for &x in &xs {
            emax = emax.max(ulp(exp_clamped(x), x.exp()));
        }
        assert!(emax <= 1, "exp_clamped drifted {emax} ULP from libm");
        let mut buf = xs.clone();
        exp_fill(&mut buf);
        let mut smax = 0i128;
        for i in 0..n {
            smax = smax.max(ulp(buf[i], xs[i].exp()));
        }
        assert!(smax <= 1, "exp_fill drifted {smax} ULP from libm");
        // Clamp behaviour at the edges stays finite.
        let mut edge = vec![-1.0e9, 1.0e9];
        exp_fill(&mut edge);
        assert!(edge[0] > 0.0 && edge[1].is_finite());
    }

    #[test]
    fn ln_fill_within_2ulp_of_libm() {
        // The skewed-marginal live domain [e^−EXP_CAP, 1) ≈ [9.5e-4, 1) — the
        // compare-ladder covers k ∈ {−11…−1} and LOG1P_H runs on m−1 ∈ [0,1).
        // Measured 2026-06-11 over this 20,003-pt grid: scalar and SIMD both
        // ≤2 ULP of libm ln — pinned at measured.
        let n = 20_003usize;
        let lo = 9.5e-4f64;
        let us: Vec<f64> = (0..n)
            .map(|k| lo + (1.0 - lo) * k as f64 / n as f64)
            .collect();
        let mut smax = 0i128;
        for &u in &us {
            smax = smax.max(ulp(ln_owned(u), u.ln()));
        }
        assert!(smax <= 2, "ln_owned drifted {smax} ULP from libm");
        let mut buf = us.clone();
        ln_fill(&mut buf);
        let mut vmax = 0i128;
        for i in 0..n {
            vmax = vmax.max(ulp(buf[i], us[i].ln()));
        }
        assert!(vmax <= 2, "ln_fill drifted {vmax} ULP from libm");
        // Clamp totality: below-domain input lands past the −EXP_CAP censor;
        // u = 1.0 maps to ln(1 − 2^−53), not −0.0.
        assert!(ln_owned(1.0e-9) < -6.96);
        assert!(ln_owned(0.0) < -6.96);
        assert!(ln_owned(1.0) < 0.0 && ln_owned(1.0) > -3.0e-16);
    }

    #[test]
    fn phi_fill_bit_identical_to_scalar_phi() {
        // phi_fill mirrors crate::distributions::phi op-for-op (A&S poly in plain
        // mul/add — fma-policy-neutral — plus the shared owned exp), so head and
        // tail must agree to the bit, the simd_fused/scalar_fused discipline.
        let n = 20_003usize;
        let z: Vec<f64> = (0..n).map(|k| -9.0 + 18.0 * k as f64 / n as f64).collect();
        let mut buf = z.clone();
        phi_fill(&mut buf);
        for i in 0..n {
            assert_eq!(
                buf[i].to_bits(),
                crate::distributions::phi(z[i]).to_bits(),
                "phi_fill diverged from scalar phi at z={}",
                z[i]
            );
        }
    }

    #[test]
    fn weight_clamped_and_finite() {
        let eta: Vec<f64> = vec![-50.0, -10.0, -1e-9, 0.0, 1e-9, 10.0, 50.0, 1e3];
        let mut p = vec![0.0; eta.len()];
        let mut w = vec![0.0; eta.len()];
        pw_and_log1pexp_sum(&eta, &mut p, &mut w);
        for i in 0..eta.len() {
            assert!(p[i].is_finite() && (0.0..=1.0).contains(&p[i]));
            assert!(w[i] >= crate::glm::WEIGHT_CLAMP && w[i].is_finite());
        }
    }
}
